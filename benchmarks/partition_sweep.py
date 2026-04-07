"""Partition-count sweep: de-risking test for budget optimizer.

For 3-4 representative slides (large/medium/small), create test tables
with B_k = 1, 2, 5, 10, 20 sub-partitions and run Q1 viewport queries.
Validates that sub-partitioning helps at any budget level before building
the full optimizer.

Usage:
    python -m benchmarks.partition_sweep
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import psycopg2

from spdb import config, hilbert
from benchmarks.framework import (
    compute_stats, load_metadata, get_slide_dimensions,
    random_viewport, time_query,
)


def _get_slide_dims_from_db(conn, slide_id):
    """Fallback: compute slide dimensions from actual coordinate range."""
    cur = conn.cursor()
    cur.execute("""
        SELECT MAX(centroid_x), MAX(centroid_y)
        FROM objects_slide_only
        WHERE slide_id = %s
    """, (slide_id,))
    row = cur.fetchone()
    cur.close()
    if row and row[0] and row[1]:
        return float(row[0]) * 1.05, float(row[1]) * 1.05  # 5% padding
    return 100000.0, 100000.0


def safe_get_slide_dimensions(conn, metadata, slide_id):
    """Get slide dimensions from metadata, falling back to DB query."""
    try:
        return get_slide_dimensions(metadata, slide_id)
    except (KeyError, TypeError):
        return _get_slide_dims_from_db(conn, slide_id)

# Representative slides: large, medium, small
# Will be populated from command-line or auto-detected
BUCKET_COUNTS = [1, 2, 5, 10, 20]
N_TRIALS = 200
VIEWPORT_FRAC = 0.05
HILBERT_ORDER = config.HILBERT_ORDER


def pick_representative_slides(conn):
    """Select 3-4 representative slides by size: large, medium, small."""
    cur = conn.cursor()
    cur.execute("""
        SELECT slide_id, COUNT(*) as n
        FROM objects_slide_only
        GROUP BY slide_id
        ORDER BY n DESC
    """)
    rows = cur.fetchall()
    cur.close()

    if not rows:
        raise RuntimeError("No slides found")

    # Large: biggest slide
    large = rows[0]
    # Small: smallest slide
    small = rows[-1]
    # Medium: closest to median
    mid_idx = len(rows) // 2
    medium = rows[mid_idx]
    # Medium-large: ~75th percentile
    q75_idx = len(rows) // 4
    medium_large = rows[q75_idx]

    picks = [
        {"slide_id": large[0], "n_objects": large[1], "label": "large"},
        {"slide_id": medium_large[0], "n_objects": medium_large[1], "label": "medium-large"},
        {"slide_id": medium[0], "n_objects": medium[1], "label": "medium"},
        {"slide_id": small[0], "n_objects": small[1], "label": "small"},
    ]
    return picks


def create_sweep_table(conn, slide_id, B_k, source_table="objects_slide_only"):
    """Create a test table with B_k sub-partitions for one slide."""
    safe = slide_id.replace("-", "_")
    parent = f"sweep_{safe}_b{B_k}"
    total_cells = 1 << (2 * HILBERT_ORDER)

    cur = conn.cursor()

    # Drop if exists
    cur.execute(f"DROP TABLE IF EXISTS {parent} CASCADE;")
    conn.commit()

    if B_k == 1:
        # Single table (SO baseline equivalent)
        cur.execute(f"""
            CREATE TABLE {parent} (
                object_id       BIGINT,
                slide_id        TEXT,
                tile_id         TEXT,
                centroid_x      DOUBLE PRECISION,
                centroid_y      DOUBLE PRECISION,
                class_label     TEXT,
                hilbert_key     BIGINT,
                geom            GEOMETRY(Point, 0)
            );
        """)
        conn.commit()

        cur.execute(f"""
            INSERT INTO {parent}
            SELECT object_id, slide_id, tile_id, centroid_x, centroid_y,
                   class_label, hilbert_key, geom
            FROM {source_table}
            WHERE slide_id = %s;
        """, (slide_id,))
        conn.commit()

        cur.execute(f"CREATE INDEX ON {parent} USING gist (geom);")
        conn.commit()
    else:
        # Partitioned table with B_k RANGE partitions on hilbert_key
        cur.execute(f"""
            CREATE TABLE {parent} (
                object_id       BIGINT,
                slide_id        TEXT,
                tile_id         TEXT,
                centroid_x      DOUBLE PRECISION,
                centroid_y      DOUBLE PRECISION,
                class_label     TEXT,
                hilbert_key     BIGINT NOT NULL,
                geom            GEOMETRY(Point, 0)
            ) PARTITION BY RANGE (hilbert_key);
        """)
        conn.commit()

        for j in range(B_k):
            lo = j * total_cells // B_k
            hi = (j + 1) * total_cells // B_k
            child = f"{parent}_h{j}"
            cur.execute(f"""
                CREATE TABLE {child}
                PARTITION OF {parent}
                FOR VALUES FROM ({lo}) TO ({hi});
            """)
        conn.commit()

        # Bulk load
        cur.execute(f"""
            INSERT INTO {parent}
            SELECT object_id, slide_id, tile_id, centroid_x, centroid_y,
                   class_label, hilbert_key, geom
            FROM {source_table}
            WHERE slide_id = %s;
        """, (slide_id,))
        conn.commit()

        # GiST indexes on each child
        for j in range(B_k):
            child = f"{parent}_h{j}"
            cur.execute(f"CREATE INDEX ON {child} USING gist (geom);")
            conn.commit()

    cur.execute(f"ANALYZE {parent};")
    conn.commit()
    cur.close()
    return parent


def drop_sweep_table(conn, slide_id, B_k):
    """Drop a sweep test table."""
    safe = slide_id.replace("-", "_")
    parent = f"sweep_{safe}_b{B_k}"
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {parent} CASCADE;")
    conn.commit()
    cur.close()


def run_sweep_queries(conn, table_name, slide_id, metadata,
                      B_k, n_trials=N_TRIALS, viewport_frac=VIEWPORT_FRAC,
                      seed=42):
    """Run Q1 viewport queries on the sweep table."""
    rng = np.random.RandomState(seed)
    latencies = []

    w, h = safe_get_slide_dimensions(conn, metadata, slide_id)

    # Build query - for B_k > 1, include hilbert_key range predicate
    if B_k > 1:
        # Use Hilbert pruning
        for trial in range(n_trials):
            x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)
            n_obj = metadata["object_counts"].get(slide_id, 1_000_000)
            num_buckets = max(1, n_obj // config.BUCKET_TARGET)
            bucket_ids = hilbert.candidate_buckets_for_bbox(
                x0, y0, x1, y1, w, h, HILBERT_ORDER, B_k
            )
            # Convert bucket IDs to hilbert_key ranges
            total_cells = 1 << (2 * HILBERT_ORDER)
            ranges = []
            for b in sorted(bucket_ids):
                lo = b * total_cells // B_k
                hi = (b + 1) * total_cells // B_k
                if ranges and ranges[-1][1] == lo:
                    ranges[-1] = (ranges[-1][0], hi)
                else:
                    ranges.append((lo, hi))

            hk_clauses = " OR ".join(
                f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
                for lo, hi in ranges
            )
            safe = slide_id.replace("-", "_")
            table = f"sweep_{safe}_b{B_k}"
            sql = f"""
                SELECT object_id, centroid_x, centroid_y, class_label
                FROM {table}
                WHERE ({hk_clauses})
                  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
            """
            _, elapsed = time_query(conn, sql, (x0, y0, x1, y1))
            latencies.append(elapsed)
    else:
        # B_k=1: just spatial query, no hilbert pruning
        safe = slide_id.replace("-", "_")
        table = f"sweep_{safe}_b{B_k}"
        sql = f"""
            SELECT object_id, centroid_x, centroid_y, class_label
            FROM {table}
            WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
        """
        for trial in range(n_trials):
            x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)
            _, elapsed = time_query(conn, sql, (x0, y0, x1, y1))
            latencies.append(elapsed)

    return latencies


def run_partition_sweep():
    """Run the full partition-count sweep."""
    print("\n" + "=" * 60)
    print("Partition-Count Sweep (De-Risking)")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = False
    metadata = load_metadata()

    # Pick representative slides
    picks = pick_representative_slides(conn)
    print(f"\nRepresentative slides:")
    for p in picks:
        print(f"  {p['label']:<14} {p['slide_id']:<30} {p['n_objects']:>10,} objects")

    results = {}

    for slide_info in picks:
        sid = slide_info["slide_id"]
        label = slide_info["label"]
        n_obj = slide_info["n_objects"]
        print(f"\n--- {label}: {sid} ({n_obj:,} objects) ---")

        slide_results = {}

        for B_k in BUCKET_COUNTS:
            print(f"  B_k={B_k:>3}: ", end="", flush=True)

            t0 = time.time()
            table = create_sweep_table(conn, sid, B_k)
            setup_time = time.time() - t0
            print(f"setup={setup_time:.1f}s  ", end="", flush=True)

            # Warmup
            lats = run_sweep_queries(conn, table, sid, metadata,
                                     B_k, n_trials=10, seed=99)

            # Actual measurement
            lats = run_sweep_queries(conn, table, sid, metadata,
                                     B_k, n_trials=N_TRIALS, seed=42)
            stats = compute_stats(lats)

            print(f"p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  "
                  f"mean={stats['mean']:.1f}ms")

            slide_results[B_k] = {
                "p50": stats["p50"],
                "p95": stats["p95"],
                "mean": stats["mean"],
                "std": stats["std"],
                "setup_sec": round(setup_time, 1),
            }

            # Cleanup
            drop_sweep_table(conn, sid, B_k)

        results[sid] = {
            "label": label,
            "n_objects": n_obj,
            "bucket_results": slide_results,
        }

        # Summary for this slide
        baseline = slide_results[1]["p50"]
        print(f"\n  Speedup vs B_k=1 (p50={baseline:.1f}ms):")
        for B_k in BUCKET_COUNTS[1:]:
            p50 = slide_results[B_k]["p50"]
            speedup = baseline / p50 if p50 > 0 else 0
            marker = " <-- HELPS" if speedup > 1.05 else " (no benefit)" if speedup > 0.95 else " WORSE"
            print(f"    B_k={B_k:>3}: p50={p50:.1f}ms  speedup={speedup:.2f}x{marker}")

    # Save results
    out_dir = config.RAW_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "partition_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    conn.close()
    return results


if __name__ == "__main__":
    run_partition_sweep()
