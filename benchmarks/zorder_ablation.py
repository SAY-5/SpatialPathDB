"""Z-order (Morton) ablation: compare Hilbert vs Z-order encoding for HCCI.

This benchmark creates a temporary Z-order composite key column, builds
a covering index on it, and runs the same viewport queries with both
Hilbert and Z-order encodings. The purpose is to validate the Hilbert
design choice by showing fewer ranges and/or lower latency.

Usage:
    python -m benchmarks.zorder_ablation
    python -m benchmarks.zorder_ablation --trials 200 --dataset pathology
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import psycopg2

from spdb import config, hcci, hilbert
from benchmarks.framework import compute_stats, time_query, wilcoxon_ranksum

TABLE = config.TABLE_SLIDE_ONLY

# ---------------------------------------------------------------------------
# Z-order (Morton) encoding
# ---------------------------------------------------------------------------

def morton_encode_batch(xs: np.ndarray, ys: np.ndarray, p: int) -> np.ndarray:
    """Vectorized Morton (Z-order) encoding: interleave bits of x and y."""
    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    d = np.zeros(len(xs), dtype=np.int64)
    for i in range(p):
        mask = np.int64(1 << i)
        d |= ((xs & mask) << np.int64(i)) | ((ys & mask) << np.int64(i + 1))
    return d


def morton_ranges_direct(
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    p: int,
    max_ranges: int = 64,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> List[Tuple[int, int]]:
    """Compute Z-order key ranges for a viewport (same logic as hilbert_ranges_direct)."""
    n = 1 << p
    gx0 = max(0, int((x0 - x_origin) * n / slide_width))
    gx1 = min(n - 1, int((x1 - x_origin) * n / slide_width))
    gy0 = max(0, int((y0 - y_origin) * n / slide_height))
    gy1 = min(n - 1, int((y1 - y_origin) * n / slide_height))

    if gx0 > gx1 or gy0 > gy1:
        return [(0, 1 << (2 * p))]

    gxs = np.arange(gx0, gx1 + 1, dtype=np.int64)
    gys = np.arange(gy0, gy1 + 1, dtype=np.int64)
    gx_mesh, gy_mesh = np.meshgrid(gxs, gys)
    gx_flat = gx_mesh.ravel()
    gy_flat = gy_mesh.ravel()

    # Morton encoding instead of Hilbert
    m_indices = morton_encode_batch(gx_flat, gy_flat, p)
    m_sorted = np.sort(m_indices)

    if len(m_sorted) == 0:
        return [(0, 1 << (2 * p))]

    diffs = np.diff(m_sorted)
    gap_idx = np.where(diffs > 1)[0]

    starts = np.empty(len(gap_idx) + 1, dtype=np.int64)
    ends = np.empty(len(gap_idx) + 1, dtype=np.int64)
    starts[0] = m_sorted[0]
    ends[-1] = m_sorted[-1] + 1
    if len(gap_idx) > 0:
        ends[:-1] = m_sorted[gap_idx] + 1
        starts[1:] = m_sorted[gap_idx + 1]

    ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]

    while len(ranges) > max_ranges:
        gaps = [(ranges[i + 1][0] - ranges[i][1], i) for i in range(len(ranges) - 1)]
        gaps.sort()
        _, merge_i = gaps[0]
        merged = (ranges[merge_i][0], ranges[merge_i + 1][1])
        ranges = ranges[:merge_i] + [merged] + ranges[merge_i + 2:]

    return ranges


# ---------------------------------------------------------------------------
# Query builders (use existing HCCI composite_key column for Hilbert,
# and zorder_composite_key column for Z-order)
# ---------------------------------------------------------------------------

def build_sfc_query(
    table: str,
    slide_id: str,
    class_label: str,
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    p: int,
    key_column: str,
    range_func,
    class_enum: dict = None,
) -> Tuple[str, tuple, int]:
    """Build UNION ALL range scan query for a given SFC encoding."""
    if class_enum is None:
        class_enum = hcci.CLASS_ENUM

    enum_val = class_enum.get(class_label, 0)
    prefix = enum_val << hcci.COMPOSITE_SHIFT

    ranges = range_func(
        x0, y0, x1, y1,
        slide_width, slide_height,
        p, max_ranges=64,
    )

    branches = []
    params = []
    for lo, hi in ranges:
        ck_lo = prefix | lo
        ck_hi = prefix | (hi - 1)  # inclusive
        branches.append(
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table} "
            f"WHERE slide_id = %s AND {key_column} >= %s AND {key_column} <= %s"
        )
        params.extend([slide_id, ck_lo, ck_hi])

    sql = " UNION ALL ".join(branches)
    return sql, tuple(params), len(ranges)


# ---------------------------------------------------------------------------
# Slide dimension helpers
# ---------------------------------------------------------------------------

_dims_cache = {}

def get_dims(conn, slide_id: str, metadata: dict = None) -> Tuple[float, float]:
    if slide_id not in _dims_cache:
        if metadata and slide_id in metadata.get("metas", {}):
            m = metadata["metas"][slide_id]
            _dims_cache[slide_id] = (float(m["image_width"]), float(m["image_height"]))
        else:
            with conn.cursor() as cur:
                cur.execute(f"SELECT MAX(centroid_x), MAX(centroid_y) FROM {TABLE} WHERE slide_id = %s", (slide_id,))
                row = cur.fetchone()
            if row and row[0] and row[1]:
                _dims_cache[slide_id] = (float(row[0]) * 1.05, float(row[1]) * 1.05)
            else:
                _dims_cache[slide_id] = (100000.0, 100000.0)
    return _dims_cache[slide_id]


# ---------------------------------------------------------------------------
# Z-order column setup
# ---------------------------------------------------------------------------

def setup_zorder_column(conn):
    """Add zorder_composite_key column and covering index if not exists."""
    with conn.cursor() as cur:
        # Check if column exists
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'zorder_composite_key'
        """, (TABLE,))
        if cur.fetchone():
            print("  zorder_composite_key column already exists")
            # Check if index exists
            cur.execute("""
                SELECT 1 FROM pg_indexes
                WHERE tablename = %s AND indexname = 'idx_zorder_covering'
            """, (TABLE,))
            if cur.fetchone():
                print("  idx_zorder_covering index already exists")
                return
            else:
                print("  Creating idx_zorder_covering index...")
        else:
            print("  Adding zorder_composite_key column...")
            cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN zorder_composite_key BIGINT")
            conn.commit()

            # Populate: read composite_key, decode class enum, re-encode with Morton
            print("  Populating zorder_composite_key (this may take a while)...")
            # We can compute Z-order key from the existing grid coords
            # Strategy: use a PL/pgSQL function or batch update from Python
            # For simplicity, do a batch UPDATE using the existing centroid coords
            _populate_zorder_keys(conn)

        # Create covering index
        print("  Creating idx_zorder_covering index...")
        cur.execute(f"""
            CREATE INDEX idx_zorder_covering ON {TABLE}
            (slide_id, zorder_composite_key)
            INCLUDE (centroid_x, centroid_y, class_label, area)
        """)
        conn.commit()
        print("  ANALYZE...")
        cur.execute(f"ANALYZE {TABLE}")
        conn.commit()
        print("  Done.")


def _populate_zorder_keys(conn):
    """Batch-populate zorder_composite_key from centroid coords + class_label."""
    p = config.HILBERT_ORDER  # same grid order

    with conn.cursor() as cur:
        # Get all distinct slide_ids
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        slides = [r[0] for r in cur.fetchall()]

    metadata_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    total_updated = 0
    for i, sid in enumerate(slides):
        w, h = get_dims(conn, sid, metadata)
        n = 1 << p

        # Batch update using SQL expression
        # Morton key = interleave bits of grid_x, grid_y
        # We'll compute in Python for correctness, then batch update
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, centroid_x, centroid_y, class_label
                FROM {TABLE}
                WHERE slide_id = %s
            """, (sid,))
            rows = cur.fetchall()

        if not rows:
            continue

        ids = np.array([r[0] for r in rows], dtype=np.int64)
        xs = np.array([float(r[1]) for r in rows], dtype=np.float64)
        ys = np.array([float(r[2]) for r in rows], dtype=np.float64)
        classes = [r[3] for r in rows]

        # Grid coords
        gx = np.clip((xs * n / w).astype(np.int64), 0, n - 1)
        gy = np.clip((ys * n / h).astype(np.int64), 0, n - 1)

        # Morton encode
        m_keys = morton_encode_batch(gx, gy, p)

        # Composite key: (class_enum << 48) | morton_key
        class_enums = np.array([hcci.CLASS_ENUM.get(c, 0) for c in classes], dtype=np.int64)
        composite = (class_enums << hcci.COMPOSITE_SHIFT) | m_keys

        # Batch update
        from psycopg2.extras import execute_values
        values = [(int(ck), int(oid)) for ck, oid in zip(composite, ids)]

        with conn.cursor() as cur:
            # Use temp table for bulk update
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _zorder_tmp (ck BIGINT, oid BIGINT)")
            cur.execute("TRUNCATE _zorder_tmp")
            execute_values(cur, "INSERT INTO _zorder_tmp (ck, oid) VALUES %s", values, page_size=10000)
            cur.execute(f"""
                UPDATE {TABLE} t
                SET zorder_composite_key = z.ck
                FROM _zorder_tmp z
                WHERE t.id = z.oid
            """)
            updated = cur.rowcount
            total_updated += updated
        conn.commit()

        if (i + 1) % 10 == 0 or i == len(slides) - 1:
            print(f"    Slide {i+1}/{len(slides)}: {total_updated:,} rows updated")

    print(f"  Total updated: {total_updated:,}")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_ablation(
    conn,
    metadata: dict,
    slides: List[str],
    n_trials: int = 200,
    viewport_frac: float = 0.05,
    class_label: str = "Tumor",
    seed: int = 42,
) -> dict:
    """Run Hilbert vs Z-order comparison on the same viewport queries."""
    rng = np.random.RandomState(seed)
    p = config.HILBERT_ORDER

    lats_hilbert = []
    lats_zorder = []
    ranges_hilbert = []
    ranges_zorder = []
    fp_hilbert = []
    fp_zorder = []

    print(f"\n  Running {n_trials} trials: Hilbert vs Z-order on class={class_label}, VP={viewport_frac}")

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # Hilbert query (uses existing composite_key column)
        h_sql, h_params, h_n = build_sfc_query(
            TABLE, sid, class_label,
            x0, y0, x1, y1, w, h, p,
            key_column="composite_key",
            range_func=hcci.hilbert_ranges_direct,
        )
        h_rows, t_h = time_query(conn, h_sql, h_params)
        lats_hilbert.append(t_h)
        ranges_hilbert.append(h_n)

        # Z-order query (uses zorder_composite_key column)
        z_sql, z_params, z_n = build_sfc_query(
            TABLE, sid, class_label,
            x0, y0, x1, y1, w, h, p,
            key_column="zorder_composite_key",
            range_func=morton_ranges_direct,
        )
        z_rows, t_z = time_query(conn, z_sql, z_params)
        lats_zorder.append(t_z)
        ranges_zorder.append(z_n)

        # Compute exact result for FP
        exact_sql = (
            f"SELECT centroid_x, centroid_y FROM {TABLE} "
            f"WHERE slide_id = %s AND class_label = %s "
            f"AND centroid_x >= %s AND centroid_x <= %s "
            f"AND centroid_y >= %s AND centroid_y <= %s"
        )
        with conn.cursor() as cur:
            cur.execute(exact_sql, (sid, class_label, x0, x1, y0, y1))
            exact_count = cur.rowcount

        if len(h_rows) > 0 and exact_count > 0:
            fp_hilbert.append((len(h_rows) - exact_count) / len(h_rows))
        else:
            fp_hilbert.append(0.0)

        if len(z_rows) > 0 and exact_count > 0:
            fp_zorder.append((len(z_rows) - exact_count) / len(z_rows))
        else:
            fp_zorder.append(0.0)

        if (trial + 1) % 50 == 0:
            print(f"    Trial {trial+1}/{n_trials}: "
                  f"Hilbert p50={np.median(lats_hilbert):.2f}ms ({np.mean(ranges_hilbert):.0f} ranges)  "
                  f"Z-order p50={np.median(lats_zorder):.2f}ms ({np.mean(ranges_zorder):.0f} ranges)")

    stats_h = compute_stats(lats_hilbert)
    stats_z = compute_stats(lats_zorder)
    _, p_val = wilcoxon_ranksum(lats_hilbert, lats_zorder)

    results = {
        "n_trials": n_trials,
        "class_label": class_label,
        "viewport_frac": viewport_frac,
        "hilbert_order": p,
        "hilbert": {
            "latency": stats_h,
            "avg_ranges": round(float(np.mean(ranges_hilbert)), 1),
            "max_ranges": int(np.max(ranges_hilbert)),
            "fp_mean": round(float(np.mean(fp_hilbert)), 4),
            "fp_p95": round(float(np.percentile(fp_hilbert, 95)), 4),
        },
        "zorder": {
            "latency": stats_z,
            "avg_ranges": round(float(np.mean(ranges_zorder)), 1),
            "max_ranges": int(np.max(ranges_zorder)),
            "fp_mean": round(float(np.mean(fp_zorder)), 4),
            "fp_p95": round(float(np.percentile(fp_zorder, 95)), 4),
        },
        "hilbert_vs_zorder_p50": round(stats_z["p50"] / stats_h["p50"], 3) if stats_h["p50"] > 0 else 0,
        "hilbert_vs_zorder_mean": round(stats_z["mean"] / stats_h["mean"], 3) if stats_h["mean"] > 0 else 0,
        "wilcoxon_p": p_val,
    }

    print(f"\n  Results:")
    print(f"  {'':>12} {'p50':>8} {'mean':>8} {'p95':>8} {'ranges':>8} {'FP%':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Hilbert':>12} {stats_h['p50']:>8.2f} {stats_h['mean']:>8.2f} {stats_h['p95']:>8.1f} "
          f"{np.mean(ranges_hilbert):>8.1f} {np.mean(fp_hilbert)*100:>7.1f}%")
    print(f"  {'Z-order':>12} {stats_z['p50']:>8.2f} {stats_z['mean']:>8.2f} {stats_z['p95']:>8.1f} "
          f"{np.mean(ranges_zorder):>8.1f} {np.mean(fp_zorder)*100:>7.1f}%")
    print(f"\n  Z-order / Hilbert p50: {results['hilbert_vs_zorder_p50']:.3f}x")
    print(f"  Wilcoxon p-value: {p_val:.2e}")
    print(f"  Hilbert produces {np.mean(ranges_hilbert)/np.mean(ranges_zorder)*100 - 100:.0f}% "
          f"{'more' if np.mean(ranges_hilbert) > np.mean(ranges_zorder) else 'fewer'} ranges than Z-order"
          if np.mean(ranges_zorder) > 0 else "")

    return results


def main():
    parser = argparse.ArgumentParser(description="Z-order ablation: Hilbert vs Morton")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    parser.add_argument("--class-label", type=str, default="Tumor")
    parser.add_argument("--setup-only", action="store_true", help="Only create Z-order column/index")
    parser.add_argument("--skip-setup", action="store_true", help="Skip Z-order column setup")
    args = parser.parse_args()

    print("=" * 60)
    print("  Z-order Ablation: Hilbert vs Morton Encoding")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True

    # Load metadata
    metadata_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    if not args.skip_setup:
        setup_zorder_column(conn)
        if args.setup_only:
            conn.close()
            return

    conn.autocommit = False

    # Get slides
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        slides = [r[0] for r in cur.fetchall()]
    print(f"  Found {len(slides)} slides")

    # Pre-cache dims
    for sid in slides:
        get_dims(conn, sid, metadata)

    # Warmup
    print("  Warming up...")
    for _ in range(10):
        sid = slides[0]
        w, h = get_dims(conn, sid, metadata)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {TABLE} WHERE slide_id = %s "
                f"AND composite_key >= 0 AND composite_key <= 1000 LIMIT 10",
                (sid,),
            )
            cur.fetchall()
            cur.execute(
                f"SELECT * FROM {TABLE} WHERE slide_id = %s "
                f"AND zorder_composite_key >= 0 AND zorder_composite_key <= 1000 LIMIT 10",
                (sid,),
            )
            cur.fetchall()

    t0 = time.time()
    results = run_ablation(
        conn, metadata, slides,
        n_trials=args.trials,
        viewport_frac=args.viewport_frac,
        class_label=args.class_label,
    )
    total = time.time() - t0

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    results["total_time_sec"] = round(total, 1)

    path = os.path.join(config.RESULTS_DIR, "raw", "zorder_ablation.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {path}")
    print(f"  Total time: {total:.0f}s")

    conn.close()


if __name__ == "__main__":
    main()
