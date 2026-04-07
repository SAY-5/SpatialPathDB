"""Write workload benchmark: insert throughput, update overhead, VACUUM impact.

Measures the write-side cost of maintaining HCCI indexes to address
the reviewer concern: "What happens at 10K inserts/second?"

Benchmarks:
1. Insert throughput: HCCI-enabled table vs GiST-only baseline
2. Composite key computation overhead (isolated)
3. Update cost: class label change requiring composite key recomputation
4. VACUUM impact on index-only scan availability

Usage:
    python -m benchmarks.write_benchmark
    python -m benchmarks.write_benchmark --batch-sizes 1000,10000,100000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from spdb import config, hilbert, hcci
from benchmarks.framework import compute_stats, save_results

HILBERT_ORDER = config.HILBERT_ORDER
COMPOSITE_SHIFT = hcci.COMPOSITE_SHIFT


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------

def _create_write_tables(conn):
    """Create two tables: one with HCCI index, one with GiST only."""
    with conn.cursor() as cur:
        # HCCI table (B-tree covering index)
        cur.execute("""
            DROP TABLE IF EXISTS write_bench_hcci CASCADE;
            CREATE TABLE write_bench_hcci (
                id BIGSERIAL PRIMARY KEY,
                slide_id TEXT NOT NULL,
                centroid_x DOUBLE PRECISION NOT NULL,
                centroid_y DOUBLE PRECISION NOT NULL,
                class_label TEXT NOT NULL,
                area DOUBLE PRECISION DEFAULT 0,
                hilbert_key BIGINT NOT NULL,
                composite_key BIGINT NOT NULL
            );
            CREATE INDEX idx_wb_hcci ON write_bench_hcci (slide_id, composite_key)
                INCLUDE (centroid_x, centroid_y, class_label, area);
        """)

        # GiST baseline table
        cur.execute("""
            DROP TABLE IF EXISTS write_bench_gist CASCADE;
            CREATE TABLE write_bench_gist (
                id BIGSERIAL PRIMARY KEY,
                slide_id TEXT NOT NULL,
                centroid_x DOUBLE PRECISION NOT NULL,
                centroid_y DOUBLE PRECISION NOT NULL,
                class_label TEXT NOT NULL,
                area DOUBLE PRECISION DEFAULT 0,
                geom geometry(Point, 0)
            );
            CREATE INDEX idx_wb_gist ON write_bench_gist USING gist (geom);
        """)
    conn.commit()


def _drop_write_tables(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS write_bench_hcci CASCADE;")
        cur.execute("DROP TABLE IF EXISTS write_bench_gist CASCADE;")
    conn.commit()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_batch(n: int, rng: np.random.RandomState,
                    slide_width: float = 100000.0,
                    slide_height: float = 100000.0) -> dict:
    """Generate n random spatial objects with class labels."""
    xs = rng.uniform(0, slide_width, n).astype(np.float64)
    ys = rng.uniform(0, slide_height, n).astype(np.float64)
    classes = rng.choice(
        list(config.CLASS_DISTRIBUTION.keys()),
        size=n,
        p=list(config.CLASS_DISTRIBUTION.values()),
    )
    areas = rng.uniform(10, 500, n).astype(np.float64)
    return {"xs": xs, "ys": ys, "classes": classes, "areas": areas}


def _compute_keys(xs, ys, classes, slide_width, slide_height, p=HILBERT_ORDER):
    """Compute hilbert_key and composite_key for a batch."""
    n_grid = 1 << p
    gxs = np.clip((xs / slide_width * n_grid).astype(np.int64), 0, n_grid - 1)
    gys = np.clip((ys / slide_height * n_grid).astype(np.int64), 0, n_grid - 1)
    h_keys = hilbert.encode_batch(gxs, gys, p)

    c_keys = np.empty(len(xs), dtype=np.int64)
    for i, cls in enumerate(classes):
        enum_val = hcci.class_to_enum(cls)
        c_keys[i] = (enum_val << COMPOSITE_SHIFT) | int(h_keys[i])

    return h_keys, c_keys


# ---------------------------------------------------------------------------
# Insert throughput benchmark
# ---------------------------------------------------------------------------

def bench_insert_throughput(
    conn,
    batch_sizes: list[int],
    n_repeats: int = 5,
    slide_width: float = 100000.0,
    slide_height: float = 100000.0,
    seed: int = config.RANDOM_SEED + 5000,
) -> dict:
    """Measure insert throughput for HCCI vs GiST tables."""
    rng = np.random.RandomState(seed)
    results = {}

    for batch_size in batch_sizes:
        print(f"\n  Insert batch_size={batch_size:,}")
        hcci_times = []
        gist_times = []
        key_compute_times = []

        for rep in range(n_repeats):
            data = _generate_batch(batch_size, rng, slide_width, slide_height)

            # Time composite key computation
            t0 = time.perf_counter()
            h_keys, c_keys = _compute_keys(
                data["xs"], data["ys"], data["classes"],
                slide_width, slide_height,
            )
            key_compute_ms = (time.perf_counter() - t0) * 1000
            key_compute_times.append(key_compute_ms)

            # Prepare HCCI rows
            hcci_rows = [
                ("slide_0", float(x), float(y), cls, float(a), int(hk), int(ck))
                for x, y, cls, a, hk, ck in zip(
                    data["xs"], data["ys"], data["classes"],
                    data["areas"], h_keys, c_keys,
                )
            ]

            # Prepare GiST rows
            gist_rows = [
                ("slide_0", float(x), float(y), cls, float(a),
                 f"SRID=0;POINT({float(x)} {float(y)})")
                for x, y, cls, a in zip(
                    data["xs"], data["ys"], data["classes"], data["areas"],
                )
            ]

            # Reset tables
            _create_write_tables(conn)

            # Insert into HCCI table
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO write_bench_hcci
                       (slide_id, centroid_x, centroid_y, class_label, area,
                        hilbert_key, composite_key)
                       VALUES %s""",
                    hcci_rows,
                    page_size=5000,
                )
            conn.commit()
            hcci_ms = (time.perf_counter() - t0) * 1000
            hcci_times.append(hcci_ms)

            # Insert into GiST table
            t0 = time.perf_counter()
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO write_bench_gist
                       (slide_id, centroid_x, centroid_y, class_label, area, geom)
                       VALUES %s""",
                    gist_rows,
                    template="(%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s))",
                    page_size=5000,
                )
            conn.commit()
            gist_ms = (time.perf_counter() - t0) * 1000
            gist_times.append(gist_ms)

            rows_per_sec_hcci = batch_size / (hcci_ms / 1000)
            rows_per_sec_gist = batch_size / (gist_ms / 1000)
            print(f"    Rep {rep+1}/{n_repeats}: "
                  f"HCCI={hcci_ms:.0f}ms ({rows_per_sec_hcci:,.0f} rows/s)  "
                  f"GiST={gist_ms:.0f}ms ({rows_per_sec_gist:,.0f} rows/s)  "
                  f"KeyComp={key_compute_ms:.1f}ms")

        results[str(batch_size)] = {
            "batch_size": batch_size,
            "n_repeats": n_repeats,
            "hcci_insert_ms": compute_stats(hcci_times),
            "gist_insert_ms": compute_stats(gist_times),
            "key_compute_ms": compute_stats(key_compute_times),
            "hcci_rows_per_sec": round(batch_size / (np.mean(hcci_times) / 1000)),
            "gist_rows_per_sec": round(batch_size / (np.mean(gist_times) / 1000)),
            "overhead_pct": round(
                (np.mean(hcci_times) / np.mean(gist_times) - 1) * 100, 1
            ),
        }

    return results


# ---------------------------------------------------------------------------
# Update overhead benchmark
# ---------------------------------------------------------------------------

def bench_update_overhead(
    conn,
    n_rows: int = 100000,
    n_updates: int = 1000,
    seed: int = config.RANDOM_SEED + 6000,
) -> dict:
    """Measure update cost: class label change requiring composite key recomputation."""
    rng = np.random.RandomState(seed)
    slide_width = slide_height = 100000.0

    print(f"\n  Update overhead: {n_rows:,} rows, {n_updates:,} updates")

    # Populate both tables
    _create_write_tables(conn)
    data = _generate_batch(n_rows, rng, slide_width, slide_height)
    h_keys, c_keys = _compute_keys(
        data["xs"], data["ys"], data["classes"], slide_width, slide_height,
    )

    hcci_rows = [
        ("slide_0", float(x), float(y), cls, float(a), int(hk), int(ck))
        for x, y, cls, a, hk, ck in zip(
            data["xs"], data["ys"], data["classes"],
            data["areas"], h_keys, c_keys,
        )
    ]
    gist_rows = [
        ("slide_0", float(x), float(y), cls, float(a),
         f"SRID=0;POINT({float(x)} {float(y)})")
        for x, y, cls, a in zip(
            data["xs"], data["ys"], data["classes"], data["areas"],
        )
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO write_bench_hcci
               (slide_id, centroid_x, centroid_y, class_label, area,
                hilbert_key, composite_key)
               VALUES %s""",
            hcci_rows, page_size=5000,
        )
        execute_values(
            cur,
            """INSERT INTO write_bench_gist
               (slide_id, centroid_x, centroid_y, class_label, area, geom)
               VALUES %s""",
            gist_rows,
            template="(%s, %s, %s, %s, %s, ST_GeomFromEWKT(%s))",
            page_size=5000,
        )
    conn.commit()

    # ANALYZE both tables
    with conn.cursor() as cur:
        cur.execute("ANALYZE write_bench_hcci;")
        cur.execute("ANALYZE write_bench_gist;")
    conn.commit()

    # Generate random updates: change class label for random rows
    update_ids = rng.randint(1, n_rows + 1, size=n_updates)
    new_classes = rng.choice(config.CLASS_LABELS, size=n_updates)

    # Time HCCI updates (must recompute composite_key)
    hcci_update_times = []
    for uid, new_cls in zip(update_ids, new_classes):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            # Get current hilbert_key
            cur.execute(
                "SELECT hilbert_key FROM write_bench_hcci WHERE id = %s",
                (int(uid),),
            )
            row = cur.fetchone()
            if row:
                hk = row[0]
                new_ck = (hcci.class_to_enum(new_cls) << COMPOSITE_SHIFT) | hk
                cur.execute(
                    "UPDATE write_bench_hcci SET class_label = %s, composite_key = %s WHERE id = %s",
                    (new_cls, new_ck, int(uid)),
                )
        conn.commit()
        hcci_update_times.append((time.perf_counter() - t0) * 1000)

    # Time GiST updates (just change class_label, no index impact)
    gist_update_times = []
    for uid, new_cls in zip(update_ids, new_classes):
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE write_bench_gist SET class_label = %s WHERE id = %s",
                (new_cls, int(uid)),
            )
        conn.commit()
        gist_update_times.append((time.perf_counter() - t0) * 1000)

    print(f"    HCCI update: p50={np.median(hcci_update_times):.2f}ms "
          f"mean={np.mean(hcci_update_times):.2f}ms")
    print(f"    GiST update: p50={np.median(gist_update_times):.2f}ms "
          f"mean={np.mean(gist_update_times):.2f}ms")

    return {
        "n_rows": n_rows,
        "n_updates": n_updates,
        "hcci_update_ms": compute_stats(hcci_update_times),
        "gist_update_ms": compute_stats(gist_update_times),
        "overhead_pct": round(
            (np.mean(hcci_update_times) / np.mean(gist_update_times) - 1) * 100, 1
        ),
    }


# ---------------------------------------------------------------------------
# VACUUM impact benchmark
# ---------------------------------------------------------------------------

def bench_vacuum_impact(
    conn,
    n_rows: int = 100000,
    n_updates_before_vacuum: int = 10000,
    seed: int = config.RANDOM_SEED + 7000,
) -> dict:
    """Measure VACUUM time and its impact on index-only scan availability."""
    rng = np.random.RandomState(seed)
    slide_width = slide_height = 100000.0

    print(f"\n  VACUUM impact: {n_rows:,} rows, {n_updates_before_vacuum:,} updates before VACUUM")

    _create_write_tables(conn)
    data = _generate_batch(n_rows, rng, slide_width, slide_height)
    h_keys, c_keys = _compute_keys(
        data["xs"], data["ys"], data["classes"], slide_width, slide_height,
    )

    hcci_rows = [
        ("slide_0", float(x), float(y), cls, float(a), int(hk), int(ck))
        for x, y, cls, a, hk, ck in zip(
            data["xs"], data["ys"], data["classes"],
            data["areas"], h_keys, c_keys,
        )
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO write_bench_hcci
               (slide_id, centroid_x, centroid_y, class_label, area,
                hilbert_key, composite_key)
               VALUES %s""",
            hcci_rows, page_size=5000,
        )
    conn.commit()

    # Initial VACUUM + ANALYZE to set baseline visibility map
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("VACUUM ANALYZE write_bench_hcci;")
    initial_vacuum_ms = (time.perf_counter() - t0) * 1000

    # Check heap fetches before updates (should be 0)
    with conn.cursor() as cur:
        tumor_enum = hcci.class_to_enum("Tumor")
        prefix = tumor_enum << COMPOSITE_SHIFT
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            f"SELECT centroid_x, centroid_y, class_label FROM write_bench_hcci "
            f"WHERE slide_id = 'slide_0' "
            f"AND composite_key >= {prefix} AND composite_key < {prefix + (1 << COMPOSITE_SHIFT)}",
        )
        plan_before = cur.fetchone()[0]

    # Do updates to dirty the visibility map
    update_ids = rng.randint(1, n_rows + 1, size=n_updates_before_vacuum)
    new_classes = rng.choice(config.CLASS_LABELS, size=n_updates_before_vacuum)
    for uid, new_cls in zip(update_ids, new_classes):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT hilbert_key FROM write_bench_hcci WHERE id = %s",
                (int(uid),),
            )
            row = cur.fetchone()
            if row:
                hk = row[0]
                new_ck = (hcci.class_to_enum(new_cls) << COMPOSITE_SHIFT) | hk
                cur.execute(
                    "UPDATE write_bench_hcci SET class_label = %s, composite_key = %s WHERE id = %s",
                    (new_cls, new_ck, int(uid)),
                )
    conn.commit()

    # Check heap fetches after updates (may be > 0 due to dirty visibility map)
    with conn.cursor() as cur:
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            f"SELECT centroid_x, centroid_y, class_label FROM write_bench_hcci "
            f"WHERE slide_id = 'slide_0' "
            f"AND composite_key >= {prefix} AND composite_key < {prefix + (1 << COMPOSITE_SHIFT)}",
        )
        plan_after_updates = cur.fetchone()[0]

    # VACUUM
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("VACUUM ANALYZE write_bench_hcci;")
    post_update_vacuum_ms = (time.perf_counter() - t0) * 1000

    # Check heap fetches after VACUUM (should be 0 again)
    with conn.cursor() as cur:
        cur.execute(
            f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            f"SELECT centroid_x, centroid_y, class_label FROM write_bench_hcci "
            f"WHERE slide_id = 'slide_0' "
            f"AND composite_key >= {prefix} AND composite_key < {prefix + (1 << COMPOSITE_SHIFT)}",
        )
        plan_after_vacuum = cur.fetchone()[0]

    # Extract heap fetches from plans
    def _extract_heap_fetches(plan_json):
        """Walk the JSON plan to find Heap Fetches."""
        if isinstance(plan_json, list):
            plan_json = plan_json[0]
        plan = plan_json.get("Plan", plan_json)
        heap_fetches = plan.get("Heap Fetches", 0)
        for child in plan.get("Plans", []):
            heap_fetches += _extract_heap_fetches({"Plan": child})
        return heap_fetches

    hf_before = _extract_heap_fetches(plan_before)
    hf_after_updates = _extract_heap_fetches(plan_after_updates)
    hf_after_vacuum = _extract_heap_fetches(plan_after_vacuum)

    print(f"    Initial VACUUM: {initial_vacuum_ms:.0f}ms")
    print(f"    Heap fetches before updates: {hf_before}")
    print(f"    Heap fetches after {n_updates_before_vacuum:,} updates: {hf_after_updates}")
    print(f"    Post-update VACUUM: {post_update_vacuum_ms:.0f}ms")
    print(f"    Heap fetches after VACUUM: {hf_after_vacuum}")

    return {
        "n_rows": n_rows,
        "n_updates": n_updates_before_vacuum,
        "initial_vacuum_ms": round(initial_vacuum_ms, 1),
        "post_update_vacuum_ms": round(post_update_vacuum_ms, 1),
        "heap_fetches_before": hf_before,
        "heap_fetches_after_updates": hf_after_updates,
        "heap_fetches_after_vacuum": hf_after_vacuum,
    }


# ---------------------------------------------------------------------------
# Composite key computation microbenchmark
# ---------------------------------------------------------------------------

def bench_key_computation(
    batch_sizes: list[int],
    n_repeats: int = 20,
    seed: int = config.RANDOM_SEED + 8000,
) -> dict:
    """Time composite key computation in isolation (no DB)."""
    rng = np.random.RandomState(seed)
    results = {}

    for n in batch_sizes:
        times = []
        for _ in range(n_repeats):
            xs = rng.uniform(0, 100000, n).astype(np.float64)
            ys = rng.uniform(0, 100000, n).astype(np.float64)
            classes = rng.choice(config.CLASS_LABELS, size=n)

            t0 = time.perf_counter()
            # Hilbert encoding (vectorized)
            n_grid = 1 << HILBERT_ORDER
            gxs = np.clip((xs / 100000 * n_grid).astype(np.int64), 0, n_grid - 1)
            gys = np.clip((ys / 100000 * n_grid).astype(np.int64), 0, n_grid - 1)
            h_keys = hilbert.encode_batch(gxs, gys, HILBERT_ORDER)

            # Composite key (vectorized)
            c_keys = np.empty(n, dtype=np.int64)
            for i, cls in enumerate(classes):
                c_keys[i] = (hcci.class_to_enum(cls) << COMPOSITE_SHIFT) | int(h_keys[i])

            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)

        throughput = n / (np.mean(times) / 1000)
        print(f"  Key computation n={n:>8,}: "
              f"mean={np.mean(times):.2f}ms  "
              f"({throughput:,.0f} keys/sec)")

        results[str(n)] = {
            "batch_size": n,
            "compute_ms": compute_stats(times),
            "keys_per_sec": round(throughput),
        }

    return results


# ---------------------------------------------------------------------------
# EXPLAIN OR vs UNION ALL (for paper figure)
# ---------------------------------------------------------------------------

def capture_explain_or_vs_union(conn) -> dict:
    """Capture EXPLAIN output for OR vs UNION ALL to demonstrate planner difference."""
    print("\n  Capturing OR vs UNION ALL EXPLAIN plans...")

    _create_write_tables(conn)

    # Insert some data
    rng = np.random.RandomState(42)
    data = _generate_batch(50000, rng, 100000.0, 100000.0)
    h_keys, c_keys = _compute_keys(
        data["xs"], data["ys"], data["classes"], 100000.0, 100000.0,
    )
    rows = [
        ("slide_0", float(x), float(y), cls, float(a), int(hk), int(ck))
        for x, y, cls, a, hk, ck in zip(
            data["xs"], data["ys"], data["classes"],
            data["areas"], h_keys, c_keys,
        )
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO write_bench_hcci
               (slide_id, centroid_x, centroid_y, class_label, area,
                hilbert_key, composite_key)
               VALUES %s""",
            rows, page_size=5000,
        )
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("VACUUM ANALYZE write_bench_hcci;")
    conn.commit()

    # Two ranges for Tumor class
    tumor_enum = hcci.class_to_enum("Tumor")
    prefix = tumor_enum << COMPOSITE_SHIFT
    r1_lo, r1_hi = prefix + 100, prefix + 200
    r2_lo, r2_hi = prefix + 500, prefix + 600

    # OR query
    or_sql = (
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) "
        f"SELECT centroid_x, centroid_y, class_label "
        f"FROM write_bench_hcci "
        f"WHERE slide_id = 'slide_0' "
        f"AND ((composite_key >= {r1_lo} AND composite_key < {r1_hi}) "
        f"  OR (composite_key >= {r2_lo} AND composite_key < {r2_hi}))"
    )

    # UNION ALL query
    union_sql = (
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) "
        f"SELECT centroid_x, centroid_y, class_label "
        f"FROM write_bench_hcci "
        f"WHERE slide_id = 'slide_0' "
        f"AND composite_key >= {r1_lo} AND composite_key < {r1_hi} "
        f"UNION ALL "
        f"SELECT centroid_x, centroid_y, class_label "
        f"FROM write_bench_hcci "
        f"WHERE slide_id = 'slide_0' "
        f"AND composite_key >= {r2_lo} AND composite_key < {r2_hi}"
    )

    with conn.cursor() as cur:
        cur.execute(or_sql)
        or_plan = "\n".join(row[0] for row in cur.fetchall())

        cur.execute(union_sql)
        union_plan = "\n".join(row[0] for row in cur.fetchall())

    print(f"\n  === OR Plan ===\n{or_plan}")
    print(f"\n  === UNION ALL Plan ===\n{union_plan}")

    return {
        "or_plan": or_plan,
        "union_all_plan": union_plan,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HCCI Write Workload Benchmark")
    parser.add_argument("--batch-sizes", type=str, default="1000,10000,100000",
                        help="Comma-separated batch sizes for insert test")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--update-rows", type=int, default=100000)
    parser.add_argument("--update-count", type=int, default=1000)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print("=" * 60)
    print("  HCCI Write Workload Benchmark")
    print(f"  Batch sizes: {batch_sizes}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True

    t_start = time.time()

    # 1. Key computation microbenchmark (no DB needed)
    print("\n--- Key Computation Microbenchmark ---")
    key_results = bench_key_computation(
        [1000, 10000, 100000, 1000000],
        n_repeats=20,
    )

    # 2. Insert throughput
    print("\n--- Insert Throughput ---")
    insert_results = bench_insert_throughput(
        conn, batch_sizes, n_repeats=args.repeats,
    )

    # 3. Update overhead
    print("\n--- Update Overhead ---")
    update_results = bench_update_overhead(
        conn, n_rows=args.update_rows, n_updates=args.update_count,
    )

    # 4. VACUUM impact
    print("\n--- VACUUM Impact ---")
    vacuum_results = bench_vacuum_impact(
        conn, n_rows=100000, n_updates_before_vacuum=10000,
    )

    # 5. OR vs UNION ALL EXPLAIN
    print("\n--- OR vs UNION ALL EXPLAIN ---")
    explain_results = capture_explain_or_vs_union(conn)

    total = time.time() - t_start

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_time_sec": round(total, 1),
        "key_computation": key_results,
        "insert_throughput": insert_results,
        "update_overhead": update_results,
        "vacuum_impact": vacuum_results,
        "explain_or_vs_union": explain_results,
    }

    path = save_results(all_results, "write_benchmark")
    print(f"\n{'='*60}")
    print(f"  Total time: {total:.0f}s")
    print(f"  Results saved to {path}")
    print(f"{'='*60}")

    _drop_write_tables(conn)
    conn.close()


if __name__ == "__main__":
    main()
