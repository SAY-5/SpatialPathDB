"""NYC Taxi HCCI Benchmark: viewport queries comparing HCCI vs GiST on taxi data.

Runs class-filtered viewport queries on the NYC taxi dataset to validate
HCCI performance on a well-known 30M+ row geospatial dataset with natural
categorical filters (payment_type).

Usage:
    python -m benchmarks.taxi_hcci_benchmark
    python -m benchmarks.taxi_hcci_benchmark --trials 500 --viewport-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci
from benchmarks.framework import compute_stats, save_results, parse_buffers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE = "taxi_trips"
DATASET_ID = "nyc_taxi"
HCCI_INDEX = "idx_taxi_hcci_covering"


def load_taxi_metadata():
    """Load taxi dataset metadata."""
    meta_path = os.path.join(config.RAW_DIR, "taxi_metadata.json")
    with open(meta_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_hcci_query(class_label: str, class_enum: dict, bounds: dict,
                     viewport: tuple, hilbert_order: int) -> tuple[str, int]:
    """Build UNION ALL HCCI query for a viewport + class filter."""
    x0, y0, x1, y1 = viewport
    width = bounds["width"]
    height = bounds["height"]
    x_min = bounds["x_min"]
    y_min = bounds["y_min"]

    enum_val = class_enum[class_label]
    ranges = hcci.hilbert_ranges_direct(
        x0, y0, x1, y1,
        width, height,
        hilbert_order, max_ranges=64,
        x_origin=x_min, y_origin=y_min,
    )

    prefix = enum_val << hcci.COMPOSITE_SHIFT
    branches = []
    for rlo, rhi in ranges:
        ck_lo = prefix | rlo
        ck_hi = prefix | rhi
        branches.append(
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {TABLE} "
            f"WHERE dataset_id = '{DATASET_ID}' "
            f"AND composite_key >= {ck_lo} AND composite_key <= {ck_hi}"
        )

    sql = " UNION ALL ".join(branches)
    return sql, len(ranges)


def build_gist_query(class_label: str, bounds: dict, viewport: tuple) -> str:
    """Build GiST + class filter query."""
    x0, y0, x1, y1 = viewport
    return (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE class_label = '{class_label}' "
        f"AND geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 4326)"
    )


def build_exact_query(class_label: str, viewport: tuple) -> str:
    """Build exact (no FP) query for FP measurement."""
    x0, y0, x1, y1 = viewport
    return (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE dataset_id = '{DATASET_ID}' "
        f"AND class_label = '{class_label}' "
        f"AND centroid_x >= {x0} AND centroid_x <= {x1} "
        f"AND centroid_y >= {y0} AND centroid_y <= {y1}"
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    conn,
    meta: dict,
    n_trials: int = 200,
    viewport_frac: float = 0.05,
    seed: int = config.RANDOM_SEED + 9000,
) -> dict:
    """Run HCCI vs GiST viewport benchmark on taxi data."""
    bounds = meta["bounds"]
    class_enum = meta["class_enum"]
    hilbert_order = meta["hilbert_order"]

    x_span = bounds["x_max"] - bounds["x_min"]
    y_span = bounds["y_max"] - bounds["y_min"]
    vw = x_span * np.sqrt(viewport_frac)
    vh = y_span * np.sqrt(viewport_frac)

    # Get class labels sorted by frequency
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) FROM {TABLE}
            GROUP BY class_label ORDER BY COUNT(*) DESC
        """)
        class_counts = dict(cur.fetchall())

    class_labels = list(class_counts.keys())
    print(f"\n  Classes: {class_labels}")
    print(f"  Viewport fraction: {viewport_frac} ({vw:.4f} x {vh:.4f} degrees)")

    rng = np.random.RandomState(seed)

    hcci_latencies = []
    gist_latencies = []
    fp_rates = []
    range_counts = []

    print(f"\n  Running {n_trials} trials...")

    for trial in range(n_trials):
        # Random viewport
        x0 = bounds["x_min"] + rng.uniform(0, max(0.001, x_span - vw))
        y0 = bounds["y_min"] + rng.uniform(0, max(0.001, y_span - vh))
        x1 = x0 + vw
        y1 = y0 + vh
        viewport = (x0, y0, x1, y1)

        # Random class label
        cl = rng.choice(class_labels)

        # HCCI query
        hcci_sql, n_ranges = build_hcci_query(cl, class_enum, bounds, viewport, hilbert_order)
        range_counts.append(n_ranges)

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(hcci_sql)
            hcci_rows = cur.fetchall()
        hcci_ms = (time.perf_counter() - t0) * 1000
        hcci_latencies.append(hcci_ms)

        # GiST query
        gist_sql = build_gist_query(cl, bounds, viewport)
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(gist_sql)
            gist_rows = cur.fetchall()
        gist_ms = (time.perf_counter() - t0) * 1000
        gist_latencies.append(gist_ms)

        # FP measurement
        exact_sql = build_exact_query(cl, viewport)
        with conn.cursor() as cur:
            cur.execute(exact_sql)
            exact_rows = cur.fetchall()

        n_hcci = len(hcci_rows)
        n_exact = len(exact_rows)
        fp = (n_hcci - n_exact) / n_hcci if n_hcci > 0 else 0.0
        fp_rates.append(max(0.0, fp))

        if (trial + 1) % 50 == 0:
            print(f"    Trial {trial+1}/{n_trials}: "
                  f"HCCI p50={np.median(hcci_latencies):.1f}ms "
                  f"GiST p50={np.median(gist_latencies):.1f}ms "
                  f"FP mean={np.mean(fp_rates):.3f}")

    # Compute speedup
    hcci_p50 = np.median(hcci_latencies)
    gist_p50 = np.median(gist_latencies)
    speedup_p50 = gist_p50 / hcci_p50 if hcci_p50 > 0 else 0

    results = {
        "dataset": "nyc_taxi",
        "table": TABLE,
        "n_trials": n_trials,
        "viewport_frac": viewport_frac,
        "hilbert_order": hilbert_order,
        "total_rows": meta["bounds"]["count"],
        "n_classes": len(class_labels),
        "hcci_latency": compute_stats(hcci_latencies),
        "gist_latency": compute_stats(gist_latencies),
        "fp_mean": round(float(np.mean(fp_rates)), 4),
        "fp_max": round(float(np.max(fp_rates)), 4),
        "fp_p95": round(float(np.percentile(fp_rates, 95)), 4),
        "fp_std": round(float(np.std(fp_rates)), 4),
        "avg_ranges": round(float(np.mean(range_counts)), 1),
        "max_ranges": int(np.max(range_counts)),
        "speedup_p50": round(speedup_p50, 2),
        "speedup_mean": round(np.mean(gist_latencies) / np.mean(hcci_latencies), 2)
            if np.mean(hcci_latencies) > 0 else 0,
    }

    print(f"\n  {'='*50}")
    print(f"  HCCI: p50={results['hcci_latency']['p50']:.2f}ms  "
          f"p95={results['hcci_latency']['p95']:.2f}ms  "
          f"mean={results['hcci_latency']['mean']:.2f}ms")
    print(f"  GiST: p50={results['gist_latency']['p50']:.2f}ms  "
          f"p95={results['gist_latency']['p95']:.2f}ms  "
          f"mean={results['gist_latency']['mean']:.2f}ms")
    print(f"  Speedup: {speedup_p50:.1f}x (p50)  "
          f"{results['speedup_mean']:.1f}x (mean)")
    print(f"  FP: mean={results['fp_mean']:.3f}  "
          f"p95={results['fp_p95']:.3f}  max={results['fp_max']:.3f}")
    print(f"  Ranges: avg={results['avg_ranges']:.1f}  max={results['max_ranges']}")
    print(f"  {'='*50}")

    return results


def run_io_decomposition(
    conn,
    meta: dict,
    n_trials: int = 50,
    viewport_frac: float = 0.05,
    seed: int = config.RANDOM_SEED + 9500,
) -> dict:
    """Run I/O decomposition analysis (EXPLAIN ANALYZE BUFFERS)."""
    bounds = meta["bounds"]
    class_enum = meta["class_enum"]
    hilbert_order = meta["hilbert_order"]

    x_span = bounds["x_max"] - bounds["x_min"]
    y_span = bounds["y_max"] - bounds["y_min"]
    vw = x_span * np.sqrt(viewport_frac)
    vh = y_span * np.sqrt(viewport_frac)

    class_labels = list(class_enum.keys())
    rng = np.random.RandomState(seed)

    hcci_buffers = []
    gist_buffers = []

    print(f"\n  I/O decomposition: {n_trials} trials...")

    for trial in range(n_trials):
        x0 = bounds["x_min"] + rng.uniform(0, max(0.001, x_span - vw))
        y0 = bounds["y_min"] + rng.uniform(0, max(0.001, y_span - vh))
        x1 = x0 + vw
        y1 = y0 + vh
        viewport = (x0, y0, x1, y1)
        cl = rng.choice(class_labels)

        # HCCI EXPLAIN
        hcci_sql, _ = build_hcci_query(cl, class_enum, bounds, viewport, hilbert_order)
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {hcci_sql}")
            plan = cur.fetchone()[0]
        hcci_buffers.append(parse_buffers(plan))

        # GiST EXPLAIN
        gist_sql = build_gist_query(cl, bounds, viewport)
        with conn.cursor() as cur:
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {gist_sql}")
            plan = cur.fetchone()[0]
        gist_buffers.append(parse_buffers(plan))

    # Aggregate
    def _avg(lst, key):
        vals = [b.get(key, 0) for b in lst]
        return round(float(np.mean(vals)), 1) if vals else 0

    return {
        "n_trials": n_trials,
        "hcci": {
            "shared_hit": _avg(hcci_buffers, "shared_hit"),
            "shared_read": _avg(hcci_buffers, "shared_read"),
            "heap_fetches": _avg(hcci_buffers, "heap_fetches"),
            "hit_ratio": _avg(hcci_buffers, "hit_ratio"),
            "execution_time": _avg(hcci_buffers, "execution_time"),
        },
        "gist": {
            "shared_hit": _avg(gist_buffers, "shared_hit"),
            "shared_read": _avg(gist_buffers, "shared_read"),
            "heap_fetches": _avg(gist_buffers, "heap_fetches"),
            "hit_ratio": _avg(gist_buffers, "hit_ratio"),
            "execution_time": _avg(gist_buffers, "execution_time"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi HCCI Benchmark")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    parser.add_argument("--io-trials", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup queries before benchmark")
    args = parser.parse_args()

    print("=" * 60)
    print("  NYC Taxi HCCI Benchmark")
    print(f"  Trials: {args.trials}, Viewport: {args.viewport_frac}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    meta = load_taxi_metadata()

    # Verify table exists
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        count = cur.fetchone()[0]
    print(f"\n  Table {TABLE}: {count:,} rows")

    # Warmup
    if args.warmup > 0:
        print(f"\n  Warming up ({args.warmup} queries)...")
        bounds = meta["bounds"]
        rng = np.random.RandomState(0)
        for _ in range(args.warmup):
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {TABLE} ORDER BY random() LIMIT 100")
                cur.fetchall()

    t_start = time.time()

    # Main benchmark
    print("\n--- Viewport Benchmark ---")
    bench_results = run_benchmark(
        conn, meta,
        n_trials=args.trials,
        viewport_frac=args.viewport_frac,
    )

    # I/O decomposition
    print("\n--- I/O Decomposition ---")
    io_results = run_io_decomposition(
        conn, meta,
        n_trials=args.io_trials,
        viewport_frac=args.viewport_frac,
    )

    total = time.time() - t_start

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_time_sec": round(total, 1),
        "benchmark": bench_results,
        "io_decomposition": io_results,
    }

    path = save_results(all_results, "taxi_hcci_benchmark")
    print(f"\n{'='*60}")
    print(f"  Total time: {total:.0f}s ({total/60:.1f}m)")
    print(f"  Results saved to {path}")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
