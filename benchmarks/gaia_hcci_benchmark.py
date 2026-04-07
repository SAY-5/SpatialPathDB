"""Gaia DR3 HCCI Benchmark: viewport queries comparing HCCI vs GiST on astronomy data.

Runs class-filtered viewport queries on the Gaia stellar catalog to validate
HCCI performance on astronomy data with natural color-based classification.

Usage:
    python -m benchmarks.gaia_hcci_benchmark
    python -m benchmarks.gaia_hcci_benchmark --trials 200 --viewport-frac 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci
from benchmarks.framework import compute_stats, save_results, parse_buffers

TABLE = "gaia_sources"
DATASET_ID = "gaia_dr3"


def load_gaia_metadata():
    meta_path = os.path.join(config.RAW_DIR, "gaia_metadata.json")
    with open(meta_path) as f:
        return json.load(f)


def build_hcci_query(class_label, class_enum, bounds, viewport, hilbert_order):
    x0, y0, x1, y1 = viewport
    enum_val = class_enum[class_label]
    ranges = hcci.hilbert_ranges_direct(
        x0, y0, x1, y1,
        bounds["width"], bounds["height"],
        hilbert_order, max_ranges=64,
        x_origin=bounds["x_min"], y_origin=bounds["y_min"],
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
    return " UNION ALL ".join(branches), len(ranges)


def build_gist_query(class_label, viewport):
    x0, y0, x1, y1 = viewport
    return (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE class_label = '{class_label}' "
        f"AND geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 0)"
    )


def build_exact_query(class_label, viewport):
    x0, y0, x1, y1 = viewport
    return (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE dataset_id = '{DATASET_ID}' "
        f"AND class_label = '{class_label}' "
        f"AND centroid_x >= {x0} AND centroid_x <= {x1} "
        f"AND centroid_y >= {y0} AND centroid_y <= {y1}"
    )


def run_benchmark(conn, meta, n_trials=200, viewport_frac=0.05,
                  seed=config.RANDOM_SEED + 11000):
    bounds = meta["bounds"]
    class_enum = meta["class_enum"]
    hilbert_order = meta["hilbert_order"]

    x_span = bounds["x_max"] - bounds["x_min"]
    y_span = bounds["y_max"] - bounds["y_min"]
    vw = x_span * np.sqrt(viewport_frac)
    vh = y_span * np.sqrt(viewport_frac)

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) FROM {TABLE}
            GROUP BY class_label ORDER BY COUNT(*) DESC
        """)
        class_counts = dict(cur.fetchall())

    class_labels = list(class_counts.keys())
    print(f"\n  Classes: {class_labels}")
    print(f"  Viewport: {viewport_frac} ({vw:.2f} x {vh:.2f} degrees)")

    rng = np.random.RandomState(seed)
    hcci_latencies = []
    gist_latencies = []
    fp_rates = []
    range_counts = []

    print(f"\n  Running {n_trials} trials...")
    for trial in range(n_trials):
        x0 = bounds["x_min"] + rng.uniform(0, max(0.001, x_span - vw))
        y0 = bounds["y_min"] + rng.uniform(0, max(0.001, y_span - vh))
        x1 = x0 + vw
        y1 = y0 + vh
        viewport = (x0, y0, x1, y1)
        cl = rng.choice(class_labels)

        hcci_sql, n_ranges = build_hcci_query(cl, class_enum, bounds, viewport, hilbert_order)
        range_counts.append(n_ranges)

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(hcci_sql)
            hcci_rows = cur.fetchall()
        hcci_latencies.append((time.perf_counter() - t0) * 1000)

        gist_sql = build_gist_query(cl, viewport)
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(gist_sql)
            gist_rows = cur.fetchall()
        gist_latencies.append((time.perf_counter() - t0) * 1000)

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
                  f"FP={np.mean(fp_rates):.3f}")

    hcci_p50 = np.median(hcci_latencies)
    gist_p50 = np.median(gist_latencies)
    speedup_p50 = gist_p50 / hcci_p50 if hcci_p50 > 0 else 0

    results = {
        "dataset": "gaia_dr3",
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
          f"p95={results['hcci_latency']['p95']:.2f}ms")
    print(f"  GiST: p50={results['gist_latency']['p50']:.2f}ms  "
          f"p95={results['gist_latency']['p95']:.2f}ms")
    print(f"  Speedup: {speedup_p50:.1f}x (p50)")
    print(f"  FP: mean={results['fp_mean']:.3f}  max={results['fp_max']:.3f}")
    print(f"  {'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Gaia DR3 HCCI Benchmark")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    args = parser.parse_args()

    print("=" * 60)
    print("  Gaia DR3 HCCI Benchmark")
    print(f"  Trials: {args.trials}, Viewport: {args.viewport_frac}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    meta = load_gaia_metadata()

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        count = cur.fetchone()[0]
    print(f"\n  Table {TABLE}: {count:,} rows")

    # Warmup
    print(f"\n  Warming up...")
    for _ in range(20):
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {TABLE} ORDER BY random() LIMIT 100")
            cur.fetchall()

    t_start = time.time()
    bench_results = run_benchmark(conn, meta, n_trials=args.trials,
                                  viewport_frac=args.viewport_frac)
    total = time.time() - t_start

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_time_sec": round(total, 1),
        "benchmark": bench_results,
    }

    vp_tag = f"{args.viewport_frac}".replace(".", "")
    path = save_results(all_results, f"gaia_hcci_benchmark_vp{vp_tag}")
    print(f"\n  Results saved to {path}")
    conn.close()


if __name__ == "__main__":
    main()
