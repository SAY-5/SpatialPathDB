"""Hilbert order sweep: test HCCI at different Hilbert curve resolutions.

Validates the FP bound across the parameter space and characterizes the
precision-performance tradeoff of space-filling curve resolution in
covering indexes.

The index is built at p=8.  For query orders p < 8, we compute coarser
ranges mapped to p=8 space (more FP, fewer ranges).  For p > 8, results
are identical to p=8 since we cannot exceed the index resolution.

Usage:
    python -m benchmarks.hcci_order_sweep
    python -m benchmarks.hcci_order_sweep --trials 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import psycopg2

from spdb import config, hcci, hilbert
from benchmarks.framework import (
    compute_stats, save_results, time_query, wilcoxon_ranksum,
)

TABLE = config.TABLE_SLIDE_ONLY
INDEX_ORDER = config.HILBERT_ORDER  # 8 — the order stored in the index


# ---------------------------------------------------------------------------
# Slide helpers
# ---------------------------------------------------------------------------

def _get_all_slides(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        return [r[0] for r in cur.fetchall()]


def _get_slide_dims(conn, slide_id: str) -> tuple[float, float]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT MAX(centroid_x), MAX(centroid_y) FROM {TABLE} WHERE slide_id = %s",
            (slide_id,),
        )
        row = cur.fetchone()
    if row and row[0] and row[1]:
        return float(row[0]) * 1.05, float(row[1]) * 1.05
    return 100000.0, 100000.0


_dims_cache: dict[str, tuple[float, float]] = {}


def get_dims(conn, sid):
    if sid not in _dims_cache:
        _dims_cache[sid] = _get_slide_dims(conn, sid)
    return _dims_cache[sid]


# ---------------------------------------------------------------------------
# Build HCCI query at a given query order
# ---------------------------------------------------------------------------

def build_hcci_query_at_order(
    table_name: str,
    slide_id: str,
    class_labels: list[str],
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    query_order: int,
    index_order: int = INDEX_ORDER,
    max_ranges: int = 64,
) -> tuple[str, tuple, int, int]:
    """Build HCCI query using ranges computed at query_order.

    Returns (sql, params, n_ranges, n_branches).
    """
    h_ranges, _ = hcci.hilbert_ranges_at_order(
        x0, y0, x1, y1,
        slide_width, slide_height,
        query_order, index_order,
        max_ranges,
    )

    # Build composite key ranges for all classes
    ck_ranges = []
    for label in class_labels:
        enum_val = hcci.class_to_enum(label)
        prefix = enum_val << hcci.COMPOSITE_SHIFT
        for h_lo, h_hi in h_ranges:
            ck_ranges.append((prefix | h_lo, prefix | h_hi))

    n_ranges = len(h_ranges)
    n_branches = len(ck_ranges)

    if not ck_ranges:
        sql = (
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE slide_id = %s AND FALSE"
        )
        return sql, (slide_id,), 0, 0

    if len(ck_ranges) == 1:
        lo, hi = ck_ranges[0]
        sql = (
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE slide_id = %s "
            f"AND composite_key >= %s AND composite_key < %s"
        )
        return sql, (slide_id, lo, hi), n_ranges, n_branches

    branches = []
    params: list = []
    for lo, hi in ck_ranges:
        branches.append(
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE slide_id = %s "
            f"AND composite_key >= %s AND composite_key < %s"
        )
        params.extend([slide_id, lo, hi])

    sql = " UNION ALL ".join(branches)
    return sql, tuple(params), n_ranges, n_branches


# ---------------------------------------------------------------------------
# Order sweep
# ---------------------------------------------------------------------------

def run_order_sweep(
    conn,
    slides: list[str],
    orders: list[int],
    class_labels: list[str],
    viewport_frac: float = 0.05,
    n_trials: int = 200,
    seed: int = config.RANDOM_SEED + 3000,
) -> dict:
    """Run HCCI at multiple orders and measure latency + FP rate."""

    rng = np.random.RandomState(seed)

    print(f"\n{'='*60}")
    print(f"  Hilbert Order Sweep: orders={orders}")
    print(f"  Classes: {class_labels}, viewport: {viewport_frac*100:.0f}%")
    print(f"  {n_trials} trials per order")
    print(f"{'='*60}")

    # Pre-generate viewports (same for all orders — paired design)
    viewports = []
    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid)
        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)
        viewports.append((sid, w, h, x0, y0, x1, y1))

    # Get exact row counts via GiST for FP measurement
    print(f"\n  Computing exact counts via GiST ({n_trials} viewports)...")
    exact_counts = []
    for trial, (sid, w, h, x0, y0, x1, y1) in enumerate(viewports):
        gist_sql, gist_params = hcci.build_baseline_gist_query(
            TABLE, sid, class_labels, x0, y0, x1, y1,
        )
        rows, _ = time_query(conn, gist_sql, gist_params)
        exact_counts.append(len(rows))
        if (trial + 1) % 50 == 0:
            print(f"    {trial + 1}/{n_trials} exact counts computed")

    results_by_order = {}

    for p in orders:
        print(f"\n  --- Order p={p} ---")
        latencies = []
        fp_rates = []
        range_counts = []
        branch_counts = []

        for trial, (sid, w, h, x0, y0, x1, y1) in enumerate(viewports):
            sql, params, n_ranges, n_branches = build_hcci_query_at_order(
                TABLE, sid, class_labels,
                x0, y0, x1, y1, w, h,
                query_order=p,
                index_order=INDEX_ORDER,
            )

            rows, elapsed = time_query(conn, sql, params)
            latencies.append(elapsed)
            range_counts.append(n_ranges)
            branch_counts.append(n_branches)

            # FP rate
            n_hcci = len(rows)
            n_exact = exact_counts[trial]
            n_fp = max(0, n_hcci - n_exact)
            fp_rate = n_fp / n_hcci if n_hcci > 0 else 0.0
            fp_rates.append(fp_rate)

            if (trial + 1) % 50 == 0:
                avg_lat = float(np.mean(latencies[-50:]))
                avg_fp = float(np.mean(fp_rates[-50:]))
                print(f"    Trial {trial+1}/{n_trials}: "
                      f"p50={np.median(latencies):.2f}ms  "
                      f"FP={avg_fp*100:.2f}%  "
                      f"ranges={n_ranges}")

        stats = compute_stats(latencies)
        theoretical_bound = hcci.false_positive_rate(viewport_frac, p)

        fp_arr = np.array(fp_rates)
        within_bound = float(np.mean(fp_arr <= theoretical_bound + 0.001))

        order_result = {
            "order": p,
            "index_order": INDEX_ORDER,
            "n_trials": n_trials,
            "latency": stats,
            "fp_mean": round(float(np.mean(fp_rates)), 4),
            "fp_max": round(float(np.max(fp_rates)), 4),
            "fp_p95": round(float(np.percentile(fp_rates, 95)), 4),
            "fp_min": round(float(np.min(fp_rates)), 4),
            "fp_std": round(float(np.std(fp_rates)), 4),
            "theoretical_bound": round(theoretical_bound, 4),
            "within_bound": round(within_bound, 4),
            "avg_ranges": round(float(np.mean(range_counts)), 1),
            "avg_branches": round(float(np.mean(branch_counts)), 1),
            "max_ranges": int(np.max(range_counts)),
            "max_branches": int(np.max(branch_counts)),
        }

        results_by_order[str(p)] = order_result

        print(f"\n  p={p}: p50={stats['p50']:.2f}ms  "
              f"FP mean={order_result['fp_mean']*100:.2f}%  "
              f"FP max={order_result['fp_max']*100:.2f}%  "
              f"bound={theoretical_bound*100:.2f}%  "
              f"within={within_bound*100:.0f}%  "
              f"ranges={order_result['avg_ranges']:.0f}")

    return results_by_order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HCCI Hilbert order sweep")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--orders", type=str, default="4,6,8,10,12",
                        help="Comma-separated Hilbert orders to test")
    args = parser.parse_args()

    orders = [int(x) for x in args.orders.split(",")]

    print("=" * 60)
    print("  HCCI Hilbert Order Sweep")
    print(f"  Orders: {orders}")
    print(f"  Index order: {INDEX_ORDER}")
    print(f"  Trials per order: {args.trials}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True

    slides = _get_all_slides(conn)
    print(f"  Found {len(slides)} slides")

    # Cache dimensions
    for sid in slides[:20]:
        get_dims(conn, sid)

    t_start = time.time()

    # Run sweep for Tumor (16.4% selectivity — primary benchmark class)
    results = run_order_sweep(
        conn, slides, orders,
        class_labels=["Tumor"],
        viewport_frac=0.05,
        n_trials=args.trials,
    )

    total = time.time() - t_start

    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table": TABLE,
        "n_slides": len(slides),
        "index_order": INDEX_ORDER,
        "class_labels": ["Tumor"],
        "viewport_frac": 0.05,
        "total_time_sec": round(total, 1),
        "orders": results,
    }

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'Order':>5} {'p50(ms)':>8} {'p95(ms)':>8} {'FP mean':>8} "
          f"{'FP max':>8} {'Bound':>8} {'Within':>8} {'Ranges':>8}")
    print(f"  {'-'*75}")
    for p in orders:
        r = results[str(p)]
        print(f"  {p:>5} {r['latency']['p50']:>8.2f} {r['latency']['p95']:>8.2f} "
              f"{r['fp_mean']*100:>7.2f}% {r['fp_max']*100:>7.2f}% "
              f"{r['theoretical_bound']*100:>7.2f}% {r['within_bound']*100:>7.0f}% "
              f"{r['avg_ranges']:>8.0f}")
    print(f"{'='*80}")
    print(f"  Total time: {total:.0f}s ({total/60:.1f}m)")

    path = save_results(all_results, "hcci_order_sweep")
    print(f"  Results saved to {path}")

    conn.close()


if __name__ == "__main__":
    main()
