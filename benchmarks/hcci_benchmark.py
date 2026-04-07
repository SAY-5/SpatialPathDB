"""HCCI benchmark: compare Hilbert-Composite Covering Index vs GiST baseline.

Tests six query types across 500 trials each, with I/O decomposition and
false positive measurement.

Usage:
    python -m benchmarks.hcci_benchmark
    python -m benchmarks.hcci_benchmark --trials 100   # quick run
    python -m benchmarks.hcci_benchmark --query A       # single query type
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import psycopg2

from spdb import config, hcci
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results,
    time_query, time_query_buffers, parse_buffers,
    wilcoxon_ranksum, print_comparison,
)

TABLE = config.TABLE_SLIDE_ONLY


# ---------------------------------------------------------------------------
# Slide dimension helpers (handle 195M dataset with >29 slides)
# ---------------------------------------------------------------------------

def _get_all_slides(conn) -> list[str]:
    """Get all distinct slide_ids from the SO table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        return [r[0] for r in cur.fetchall()]


def _get_slide_dims(conn, slide_id: str) -> tuple[float, float]:
    """Get slide dimensions from data (MAX centroid + 5% padding)."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT MAX(centroid_x), MAX(centroid_y)
            FROM {TABLE}
            WHERE slide_id = %s
        """, (slide_id,))
        row = cur.fetchone()
    if row and row[0] and row[1]:
        return float(row[0]) * 1.05, float(row[1]) * 1.05
    return 100000.0, 100000.0


def _get_slide_object_count(conn, slide_id: str) -> int:
    """Estimate object count for a slide (fast via pg_class)."""
    with conn.cursor() as cur:
        safe = slide_id.replace("-", "_").replace(".", "_").lower()
        part_name = f"{TABLE}_{safe}"
        cur.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s",
            (part_name,)
        )
        row = cur.fetchone()
        if row and row[0] and row[0] > 0:
            return int(row[0])
        cur.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE slide_id = %s",
            (slide_id,)
        )
        return cur.fetchone()[0]


_slide_dims_cache: dict[str, tuple[float, float]] = {}
_slide_counts_cache: dict[str, int] = {}


def get_dims(conn, slide_id):
    if slide_id not in _slide_dims_cache:
        _slide_dims_cache[slide_id] = _get_slide_dims(conn, slide_id)
    return _slide_dims_cache[slide_id]


def get_count(conn, slide_id):
    if slide_id not in _slide_counts_cache:
        _slide_counts_cache[slide_id] = _get_slide_object_count(conn, slide_id)
    return _slide_counts_cache[slide_id]


# ---------------------------------------------------------------------------
# Query type definitions
# ---------------------------------------------------------------------------

QUERY_TYPES = {
    "A": {
        "name": "Tumor viewport (16.4%)",
        "classes": ["Tumor"],
        "viewport_frac": 0.05,
        "description": "Single rare class - primary HCCI win scenario",
    },
    "B": {
        "name": "Lymphocyte viewport (11.4%)",
        "classes": ["Lymphocyte"],
        "viewport_frac": 0.05,
        "description": "Rarest class - largest HCCI advantage expected",
    },
    "C": {
        "name": "Tumor+Lymphocyte viewport (27.8%)",
        "classes": ["Tumor", "Lymphocyte"],
        "viewport_frac": 0.05,
        "description": "Two-class immune query - existing Q4 pattern",
    },
    "D": {
        "name": "Epithelial viewport (42.5%)",
        "classes": ["Epithelial"],
        "viewport_frac": 0.05,
        "description": "Majority class - smallest HCCI advantage",
    },
    "E": {
        "name": "All classes (control)",
        "classes": ["Epithelial", "Stromal", "Tumor", "Lymphocyte"],
        "viewport_frac": 0.05,
        "description": "Control - HCCI should not hurt unfiltered queries",
    },
    "F": {
        "name": "Tumor small viewport (1%)",
        "classes": ["Tumor"],
        "viewport_frac": 0.01,
        "description": "High spatial selectivity + class filter",
    },
}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_query_type(
    conn,
    query_id: str,
    slides: list[str],
    n_trials: int = 500,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run HCCI vs GiST baseline for a single query type."""
    qt = QUERY_TYPES[query_id]
    class_labels = qt["classes"]
    viewport_frac = qt["viewport_frac"]
    rng = np.random.RandomState(seed)

    print(f"\n{'='*60}")
    print(f"  Query {query_id}: {qt['name']}")
    print(f"  Classes: {class_labels}  Viewport: {viewport_frac*100:.0f}%")
    print(f"  {qt['description']}")
    print(f"  {n_trials} trials")
    print(f"{'='*60}")

    lats_hcci: list[float] = []
    lats_gist: list[float] = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # HCCI query (direct Hilbert ranges — no bucket discretization)
        hcci_sql, hcci_params = hcci.build_hcci_query(
            TABLE, sid, class_labels,
            x0, y0, x1, y1, w, h,
            config.HILBERT_ORDER, use_direct=True,
        )
        _, t_hcci = time_query(conn, hcci_sql, hcci_params)
        lats_hcci.append(t_hcci)

        # GiST baseline
        if query_id == "E":
            gist_sql, gist_params = hcci.build_baseline_gist_query_all_classes(
                TABLE, sid, x0, y0, x1, y1,
            )
        else:
            gist_sql, gist_params = hcci.build_baseline_gist_query(
                TABLE, sid, class_labels, x0, y0, x1, y1,
            )
        _, t_gist = time_query(conn, gist_sql, gist_params)
        lats_gist.append(t_gist)

        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial+1}/{n_trials}: "
                  f"HCCI p50={np.median(lats_hcci):.1f}ms  "
                  f"GiST p50={np.median(lats_gist):.1f}ms")

    stats_hcci = compute_stats(lats_hcci)
    stats_gist = compute_stats(lats_gist)

    _, wsr_p = wilcoxon_ranksum(lats_hcci, lats_gist)

    speedup_p50 = stats_gist["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0
    speedup_mean = stats_gist["mean"] / stats_hcci["mean"] if stats_hcci["mean"] > 0 else 0

    save_raw_latencies(lats_hcci, f"hcci_{query_id}", "HCCI")
    save_raw_latencies(lats_gist, f"hcci_{query_id}", "GiST")

    print(f"\n  Results for Query {query_id}:")
    print(f"  {'':>16} {'p50':>8} {'p95':>8} {'mean':>8} {'std':>8}")
    print(f"  {'HCCI':>16} {stats_hcci['p50']:>8.1f} {stats_hcci['p95']:>8.1f} "
          f"{stats_hcci['mean']:>8.1f} {stats_hcci['std']:>8.1f}")
    print(f"  {'GiST':>16} {stats_gist['p50']:>8.1f} {stats_gist['p95']:>8.1f} "
          f"{stats_gist['mean']:>8.1f} {stats_gist['std']:>8.1f}")
    print(f"  Speedup: {speedup_p50:.2f}x (p50), {speedup_mean:.2f}x (mean)")
    print(f"  Wilcoxon p = {wsr_p:.2e}")

    return {
        "query_id": query_id,
        "query_name": qt["name"],
        "classes": class_labels,
        "viewport_frac": viewport_frac,
        "n_trials": n_trials,
        "hcci": stats_hcci,
        "gist": stats_gist,
        "speedup_p50": round(speedup_p50, 3),
        "speedup_mean": round(speedup_mean, 3),
        "wilcoxon_p": wsr_p,
    }


# ---------------------------------------------------------------------------
# I/O decomposition
# ---------------------------------------------------------------------------

def run_io_decomposition(
    conn,
    query_ids: list[str],
    slides: list[str],
    n_trials: int = 50,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run EXPLAIN (ANALYZE, BUFFERS) for I/O decomposition."""
    rng = np.random.RandomState(seed + 999)
    results = {}

    for qid in query_ids:
        qt = QUERY_TYPES[qid]
        print(f"\n  I/O decomposition for Query {qid} ({n_trials} trials)...")

        hcci_buffers = []
        gist_buffers = []

        for _ in range(n_trials):
            sid = rng.choice(slides)
            w, h = get_dims(conn, sid)

            vf = qt["viewport_frac"]
            vw = w * float(np.sqrt(vf))
            vh = h * float(np.sqrt(vf))
            x0 = float(rng.uniform(0, max(1, w - vw)))
            y0 = float(rng.uniform(0, max(1, h - vh)))
            x1 = float(x0 + vw)
            y1 = float(y0 + vh)

            hcci_sql, hcci_p = hcci.build_hcci_query(
                TABLE, sid, qt["classes"],
                x0, y0, x1, y1, w, h,
                config.HILBERT_ORDER, use_direct=True,
            )
            _, hb = time_query_buffers(conn, hcci_sql, hcci_p)
            hcci_buffers.append(hb)

            if qid == "E":
                gist_sql, gist_p = hcci.build_baseline_gist_query_all_classes(
                    TABLE, sid, x0, y0, x1, y1,
                )
            else:
                gist_sql, gist_p = hcci.build_baseline_gist_query(
                    TABLE, sid, qt["classes"], x0, y0, x1, y1,
                )
            _, gb = time_query_buffers(conn, gist_sql, gist_p)
            gist_buffers.append(gb)

        def _avg_buffers(buf_list):
            keys = ["shared_hit", "shared_read", "heap_fetches",
                    "rows_removed_by_filter",
                    "actual_rows", "planning_time", "execution_time"]
            return {
                k: round(np.mean([b.get(k, 0) for b in buf_list]), 1)
                for k in keys
            }

        hcci_avg = _avg_buffers(hcci_buffers)
        gist_avg = _avg_buffers(gist_buffers)

        # Check for index-only scan (heap fetches = 0)
        hcci_node_types = set()
        for b in hcci_buffers:
            hcci_node_types.update(b.get("node_types", []))

        results[qid] = {
            "hcci": hcci_avg,
            "gist": gist_avg,
            "hcci_index_only": "Index Only Scan" in hcci_node_types,
            "buffer_reduction_pct": round(
                (1 - (hcci_avg["shared_hit"] + hcci_avg["shared_read"]) /
                 max(1, gist_avg["shared_hit"] + gist_avg["shared_read"])) * 100, 1
            ),
        }

        print(f"    GiST: shared_hit={gist_avg['shared_hit']:.0f}  "
              f"shared_read={gist_avg['shared_read']:.0f}  "
              f"heap_fetches={gist_avg['heap_fetches']:.0f}  "
              f"rows_removed_by_filter={gist_avg['rows_removed_by_filter']:.0f}  "
              f"actual_rows={gist_avg['actual_rows']:.0f}")
        print(f"    HCCI: shared_hit={hcci_avg['shared_hit']:.0f}  "
              f"shared_read={hcci_avg['shared_read']:.0f}  "
              f"heap_fetches={hcci_avg['heap_fetches']:.0f}  "
              f"rows_removed_by_filter={hcci_avg['rows_removed_by_filter']:.0f}  "
              f"actual_rows={hcci_avg['actual_rows']:.0f}")
        print(f"    Index-only scan: {'YES' if results[qid]['hcci_index_only'] else 'NO'}")
        print(f"    Heap fetches (HCCI): {hcci_avg['heap_fetches']:.0f} "
              f"{'(VERIFIED: zero heap access)' if hcci_avg['heap_fetches'] < 1 else '(WARNING: heap fetches detected — run VACUUM)'}")
        print(f"    Buffer reduction: {results[qid]['buffer_reduction_pct']:.1f}%")

    return results


# ---------------------------------------------------------------------------
# False positive measurement
# ---------------------------------------------------------------------------

def run_fp_measurement(
    conn,
    slides: list[str],
    n_trials: int = 50,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Measure empirical false positive rate vs theoretical bound."""
    rng = np.random.RandomState(seed + 777)
    print(f"\n  False positive measurement ({n_trials} trials, Tumor 5% viewport)...")

    fp_rates = []
    theoretical = hcci.false_positive_rate(0.05, config.HILBERT_ORDER)

    for i in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid)

        vw = w * float(np.sqrt(0.05))
        vh = h * float(np.sqrt(0.05))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))

        result = hcci.measure_false_positive_rate(
            conn, TABLE, sid, ["Tumor"],
            x0, y0, float(x0 + vw), float(y0 + vh),
            w, h, config.HILBERT_ORDER,
            use_direct=True,
        )
        fp_rates.append(result["fp_rate"])

    return {
        "n_trials": n_trials,
        "theoretical_bound": round(theoretical, 4),
        "empirical_mean": round(float(np.mean(fp_rates)), 4),
        "empirical_max": round(float(np.max(fp_rates)), 4),
        "empirical_p95": round(float(np.percentile(fp_rates, 95)), 4),
        "within_bound": sum(1 for r in fp_rates if r <= theoretical * 1.1) / len(fp_rates),
    }


# ---------------------------------------------------------------------------
# Bounding-box baseline comparison (Issue 2: apples-to-apples)
# ---------------------------------------------------------------------------

def run_bbox_comparison(
    conn,
    slides: list[str],
    n_trials: int = 200,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Compare HCCI vs GiST+ST_Intersects vs GiST+bbox on Query A.

    Isolates heap I/O cost from geometry computation cost. If GiST+bbox
    ≈ GiST+ST_Intersects, the heap I/O dominates and HCCI's speedup is
    genuine. If GiST+bbox << GiST+ST_Intersects, geometry computation
    is a significant factor in the measured speedup.
    """
    rng = np.random.RandomState(seed + 888)
    class_labels = ["Tumor"]
    viewport_frac = 0.05

    print(f"\n{'='*60}")
    print(f"  Bbox Baseline Comparison (Query A, {n_trials} trials)")
    print(f"  HCCI vs GiST+ST_Intersects vs GiST+bbox")
    print(f"{'='*60}")

    lats_hcci: list[float] = []
    lats_gist: list[float] = []
    lats_bbox: list[float] = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # HCCI (direct ranges)
        hcci_sql, hcci_params = hcci.build_hcci_query(
            TABLE, sid, class_labels,
            x0, y0, x1, y1, w, h,
            config.HILBERT_ORDER, use_direct=True,
        )
        _, t_hcci = time_query(conn, hcci_sql, hcci_params)
        lats_hcci.append(t_hcci)

        # GiST + ST_Intersects
        gist_sql, gist_params = hcci.build_baseline_gist_query(
            TABLE, sid, class_labels, x0, y0, x1, y1,
        )
        _, t_gist = time_query(conn, gist_sql, gist_params)
        lats_gist.append(t_gist)

        # GiST + bbox (&&)
        bbox_sql, bbox_params = hcci.build_baseline_bbox_query(
            TABLE, sid, class_labels, x0, y0, x1, y1,
        )
        _, t_bbox = time_query(conn, bbox_sql, bbox_params)
        lats_bbox.append(t_bbox)

        if (trial + 1) % 50 == 0:
            print(f"  Trial {trial+1}/{n_trials}: "
                  f"HCCI p50={np.median(lats_hcci):.1f}ms  "
                  f"GiST p50={np.median(lats_gist):.1f}ms  "
                  f"bbox p50={np.median(lats_bbox):.1f}ms")

    stats_hcci = compute_stats(lats_hcci)
    stats_gist = compute_stats(lats_gist)
    stats_bbox = compute_stats(lats_bbox)

    # Speedups
    sp_hcci_p50 = stats_gist["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0
    sp_bbox_p50 = stats_gist["p50"] / stats_bbox["p50"] if stats_bbox["p50"] > 0 else 0
    sp_hcci_vs_bbox_p50 = stats_bbox["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0

    # ST_Intersects fraction: how much does geometry computation add?
    geom_overhead_pct = (
        (stats_gist["p50"] - stats_bbox["p50"]) / stats_gist["p50"] * 100
        if stats_gist["p50"] > 0 else 0
    )

    print(f"\n  Results:")
    print(f"  {'':>20} {'p50':>8} {'p95':>8} {'mean':>8}")
    print(f"  {'HCCI':>20} {stats_hcci['p50']:>8.1f} {stats_hcci['p95']:>8.1f} {stats_hcci['mean']:>8.1f}")
    print(f"  {'GiST+ST_Intersects':>20} {stats_gist['p50']:>8.1f} {stats_gist['p95']:>8.1f} {stats_gist['mean']:>8.1f}")
    print(f"  {'GiST+bbox (&&)':>20} {stats_bbox['p50']:>8.1f} {stats_bbox['p95']:>8.1f} {stats_bbox['mean']:>8.1f}")
    print(f"\n  HCCI vs GiST+STI:  {sp_hcci_p50:.1f}x (p50)")
    print(f"  bbox vs GiST+STI:  {sp_bbox_p50:.1f}x (p50)")
    print(f"  HCCI vs GiST+bbox: {sp_hcci_vs_bbox_p50:.1f}x (p50)")
    print(f"  ST_Intersects overhead: {geom_overhead_pct:.1f}% of GiST latency")

    result = {
        "n_trials": n_trials,
        "hcci": stats_hcci,
        "gist_st_intersects": stats_gist,
        "gist_bbox": stats_bbox,
        "speedup_hcci_vs_gist_p50": round(sp_hcci_p50, 3),
        "speedup_bbox_vs_gist_p50": round(sp_bbox_p50, 3),
        "speedup_hcci_vs_bbox_p50": round(sp_hcci_vs_bbox_p50, 3),
        "st_intersects_overhead_pct": round(geom_overhead_pct, 1),
    }

    save_raw_latencies(lats_hcci, "bbox_comparison", "HCCI")
    save_raw_latencies(lats_gist, "bbox_comparison", "GiST_STI")
    save_raw_latencies(lats_bbox, "bbox_comparison", "GiST_bbox")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HCCI benchmark suite")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--query", type=str, default=None,
                        help="Run single query type (A-F)")
    parser.add_argument("--skip-io", action="store_true",
                        help="Skip I/O decomposition")
    parser.add_argument("--skip-fp", action="store_true",
                        help="Skip false positive measurement")
    parser.add_argument("--fp-only", action="store_true",
                        help="Run only false positive measurement (50 trials)")
    parser.add_argument("--bbox-only", action="store_true",
                        help="Run only bbox baseline comparison (200 trials)")
    parser.add_argument("--bbox-trials", type=int, default=200,
                        help="Number of trials for bbox comparison")
    args = parser.parse_args()

    print("=" * 60)
    print("  HCCI Benchmark: Composite Covering Index vs GiST")
    print("  (using direct Hilbert ranges — no bucket discretization)")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = False

    # Get slide list
    slides = _get_all_slides(conn)
    print(f"\n  Found {len(slides)} slides")

    # Pre-cache dimensions for a sample of slides
    print("  Pre-caching slide dimensions...")
    for sid in slides[:20]:
        get_dims(conn, sid)
        get_count(conn, sid)

    t_start = time.time()

    # --- Fast mode: FP-only ---
    if args.fp_only:
        print(f"\n{'='*60}")
        print(f"  False Positive Rate (direct Hilbert ranges)")
        print(f"{'='*60}")

        # Also run Query A to verify latency isn't hurt
        print("\n  Running Query A (500 trials) with direct ranges...")
        result_a = run_query_type(conn, "A", slides, n_trials=args.trials)

        fp_results = run_fp_measurement(conn, slides, n_trials=50)
        print(f"\n  Theoretical bound: {fp_results['theoretical_bound']:.1%}")
        print(f"  Empirical mean:    {fp_results['empirical_mean']:.1%}")
        print(f"  Empirical max:     {fp_results['empirical_max']:.1%}")
        print(f"  Within bound:      {fp_results['within_bound']:.0%} of trials")

        fp_results["query_a"] = result_a
        total = time.time() - t_start
        print(f"\n  Total time: {total:.0f}s ({total/60:.1f}m)")
        path = save_results(fp_results, "hcci_fp_direct")
        print(f"  Results saved to {path}")
        conn.close()
        return

    # --- Fast mode: bbox-only ---
    if args.bbox_only:
        bbox_results = run_bbox_comparison(
            conn, slides, n_trials=args.bbox_trials,
        )
        total = time.time() - t_start
        print(f"\n  Total time: {total:.0f}s ({total/60:.1f}m)")
        path = save_results(bbox_results, "hcci_bbox_comparison")
        print(f"  Results saved to {path}")
        conn.close()
        return

    # --- Full benchmark ---
    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table": TABLE,
        "n_slides": len(slides),
        "hilbert_order": config.HILBERT_ORDER,
        "hilbert_range_mode": "direct",
        "queries": {},
    }

    # Run query benchmarks
    query_ids = [args.query] if args.query else list(QUERY_TYPES.keys())
    for qid in query_ids:
        result = run_query_type(conn, qid, slides, n_trials=args.trials)
        all_results["queries"][qid] = result

    # I/O decomposition
    if not args.skip_io:
        print(f"\n{'='*60}")
        print(f"  I/O Decomposition (EXPLAIN ANALYZE BUFFERS)")
        print(f"{'='*60}")
        io_results = run_io_decomposition(conn, ["A", "C"], slides, n_trials=50)
        all_results["io_decomposition"] = io_results

    # False positive measurement
    if not args.skip_fp:
        print(f"\n{'='*60}")
        print(f"  False Positive Rate Measurement")
        print(f"{'='*60}")
        fp_results = run_fp_measurement(conn, slides, n_trials=50)
        all_results["false_positive"] = fp_results
        print(f"\n  Theoretical bound: {fp_results['theoretical_bound']:.1%}")
        print(f"  Empirical mean:    {fp_results['empirical_mean']:.1%}")
        print(f"  Empirical max:     {fp_results['empirical_max']:.1%}")
        print(f"  Within bound:      {fp_results['within_bound']:.0%} of trials")

    # Bbox baseline comparison
    print(f"\n{'='*60}")
    print(f"  Bbox Baseline Comparison")
    print(f"{'='*60}")
    bbox_results = run_bbox_comparison(conn, slides, n_trials=200)
    all_results["bbox_comparison"] = bbox_results

    # Cost model predictions
    print(f"\n{'='*60}")
    print(f"  Cost Model Predictions")
    print(f"{'='*60}")
    for label, sel in [("Tumor", 0.164), ("Lymphocyte", 0.114),
                       ("Tumor+Lymph", 0.278), ("Epithelial", 0.425)]:
        pred = hcci.hcci_cost_model(1_500_000, 0.05, sel)
        print(f"  {label:<14}: predicted speedup = {pred['speedup']:.1f}x  "
              f"(GiST {pred['gist']['total_ms']:.1f}ms, "
              f"HCCI {pred['hcci']['total_ms']:.1f}ms)")
    all_results["cost_model"] = {
        label: hcci.hcci_cost_model(1_500_000, 0.05, sel)
        for label, sel in [("Tumor", 0.164), ("Lymphocyte", 0.114),
                           ("Tumor+Lymph", 0.278), ("Epithelial", 0.425)]
    }

    # Summary
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  HCCI Benchmark Summary")
    print(f"{'='*60}")
    for qid, qr in all_results["queries"].items():
        sp = qr["speedup_p50"]
        wp = qr["wilcoxon_p"]
        sig = "***" if wp < 0.001 else "**" if wp < 0.01 else "*" if wp < 0.05 else "ns"
        print(f"  Query {qid} ({qr['query_name']:<35}): "
              f"{sp:.2f}x speedup  p={wp:.2e} {sig}")

    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}m)")

    path = save_results(all_results, "hcci_benchmark")
    print(f"  Results saved to {path}")

    conn.close()


if __name__ == "__main__":
    main()
