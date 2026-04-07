"""OSM HCCI benchmark: validate HCCI on urban geospatial data.

Tests HCCI covering index vs GiST baseline on NYC OpenStreetMap POIs
with 50+ category types and diverse selectivity distributions.

Demonstrates HCCI generalizability beyond pathology domain.

Usage:
    python -m benchmarks.osm_hcci_benchmark
    python -m benchmarks.osm_hcci_benchmark --trials 100
    python -m benchmarks.osm_hcci_benchmark --quick    # 100 trials, skip I/O
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
    wilcoxon_ranksum,
)

TABLE = "osm_pois"
DATASET_ID = "nyc"


# ---------------------------------------------------------------------------
# Load metadata
# ---------------------------------------------------------------------------

def load_osm_metadata() -> dict:
    """Load OSM dataset metadata (bounds, class_enum)."""
    path = os.path.join(config.RAW_DIR, "osm_metadata.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Category analysis and query type selection
# ---------------------------------------------------------------------------

def analyze_categories(conn) -> list[dict]:
    """Discover categories with counts and selectivities."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        rows = cur.fetchall()
        total = sum(r[1] for r in rows)

    cats = []
    for cat, cnt in rows:
        cats.append({
            "category": cat,
            "count": cnt,
            "selectivity": cnt / total,
        })
    return cats


def select_query_types(categories: list[dict]) -> dict[str, dict]:
    """Select query types spanning different selectivity ranges.

    Picks categories to match the pathology benchmark pattern:
    A: rare class (~5-15%), B: rarest meaningful class (<5%),
    C: two-class combo, D: common class (>20%), E: all classes (control),
    F: rare class + small viewport (1%).
    """
    # Sort by selectivity descending
    by_sel = sorted(categories, key=lambda c: c["selectivity"], reverse=True)

    # Find candidates
    rare = [c for c in by_sel if 0.02 < c["selectivity"] < 0.15]
    very_rare = [c for c in by_sel if 0.005 < c["selectivity"] < 0.03]
    common = [c for c in by_sel if c["selectivity"] > 0.15]

    # Pick specific categories
    cat_a = rare[0] if rare else by_sel[min(5, len(by_sel) - 1)]
    cat_b = very_rare[0] if very_rare else by_sel[min(10, len(by_sel) - 1)]
    cat_d = common[0] if common else by_sel[0]

    # Two-class combo
    combo_classes = [cat_a["category"], cat_b["category"]]
    combo_sel = cat_a["selectivity"] + cat_b["selectivity"]

    # All classes for control — top 10 categories
    all_classes = [c["category"] for c in by_sel[:10]]
    all_sel = sum(c["selectivity"] for c in by_sel[:10])

    queries = {
        "A": {
            "name": f"{cat_a['category']} ({cat_a['selectivity']:.1%})",
            "classes": [cat_a["category"]],
            "viewport_frac": 0.05,
            "description": f"Moderate selectivity ({cat_a['selectivity']:.1%})",
        },
        "B": {
            "name": f"{cat_b['category']} ({cat_b['selectivity']:.1%})",
            "classes": [cat_b["category"]],
            "viewport_frac": 0.05,
            "description": f"Low selectivity ({cat_b['selectivity']:.1%})",
        },
        "C": {
            "name": f"Two-class combo ({combo_sel:.1%})",
            "classes": combo_classes,
            "viewport_frac": 0.05,
            "description": f"{combo_classes[0]} + {combo_classes[1]}",
        },
        "D": {
            "name": f"{cat_d['category']} ({cat_d['selectivity']:.1%})",
            "classes": [cat_d["category"]],
            "viewport_frac": 0.05,
            "description": f"High selectivity ({cat_d['selectivity']:.1%})",
        },
        "E": {
            "name": f"Top-10 classes ({all_sel:.1%})",
            "classes": all_classes,
            "viewport_frac": 0.05,
            "description": "Control — most common categories",
        },
        "F": {
            "name": f"{cat_a['category']} @1% viewport",
            "classes": [cat_a["category"]],
            "viewport_frac": 0.01,
            "description": "High spatial selectivity + class filter",
        },
    }
    return queries


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_query_type(
    conn,
    query_id: str,
    qt: dict,
    meta: dict,
    n_trials: int = 500,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run HCCI vs GiST + GiST-bbox for a single query type."""
    class_labels = qt["classes"]
    viewport_frac = qt["viewport_frac"]
    class_enum = meta["class_enum"]
    bounds = meta["bounds"]
    width = bounds["width"]
    height = bounds["height"]
    x_min = bounds["x_min"]
    y_min = bounds["y_min"]
    rng = np.random.RandomState(seed)

    print(f"\n{'='*60}")
    print(f"  Query {query_id}: {qt['name']}")
    print(f"  Classes: {class_labels[:3]}{'...' if len(class_labels) > 3 else ''}")
    print(f"  Viewport: {viewport_frac*100:.0f}%  |  {n_trials} trials")
    print(f"{'='*60}")

    lats_hcci = []
    lats_gist = []
    lats_bbox = []

    for trial in range(n_trials):
        vw = width * float(np.sqrt(viewport_frac))
        vh = height * float(np.sqrt(viewport_frac))
        x0 = float(x_min + rng.uniform(0, max(0.0001, width - vw)))
        y0 = float(y_min + rng.uniform(0, max(0.0001, height - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # HCCI query (direct Hilbert ranges)
        hcci_sql, hcci_params = hcci.build_hcci_query(
            TABLE, DATASET_ID, class_labels,
            x0, y0, x1, y1, width, height,
            config.HILBERT_ORDER, use_direct=True,
            class_enum=class_enum,
            id_column="dataset_id",
            x_origin=x_min, y_origin=y_min,
        )
        _, t_hcci = time_query(conn, hcci_sql, hcci_params)
        lats_hcci.append(t_hcci)

        # GiST + ST_Intersects baseline
        if len(class_labels) >= 10:
            gist_sql, gist_params = hcci.build_baseline_gist_query_all_classes(
                TABLE, DATASET_ID, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        else:
            gist_sql, gist_params = hcci.build_baseline_gist_query(
                TABLE, DATASET_ID, class_labels, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        _, t_gist = time_query(conn, gist_sql, gist_params)
        lats_gist.append(t_gist)

        # GiST + bbox baseline
        if len(class_labels) >= 10:
            bbox_sql, bbox_params = hcci.build_baseline_bbox_query_all_classes(
                TABLE, DATASET_ID, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        else:
            bbox_sql, bbox_params = hcci.build_baseline_bbox_query(
                TABLE, DATASET_ID, class_labels, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        _, t_bbox = time_query(conn, bbox_sql, bbox_params)
        lats_bbox.append(t_bbox)

        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial+1}/{n_trials}: "
                  f"HCCI={np.median(lats_hcci):.2f}ms  "
                  f"GiST={np.median(lats_gist):.2f}ms  "
                  f"bbox={np.median(lats_bbox):.2f}ms")

    stats_h = compute_stats(lats_hcci)
    stats_g = compute_stats(lats_gist)
    stats_b = compute_stats(lats_bbox)

    _, wsr_p = wilcoxon_ranksum(lats_hcci, lats_gist)

    sp_p50 = stats_g["p50"] / stats_h["p50"] if stats_h["p50"] > 0 else 0
    sp_mean = stats_g["mean"] / stats_h["mean"] if stats_h["mean"] > 0 else 0
    sp_bbox = stats_b["p50"] / stats_h["p50"] if stats_h["p50"] > 0 else 0

    print(f"\n  {'':>20} {'p50':>8} {'p95':>8} {'mean':>8}")
    print(f"  {'HCCI':>20} {stats_h['p50']:>8.2f} {stats_h['p95']:>8.2f} {stats_h['mean']:>8.2f}")
    print(f"  {'GiST+STI':>20} {stats_g['p50']:>8.2f} {stats_g['p95']:>8.2f} {stats_g['mean']:>8.2f}")
    print(f"  {'GiST+bbox':>20} {stats_b['p50']:>8.2f} {stats_b['p95']:>8.2f} {stats_b['mean']:>8.2f}")
    print(f"  HCCI vs GiST: {sp_p50:.2f}x (p50), {sp_mean:.2f}x (mean)")
    print(f"  HCCI vs bbox: {sp_bbox:.2f}x (p50)")
    print(f"  Wilcoxon p = {wsr_p:.2e}")

    save_raw_latencies(lats_hcci, f"osm_{query_id}", "HCCI")
    save_raw_latencies(lats_gist, f"osm_{query_id}", "GiST")
    save_raw_latencies(lats_bbox, f"osm_{query_id}", "bbox")

    return {
        "query_id": query_id,
        "query_name": qt["name"],
        "classes": class_labels,
        "viewport_frac": viewport_frac,
        "n_trials": n_trials,
        "hcci": stats_h,
        "gist_st_intersects": stats_g,
        "gist_bbox": stats_b,
        "speedup_p50": round(sp_p50, 3),
        "speedup_mean": round(sp_mean, 3),
        "speedup_vs_bbox_p50": round(sp_bbox, 3),
        "wilcoxon_p": wsr_p,
    }


# ---------------------------------------------------------------------------
# FP measurement
# ---------------------------------------------------------------------------

def run_fp_measurement(conn, meta: dict, n_trials: int = 50) -> dict:
    """Measure empirical FP rate on OSM data."""
    class_enum = meta["class_enum"]
    bounds = meta["bounds"]
    width = bounds["width"]
    height = bounds["height"]
    x_min = bounds["x_min"]
    y_min = bounds["y_min"]

    # Pick the most common category for FP measurement
    cats = analyze_categories(conn)
    test_cat = cats[0]["category"]

    rng = np.random.RandomState(config.RANDOM_SEED + 777)
    print(f"\n  FP measurement ({n_trials} trials, category='{test_cat}', 5% viewport)...")

    fp_rates = []
    theoretical = hcci.false_positive_rate(0.05, config.HILBERT_ORDER)

    for _ in range(n_trials):
        vw = width * float(np.sqrt(0.05))
        vh = height * float(np.sqrt(0.05))
        x0 = float(x_min + rng.uniform(0, max(0.0001, width - vw)))
        y0 = float(y_min + rng.uniform(0, max(0.0001, height - vh)))

        result = hcci.measure_false_positive_rate(
            conn, TABLE, DATASET_ID, [test_cat],
            x0, y0, float(x0 + vw), float(y0 + vh),
            width, height, config.HILBERT_ORDER,
            use_direct=True,
            id_column="dataset_id",
            class_enum=class_enum,
            srid=4326,
            x_origin=x_min, y_origin=y_min,
        )
        fp_rates.append(result["fp_rate"])

    return {
        "n_trials": n_trials,
        "category": test_cat,
        "theoretical_bound": round(theoretical, 4),
        "empirical_mean": round(float(np.mean(fp_rates)), 4),
        "empirical_max": round(float(np.max(fp_rates)), 4),
        "empirical_p95": round(float(np.percentile(fp_rates, 95)), 4),
        "within_bound": sum(1 for r in fp_rates if r <= theoretical * 1.1) / len(fp_rates),
    }


# ---------------------------------------------------------------------------
# I/O decomposition
# ---------------------------------------------------------------------------

def run_io_decomposition(conn, query_id: str, qt: dict, meta: dict,
                         n_trials: int = 50) -> dict:
    """EXPLAIN BUFFERS decomposition for one query type."""
    class_enum = meta["class_enum"]
    bounds = meta["bounds"]
    width = bounds["width"]
    height = bounds["height"]
    x_min = bounds["x_min"]
    y_min = bounds["y_min"]
    class_labels = qt["classes"]
    viewport_frac = qt["viewport_frac"]
    rng = np.random.RandomState(config.RANDOM_SEED + 999)

    print(f"\n  I/O decomposition for Query {query_id} ({n_trials} trials)...")

    hcci_buffers = []
    gist_buffers = []

    for _ in range(n_trials):
        vw = width * float(np.sqrt(viewport_frac))
        vh = height * float(np.sqrt(viewport_frac))
        x0 = float(x_min + rng.uniform(0, max(0.0001, width - vw)))
        y0 = float(y_min + rng.uniform(0, max(0.0001, height - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        hcci_sql, hcci_p = hcci.build_hcci_query(
            TABLE, DATASET_ID, class_labels,
            x0, y0, x1, y1, width, height,
            config.HILBERT_ORDER, use_direct=True,
            class_enum=class_enum,
            id_column="dataset_id",
            x_origin=x_min, y_origin=y_min,
        )
        _, hb = time_query_buffers(conn, hcci_sql, hcci_p)
        hcci_buffers.append(hb)

        if len(class_labels) >= 10:
            gist_sql, gist_p = hcci.build_baseline_gist_query_all_classes(
                TABLE, DATASET_ID, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        else:
            gist_sql, gist_p = hcci.build_baseline_gist_query(
                TABLE, DATASET_ID, class_labels, x0, y0, x1, y1,
                id_column="dataset_id", srid=4326,
            )
        _, gb = time_query_buffers(conn, gist_sql, gist_p)
        gist_buffers.append(gb)

    def _avg(buf_list):
        keys = ["shared_hit", "shared_read", "heap_fetches",
                "rows_removed_by_filter", "actual_rows",
                "planning_time", "execution_time"]
        return {k: round(np.mean([b.get(k, 0) for b in buf_list]), 1) for k in keys}

    hcci_avg = _avg(hcci_buffers)
    gist_avg = _avg(gist_buffers)

    hcci_total = hcci_avg["shared_hit"] + hcci_avg["shared_read"]
    gist_total = gist_avg["shared_hit"] + gist_avg["shared_read"]
    buf_reduction = (1 - hcci_total / max(1, gist_total)) * 100

    hcci_types = set()
    for b in hcci_buffers:
        hcci_types.update(b.get("node_types", []))

    print(f"    GiST: hit={gist_avg['shared_hit']:.0f} read={gist_avg['shared_read']:.0f} "
          f"heap_fetch={gist_avg['heap_fetches']:.0f} filter={gist_avg['rows_removed_by_filter']:.0f}")
    print(f"    HCCI: hit={hcci_avg['shared_hit']:.0f} read={hcci_avg['shared_read']:.0f} "
          f"heap_fetch={hcci_avg['heap_fetches']:.0f} filter={hcci_avg['rows_removed_by_filter']:.0f}")
    print(f"    Index-only scan: {'YES' if 'Index Only Scan' in hcci_types else 'NO'}")
    print(f"    Buffer reduction: {buf_reduction:.1f}%")

    return {
        "hcci": hcci_avg,
        "gist": gist_avg,
        "hcci_index_only": "Index Only Scan" in hcci_types,
        "buffer_reduction_pct": round(buf_reduction, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OSM HCCI benchmark")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 100 trials, skip I/O")
    parser.add_argument("--skip-io", action="store_true")
    parser.add_argument("--skip-fp", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.trials = 100
        args.skip_io = True

    print("=" * 60)
    print("  OSM HCCI Benchmark: Urban Geospatial Validation")
    print("  Dataset: NYC OpenStreetMap POIs")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = False

    # Load metadata
    meta = load_osm_metadata()
    print(f"\n  Table: {TABLE}")
    print(f"  Bounds: x=[{meta['bounds']['x_min']:.4f}, {meta['bounds']['x_max']:.4f}]")
    print(f"  Bounds: y=[{meta['bounds']['y_min']:.4f}, {meta['bounds']['y_max']:.4f}]")
    print(f"  Categories: {len(meta['class_enum'])}")

    # Analyze categories and select query types
    categories = analyze_categories(conn)
    total_pois = sum(c["count"] for c in categories)
    print(f"  Total POIs: {total_pois:,}")
    print(f"\n  Category selectivity distribution:")
    for c in categories[:10]:
        print(f"    {c['category']:<35} {c['count']:>8,}  ({c['selectivity']:.1%})")
    if len(categories) > 10:
        print(f"    ... {len(categories) - 10} more categories")

    query_types = select_query_types(categories)

    t_start = time.time()
    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table": TABLE,
        "dataset": "NYC OpenStreetMap",
        "total_pois": total_pois,
        "n_categories": len(meta["class_enum"]),
        "hilbert_order": config.HILBERT_ORDER,
        "queries": {},
    }

    # Run query benchmarks
    for qid in sorted(query_types.keys()):
        qt = query_types[qid]
        result = run_query_type(conn, qid, qt, meta, n_trials=args.trials)
        all_results["queries"][qid] = result

    # I/O decomposition
    if not args.skip_io:
        print(f"\n{'='*60}")
        print(f"  I/O Decomposition")
        print(f"{'='*60}")
        io_results = {}
        for qid in ["A", "D"]:
            io_results[qid] = run_io_decomposition(
                conn, qid, query_types[qid], meta, n_trials=50,
            )
        all_results["io_decomposition"] = io_results

    # FP measurement
    if not args.skip_fp:
        print(f"\n{'='*60}")
        print(f"  False Positive Rate")
        print(f"{'='*60}")
        fp_results = run_fp_measurement(conn, meta, n_trials=50)
        all_results["false_positive"] = fp_results
        print(f"\n  Theoretical bound: {fp_results['theoretical_bound']:.1%}")
        print(f"  Empirical mean:    {fp_results['empirical_mean']:.1%}")
        print(f"  Empirical max:     {fp_results['empirical_max']:.1%}")
        print(f"  Within bound:      {fp_results['within_bound']:.0%} of trials")

    # Summary
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  OSM HCCI Benchmark Summary")
    print(f"{'='*60}")
    for qid, qr in all_results["queries"].items():
        sp = qr["speedup_p50"]
        print(f"  Query {qid} ({qr['query_name']:<40}): "
              f"HCCI={qr['hcci']['p50']:.2f}ms  GiST={qr['gist_st_intersects']['p50']:.2f}ms  "
              f"speedup={sp:.1f}x")

    print(f"\n  Total time: {total:.0f}s ({total/60:.1f}m)")

    path = save_results(all_results, "osm_hcci_benchmark")
    print(f"  Results saved to {path}")

    conn.close()


if __name__ == "__main__":
    main()
