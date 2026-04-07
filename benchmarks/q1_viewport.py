"""Q1: Viewport (spatial range) query benchmark.

For SPDB tables, the query includes a Hilbert key range predicate derived
from the viewport bounding box, enabling Level-2 partition pruning.
"""

import numpy as np
import psycopg2

from spdb import config, hilbert
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    get_slide_dimensions, random_viewport, time_query, warmup_cache,
    print_comparison, wilcoxon_ranksum,
)

CONFIGS = config.BENCH_CONFIGS

SPDB_TABLES = {config.TABLE_SPDB, config.TABLE_SPDB_ZORDER}


def _hilbert_key_ranges(bucket_ids, num_buckets, p):
    """Convert bucket IDs to contiguous hilbert_key ranges for SQL predicates.

    Groups adjacent buckets into contiguous ranges to minimise OR clauses.
    Returns list of (lo, hi) tuples.
    """
    total_cells = 1 << (2 * p)
    ranges = []
    for b in sorted(bucket_ids):
        lo = b * total_cells // num_buckets
        hi = (b + 1) * total_cells // num_buckets
        if ranges and ranges[-1][1] == lo:
            ranges[-1] = (ranges[-1][0], hi)
        else:
            ranges.append((lo, hi))
    return ranges


def _build_spdb_query(table_name, key_ranges):
    """Build Q1 SQL with Hilbert key range predicates for partition pruning."""
    hk_clauses = " OR ".join(
        f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
        for lo, hi in key_ranges
    )
    return f"""
        SELECT object_id, centroid_x, centroid_y, class_label
        FROM {table_name}
        WHERE slide_id = %s
          AND ({hk_clauses})
          AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
    """


def viewport_query_sql(table_name):
    return f"""
        SELECT object_id, centroid_x, centroid_y, class_label
        FROM {table_name}
        WHERE slide_id = %s
          AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
    """


def run_q1(conn, table_name, slide_ids, metadata, n_trials=500,
           viewport_frac=0.05, seed=42, hilbert_order=None):
    """Run Q1 viewport benchmark on a single table config.

    For SPDB tables, automatically adds Hilbert key range predicates.
    """
    rng = np.random.RandomState(seed)
    latencies = []
    is_spdb = table_name in SPDB_TABLES or table_name.startswith("objects_spdb_h")

    if hilbert_order is None:
        hilbert_order = config.HILBERT_ORDER

    warmup_cache(conn, table_name)

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)

        if is_spdb:
            n_obj = metadata["object_counts"].get(sid, 1_000_000)
            num_buckets = max(1, n_obj // config.BUCKET_TARGET)
            bucket_ids = hilbert.candidate_buckets_for_bbox(
                x0, y0, x1, y1, w, h, hilbert_order, num_buckets
            )
            key_ranges = _hilbert_key_ranges(bucket_ids, num_buckets, hilbert_order)
            sql = _build_spdb_query(table_name, key_ranges)
        else:
            sql = viewport_query_sql(table_name)

        _, elapsed = time_query(conn, sql, (sid, x0, y0, x1, y1))
        latencies.append(elapsed)

    return latencies


def run_q1_all_configs(n_trials=500, viewport_frac=0.05, seed=42):
    """Run Q1 across all configurations and compute statistics."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    all_results = {}
    all_latencies = {}

    for name, table in CONFIGS.items():
        print(f"  Running Q1 on {name} ({table})...")
        lats = run_q1(conn, table, slide_ids, metadata,
                      n_trials=n_trials, viewport_frac=viewport_frac, seed=seed)
        stats = compute_stats(lats)
        all_results[name] = stats
        all_latencies[name] = lats
        save_raw_latencies(lats, "q1_viewport", name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  "
              f"mean={stats['mean']:.1f}ms  std={stats['std']:.1f}ms")

    stat_tests = {}
    pairs = [("Mono", "SPDB"), ("SO", "SPDB"), ("Mono", "SO"),
             ("Mono-C", "SO-C"), ("SO-C", "SPDB"), ("Mono", "SO-C")]
    for a, b in pairs:
        if a in all_latencies and b in all_latencies:
            stat, p = wilcoxon_ranksum(all_latencies[a], all_latencies[b])
            stat_tests[f"{a}_vs_{b}"] = {"statistic": stat, "p_value": p}
            print(f"    Wilcoxon {a} vs {b}: p={p:.2e}")

    results = {
        "query": "Q1_viewport",
        "n_trials": n_trials,
        "viewport_frac": viewport_frac,
        "configs": all_results,
        "statistical_tests": stat_tests,
    }
    save_results(results, "q1_viewport")
    print_comparison(all_results)
    conn.close()
    return results, all_latencies


if __name__ == "__main__":
    run_q1_all_configs()
