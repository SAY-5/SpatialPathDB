"""Q2: k-Nearest Neighbor query benchmark."""

import numpy as np
import psycopg2

from spdb import config
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    get_slide_dimensions, random_point, time_query, warmup_cache,
    print_comparison,
)

CONFIGS = config.BENCH_CONFIGS


def knn_query_sql(table_name, k):
    return f"""
        SELECT object_id, centroid_x, centroid_y, class_label,
               geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) AS dist
        FROM {table_name}
        WHERE slide_id = %s
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0)
        LIMIT {k}
    """


def run_q2(conn, table_name, slide_ids, metadata, k=50, n_trials=500, seed=42):
    """Run Q2 kNN benchmark."""
    rng = np.random.RandomState(seed)
    latencies = []
    rings_needed = []

    warmup_cache(conn, table_name)

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        qx, qy = random_point(w, h, rng)

        sql = knn_query_sql(table_name, k)
        rows, elapsed = time_query(conn, sql, (qx, qy, sid, qx, qy))
        latencies.append(elapsed)
        rings_needed.append(1)

    return latencies, rings_needed


def run_q2_all_configs(k=50, n_trials=500, seed=42):
    """Run Q2 across all configurations."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    all_results = {}
    all_latencies = {}

    for name, table in CONFIGS.items():
        print(f"  Running Q2 (k={k}) on {name}...")
        lats, rings = run_q2(conn, table, slide_ids, metadata,
                             k=k, n_trials=n_trials, seed=seed)
        stats = compute_stats(lats)
        stats["rings_mean"] = float(np.mean(rings))
        all_results[name] = stats
        all_latencies[name] = lats
        save_raw_latencies(lats, f"q2_knn_k{k}", name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    results = {
        "query": "Q2_knn",
        "k": k,
        "n_trials": n_trials,
        "configs": all_results,
    }
    save_results(results, f"q2_knn_k{k}")
    print_comparison(all_results)
    conn.close()
    return results, all_latencies


if __name__ == "__main__":
    run_q2_all_configs()
