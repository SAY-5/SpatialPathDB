"""Q3: Aggregation (density estimation) benchmark."""

import numpy as np
import psycopg2

from spdb import config
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    time_query, warmup_cache, print_comparison,
)

CONFIGS = config.BENCH_CONFIGS


def aggregation_query_sql(table_name):
    return f"""
        SELECT tile_id, class_label, COUNT(*) AS cnt
        FROM {table_name}
        WHERE slide_id = %s
        GROUP BY tile_id, class_label
        ORDER BY cnt DESC
    """


def run_q3(conn, table_name, slide_ids, n_trials=500, seed=42):
    rng = np.random.RandomState(seed)
    latencies = []

    warmup_cache(conn, table_name)

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        sql = aggregation_query_sql(table_name)
        _, elapsed = time_query(conn, sql, (sid,))
        latencies.append(elapsed)

    return latencies


def run_q3_all_configs(n_trials=500, seed=42):
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    all_results = {}

    for name, table in CONFIGS.items():
        print(f"  Running Q3 on {name}...")
        lats = run_q3(conn, table, slide_ids, n_trials=n_trials, seed=seed)
        stats = compute_stats(lats)
        all_results[name] = stats
        save_raw_latencies(lats, "q3_aggregation", name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    results = {"query": "Q3_aggregation", "n_trials": n_trials, "configs": all_results}
    save_results(results, "q3_aggregation")
    print_comparison(all_results)
    conn.close()
    return results


if __name__ == "__main__":
    run_q3_all_configs()
