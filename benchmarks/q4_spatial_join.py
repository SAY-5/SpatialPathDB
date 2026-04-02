"""Q4: Spatial join benchmark."""

import numpy as np
import psycopg2

from spdb import config
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    get_slide_dimensions, random_viewport, time_query, warmup_cache,
    print_comparison,
)

CONFIGS = {
    "Mono": config.TABLE_MONO,
    "Mono-T": config.TABLE_MONO_TUNED,
    "SO": config.TABLE_SLIDE_ONLY,
    "SPDB": config.TABLE_SPDB,
}


def spatial_join_query_sql(table_name):
    return f"""
        SELECT a.object_id, a.class_label, a.centroid_x, a.centroid_y
        FROM {table_name} a
        WHERE a.slide_id = %s
          AND ST_Intersects(a.geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
          AND a.class_label IN ('Tumor', 'Lymphocyte')
        LIMIT 500
    """


def run_q4(conn, table_name, slide_ids, metadata, n_trials=100,
           viewport_frac=0.02, seed=42):
    rng = np.random.RandomState(seed)
    latencies = []

    warmup_cache(conn, table_name)

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)

        sql = spatial_join_query_sql(table_name)
        rows, elapsed = time_query(conn, sql, (sid, x0, y0, x1, y1))
        latencies.append(elapsed)

    return latencies


def run_q4_all_configs(n_trials=100, seed=42):
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    all_results = {}

    for name, table in CONFIGS.items():
        print(f"  Running Q4 on {name}...")
        lats = run_q4(conn, table, slide_ids, metadata, n_trials=n_trials, seed=seed)
        stats = compute_stats(lats)
        all_results[name] = stats
        save_raw_latencies(lats, "q4_spatial_join", name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    results = {"query": "Q4_spatial_join", "n_trials": n_trials, "configs": all_results}
    save_results(results, "q4_spatial_join")
    print_comparison(all_results)
    conn.close()
    return results


if __name__ == "__main__":
    run_q4_all_configs()
