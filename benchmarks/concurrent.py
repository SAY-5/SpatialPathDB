"""Concurrent throughput benchmark using asyncpg connection pool.

Uses prepared statements and plan_cache_mode=force_generic_plan to
mitigate partition planning overhead for highly partitioned tables.
For SPDB tables, queries include Hilbert key range predicates.
"""

import asyncio
import time
import numpy as np
import psycopg2

from spdb import config, hilbert
from benchmarks.framework import (
    compute_stats, save_results, load_metadata,
    get_slide_dimensions, random_viewport,
)


def _hilbert_key_ranges(bucket_ids, num_buckets, p):
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


async def _run_concurrent_worker(pool, table_name, slide_ids, metadata,
                                  viewport_frac, duration_sec, rng_seed, results_list,
                                  is_spdb=False):
    """Single async worker that runs viewport queries in a loop."""
    rng = np.random.RandomState(rng_seed)
    start = time.monotonic()
    count = 0
    errors = 0

    base_sql = (
        f"SELECT object_id, centroid_x, centroid_y "
        f"FROM {table_name} "
        f"WHERE slide_id = $1 "
        f"AND ST_Intersects(geom, ST_MakeEnvelope($2, $3, $4, $5, 0))"
    )

    while time.monotonic() - start < duration_sec:
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)

        if is_spdb:
            n_obj = metadata["object_counts"].get(sid, 1_000_000)
            num_buckets = max(1, n_obj // config.BUCKET_TARGET)
            bucket_ids = hilbert.candidate_buckets_for_bbox(
                x0, y0, x1, y1, w, h, config.HILBERT_ORDER, num_buckets
            )
            key_ranges = _hilbert_key_ranges(bucket_ids, num_buckets, config.HILBERT_ORDER)
            hk = " OR ".join(f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
                             for lo, hi in key_ranges)
            sql = (
                f"SELECT object_id, centroid_x, centroid_y "
                f"FROM {table_name} "
                f"WHERE slide_id = $1 AND ({hk}) "
                f"AND ST_Intersects(geom, ST_MakeEnvelope($2, $3, $4, $5, 0))"
            )
        else:
            sql = base_sql

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, sid, x0, y0, x1, y1)
                count += 1
        except Exception:
            errors += 1
            await asyncio.sleep(0.05)

    results_list.append({"queries": count, "errors": errors})


async def _run_concurrent_bench(table_name, n_clients, slide_ids, metadata,
                                 viewport_frac=0.05, duration_sec=30, seed=42):
    """Run concurrent throughput test with n_clients."""
    import asyncpg

    pool = await asyncpg.create_pool(
        config.asyncpg_dsn(),
        min_size=min(n_clients, 10),
        max_size=min(n_clients + 5, 80),
        command_timeout=60,
        server_settings={
            "plan_cache_mode": "force_generic_plan",
            "statement_timeout": "30000",
        },
    )

    is_spdb = table_name in (config.TABLE_SPDB, config.TABLE_SPDB_ZORDER)

    results_list = []
    tasks = []
    for i in range(n_clients):
        tasks.append(
            _run_concurrent_worker(
                pool, table_name, slide_ids, metadata,
                viewport_frac, duration_sec, seed + i, results_list,
                is_spdb=is_spdb,
            )
        )

    await asyncio.gather(*tasks)
    await pool.close()

    total_queries = sum(r["queries"] for r in results_list)
    total_errors = sum(r["errors"] for r in results_list)
    qps = total_queries / duration_sec

    return {
        "n_clients": n_clients,
        "duration_sec": duration_sec,
        "total_queries": total_queries,
        "total_errors": total_errors,
        "qps": round(qps, 2),
    }


def run_concurrent_all(levels=None, duration_sec=30, seed=42):
    """Run concurrency sweep across all configurations."""
    if levels is None:
        levels = config.CONCURRENCY_LEVELS

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]

    configs = {
        "Mono": config.TABLE_MONO,
        "SO": config.TABLE_SLIDE_ONLY,
        "SPDB": config.TABLE_SPDB,
    }

    all_results = {}

    for cfg_name, table in configs.items():
        all_results[cfg_name] = {}
        for n_clients in levels:
            print(f"  {cfg_name} @ {n_clients} clients...")
            result = asyncio.run(
                _run_concurrent_bench(
                    table, n_clients, slide_ids, metadata,
                    duration_sec=duration_sec, seed=seed,
                )
            )
            all_results[cfg_name][n_clients] = result
            print(f"    QPS={result['qps']:.1f}  errors={result['total_errors']}")

    save_results(all_results, "concurrent_throughput")

    print(f"\n{'Clients':<10}", end="")
    for cfg in configs:
        print(f"{cfg+' QPS':>12}", end="")
    print()
    print("-" * (10 + 12 * len(configs)))
    for n_clients in levels:
        print(f"{n_clients:<10}", end="")
        for cfg in configs:
            qps = all_results[cfg][n_clients]["qps"]
            print(f"{qps:>12.1f}", end="")
        print()

    return all_results


if __name__ == "__main__":
    run_concurrent_all()
