"""Parallel Gaia DR3 ingestion: run multiple TAP sessions for faster download.

Splits sky patches across N workers, each downloading to separate cache files
and temp tables, then merges into the main gaia_sources table.

Usage:
    python -m benchmarks.gaia_parallel_ingest
    python -m benchmarks.gaia_parallel_ingest --workers 4 --target-rows 50000000
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci
from benchmarks.gaia_ingest import (
    TABLE, INDEX_HCCI, INDEX_GIST, DATASET_ID,
    CACHE_DIR, SKY_PATCHES, COLOR_CLASSES,
    query_gaia_patch, _parse_csv_cache,
    create_table, load_data, build_geometry,
    compute_hilbert_keys, build_class_enum, compute_composite_keys,
    build_gist_index, build_hcci_index, vacuum_analyze,
    verify, save_metadata,
)


def download_worker(args):
    """Worker function: download a subset of patches."""
    worker_id, patches, max_rows_per_patch = args
    print(f"  [Worker {worker_id}] Starting with {len(patches)} patches...")

    all_records = []
    for dec_min, dec_max, ra_min, ra_max, mag_limit in patches:
        try:
            records = query_gaia_patch(
                dec_min, dec_max, ra_min, ra_max,
                mag_limit, max_rows=max_rows_per_patch,
            )
            all_records.extend(records)
            print(f"  [W{worker_id}] dec [{dec_min:+.0f},{dec_max:+.0f}] "
                  f"ra [{ra_min:.0f},{ra_max:.0f}]: {len(records):,} "
                  f"(subtotal: {len(all_records):,})")
        except Exception as e:
            print(f"  [W{worker_id}] FAILED dec [{dec_min},{dec_max}] "
                  f"ra [{ra_min},{ra_max}]: {e}")

    print(f"  [Worker {worker_id}] Done: {len(all_records):,} total records")
    return all_records


def main():
    parser = argparse.ArgumentParser(description="Parallel Gaia DR3 Ingestion")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-rows", type=int, default=50_000_000)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gaia DR3 Parallel Ingestion")
    print(f"  Target: {args.target_rows:,} sources")
    print(f"  Workers: {args.workers}")
    print(f"  Patches: {len(SKY_PATCHES)}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())

    if args.stats_only:
        class_enum = build_class_enum(conn)
        verify(conn, class_enum)
        conn.close()
        return

    t_start = time.time()

    if not args.index_only:
        # Split patches across workers
        max_rows_per_patch = args.target_rows // len(SKY_PATCHES) + 1
        worker_args = []
        for i in range(args.workers):
            worker_patches = SKY_PATCHES[i::args.workers]
            worker_args.append((i, worker_patches, max_rows_per_patch))

        print(f"\n[Download] Launching {args.workers} parallel download workers...")

        # Use multiprocessing for true parallelism
        with multiprocessing.Pool(args.workers) as pool:
            results = pool.map(download_worker, worker_args)

        # Merge all records
        all_records = []
        for worker_records in results:
            all_records.extend(worker_records)

        # Deduplicate by source_id
        seen = set()
        unique_records = []
        for r in all_records:
            if r["source_id"] not in seen:
                seen.add(r["source_id"])
                unique_records.append(r)

        print(f"\n  Total: {len(all_records):,} records "
              f"({len(unique_records):,} unique)")

        if len(unique_records) > args.target_rows:
            unique_records = unique_records[:args.target_rows]
            print(f"  Trimmed to {args.target_rows:,}")

        all_records = unique_records
        dl_time = time.time() - t_start
        print(f"  Download time: {dl_time:.0f}s ({dl_time/60:.1f}m)")

        # Load into database
        create_table(conn)
        load_data(conn, all_records)
        build_geometry(conn)

    # Build indexes
    bounds = compute_hilbert_keys(conn)
    class_enum = build_class_enum(conn)
    compute_composite_keys(conn, class_enum)
    build_hcci_index(conn)
    build_gist_index(conn)
    vacuum_analyze(conn)
    verify(conn, class_enum)

    # Save metadata
    meta_path = os.path.join(config.RAW_DIR, "gaia_metadata.json")
    save_metadata(bounds, class_enum, meta_path)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total time: {total:.0f}s ({total/60:.1f}m)")
    print(f"{'='*60}")

    conn.close()


if __name__ == "__main__":
    main()
