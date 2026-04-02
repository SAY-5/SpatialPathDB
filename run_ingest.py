#!/usr/bin/env python3
"""Run full SpatialPathDB ingestion pipeline with hardcoded patient list.

Ingests data into all 7 database configurations:
  Mono, Mono-T, Mono-C, SO, SO-C, SPDB, SPDB-Z
"""

import sys
import os
import time
import json
import gc

sys.path.insert(0, os.path.dirname(__file__))

from spdb import config, schema
from spdb.ingest import download_patient, transform_patient, _copy_chunk_numpy

SELECTED_PATIENTS = [
    "bcr_patient_barcode=TCGA-2F-A9KO",
    "bcr_patient_barcode=TCGA-4Z-AA7R",
    "bcr_patient_barcode=TCGA-5N-A9KI",
    "bcr_patient_barcode=TCGA-BT-A0YX",
    "bcr_patient_barcode=TCGA-BT-A42B",
    "bcr_patient_barcode=TCGA-CF-A27C",
    "bcr_patient_barcode=TCGA-CF-A5UA",
    "bcr_patient_barcode=TCGA-CU-A0YO",
    "bcr_patient_barcode=TCGA-E7-A4IJ",
    "bcr_patient_barcode=TCGA-E7-A7PW",
    "bcr_patient_barcode=TCGA-FD-A3B6",
    "bcr_patient_barcode=TCGA-FD-A43N",
    "bcr_patient_barcode=TCGA-FD-A5BT",
    "bcr_patient_barcode=TCGA-FD-A62P",
    "bcr_patient_barcode=TCGA-FD-A6TK",
    "bcr_patient_barcode=TCGA-G2-A2EJ",
    "bcr_patient_barcode=TCGA-G2-AA3B",
    "bcr_patient_barcode=TCGA-GC-A3RD",
    "bcr_patient_barcode=TCGA-GD-A3OP",
    "bcr_patient_barcode=TCGA-GU-A764",
    "bcr_patient_barcode=TCGA-GV-A3QK",
    "bcr_patient_barcode=TCGA-K4-A3WU",
    "bcr_patient_barcode=TCGA-K4-A6MB",
    "bcr_patient_barcode=TCGA-UY-A8OB",
    "bcr_patient_barcode=TCGA-XF-A8HH",
    "bcr_patient_barcode=TCGA-XF-A9ST",
    "bcr_patient_barcode=TCGA-XF-A9T4",
    "bcr_patient_barcode=TCGA-XF-AAMJ",
    "bcr_patient_barcode=TCGA-XF-AAMZ",
]

TABLES = config.ALL_TABLES


def main():
    P = config.HILBERT_ORDER
    T = config.BUCKET_TARGET
    print(f"=== SpatialPathDB Ingestion: {len(SELECTED_PATIENTS)} slides, "
          f"p={P}, T={T}, {len(TABLES)} configs ===")

    # download + transform
    all_metas = {}
    object_counts = {}
    slide_ids = []
    total_objects = 0
    parquet_paths = {}

    for i, patient_dir in enumerate(SELECTED_PATIENTS):
        print(f"\n[{i+1}/{len(SELECTED_PATIENTS)}] {patient_dir}")
        try:
            t0 = time.time()
            path = download_patient(patient_dir)
            dl = time.time() - t0

            t0 = time.time()
            df, meta = transform_patient(path, p=P, bucket_target=T)
            tx = time.time() - t0

            sid = meta["slide_id"]
            slide_ids.append(sid)
            all_metas[sid] = meta
            object_counts[sid] = meta["num_objects"]
            total_objects += meta["num_objects"]
            parquet_paths[sid] = path

            del df
            gc.collect()

            print(f"  {sid}: {meta['num_objects']:,} objects "
                  f"({meta['image_width']:.0f}x{meta['image_height']:.0f}px) "
                  f"[dl={dl:.1f}s tx={tx:.1f}s]")
        except Exception as e:
            print(f"  SKIP: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total: {total_objects:,} objects across {len(slide_ids)} slides")
    print(f"{'='*60}")

    # setup schemas
    conn = schema.get_connection()
    conn.autocommit = True
    schema.drop_all(conn)
    conn.autocommit = False

    print("\nCreating schemas...")

    # Unpartitioned tables
    schema.create_monolithic(conn)
    schema.create_monolithic(conn, config.TABLE_MONO_TUNED)
    schema.create_monolithic_clustered(conn)

    # Partitioned tables
    schema.create_slide_only(conn)
    schema.create_slide_only_clustered(conn)
    schema.create_spdb(conn)
    schema.create_spdb(conn, config.TABLE_SPDB_ZORDER)

    for sid in slide_ids:
        n = object_counts[sid]
        num_buckets = max(1, n // T)

        schema.add_slide_partition_so(conn, sid)
        schema.add_slide_partition_soc(conn, sid)
        schema.add_slide_hilbert_partitions(conn, sid, num_buckets)
        schema.add_slide_hilbert_partitions(
            conn, sid, num_buckets,
            table_name=config.TABLE_SPDB_ZORDER,
            key_col="zorder_key",
        )
    print(f"  Partitions created for {len(slide_ids)} slides.")

    # ingest slides
    print(f"\nIngesting into {len(TABLES)} tables...")
    t_ingest_total = time.time()

    for idx, sid in enumerate(slide_ids):
        t0 = time.time()
        df, _ = transform_patient(parquet_paths[sid], p=P, bucket_target=T)

        for tbl in TABLES:
            _copy_chunk_numpy(conn, tbl, df)

        elapsed = time.time() - t0
        n = len(df)
        rate = n * len(TABLES) / elapsed
        del df
        gc.collect()
        print(f"  [{idx+1}/{len(slide_ids)}] {sid}: {n:,} rows x {len(TABLES)} tables "
              f"in {elapsed:.1f}s ({rate:.0f} rows/sec)")

    total_elapsed = time.time() - t_ingest_total
    total_rows = total_objects * len(TABLES)
    print(f"\n  Ingestion: {total_rows:,} total rows in {total_elapsed:.1f}s "
          f"({total_rows/total_elapsed:.0f} rows/sec)")

    # build indexes
    print("\nBuilding indexes...")
    t0 = time.time()

    schema.index_monolithic(conn)
    schema.index_monolithic(conn, config.TABLE_MONO_TUNED)
    print("  Mono + Mono-T indexes done.")

    print("  Clustering Mono-C (this reorders the full table by Hilbert via GiST)...")
    schema.index_monolithic_clustered(conn)
    print("  Mono-C CLUSTER + BRIN done.")

    schema.index_slide_only(conn, slide_ids)
    print("  SO indexes done.")

    print("  Clustering SO-C partitions...")
    schema.index_slide_only_clustered(conn, slide_ids)
    print("  SO-C CLUSTER + BRIN done.")

    for sid in slide_ids:
        n = object_counts[sid]
        num_buckets = max(1, n // T)
        schema.index_spdb(conn, [sid], num_buckets)
        schema.index_spdb(conn, [sid], num_buckets,
                         table_name=config.TABLE_SPDB_ZORDER,
                         key_col="zorder_key")
    print(f"  SPDB + SPDB-Z indexes done.")
    print(f"  Total index time: {time.time()-t0:.1f}s")

    print("\nANALYZE...")
    schema.analyze_all(conn)

    # Verify
    print("\nVerification:")
    with conn.cursor() as cur:
        for tbl in TABLES:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                cnt = cur.fetchone()[0]
                print(f"  {tbl}: {cnt:,} rows")
            except Exception as e:
                print(f"  {tbl}: ERROR ({e})")
                conn.rollback()

    # Save metadata
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    meta_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "slide_ids": slide_ids,
            "object_counts": object_counts,
            "total_objects": total_objects,
            "metas": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in all_metas.items()},
            "hilbert_order": P,
            "bucket_target": T,
            "tables": TABLES,
        }, f, indent=2)
    print(f"\nMetadata: {meta_path}")

    conn.close()
    print("\n=== Ingestion Complete ===")


if __name__ == "__main__":
    main()
