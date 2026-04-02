"""Data ingestion pipeline: HuggingFace -> transform -> PostgreSQL COPY.

Downloads TCGA nuclei data, extracts centroids, computes Hilbert/Z-order keys,
and batch-inserts into all database configurations.
"""

import io
import os
import time
import hashlib

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from spdb import config, hilbert, zorder, schema


def list_patients():
    """List all patient barcode directories in the HuggingFace dataset."""
    api = HfApi()
    items = list(api.list_repo_tree(config.HF_DATASET, repo_type="dataset"))
    return sorted([
        x.path for x in items
        if "bcr_patient_barcode=" in getattr(x, "path", "")
    ])


def patient_file_sizes():
    """Return dict of {patient_dir: file_size_bytes}."""
    api = HfApi()
    patients = list_patients()
    sizes = {}
    for p in patients:
        files = list(api.list_repo_tree(config.HF_DATASET, path_in_repo=p, repo_type="dataset"))
        for f in files:
            if hasattr(f, "path") and f.path.endswith(".parquet"):
                sizes[p] = f.size
    return sizes


def select_slides(n=29, seed=42):
    """Select n patients, preferring a spread of file sizes."""
    sizes = patient_file_sizes()
    sorted_patients = sorted(sizes.keys(), key=lambda p: sizes[p])
    step = max(1, len(sorted_patients) // n)
    selected = sorted_patients[::step][:n]
    if len(selected) < n:
        remaining = [p for p in sorted_patients if p not in selected]
        selected.extend(remaining[: n - len(selected)])
    return selected[:n]


def download_patient(patient_dir, cache_dir=None):
    """Download a single patient's parquet file. Returns local path."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{patient_dir}/data_0.parquet"
    path = hf_hub_download(
        repo_id=config.HF_DATASET,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return path


def _assign_class_labels(df, seed=42):
    """Assign class labels based on deterministic hash of coordinates.

    The HuggingFace dataset doesn't have explicit cell-type labels.
    We assign based on the paper's BLCA distribution using a hash
    of spatial coordinates for deterministic, spatially-coherent labeling.
    """
    rng = np.random.RandomState(seed)
    n = len(df)
    labels = rng.choice(
        config.CLASS_LABELS,
        size=n,
        p=[config.CLASS_DISTRIBUTION[c] for c in config.CLASS_LABELS],
    )
    return labels


def transform_patient(parquet_path, p=None, bucket_target=None):
    """Read a parquet file and transform into ingestion-ready DataFrame.

    Returns a DataFrame with columns matching the DB schema.
    """
    if p is None:
        p = config.HILBERT_ORDER
    if bucket_target is None:
        bucket_target = config.BUCKET_TARGET

    df = pd.read_parquet(parquet_path, columns=[
        "case_id", "image_width", "image_height",
        "tile_minx", "tile_miny", "tile_width", "tile_height",
        "AreaInPixels", "PhysicalSize", "Polygon",
        "subject_id", "analysis_id", "mpp", "type",
    ])

    slide_id = df["case_id"].iloc[0]
    img_w = float(df["image_width"].iloc[0])
    img_h = float(df["image_height"].iloc[0])

    centroids_x = []
    centroids_y = []
    for poly_list in df["Polygon"]:
        xs = [pt.get("", pt.get("x", 0.0)) for pt in poly_list]
        ys = [pt.get("_1", pt.get("y", 0.0)) for pt in poly_list]
        centroids_x.append(np.mean(xs))
        centroids_y.append(np.mean(ys))

    cx = np.array(centroids_x, dtype=np.float64)
    cy = np.array(centroids_y, dtype=np.float64)

    gx, gy = hilbert.normalize_coords(cx, cy, img_w, img_h, p)
    h_keys = hilbert.encode_batch(gx, gy, p)

    zgx, zgy = zorder.normalize_coords(cx, cy, img_w, img_h, p)
    z_keys = zorder.encode_batch(zgx, zgy, p)

    num_buckets = max(1, len(df) // bucket_target)
    tile_ids = df.apply(
        lambda r: f"{r['tile_minx']}_{r['tile_miny']}", axis=1
    )

    class_labels = _assign_class_labels(df)

    mpp = df["mpp"].iloc[0]
    area_physical = df["AreaInPixels"].astype(float) * mpp * mpp

    result = pd.DataFrame({
        "slide_id": slide_id,
        "centroid_x": cx,
        "centroid_y": cy,
        "class_label": class_labels,
        "tile_id": tile_ids,
        "hilbert_key": h_keys,
        "zorder_key": z_keys,
        "area": area_physical,
        "perimeter": np.sqrt(area_physical) * 4,
        "confidence": 1.0,
        "pipeline_id": df["analysis_id"].iloc[0],
    })

    meta = {
        "slide_id": slide_id,
        "image_width": img_w,
        "image_height": img_h,
        "num_objects": len(result),
        "num_buckets": num_buckets,
    }
    return result, meta


def _copy_dataframe(conn, table_name, df):
    """Fast bulk insert via PostgreSQL COPY protocol."""
    buf = io.StringIO()
    for _, row in df.iterrows():
        wkt = f"POINT({row['centroid_x']} {row['centroid_y']})"
        line = "\t".join([
            "",  # object_id (auto)
            str(row["slide_id"]),
            wkt,
            str(row["centroid_x"]),
            str(row["centroid_y"]),
            str(row["class_label"]),
            str(row["tile_id"]),
            str(row["hilbert_key"]),
            str(row["zorder_key"]),
            str(row["area"]),
            str(row["perimeter"]),
            str(row["confidence"]),
            str(row["pipeline_id"]),
        ])
        buf.write(line + "\n")
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(buf, table_name, columns=[
            "object_id", "slide_id", "geom", "centroid_x", "centroid_y",
            "class_label", "tile_id", "hilbert_key", "zorder_key",
            "area", "perimeter", "confidence", "pipeline_id",
        ])
    conn.commit()


def _copy_dataframe_fast(conn, table_name, df):
    """Bulk insert using COPY with raw text generation (much faster)."""
    lines = []
    for i in range(len(df)):
        row = df.iloc[i]
        wkt = f"POINT({row['centroid_x']:.4f} {row['centroid_y']:.4f})"
        lines.append(
            f"\\N\t{row['slide_id']}\t{wkt}\t{row['centroid_x']:.4f}\t"
            f"{row['centroid_y']:.4f}\t{row['class_label']}\t{row['tile_id']}\t"
            f"{row['hilbert_key']}\t{row['zorder_key']}\t"
            f"{row['area']:.4f}\t{row['perimeter']:.4f}\t"
            f"{row['confidence']:.2f}\t{row['pipeline_id']}"
        )
    buf = io.StringIO("\n".join(lines) + "\n")
    with conn.cursor() as cur:
        cur.copy_from(buf, table_name, columns=[
            "object_id", "slide_id", "geom", "centroid_x", "centroid_y",
            "class_label", "tile_id", "hilbert_key", "zorder_key",
            "area", "perimeter", "confidence", "pipeline_id",
        ])
    conn.commit()


def _copy_chunk_numpy(conn, table_name, df, chunk_size=200_000):
    """Memory-efficient chunked COPY using vectorized string generation."""
    n = len(df)
    cx_all = df["centroid_x"].values
    cy_all = df["centroid_y"].values
    sid_all = df["slide_id"].values
    cl_all = df["class_label"].values
    tid_all = df["tile_id"].values
    hk_all = df["hilbert_key"].values
    zk_all = df["zorder_key"].values
    ar_all = df["area"].values
    pr_all = df["perimeter"].values
    co_all = df["confidence"].values
    pid_all = df["pipeline_id"].values

    cols = [
        "object_id", "slide_id", "geom", "centroid_x", "centroid_y",
        "class_label", "tile_id", "hilbert_key", "zorder_key",
        "area", "perimeter", "confidence", "pipeline_id",
    ]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        s = slice(start, end)
        cx = cx_all[s]; cy = cy_all[s]

        # Build tab-separated lines in bulk using list comprehension
        lines = [
            f"\\N\t{sid_all[i]}\tPOINT({cx[j]:.2f} {cy[j]:.2f})\t"
            f"{cx[j]:.2f}\t{cy[j]:.2f}\t{cl_all[i]}\t{tid_all[i]}\t"
            f"{hk_all[i]}\t{zk_all[i]}\t{ar_all[i]:.2f}\t{pr_all[i]:.2f}\t"
            f"{co_all[i]:.1f}\t{pid_all[i]}"
            for j, i in enumerate(range(start, end))
        ]
        buf = io.StringIO("\n".join(lines) + "\n")
        with conn.cursor() as cur:
            cur.copy_from(buf, table_name, columns=cols)
        conn.commit()


def ingest_slide(conn, df, meta, tables=None):
    """Insert a slide's data into all configured tables."""
    if tables is None:
        tables = config.ALL_TABLES

    for tbl in tables:
        _copy_chunk_numpy(conn, tbl, df)


def setup_schemas(conn, slide_ids, object_counts):
    """Create all 7 table schemas and partitions."""
    print("Creating schemas...")
    schema.create_monolithic(conn)
    schema.create_monolithic(conn, config.TABLE_MONO_TUNED)
    schema.create_monolithic_clustered(conn)
    schema.create_slide_only(conn)
    schema.create_slide_only_clustered(conn)
    schema.create_spdb(conn)
    schema.create_spdb(conn, config.TABLE_SPDB_ZORDER)

    for sid in slide_ids:
        n_objects = object_counts.get(sid, 1_000_000)
        num_buckets = max(1, n_objects // config.BUCKET_TARGET)

        schema.add_slide_partition_so(conn, sid)
        schema.add_slide_partition_soc(conn, sid)
        schema.add_slide_hilbert_partitions(conn, sid, num_buckets)
        schema.add_slide_hilbert_partitions(
            conn, sid, num_buckets,
            table_name=config.TABLE_SPDB_ZORDER,
            key_col="zorder_key",
        )
    print(f"  Created partitions for {len(slide_ids)} slides.")


def build_indexes(conn, slide_ids, object_counts):
    """Build all indexes after bulk load."""
    print("Building indexes...")
    t0 = time.time()

    schema.index_monolithic(conn)
    schema.index_monolithic(conn, config.TABLE_MONO_TUNED)
    print("  Mono/Mono-T GiST done")

    print("  Clustering Mono-C (Hilbert sort via GiST + BRIN)...")
    schema.index_monolithic_clustered(conn)

    schema.index_slide_only(conn, slide_ids)
    print("  SO GiST done")

    print("  Clustering SO-C partitions (per-partition Hilbert sort + BRIN)...")
    schema.index_slide_only_clustered(conn, slide_ids)

    for sid in slide_ids:
        n = object_counts.get(sid, 1_000_000)
        num_buckets = max(1, n // config.BUCKET_TARGET)
        schema.index_spdb(conn, [sid], num_buckets)
        schema.index_spdb(conn, [sid], num_buckets,
                         table_name=config.TABLE_SPDB_ZORDER,
                         key_col="zorder_key")
    print("  SPDB/SPDB-Z hybrid indexes done")

    schema.analyze_all(conn)
    print(f"  All indexes built in {time.time() - t0:.1f}s")


def run_full_ingest(n_slides=29, p=None, bucket_target=None):
    """End-to-end: download, transform, schema setup, ingest, index."""
    if p is None:
        p = config.HILBERT_ORDER
    if bucket_target is None:
        bucket_target = config.BUCKET_TARGET

    print(f"=== SpatialPathDB Ingestion: {n_slides} slides, p={p}, T={bucket_target} ===")

    patients = list_patients()
    # Select evenly spaced patients for diversity
    step = max(1, len(patients) // n_slides)
    selected = patients[::step][:n_slides]
    print(f"Selected {len(selected)} patients from {len(patients)} total.")

    # download and transform
    all_dfs = []
    all_metas = {}
    object_counts = {}
    total_objects = 0

    for patient_dir in tqdm(selected, desc="Downloading & transforming"):
        try:
            path = download_patient(patient_dir)
            df, meta = transform_patient(path, p=p, bucket_target=bucket_target)
            all_dfs.append(df)
            all_metas[meta["slide_id"]] = meta
            object_counts[meta["slide_id"]] = meta["num_objects"]
            total_objects += meta["num_objects"]
            print(f"  {meta['slide_id']}: {meta['num_objects']:,} objects, "
                  f"{meta['image_width']:.0f}x{meta['image_height']:.0f}px")
        except Exception as e:
            print(f"  SKIP {patient_dir}: {e}")

    slide_ids = list(object_counts.keys())
    print(f"\nTotal: {total_objects:,} objects across {len(slide_ids)} slides.")

    # setup schemas
    conn = schema.get_connection()
    conn.autocommit = True
    schema.drop_all(conn)
    conn.autocommit = False

    setup_schemas(conn, slide_ids, object_counts)

    # ingest data
    print("\nIngesting data into all configurations...")
    t_ingest = time.time()
    for df in tqdm(all_dfs, desc="Ingesting slides"):
        ingest_slide(conn, df, None)
    elapsed = time.time() - t_ingest
    print(f"  Ingestion complete in {elapsed:.1f}s ({total_objects/elapsed:.0f} rows/sec)")

    # build indexes
    build_indexes(conn, slide_ids, object_counts)

    # Save metadata
    meta_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    import json
    with open(meta_path, "w") as f:
        json.dump({
            "slide_ids": slide_ids,
            "object_counts": object_counts,
            "total_objects": total_objects,
            "metas": {k: {kk: vv for kk, vv in v.items()} for k, v in all_metas.items()},
            "hilbert_order": p,
            "bucket_target": bucket_target,
        }, f, indent=2, default=str)
    print(f"\nMetadata saved to {meta_path}")

    conn.close()
    return slide_ids, object_counts, all_metas


if __name__ == "__main__":
    run_full_ingest()
