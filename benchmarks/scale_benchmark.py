#!/usr/bin/env python3
"""
scale_benchmark.py  --  Scale SpatialPathDB to 200M+ nuclei and run full benchmark suite.

Target environment: AWS r5.4xlarge (128 GB RAM), PostgreSQL + PostGIS, database "spdb".
Data source: HuggingFace dataset "longevity-db/pan-cancer-nuclei-seg" (BLCA cohort).

Usage:
    python scale_benchmark.py            # ingest + benchmark
    python scale_benchmark.py --bench    # skip ingestion, only benchmark
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import textwrap
import time
from io import StringIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure required packages
# ---------------------------------------------------------------------------

def _ensure_packages():
    """Install missing Python packages."""
    required = {
        "psycopg2": "psycopg2-binary",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "huggingface_hub": "huggingface_hub",
    }
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"[setup] Installing {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg]
            )

_ensure_packages()

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_NAME = "spdb"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

HF_DATASET = "longevity-db/pan-cancer-nuclei-seg"
HF_REPO_TYPE = "dataset"
BLCA_PREFIX = "BLCA/"

HILBERT_ORDER = 8            # p = 8  => grid 256x256
HILBERT_N = 1 << HILBERT_ORDER   # 256
HILBERT_MAX_D = HILBERT_N * HILBERT_N  # 65536

TARGET_NUCLEI = 200_000_000  # 200M target
BATCH_SIZE = 10              # slides per memory batch
BENCHMARK_TRIALS = 500
COLD_CACHE_TRIALS = 100
KNN_K = 50

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

# ---------------------------------------------------------------------------
# Hilbert & Z-order helpers (pure Python, no external deps)
# ---------------------------------------------------------------------------

def xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) on an n x n Hilbert grid to distance d."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d


def z_order(gx: int, gy: int) -> int:
    """Bit-interleave two 8-bit grid coordinates into a 16-bit Z-value."""
    z = 0
    for i in range(8):
        z |= ((gx >> i) & 1) << (2 * i)
        z |= ((gy >> i) & 1) << (2 * i + 1)
    return z


def compute_keys(
    cx: pd.Series, cy: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return (hilbert_key, z_key) series for centroid arrays.

    Maps centroids into a [0, 255] grid based on the slide's own extent,
    then computes Hilbert distance (order 8) and Z-order key.
    """
    xmin, xmax = cx.min(), cx.max()
    ymin, ymax = cy.min(), cy.max()
    span_x = max(xmax - xmin, 1.0)
    span_y = max(ymax - ymin, 1.0)

    gx = ((cx - xmin) / span_x * (HILBERT_N - 1)).astype(int).clip(0, HILBERT_N - 1)
    gy = ((cy - ymin) / span_y * (HILBERT_N - 1)).astype(int).clip(0, HILBERT_N - 1)

    h_keys = [xy2d(HILBERT_N, int(x), int(y)) for x, y in zip(gx, gy)]
    z_keys = [z_order(int(x), int(y)) for x, y in zip(gx, gy)]

    return pd.Series(h_keys, dtype="int64"), pd.Series(z_keys, dtype="int64")


# ---------------------------------------------------------------------------
# TCGA barcode → slide_id
# ---------------------------------------------------------------------------

def barcode_to_patient_id(patient_dir: str) -> str:
    """Extract patient barcode from HuggingFace directory name.

    Example: 'bcr_patient_barcode=TCGA-2F-A9KO' -> 'TCGA-2F-A9KO'
    """
    name = patient_dir.split("=", 1)[-1] if "=" in patient_dir else patient_dir
    return name


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT
    )


def get_loaded_slides(conn) -> set[str]:
    """Return set of patient barcodes already present in objects_mono.

    Slide IDs in DB are like 'TCGA-2F-A9KO-01Z-00-DX1'.
    We extract patient barcode 'TCGA-2F-A9KO' (first 3 segments) for matching.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT slide_id FROM objects_mono;")
        loaded = set()
        for (sid,) in cur.fetchall():
            # Extract patient barcode: first 3 hyphen-separated segments
            parts = sid.split("-")
            if len(parts) >= 3:
                loaded.add("-".join(parts[:3]))
            else:
                loaded.add(sid)
        return loaded


def count_rows(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table};")
        return cur.fetchone()[0]


def estimate_rows(conn, table: str) -> int:
    """Fast row estimate from pg_class (avoids full scan)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = %s;",
            (table,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def create_so_partition(conn, slide_id: str, safe_name: str = None):
    """Create a LIST partition in objects_slide_only for the given slide_id."""
    safe_val = slide_id.replace("'", "''")
    part_name = f"objects_slide_only_{safe_name or slide_id.replace('-', '_').lower()}"
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {part_name}
            PARTITION OF objects_slide_only
            FOR VALUES IN ('{safe_val}');
        """)
    conn.commit()


def create_spdb_partition(conn, slide_id: str, safe_name: str = None, n_sub: int = 30):
    """Create LIST partition in objects_spdb, then RANGE sub-partitions on hilbert_key."""
    safe_val = slide_id.replace("'", "''")
    parent = f"objects_spdb_{safe_name or slide_id.replace('-', '_').lower()}"

    with conn.cursor() as cur:
        # Top-level LIST partition (partitioned further by RANGE on hilbert_key)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {parent}
            PARTITION OF objects_spdb
            FOR VALUES IN ('{safe_val}')
            PARTITION BY RANGE (hilbert_key);
        """)

        # Sub-partitions: equal-width ranges of hilbert_key [0, HILBERT_MAX_D)
        step = math.ceil(HILBERT_MAX_D / n_sub)
        for i in range(n_sub):
            lo = i * step
            hi = min((i + 1) * step, HILBERT_MAX_D + 1)  # upper bound is exclusive
            sub_name = f"{parent}_h{i}"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {sub_name}
                PARTITION OF {parent}
                FOR VALUES FROM ({lo}) TO ({hi});
            """)

    conn.commit()


def bulk_insert(conn, table: str, df: pd.DataFrame):
    """Use COPY FROM with StringIO for fast bulk insert.

    Expected df columns: slide_id, centroid_x, centroid_y, area,
                         class_label, hilbert_key, zorder_key
    Geometry is generated server-side via ST_MakePoint.
    """
    # We insert into a temp table first, then INSERT ... SELECT with geom.
    tmp = f"_tmp_{table}_{os.getpid()}"
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TEMP TABLE {tmp} (
                slide_id   TEXT,
                centroid_x DOUBLE PRECISION,
                centroid_y DOUBLE PRECISION,
                area       DOUBLE PRECISION,
                class_label TEXT,
                hilbert_key INTEGER,
                zorder_key  INTEGER
            ) ON COMMIT DROP;
        """)

        buf = StringIO()
        for row in df.itertuples(index=False):
            line = "\t".join(str(v) for v in row)
            buf.write(line + "\n")
        buf.seek(0)

        cur.copy_from(
            buf,
            tmp,
            columns=[
                "slide_id",
                "centroid_x",
                "centroid_y",
                "area",
                "class_label",
                "hilbert_key",
                "zorder_key",
            ],
        )

        cur.execute(f"""
            INSERT INTO {table}
                (slide_id, centroid_x, centroid_y, area,
                 class_label, hilbert_key, zorder_key, geom)
            SELECT slide_id, centroid_x, centroid_y, area,
                   class_label, hilbert_key, zorder_key,
                   ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 0)
            FROM {tmp};
        """)

    conn.commit()


# ---------------------------------------------------------------------------
# HuggingFace data download helpers
# ---------------------------------------------------------------------------

def list_blca_patients() -> list[str]:
    """Return sorted list of BLCA patient directory names from the HF dataset."""
    api = HfApi()
    # Dataset has directories at root: bcr_patient_barcode=TCGA-XXXX/data_0.parquet
    files = api.list_repo_tree(
        repo_id=HF_DATASET,
        repo_type=HF_REPO_TYPE,
    )
    patients: list[str] = []
    for entry in files:
        path = entry.path if hasattr(entry, "path") else str(entry)
        if path.startswith("bcr_patient_barcode="):
            patients.append(path)
    return sorted(patients)


def download_parquet(patient_dir: str) -> pd.DataFrame:
    """Download and read a single patient's data_0.parquet from HuggingFace."""
    path_in_repo = f"{patient_dir}/data_0.parquet"
    try:
        local = hf_hub_download(
            repo_id=HF_DATASET,
            repo_type=HF_REPO_TYPE,
            filename=path_in_repo,
        )
        return pd.read_parquet(local)
    except Exception as e:
        print(f"  [warn] Failed to download {path_in_repo}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Schema discovery: map parquet columns to our canonical names
# ---------------------------------------------------------------------------

_CENTROID_X_CANDIDATES = ["centroid_x", "Centroid_X", "cx", "x", "X"]
_CENTROID_Y_CANDIDATES = ["centroid_y", "Centroid_Y", "cy", "y", "Y"]
_AREA_CANDIDATES = ["area", "Area", "cell_area", "nucleus_area"]
_CLASS_CANDIDATES = ["class_label", "type", "cell_type", "label", "class"]


def _pick(df_cols: list[str], candidates: list[str], fallback: str | None = None):
    for c in candidates:
        if c in df_cols:
            return c
    return fallback


def normalise_parquet(df: pd.DataFrame, slide_id: str) -> pd.DataFrame:
    """Map raw parquet columns to canonical schema and compute keys.

    The HuggingFace pan-cancer-nuclei-seg parquet has:
    - Polygon: numpy array of dicts [{'': x, '_1': y}, ...] in absolute slide coords
    - AreaInPixels: nucleus area
    - case_id: full TCGA slide barcode (e.g., TCGA-2F-A9KO-01Z-00-DX1)
    - type: histological type (can serve as class_label)
    """
    cols = list(df.columns)

    # Use case_id as slide_id if available (overrides the passed-in slide_id)
    # Keep original format (with hyphens) to match existing data in DB
    if "case_id" in cols:
        slide_id = str(df["case_id"].iloc[0])

    # Extract centroids from Polygon column (array of {'': x, '_1': y} dicts)
    if "Polygon" in cols:
        # Process in chunks to avoid memory explosion with .apply()
        n_rows = len(df)
        cx_arr = np.zeros(n_rows, dtype=np.float64)
        cy_arr = np.zeros(n_rows, dtype=np.float64)
        polys = df["Polygon"].values
        for i in range(n_rows):
            poly = polys[i]
            if poly is None or len(poly) == 0:
                continue
            try:
                xs = [p[''] for p in poly]
                ys = [p['_1'] for p in poly]
                cx_arr[i] = sum(xs) / len(xs)
                cy_arr[i] = sum(ys) / len(ys)
            except (KeyError, TypeError):
                pass
        cx = pd.Series(cx_arr)
        cy = pd.Series(cy_arr)
    else:
        # Fallback to explicit centroid columns
        cx_col = _pick(cols, _CENTROID_X_CANDIDATES)
        cy_col = _pick(cols, _CENTROID_Y_CANDIDATES)
        if cx_col is None or cy_col is None:
            print(f"  [warn] Cannot find centroid columns in {cols}. Skipping.")
            return pd.DataFrame()
        cx = df[cx_col].astype(float)
        cy = df[cy_col].astype(float)

    area_col = _pick(cols, _AREA_CANDIDATES + ["AreaInPixels"])
    class_col = _pick(cols, _CLASS_CANDIDATES)

    out = pd.DataFrame({
        "centroid_x": cx.astype(float).values,
        "centroid_y": cy.astype(float).values,
    })
    out["slide_id"] = slide_id
    out["area"] = df[area_col].astype(float).values if area_col else 0.0
    out["class_label"] = df[class_col].astype(str).values if class_col else "unknown"

    h, z = compute_keys(out["centroid_x"], out["centroid_y"])
    out["hilbert_key"] = h.astype(int)
    out["zorder_key"] = z.astype(int)

    # Reorder columns to match bulk_insert expectations
    return out[["slide_id", "centroid_x", "centroid_y", "area", "class_label", "hilbert_key", "zorder_key"]]


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_slides(
    conn,
    patients: list[str],
    loaded: set[str],
    target: int = TARGET_NUCLEI,
):
    """Download + insert slides until we exceed *target* total nuclei."""
    total_before = estimate_rows(conn, "objects_mono")
    total = total_before
    print(f"\n[ingest] Starting ingestion. Rows in objects_mono (estimate): {total:,}")
    print(f"[ingest] Target: {target:,} nuclei")
    print(f"[ingest] Candidates: {len(patients)} BLCA patients, {len(loaded)} already loaded\n")

    new_patients = [p for p in patients if barcode_to_patient_id(p) not in loaded]
    if not new_patients:
        print("[ingest] No new patients available. All loaded.")
        return total

    slides_added = 0

    for batch_start in range(0, len(new_patients), BATCH_SIZE):
        if total >= target:
            break

        batch = new_patients[batch_start : batch_start + BATCH_SIZE]
        print(f"[ingest] Batch {batch_start // BATCH_SIZE + 1}: "
              f"slides {batch_start+1}-{batch_start+len(batch)} of {len(new_patients)}")

        for patient_dir in batch:
            if total >= target:
                break

            patient_id = barcode_to_patient_id(patient_dir)
            print(f"  Downloading {patient_dir} ...", end=" ", flush=True)
            raw = download_parquet(patient_dir)
            if raw.empty:
                print("EMPTY, skipping.")
                continue

            print(f"{len(raw):,} rows.", end=" ", flush=True)
            df = normalise_parquet(raw, patient_id)
            if df.empty:
                print("NORM FAILED, skipping.")
                continue

            # Get actual slide_id from normalised data (set by case_id)
            slide_id = df["slide_id"].iloc[0]

            # Create partitions first (use SQL-safe name for partition tables)
            safe_slide = slide_id.replace("-", "_").lower()
            try:
                create_so_partition(conn, slide_id, safe_slide)
                create_spdb_partition(conn, slide_id, safe_slide)
            except psycopg2.Error as e:
                # Partition may already exist — or connection dropped
                try:
                    conn.rollback()
                except Exception:
                    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
                    print("\n  [reconnected]", end=" ")
                    try:
                        create_so_partition(conn, slide_id, safe_slide)
                        create_spdb_partition(conn, slide_id, safe_slide)
                    except psycopg2.Error:
                        conn.rollback()
                print(f"(partition note: {e.pgerror.strip() if e.pgerror else e})", end=" ")

            # Bulk insert into all three tables
            insert_ok = True
            for tbl in ("objects_mono", "objects_slide_only", "objects_spdb"):
                try:
                    bulk_insert(conn, tbl, df)
                except (psycopg2.OperationalError, psycopg2.InterfaceError):
                    # Connection lost — reconnect and retry once
                    print(f"\n  [reconnecting for {tbl}]", end=" ", flush=True)
                    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)
                    try:
                        bulk_insert(conn, tbl, df)
                    except psycopg2.Error as e2:
                        conn.rollback()
                        print(f"\n  [ERROR] Insert into {tbl} failed after reconnect: {e2}")
                        insert_ok = False
                        continue
                except psycopg2.Error as e:
                    conn.rollback()
                    print(f"\n  [ERROR] Insert into {tbl} failed: {e}")
                    insert_ok = False
                    continue

            total += len(df)
            slides_added += 1
            loaded.add(slide_id)
            print(f"OK. Running total: {total:,}")

            # Free memory
            del raw, df

        # ANALYZE periodically for planner statistics
        if slides_added % 20 == 0 and slides_added > 0:
            print("[ingest] Running ANALYZE ...")
            with conn.cursor() as cur:
                for tbl in ("objects_mono", "objects_slide_only", "objects_spdb"):
                    cur.execute(f"ANALYZE {tbl};")
            conn.commit()

    # Final ANALYZE
    print("\n[ingest] Final ANALYZE on all tables ...")
    with conn.cursor() as cur:
        for tbl in ("objects_mono", "objects_slide_only", "objects_spdb"):
            cur.execute(f"ANALYZE {tbl};")
    conn.commit()

    total_final = estimate_rows(conn, "objects_mono")
    print(f"[ingest] Ingestion complete. Estimated rows in objects_mono: {total_final:,}")
    print(f"[ingest] Slides added this run: {slides_added}")
    return total_final


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _random_viewport(conn, slide_id: str, frac: float):
    """Generate a random viewport covering *frac* of the slide's extent."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(centroid_x), max(centroid_x), "
            "min(centroid_y), max(centroid_y) "
            "FROM objects_mono WHERE slide_id = %s;",
            (slide_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    xmin, xmax, ymin, ymax = row
    w = (xmax - xmin) * math.sqrt(frac)
    h = (ymax - ymin) * math.sqrt(frac)
    x0 = random.uniform(xmin, xmax - w) if xmax - w > xmin else xmin
    y0 = random.uniform(ymin, ymax - h) if ymax - h > ymin else ymin
    return x0, y0, x0 + w, y0 + h


def _get_sample_slides(conn, n: int = 10) -> list[str]:
    """Pick up to n random loaded slide_ids."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT slide_id FROM objects_mono;")
        all_slides = [r[0] for r in cur.fetchall()]
    if len(all_slides) <= n:
        return all_slides
    return random.sample(all_slides, n)


def _precompute_viewports(conn, slides: list[str], frac: float, n: int):
    """Pre-generate n viewports spread across slides."""
    viewports = []
    for _ in range(n):
        sid = random.choice(slides)
        vp = _random_viewport(conn, sid, frac)
        if vp:
            viewports.append((sid, vp))
    return viewports


def _random_point(conn, slide_id: str):
    """Pick a random nuclei centroid from the slide as kNN center."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT centroid_x, centroid_y FROM objects_mono "
            "WHERE slide_id = %s ORDER BY random() LIMIT 1;",
            (slide_id,),
        )
        row = cur.fetchone()
    return row if row else None


def _precompute_knn_centers(conn, slides: list[str], n: int):
    """Pre-generate n (slide_id, cx, cy) tuples for kNN queries."""
    centers = []
    for _ in range(n):
        sid = random.choice(slides)
        pt = _random_point(conn, sid)
        if pt:
            centers.append((sid, pt[0], pt[1]))
    return centers


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------

def q_viewport(conn, table: str, slide_id: str, vp: tuple) -> float:
    """Q1: viewport (window) query. Returns latency in ms."""
    x0, y0, x1, y1 = vp
    sql = f"""
        SELECT count(*) FROM {table}
        WHERE slide_id = %s
          AND centroid_x BETWEEN %s AND %s
          AND centroid_y BETWEEN %s AND %s;
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (slide_id, x0, x1, y0, y1))
        cur.fetchone()
    return (time.perf_counter() - t0) * 1000


def q_knn(conn, table: str, slide_id: str, cx: float, cy: float, k: int) -> float:
    """Q2: kNN query using PostGIS <-> distance operator. Returns latency in ms."""
    sql = f"""
        SELECT centroid_x, centroid_y,
               geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) AS dist
        FROM {table}
        WHERE slide_id = %s
        ORDER BY dist
        LIMIT %s;
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (cx, cy, slide_id, k))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000


def q_agg(conn, table: str, slide_id: str) -> float:
    """Q3: aggregation query -- per-class statistics for a slide. Returns ms."""
    sql = f"""
        SELECT class_label, count(*), avg(area), stddev(area)
        FROM {table}
        WHERE slide_id = %s
        GROUP BY class_label;
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (slide_id,))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def summarise(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": 0, "p95": 0, "mean": 0, "std": 0, "n": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "p50": s[int(n * 0.5)],
        "p95": s[int(n * 0.95)],
        "mean": statistics.mean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def run_benchmarks(conn) -> dict[str, Any]:
    """Execute Q1, Q2, Q3, and cold-cache benchmarks. Return results dict."""
    tables = [
        "objects_mono", "objects_mono_clustered",
        "objects_slide_only", "objects_so_clustered",
        "objects_spdb",
    ]
    results: dict[str, Any] = {}

    # Row counts
    print("\n[bench] Counting rows ...")
    for t in tables:
        est = estimate_rows(conn, t)
        results[f"{t}_rows_est"] = est
        print(f"  {t}: ~{est:,}")

    slides = _get_sample_slides(conn, n=15)
    print(f"[bench] Using {len(slides)} sample slides for queries.\n")

    # ---- Warm cache: ensure shared_buffers are populated ------------------
    print("[bench] Warming cache ...")
    for sid in slides[:3]:
        for t in tables:
            q_viewport(conn, t, sid, _random_viewport(conn, sid, 0.05) or (0, 0, 1, 1))

    # ---- Q1: viewport queries ---------------------------------------------
    for frac, label in [(0.05, "5pct"), (0.01, "1pct")]:
        print(f"[bench] Q1 viewport f={label} ({BENCHMARK_TRIALS} trials) ...")
        viewports = _precompute_viewports(conn, slides, frac, BENCHMARK_TRIALS)
        for t in tables:
            lats = [q_viewport(conn, t, sid, vp) for sid, vp in viewports]
            key = f"Q1_{label}_{t}"
            results[key] = summarise(lats)
            print(f"  {t}: p50={results[key]['p50']:.2f}ms  p95={results[key]['p95']:.2f}ms")

    # ---- Q2: kNN ----------------------------------------------------------
    print(f"\n[bench] Q2 kNN k={KNN_K} ({BENCHMARK_TRIALS} trials) ...")
    centers = _precompute_knn_centers(conn, slides, BENCHMARK_TRIALS)
    for t in tables:
        lats = [q_knn(conn, t, sid, cx, cy, KNN_K) for sid, cx, cy in centers]
        key = f"Q2_knn_{t}"
        results[key] = summarise(lats)
        print(f"  {t}: p50={results[key]['p50']:.2f}ms  p95={results[key]['p95']:.2f}ms")

    # ---- Q3: aggregation --------------------------------------------------
    print(f"\n[bench] Q3 aggregation ({BENCHMARK_TRIALS} trials) ...")
    for t in tables:
        lats = []
        for _ in range(BENCHMARK_TRIALS):
            sid = random.choice(slides)
            lats.append(q_agg(conn, t, sid))
        key = f"Q3_agg_{t}"
        results[key] = summarise(lats)
        print(f"  {t}: p50={results[key]['p50']:.2f}ms  p95={results[key]['p95']:.2f}ms")

    # ---- Speedups ---------------------------------------------------------
    print("\n[bench] Computing speedups ...")
    for qname in ["Q1_5pct", "Q1_1pct", "Q2_knn", "Q3_agg"]:
        mono_key = f"{qname}_objects_mono"
        if mono_key not in results:
            continue
        mono_p50 = results[mono_key]["p50"]
        for t in ["objects_mono_clustered", "objects_slide_only",
                  "objects_so_clustered", "objects_spdb"]:
            other_key = f"{qname}_{t}"
            if other_key in results and results[other_key]["p50"] > 0:
                sp = mono_p50 / results[other_key]["p50"]
                results[f"speedup_{qname}_{t}"] = round(sp, 2)
                print(f"  {qname} {t}: {sp:.2f}x vs mono")

    # ---- Cold-cache benchmark ---------------------------------------------
    print(f"\n[bench] Cold-cache benchmark ({COLD_CACHE_TRIALS} viewport trials) ...")
    cold_viewports = _precompute_viewports(conn, slides, 0.05, COLD_CACHE_TRIALS)

    # Flush OS page cache and restart PostgreSQL
    try:
        print("  Flushing OS caches ...")
        subprocess.run(
            ["sudo", "sync"],
            check=True, timeout=30,
        )
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True, timeout=30,
        )
        print("  Restarting PostgreSQL ...")
        subprocess.run(
            ["sudo", "systemctl", "restart", "postgresql"],
            check=True, timeout=60,
        )
        time.sleep(3)  # allow PG to come up

        # Reconnect
        conn_cold = get_conn()

        for t in tables:
            lats = [q_viewport(conn_cold, t, sid, vp) for sid, vp in cold_viewports]
            key = f"cold_Q1_5pct_{t}"
            results[key] = summarise(lats)
            print(f"  {t}: p50={results[key]['p50']:.2f}ms  p95={results[key]['p95']:.2f}ms")

            # Flush again between tables so each table starts cold
            subprocess.run(["sudo", "sync"], check=True, timeout=30)
            subprocess.run(
                ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                check=True, timeout=30,
            )
            subprocess.run(
                ["sudo", "systemctl", "restart", "postgresql"],
                check=True, timeout=60,
            )
            time.sleep(3)
            conn_cold = get_conn()

        conn_cold.close()

    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        print(f"  [warn] Cold-cache flush failed ({e}). "
              "Skipping cold-cache results (requires sudo).")
        for t in tables:
            results[f"cold_Q1_5pct_{t}"] = {"skipped": True, "reason": str(e)}

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, total_rows: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "scale_test_200M.json"

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_rows_estimate": total_rows,
        "hilbert_order": HILBERT_ORDER,
        "benchmark_trials": BENCHMARK_TRIALS,
        "cold_cache_trials": COLD_CACHE_TRIALS,
        "knn_k": KNN_K,
        "metrics": results,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[results] Saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scale SpatialPathDB to 200M+ nuclei and benchmark."
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Skip ingestion, run benchmark only.",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_NUCLEI,
        help="Target nuclei count (default: 200M).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  SpatialPathDB Scale Benchmark")
    print("=" * 70)

    conn = get_conn()
    conn.set_session(autocommit=False)

    # --- Check current state -----------------------------------------------
    loaded = get_loaded_slides(conn)
    current_est = estimate_rows(conn, "objects_mono")
    print(f"\n[status] Slides loaded: {len(loaded)}")
    print(f"[status] Estimated rows in objects_mono: {current_est:,}")

    # --- Ingestion (unless --bench) ----------------------------------------
    if not args.bench:
        print("\n[hf] Listing BLCA patients on HuggingFace ...")
        patients = list_blca_patients()
        print(f"[hf] Found {len(patients)} patient directories.")

        total = ingest_slides(conn, patients, loaded, target=args.target)
    else:
        total = current_est
        print("[bench] Skipping ingestion (--bench flag).")

    if total < args.target:
        print(f"\n[warn] Only {total:,} rows loaded (target was {args.target:,}).")
        print("[warn] Proceeding with benchmark anyway.\n")

    # --- Run benchmarks ----------------------------------------------------
    results = run_benchmarks(conn)

    # --- Save results ------------------------------------------------------
    save_results(results, total)

    conn.close()
    print("\n[done] Scale benchmark complete.")


if __name__ == "__main__":
    main()
