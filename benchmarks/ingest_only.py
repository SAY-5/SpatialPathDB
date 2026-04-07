#!/usr/bin/env python3
"""
ingest_only.py -- Lean ingest-only script. NO ANALYZE. NO benchmarks.

Phase A: ingest remaining slides to cross 200M target.
Phase B (separate): run ANALYZE once, then benchmark with --bench flag.

Usage:
    python3 ingest_only.py              # ingest until 200M+
    python3 ingest_only.py --target 210000000  # custom target
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from io import StringIO

# ---------------------------------------------------------------------------
# Ensure required packages
# ---------------------------------------------------------------------------
def _ensure_packages():
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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure_packages()

import numpy as np
import pandas as pd
import psycopg2
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

HILBERT_ORDER = 8
HILBERT_N = 1 << HILBERT_ORDER  # 256
HILBERT_MAX_D = HILBERT_N * HILBERT_N  # 65536

TARGET_NUCLEI = 200_000_000

# ---------------------------------------------------------------------------
# Hilbert & Z-order
# ---------------------------------------------------------------------------
def xy2d(n, x, y):
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d

def z_order(gx, gy):
    z = 0
    for i in range(8):
        z |= ((gx >> i) & 1) << (2 * i)
        z |= ((gy >> i) & 1) << (2 * i + 1)
    return z

def compute_keys(cx, cy):
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
# DB helpers
# ---------------------------------------------------------------------------
def get_conn():
    return psycopg2.connect(dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT)

def get_loaded_slides(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT slide_id FROM objects_mono;")
        loaded = set()
        for (sid,) in cur.fetchall():
            parts = sid.split("-")
            if len(parts) >= 3:
                loaded.add("-".join(parts[:3]))
            else:
                loaded.add(sid)
        return loaded

def count_rows_fast(conn, table):
    """Exact count via pg_class estimate (fast, no full scan)."""
    with conn.cursor() as cur:
        cur.execute("SELECT reltuples::bigint FROM pg_class WHERE relname = %s;", (table,))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] > 0 else 0

def count_rows_exact(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table};")
        return cur.fetchone()[0]

def create_so_partition(conn, slide_id, safe_name):
    safe_val = slide_id.replace("'", "''")
    part_name = f"objects_slide_only_{safe_name}"
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {part_name}
            PARTITION OF objects_slide_only
            FOR VALUES IN ('{safe_val}');
        """)
    conn.commit()

def create_spdb_partition(conn, slide_id, safe_name, n_sub=30):
    safe_val = slide_id.replace("'", "''")
    parent = f"objects_spdb_{safe_name}"
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {parent}
            PARTITION OF objects_spdb
            FOR VALUES IN ('{safe_val}')
            PARTITION BY RANGE (hilbert_key);
        """)
        step = math.ceil(HILBERT_MAX_D / n_sub)
        for i in range(n_sub):
            lo = i * step
            hi = min((i + 1) * step, HILBERT_MAX_D + 1)
            sub_name = f"{parent}_h{i}"
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {sub_name}
                PARTITION OF {parent}
                FOR VALUES FROM ({lo}) TO ({hi});
            """)
    conn.commit()

def bulk_insert(conn, table, df):
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
            buf.write("\t".join(str(v) for v in row) + "\n")
        buf.seek(0)
        cur.copy_from(buf, tmp, columns=["slide_id","centroid_x","centroid_y","area","class_label","hilbert_key","zorder_key"])
        cur.execute(f"""
            INSERT INTO {table}
                (slide_id, centroid_x, centroid_y, area, class_label, hilbert_key, zorder_key, geom)
            SELECT slide_id, centroid_x, centroid_y, area, class_label, hilbert_key, zorder_key,
                   ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 0)
            FROM {tmp};
        """)
    conn.commit()

# ---------------------------------------------------------------------------
# HuggingFace helpers
# ---------------------------------------------------------------------------
def list_blca_patients():
    api = HfApi()
    files = api.list_repo_tree(repo_id=HF_DATASET, repo_type=HF_REPO_TYPE)
    patients = []
    for entry in files:
        path = entry.path if hasattr(entry, "path") else str(entry)
        if path.startswith("bcr_patient_barcode="):
            patients.append(path)
    return sorted(patients)

def download_parquet(patient_dir):
    path_in_repo = f"{patient_dir}/data_0.parquet"
    try:
        local = hf_hub_download(repo_id=HF_DATASET, repo_type=HF_REPO_TYPE, filename=path_in_repo)
        return pd.read_parquet(local)
    except Exception as e:
        print(f"  [warn] Download failed {path_in_repo}: {e}")
        return pd.DataFrame()

def normalise_parquet(df, slide_id):
    cols = list(df.columns)
    if "case_id" in cols:
        slide_id = str(df["case_id"].iloc[0])
    if "Polygon" in cols:
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
        for c in ["centroid_x","Centroid_X","cx","x","X"]:
            if c in cols:
                cx = df[c].astype(float); break
        else:
            return pd.DataFrame()
        for c in ["centroid_y","Centroid_Y","cy","y","Y"]:
            if c in cols:
                cy = df[c].astype(float); break
        else:
            return pd.DataFrame()
    area_col = None
    for c in ["area","Area","cell_area","nucleus_area","AreaInPixels"]:
        if c in cols:
            area_col = c; break
    class_col = None
    for c in ["class_label","type","cell_type","label","class"]:
        if c in cols:
            class_col = c; break
    out = pd.DataFrame({"centroid_x": cx.astype(float).values, "centroid_y": cy.astype(float).values})
    out["slide_id"] = slide_id
    out["area"] = df[area_col].astype(float).values if area_col else 0.0
    out["class_label"] = df[class_col].astype(str).values if class_col else "unknown"
    h, z = compute_keys(out["centroid_x"], out["centroid_y"])
    out["hilbert_key"] = h.astype(int)
    out["zorder_key"] = z.astype(int)
    return out[["slide_id","centroid_x","centroid_y","area","class_label","hilbert_key","zorder_key"]]

def barcode_to_patient_id(patient_dir):
    return patient_dir.split("=", 1)[-1] if "=" in patient_dir else patient_dir

# ---------------------------------------------------------------------------
# Main: ingest only, NO ANALYZE
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_NUCLEI)
    args = parser.parse_args()

    print("=" * 60)
    print("  INGEST ONLY — no ANALYZE, no benchmarks")
    print("=" * 60)

    conn = get_conn()
    conn.set_session(autocommit=False)

    # Step 1: Disable autovacuum on leaf partitions (partitioned parents can't have storage params)
    print("\n[setup] Disabling autovacuum on leaf partitions ...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.relname FROM pg_class c
            JOIN pg_inherits i ON c.oid = i.inhrelid
            WHERE i.inhparent IN (
                'objects_mono'::regclass,
                'objects_slide_only'::regclass,
                'objects_spdb'::regclass
            )
            AND c.relkind = 'r';
        """)
        # Also get sub-sub-partitions (SPDB has 2 levels)
        cur.execute("""
            WITH RECURSIVE parts AS (
                SELECT c.oid, c.relname, c.relkind FROM pg_class c
                WHERE c.relname IN ('objects_mono', 'objects_slide_only', 'objects_spdb')
                UNION ALL
                SELECT c2.oid, c2.relname, c2.relkind
                FROM pg_class c2 JOIN pg_inherits i ON c2.oid = i.inhrelid
                JOIN parts p ON i.inhparent = p.oid
            )
            SELECT relname FROM parts WHERE relkind = 'r';
        """)
        leaves = [r[0] for r in cur.fetchall()]
        for leaf in leaves:
            cur.execute(f"ALTER TABLE {leaf} SET (autovacuum_enabled = false);")
    conn.commit()
    print(f"[setup] Autovacuum disabled on {len(leaves)} leaf partitions.")

    # Step 2: Check current state
    loaded = get_loaded_slides(conn)
    # Use exact count for first slide count
    total = count_rows_exact(conn, "objects_mono")
    print(f"\n[status] Slides loaded: {len(loaded)}")
    print(f"[status] Exact rows in objects_mono: {total:,}")
    print(f"[status] Target: {args.target:,}")

    if total >= args.target:
        print(f"\n[done] Already at {total:,} >= {args.target:,}. Nothing to ingest.")
        conn.close()
        return

    # Step 3: List patients and find new ones
    print("\n[hf] Listing BLCA patients ...")
    patients = list_blca_patients()
    print(f"[hf] Found {len(patients)} patient directories.")

    new_patients = [p for p in patients if barcode_to_patient_id(p) not in loaded]
    print(f"[ingest] {len(new_patients)} new patients to load.")

    if not new_patients:
        print("[ingest] No new patients. Exiting.")
        conn.close()
        return

    # Step 4: Ingest slide by slide — NO ANALYZE
    slides_added = 0
    for idx, patient_dir in enumerate(new_patients):
        if total >= args.target:
            break

        patient_id = barcode_to_patient_id(patient_dir)
        print(f"\n  [{idx+1}/{len(new_patients)}] Downloading {patient_dir} ...", end=" ", flush=True)

        raw = download_parquet(patient_dir)
        if raw.empty:
            print("EMPTY, skipping.")
            continue

        print(f"{len(raw):,} rows.", end=" ", flush=True)
        df = normalise_parquet(raw, patient_id)
        if df.empty:
            print("NORM FAILED, skipping.")
            continue

        slide_id = df["slide_id"].iloc[0]
        if slide_id == "nan" or pd.isna(slide_id):
            print("BAD slide_id (nan), skipping.")
            del raw, df
            continue

        safe_slide = slide_id.replace("-", "_").lower()

        # Create partitions
        try:
            create_so_partition(conn, slide_id, safe_slide)
            create_spdb_partition(conn, slide_id, safe_slide)
        except psycopg2.Error as e:
            try:
                conn.rollback()
            except Exception:
                conn = get_conn()
                conn.set_session(autocommit=False)
                try:
                    create_so_partition(conn, slide_id, safe_slide)
                    create_spdb_partition(conn, slide_id, safe_slide)
                except psycopg2.Error:
                    conn.rollback()
            print(f"(partition: {str(e).strip()[:60]})", end=" ")

        # Bulk insert into all three tables
        ok = True
        for tbl in ("objects_mono", "objects_slide_only", "objects_spdb"):
            try:
                bulk_insert(conn, tbl, df)
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                print(f"\n  [reconnect for {tbl}]", end=" ", flush=True)
                conn = get_conn()
                conn.set_session(autocommit=False)
                try:
                    bulk_insert(conn, tbl, df)
                except psycopg2.Error as e2:
                    conn.rollback()
                    print(f"\n  [ERROR] {tbl}: {e2}")
                    ok = False
            except psycopg2.Error as e:
                conn.rollback()
                print(f"\n  [ERROR] {tbl}: {e}")
                ok = False

        total += len(df)
        slides_added += 1
        loaded.add(slide_id)
        print(f"OK. Total: {total:,}")

        del raw, df

    print(f"\n[ingest] Done. Slides added: {slides_added}. Total rows: {total:,}")

    # Step 5: Re-enable autovacuum on leaf partitions
    print("[cleanup] Re-enabling autovacuum ...")
    with conn.cursor() as cur:
        cur.execute("""
            WITH RECURSIVE parts AS (
                SELECT c.oid, c.relname, c.relkind FROM pg_class c
                WHERE c.relname IN ('objects_mono', 'objects_slide_only', 'objects_spdb')
                UNION ALL
                SELECT c2.oid, c2.relname, c2.relkind
                FROM pg_class c2 JOIN pg_inherits i ON c2.oid = i.inhrelid
                JOIN parts p ON i.inhparent = p.oid
            )
            SELECT relname FROM parts WHERE relkind = 'r';
        """)
        leaves = [r[0] for r in cur.fetchall()]
        for leaf in leaves:
            cur.execute(f"ALTER TABLE {leaf} SET (autovacuum_enabled = true);")
    conn.commit()
    print(f"[cleanup] Autovacuum re-enabled on {len(leaves)} leaves.")

    print(f"\n[NEXT STEPS]")
    print(f"  1. Run: sudo -u postgres psql -d spdb -c 'ANALYZE objects_mono; ANALYZE objects_slide_only; ANALYZE objects_spdb;'")
    print(f"  2. Run: python3 benchmarks/scale_benchmark.py --bench")

    conn.close()

if __name__ == "__main__":
    main()
