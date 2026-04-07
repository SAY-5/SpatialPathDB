#!/usr/bin/env python3
"""
pannuke_benchmark.py  --  PanNuke multi-cancer-type benchmark.

Demonstrates that SPDB's two-level partitioning (LIST by tissue_type,
RANGE by hilbert_key) generalises across 19 tissue types from the
PanNuke pan-cancer nuclei segmentation dataset (~190K nuclei).

Key question: does tissue_type as the L1 partition key (analogous to
slide_id in pathology) yield the same partition-pruning speedups?

Data source: /data/pan_cancer_nuclei_seg/pannuke_polygons.csv
Target DB:   spdb (PostgreSQL + PostGIS, user=postgres, host=localhost)

Tables created:
  pannuke_mono  — flat table, GiST index on geom
  pannuke_so    — PARTITION BY LIST (tissue_type), per-partition GiST
  pannuke_spdb  — PARTITION BY LIST (tissue_type),
                  then RANGE (hilbert_key) with ~5 sub-partitions per tissue

Usage:
    python3 pannuke_benchmark.py              # full: ingest + benchmark
    python3 pannuke_benchmark.py --bench      # skip ingestion, benchmark only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure required packages
# ---------------------------------------------------------------------------

def _ensure_packages():
    required = {
        "psycopg2": "psycopg2-binary",
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

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_NAME = "spdb"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

CSV_PATH = "/data/pan_cancer_nuclei_seg/pannuke_polygons.csv"

HILBERT_ORDER = 8
HILBERT_N = 1 << HILBERT_ORDER   # 256
HILBERT_MAX_D = HILBERT_N * HILBERT_N  # 65536

N_HILBERT_SUB = 5   # sub-partitions per tissue in SPDB

BENCHMARK_TRIALS = 200
KNN_K = 50

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "raw"

TABLE_MONO = "pannuke_mono"
TABLE_SO   = "pannuke_so"
TABLE_SPDB = "pannuke_spdb"

# ---------------------------------------------------------------------------
# Hilbert curve helpers (pure Python, order-8)
# ---------------------------------------------------------------------------

def xy2d(n: int, x: int, y: int) -> int:
    """Convert (x, y) on an n x n Hilbert grid to distance d."""
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


def compute_hilbert_key(cx: float, cy: float,
                        xmin: float, xmax: float,
                        ymin: float, ymax: float) -> int:
    """Map (cx, cy) into the Hilbert curve using the extent as normalisation."""
    span_x = max(xmax - xmin, 1.0)
    span_y = max(ymax - ymin, 1.0)
    gx = int((cx - xmin) / span_x * (HILBERT_N - 1))
    gy = int((cy - ymin) / span_y * (HILBERT_N - 1))
    gx = max(0, min(HILBERT_N - 1, gx))
    gy = max(0, min(HILBERT_N - 1, gy))
    return xy2d(HILBERT_N, gx, gy)


def hilbert_range_for_bbox(x0: float, y0: float, x1: float, y1: float,
                           xmin: float, xmax: float,
                           ymin: float, ymax: float) -> tuple[int, int]:
    """Return (lo, hi) Hilbert key range that conservatively covers a bbox.

    Computes keys at all four corners + centre, then pads the range.
    """
    points = [
        (x0, y0), (x0, y1), (x1, y0), (x1, y1),
        ((x0 + x1) / 2, (y0 + y1) / 2),
    ]
    keys = [compute_hilbert_key(px, py, xmin, xmax, ymin, ymax)
            for px, py in points]
    kmin, kmax = min(keys), max(keys)
    # Pad by the range itself (Hilbert curve can be non-contiguous)
    pad = max(kmax - kmin, 1)
    return max(0, kmin - pad), min(HILBERT_MAX_D, kmax + pad)


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    """Load the PanNuke CSV into a list of dicts."""
    print(f"[data] Reading {path} ...")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[data] Loaded {len(rows):,} nuclei")
    return rows


def compute_extents(rows: list[dict]) -> dict[str, tuple]:
    """Compute per-tissue-type bounding boxes and global extent.

    Returns dict with 'global' and per-tissue keys, each mapping to
    (xmin, xmax, ymin, ymax).
    """
    tissue_coords: dict[str, list] = {}
    for r in rows:
        tt = r["tissue_type"]
        cx = float(r["centroid_x"])
        cy = float(r["centroid_y"])
        if tt not in tissue_coords:
            tissue_coords[tt] = [cx, cx, cy, cy]
        else:
            e = tissue_coords[tt]
            if cx < e[0]: e[0] = cx
            if cx > e[1]: e[1] = cx
            if cy < e[2]: e[2] = cy
            if cy > e[3]: e[3] = cy

    extents = {}
    gx0, gx1, gy0, gy1 = float("inf"), float("-inf"), float("inf"), float("-inf")
    for tt, (x0, x1, y0, y1) in tissue_coords.items():
        extents[tt] = (x0, x1, y0, y1)
        gx0 = min(gx0, x0)
        gx1 = max(gx1, x1)
        gy0 = min(gy0, y0)
        gy1 = max(gy1, y1)
    extents["global"] = (gx0, gx1, gy0, gy1)
    return extents


# ---------------------------------------------------------------------------
# Table creation + ingestion
# ---------------------------------------------------------------------------

def safe_name(tissue_type: str) -> str:
    """Convert a tissue type string to a safe SQL identifier."""
    return tissue_type.lower().replace(" ", "_").replace("&", "and").replace("-", "_")


def create_tables(conn, tissue_types: list[str], extents: dict):
    """Create pannuke_mono, pannuke_so, pannuke_spdb tables."""
    cur = conn.cursor()

    # ---- pannuke_mono: flat table ----
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_MONO} CASCADE")
    cur.execute(f"""
        CREATE TABLE {TABLE_MONO} (
            id SERIAL PRIMARY KEY,
            image_id       INT NOT NULL,
            tissue_type    TEXT NOT NULL,
            nucleus_id     INT,
            category       TEXT,
            centroid_x     DOUBLE PRECISION NOT NULL,
            centroid_y     DOUBLE PRECISION NOT NULL,
            area           DOUBLE PRECISION,
            perimeter      DOUBLE PRECISION,
            hilbert_key    INT NOT NULL,
            geom           geometry(Point, 0) NOT NULL
        )
    """)
    print(f"  Created {TABLE_MONO}")

    # ---- pannuke_so: PARTITION BY LIST (tissue_type) ----
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_SO} CASCADE")
    cur.execute(f"""
        CREATE TABLE {TABLE_SO} (
            id SERIAL,
            image_id       INT NOT NULL,
            tissue_type    TEXT NOT NULL,
            nucleus_id     INT,
            category       TEXT,
            centroid_x     DOUBLE PRECISION NOT NULL,
            centroid_y     DOUBLE PRECISION NOT NULL,
            area           DOUBLE PRECISION,
            perimeter      DOUBLE PRECISION,
            hilbert_key    INT NOT NULL,
            geom           geometry(Point, 0) NOT NULL
        ) PARTITION BY LIST (tissue_type)
    """)
    for tt in tissue_types:
        sn = safe_name(tt)
        safe_val = tt.replace("'", "''")
        cur.execute(f"""
            CREATE TABLE {TABLE_SO}_{sn}
            PARTITION OF {TABLE_SO}
            FOR VALUES IN ('{safe_val}')
        """)
    print(f"  Created {TABLE_SO} with {len(tissue_types)} partitions")

    # ---- pannuke_spdb: LIST (tissue_type) -> RANGE (hilbert_key) ----
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_SPDB} CASCADE")
    cur.execute(f"""
        CREATE TABLE {TABLE_SPDB} (
            id SERIAL,
            image_id       INT NOT NULL,
            tissue_type    TEXT NOT NULL,
            nucleus_id     INT,
            category       TEXT,
            centroid_x     DOUBLE PRECISION NOT NULL,
            centroid_y     DOUBLE PRECISION NOT NULL,
            area           DOUBLE PRECISION,
            perimeter      DOUBLE PRECISION,
            hilbert_key    INT NOT NULL,
            geom           geometry(Point, 0) NOT NULL
        ) PARTITION BY LIST (tissue_type)
    """)

    step = math.ceil(HILBERT_MAX_D / N_HILBERT_SUB)
    for tt in tissue_types:
        sn = safe_name(tt)
        safe_val = tt.replace("'", "''")
        cur.execute(f"""
            CREATE TABLE {TABLE_SPDB}_{sn}
            PARTITION OF {TABLE_SPDB}
            FOR VALUES IN ('{safe_val}')
            PARTITION BY RANGE (hilbert_key)
        """)
        for i in range(N_HILBERT_SUB):
            lo = i * step
            hi = min((i + 1) * step, HILBERT_MAX_D + 1)
            cur.execute(f"""
                CREATE TABLE {TABLE_SPDB}_{sn}_h{i}
                PARTITION OF {TABLE_SPDB}_{sn}
                FOR VALUES FROM ({lo}) TO ({hi})
            """)

    total_leaf = len(tissue_types) * N_HILBERT_SUB
    print(f"  Created {TABLE_SPDB} with {len(tissue_types)} L1 x {N_HILBERT_SUB} L2 = {total_leaf} leaf partitions")

    conn.commit()
    return cur


def ingest_data(conn, rows: list[dict], extents: dict):
    """Bulk-insert nuclei into all three tables using COPY."""
    print("[ingest] Preparing data for COPY ...")

    # Pre-compute Hilbert keys per tissue type
    tissue_rows: dict[str, list] = {}
    for r in rows:
        tt = r["tissue_type"]
        if tt not in tissue_rows:
            tissue_rows[tt] = []
        tissue_rows[tt].append(r)

    # Build TSV in memory
    # Columns: image_id, tissue_type, nucleus_id, category,
    #          centroid_x, centroid_y, area, perimeter, hilbert_key
    tsv_lines = []
    for tt, tt_rows in tissue_rows.items():
        ext = extents.get(tt, extents["global"])
        xmin, xmax, ymin, ymax = ext
        for r in tt_rows:
            cx = float(r["centroid_x"])
            cy = float(r["centroid_y"])
            hk = compute_hilbert_key(cx, cy, xmin, xmax, ymin, ymax)
            tsv_lines.append(
                f"{r['image_id']}\t{tt}\t{r['nucleus_id']}\t{r['category']}\t"
                f"{cx}\t{cy}\t{r['area']}\t{r['perimeter']}\t{hk}"
            )

    tsv_blob = "\n".join(tsv_lines) + "\n"
    print(f"[ingest] {len(tsv_lines):,} rows prepared")

    cols = [
        "image_id", "tissue_type", "nucleus_id", "category",
        "centroid_x", "centroid_y", "area", "perimeter", "hilbert_key",
    ]

    for table in [TABLE_MONO, TABLE_SO, TABLE_SPDB]:
        print(f"  Inserting into {table} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        # Use a temp table + INSERT ... SELECT to generate geometry server-side
        tmp = f"_tmp_{table}_{os.getpid()}"
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TEMP TABLE {tmp} (
                image_id       INT,
                tissue_type    TEXT,
                nucleus_id     INT,
                category       TEXT,
                centroid_x     DOUBLE PRECISION,
                centroid_y     DOUBLE PRECISION,
                area           DOUBLE PRECISION,
                perimeter      DOUBLE PRECISION,
                hilbert_key    INT
            ) ON COMMIT DROP
        """)

        buf = StringIO(tsv_blob)
        cur.copy_from(buf, tmp, columns=cols)

        cur.execute(f"""
            INSERT INTO {table}
                (image_id, tissue_type, nucleus_id, category,
                 centroid_x, centroid_y, area, perimeter, hilbert_key, geom)
            SELECT image_id, tissue_type, nucleus_id, category,
                   centroid_x, centroid_y, area, perimeter, hilbert_key,
                   ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 0)
            FROM {tmp}
        """)
        conn.commit()
        elapsed = time.perf_counter() - t0
        print(f"{elapsed:.1f}s")

    return len(tsv_lines)


def create_indexes(conn, tissue_types: list[str]):
    """Create GiST indexes on all tables."""
    cur = conn.cursor()
    print("[index] Creating GiST indexes ...")

    # Mono
    t0 = time.perf_counter()
    cur.execute(f"CREATE INDEX idx_{TABLE_MONO}_geom ON {TABLE_MONO} USING gist(geom)")
    conn.commit()
    print(f"  {TABLE_MONO}: {time.perf_counter()-t0:.1f}s")

    # SO: per-partition
    t0 = time.perf_counter()
    for tt in tissue_types:
        sn = safe_name(tt)
        cur.execute(
            f"CREATE INDEX idx_{TABLE_SO}_{sn}_geom "
            f"ON {TABLE_SO}_{sn} USING gist(geom)"
        )
    conn.commit()
    print(f"  {TABLE_SO}: {time.perf_counter()-t0:.1f}s ({len(tissue_types)} partitions)")

    # SPDB: per-leaf-partition
    t0 = time.perf_counter()
    for tt in tissue_types:
        sn = safe_name(tt)
        for i in range(N_HILBERT_SUB):
            cur.execute(
                f"CREATE INDEX idx_{TABLE_SPDB}_{sn}_h{i}_geom "
                f"ON {TABLE_SPDB}_{sn}_h{i} USING gist(geom)"
            )
    conn.commit()
    total_idx = len(tissue_types) * N_HILBERT_SUB
    print(f"  {TABLE_SPDB}: {time.perf_counter()-t0:.1f}s ({total_idx} leaf partitions)")

    # ANALYZE all
    print("[index] Running ANALYZE ...")
    for table in [TABLE_MONO, TABLE_SO, TABLE_SPDB]:
        cur.execute(f"ANALYZE {table}")
    conn.commit()
    print("[index] Done.")


# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------

def _get_tissue_extent(conn, table: str, tissue_type: str) -> tuple:
    """Get the (xmin, xmax, ymin, ymax) extent for a tissue type."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT min(centroid_x), max(centroid_x), "
            f"min(centroid_y), max(centroid_y) "
            f"FROM {TABLE_MONO} WHERE tissue_type = %s",
            (tissue_type,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return (0, 1, 0, 1)
    return row


def _random_viewport(ext: tuple, frac: float) -> tuple:
    """Generate a random viewport covering frac of the tissue extent."""
    xmin, xmax, ymin, ymax = ext
    dx = xmax - xmin
    dy = ymax - ymin
    side = math.sqrt(frac)
    wx = max(dx * side, 1.0)
    wy = max(dy * side, 1.0)
    x0 = random.uniform(xmin, max(xmin, xmax - wx))
    y0 = random.uniform(ymin, max(ymin, ymax - wy))
    return x0, y0, x0 + wx, y0 + wy


def _random_point(ext: tuple) -> tuple:
    """Random point within the tissue extent."""
    xmin, xmax, ymin, ymax = ext
    return random.uniform(xmin, xmax), random.uniform(ymin, ymax)


def warmup(conn, table: str):
    """Warm PostgreSQL shared buffers for a table."""
    with conn.cursor() as cur:
        for _ in range(3):
            cur.execute(f"SELECT count(*) FROM {table}")
            cur.fetchone()
            cur.execute(f"SELECT * FROM {table} ORDER BY random() LIMIT 500")
            cur.fetchall()


def summarise(latencies: list[float]) -> dict:
    if not latencies:
        return {"p50": 0, "p95": 0, "mean": 0, "std": 0, "n": 0}
    s = sorted(latencies)
    n = len(s)
    return {
        "p50": s[int(n * 0.5)],
        "p95": s[min(int(n * 0.95), n - 1)],
        "p99": s[min(int(n * 0.99), n - 1)],
        "mean": statistics.mean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "n": n,
    }


# ----- Q1: Viewport -------------------------------------------------------

def q1_viewport(conn, table: str, tissue_type: str, vp: tuple,
                ext: tuple) -> float:
    """Execute a viewport (bounding-box) count query. Returns latency in ms."""
    x0, y0, x1, y1 = vp

    if "spdb" in table:
        xmin, xmax, ymin, ymax = ext
        hlo, hhi = hilbert_range_for_bbox(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
        sql = f"""
            SELECT count(*) FROM {table}
            WHERE tissue_type = %s
              AND hilbert_key BETWEEN %s AND %s
              AND geom && ST_MakeEnvelope(%s, %s, %s, %s, 0)
        """
        params = (tissue_type, hlo, hhi, x0, y0, x1, y1)
    else:
        sql = f"""
            SELECT count(*) FROM {table}
            WHERE tissue_type = %s
              AND geom && ST_MakeEnvelope(%s, %s, %s, %s, 0)
        """
        params = (tissue_type, x0, y0, x1, y1)

    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cur.fetchone()
    return (time.perf_counter() - t0) * 1000


# ----- Q2: kNN -------------------------------------------------------------

def q2_knn(conn, table: str, tissue_type: str,
           cx: float, cy: float, k: int) -> float:
    """Execute k-nearest-neighbor query. Returns latency in ms."""
    sql = f"""
        SELECT centroid_x, centroid_y, category,
               geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) AS dist
        FROM {table}
        WHERE tissue_type = %s
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0)
        LIMIT %s
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (cx, cy, tissue_type, cx, cy, k))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000


# ----- Q3: Aggregation -----------------------------------------------------

def q3_agg(conn, table: str, tissue_type: str) -> float:
    """Per-category count + avg area aggregation. Returns latency in ms."""
    sql = f"""
        SELECT category, count(*), avg(area)
        FROM {table}
        WHERE tissue_type = %s
        GROUP BY category
    """
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, (tissue_type,))
        cur.fetchall()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def run_benchmarks(conn, tissue_types: list[str], extents: dict) -> dict[str, Any]:
    """Run all benchmark queries across Mono / SO / SPDB."""
    tables = {
        "Mono": TABLE_MONO,
        "SO":   TABLE_SO,
        "SPDB": TABLE_SPDB,
    }

    results: dict[str, Any] = {}
    random.seed(42)

    # Row counts
    print("\n[bench] Row counts:")
    for label, table in tables.items():
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {table}")
            cnt = cur.fetchone()[0]
        results[f"{label}_rows"] = cnt
        print(f"  {label} ({table}): {cnt:,}")

    # Pre-compute tissue extents from DB
    tissue_extents: dict[str, tuple] = {}
    for tt in tissue_types:
        tissue_extents[tt] = _get_tissue_extent(conn, TABLE_MONO, tt)

    # Tissue-type counts for reporting
    tissue_counts: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT tissue_type, count(*) FROM {TABLE_MONO} "
            f"GROUP BY tissue_type ORDER BY count(*) DESC"
        )
        for tt, cnt in cur.fetchall():
            tissue_counts[tt] = cnt
    results["tissue_counts"] = tissue_counts
    print(f"\n[bench] Tissue types ({len(tissue_counts)}):")
    for tt, cnt in tissue_counts.items():
        print(f"  {tt}: {cnt:,}")

    # Warm caches
    print("\n[bench] Warming caches ...")
    for label, table in tables.items():
        warmup(conn, table)

    # ==== Q1: Viewport at f=5% and f=1% ====
    for frac, frac_label in [(0.05, "5pct"), (0.01, "1pct")]:
        print(f"\n[bench] Q1 viewport f={frac_label} ({BENCHMARK_TRIALS} trials) ...")

        # Pre-generate random viewports (same for all configs)
        vp_list = []
        for _ in range(BENCHMARK_TRIALS):
            tt = random.choice(tissue_types)
            ext = tissue_extents[tt]
            vp = _random_viewport(ext, frac)
            vp_list.append((tt, ext, vp))

        for label, table in tables.items():
            lats = []
            for tt, ext, vp in vp_list:
                lats.append(q1_viewport(conn, table, tt, vp, ext))
            key = f"Q1_{frac_label}_{label}"
            results[key] = summarise(lats)
            results[key]["raw_latencies"] = lats
            print(f"  {label:5s}: p50={results[key]['p50']:.2f}ms  "
                  f"p95={results[key]['p95']:.2f}ms  "
                  f"mean={results[key]['mean']:.2f}ms")

    # ==== Q2: kNN (k=50) ====
    print(f"\n[bench] Q2 kNN k={KNN_K} ({BENCHMARK_TRIALS} trials) ...")
    knn_list = []
    for _ in range(BENCHMARK_TRIALS):
        tt = random.choice(tissue_types)
        ext = tissue_extents[tt]
        cx, cy = _random_point(ext)
        knn_list.append((tt, cx, cy))

    for label, table in tables.items():
        lats = []
        for tt, cx, cy in knn_list:
            lats.append(q2_knn(conn, table, tt, cx, cy, KNN_K))
        key = f"Q2_knn_{label}"
        results[key] = summarise(lats)
        results[key]["raw_latencies"] = lats
        print(f"  {label:5s}: p50={results[key]['p50']:.2f}ms  "
              f"p95={results[key]['p95']:.2f}ms  "
              f"mean={results[key]['mean']:.2f}ms")

    # ==== Q3: Aggregation ====
    print(f"\n[bench] Q3 aggregation ({BENCHMARK_TRIALS} trials) ...")
    agg_tissues = [random.choice(tissue_types) for _ in range(BENCHMARK_TRIALS)]

    for label, table in tables.items():
        lats = []
        for tt in agg_tissues:
            lats.append(q3_agg(conn, table, tt))
        key = f"Q3_agg_{label}"
        results[key] = summarise(lats)
        results[key]["raw_latencies"] = lats
        print(f"  {label:5s}: p50={results[key]['p50']:.2f}ms  "
              f"p95={results[key]['p95']:.2f}ms  "
              f"mean={results[key]['mean']:.2f}ms")

    # ==== Per-tissue-type breakdown (SPDB vs Mono) ====
    print(f"\n[bench] Per-tissue breakdown Q1 f=5% (50 trials per tissue) ...")
    per_tissue: dict[str, dict] = {}
    for tt in tissue_types:
        ext = tissue_extents[tt]
        mono_lats = []
        spdb_lats = []
        for _ in range(50):
            vp = _random_viewport(ext, 0.05)
            mono_lats.append(q1_viewport(conn, TABLE_MONO, tt, vp, ext))
            spdb_lats.append(q1_viewport(conn, TABLE_SPDB, tt, vp, ext))

        mono_s = summarise(mono_lats)
        spdb_s = summarise(spdb_lats)
        speedup = mono_s["p50"] / spdb_s["p50"] if spdb_s["p50"] > 0 else float("inf")
        per_tissue[tt] = {
            "mono_p50": round(mono_s["p50"], 3),
            "spdb_p50": round(spdb_s["p50"], 3),
            "speedup": round(speedup, 2),
            "n_nuclei": tissue_counts.get(tt, 0),
        }
        print(f"  {tt:20s}: Mono p50={mono_s['p50']:6.2f}ms  "
              f"SPDB p50={spdb_s['p50']:6.2f}ms  "
              f"speedup={speedup:.1f}x  (n={tissue_counts.get(tt, 0):,})")

    results["per_tissue_q1_5pct"] = per_tissue

    # ==== Speedup summary ====
    print("\n[bench] Speedup summary (p50, Mono vs SPDB):")
    for qname in ["Q1_5pct", "Q1_1pct", "Q2_knn", "Q3_agg"]:
        mono_key = f"{qname}_Mono"
        spdb_key = f"{qname}_SPDB"
        so_key   = f"{qname}_SO"
        if mono_key in results and spdb_key in results:
            mono_p50 = results[mono_key]["p50"]
            spdb_p50 = results[spdb_key]["p50"]
            so_p50   = results[so_key]["p50"] if so_key in results else 0
            sp_spdb = mono_p50 / spdb_p50 if spdb_p50 > 0 else 0
            sp_so = mono_p50 / so_p50 if so_p50 > 0 else 0
            results[f"speedup_{qname}_SPDB_vs_Mono"] = round(sp_spdb, 2)
            results[f"speedup_{qname}_SO_vs_Mono"] = round(sp_so, 2)
            print(f"  {qname:10s}: SO {sp_so:.2f}x, SPDB {sp_spdb:.2f}x vs Mono")

    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, n_rows: int, tissue_types: list[str]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pannuke_multi_cancer.json"

    # Strip raw latencies for JSON (too large) -- save separately
    results_clean = {}
    raw_latencies = {}
    for k, v in results.items():
        if isinstance(v, dict) and "raw_latencies" in v:
            raw_latencies[k] = v.pop("raw_latencies")
        results_clean[k] = v

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": "PanNuke pan-cancer nuclei segmentation",
        "csv_path": CSV_PATH,
        "total_nuclei": n_rows,
        "n_tissue_types": len(tissue_types),
        "tissue_types": tissue_types,
        "hilbert_order": HILBERT_ORDER,
        "n_hilbert_sub": N_HILBERT_SUB,
        "benchmark_trials": BENCHMARK_TRIALS,
        "knn_k": KNN_K,
        "tables": {
            "mono": TABLE_MONO,
            "so": TABLE_SO,
            "spdb": TABLE_SPDB,
        },
        "metrics": results_clean,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[results] Saved to {out_path}")

    # Also save raw latencies CSV for reproducibility
    raw_csv_path = RESULTS_DIR / "pannuke_raw_latencies.csv"
    with open(raw_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_config", "trial", "latency_ms"])
        for qname, lats in raw_latencies.items():
            for i, lat in enumerate(lats):
                w.writerow([qname, i, round(lat, 4)])
    print(f"[results] Raw latencies saved to {raw_csv_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PanNuke multi-cancer-type benchmark for SpatialPathDB."
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Skip ingestion, run benchmark only.",
    )
    parser.add_argument(
        "--csv", type=str, default=CSV_PATH,
        help=f"Path to PanNuke CSV (default: {CSV_PATH}).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PanNuke Multi-Cancer Benchmark — SpatialPathDB")
    print("=" * 70)
    print(f"  Hilbert order: {HILBERT_ORDER}  |  Sub-partitions/tissue: {N_HILBERT_SUB}")
    print(f"  Trials: {BENCHMARK_TRIALS}  |  kNN k: {KNN_K}")
    print("=" * 70)

    conn = get_conn()

    if not args.bench:
        # ---- Full pipeline: load CSV -> create tables -> ingest -> benchmark
        rows = load_csv(args.csv)
        extents = compute_extents(rows)

        tissue_types = sorted(set(r["tissue_type"] for r in rows))
        print(f"\n[data] {len(tissue_types)} tissue types: {tissue_types}")

        print("\n[setup] Creating tables ...")
        create_tables(conn, tissue_types, extents)

        n_rows = ingest_data(conn, rows, extents)
        create_indexes(conn, tissue_types)

        # Free CSV memory
        del rows

    else:
        print("\n[bench] Skipping ingestion (--bench flag).")
        # Discover tissue types from existing data
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT DISTINCT tissue_type FROM {TABLE_MONO} ORDER BY tissue_type"
            )
            tissue_types = [r[0] for r in cur.fetchall()]
            cur.execute(f"SELECT count(*) FROM {TABLE_MONO}")
            n_rows = cur.fetchone()[0]
        print(f"[bench] Found {len(tissue_types)} tissue types, {n_rows:,} rows")

        extents = {}
        for tt in tissue_types:
            extents[tt] = _get_tissue_extent(conn, TABLE_MONO, tt)
        gx0 = min(e[0] for e in extents.values())
        gx1 = max(e[1] for e in extents.values())
        gy0 = min(e[2] for e in extents.values())
        gy1 = max(e[3] for e in extents.values())
        extents["global"] = (gx0, gx1, gy0, gy1)

    # ---- Run benchmarks ----
    results = run_benchmarks(conn, tissue_types, extents)

    # ---- Save ----
    save_results(results, n_rows, tissue_types)

    conn.close()
    print("\n[done] PanNuke benchmark complete.")


if __name__ == "__main__":
    main()
