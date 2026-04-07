#!/usr/bin/env python3
"""Standalone SO (slide-only) viewport sensitivity benchmark.

Runs Q1 viewport queries against objects_slide_only at varying viewport
fractions to fill the missing SO column in the viewport sensitivity table.

This script is self-contained and does not import from the spdb package,
so it can run without disturbing the main benchmark process.
"""

import json
import os
import sys
import time
import csv

import numpy as np
from scipy import stats
import psycopg2

# ── Configuration ──────────────────────────────────────────────────────────
DB_DSN = f"host=localhost port=5432 dbname=spdb user={os.getenv('USER', 'ubuntu')}"
TABLE = "objects_slide_only"
METADATA_PATH = os.path.expanduser(
    "~/SpatialPathDB_CLEAN/results/ingest_metadata.json"
)
OUTPUT_PATH = os.path.expanduser(
    "~/SpatialPathDB_CLEAN/results/raw/viewport_sensitivity_so.json"
)
RAW_DIR = os.path.expanduser("~/SpatialPathDB_CLEAN/results/raw")

VIEWPORT_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
N_TRIALS = 200
SEED = 42


# ── Helper functions (mirrors benchmarks/framework.py) ─────────────────────

def compute_stats(latencies_ms):
    arr = np.array(latencies_ms, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_half = t_crit * se
    else:
        ci_half = 0.0
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": std,
        "cv": std / mean if mean > 0 else 0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_lower": round(mean - ci_half, 3),
        "ci95_upper": round(mean + ci_half, 3),
        "ci95_half": round(ci_half, 3),
    }


def random_viewport(width, height, frac, rng):
    vw = width * np.sqrt(frac)
    vh = height * np.sqrt(frac)
    x0 = float(rng.uniform(0, max(1, width - vw)))
    y0 = float(rng.uniform(0, max(1, height - vh)))
    return x0, y0, float(x0 + vw), float(y0 + vh)


def time_query(conn, sql, params=None):
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        cur.execute(sql, params)
        rows = cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
    return rows, elapsed


def warmup_cache(conn, table_name, n_passes=3):
    with conn.cursor() as cur:
        for _ in range(n_passes):
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            cur.fetchone()
            cur.execute(
                f"SELECT * FROM {table_name} ORDER BY random() LIMIT 1000"
            )
            cur.fetchall()


def save_raw_latencies(latencies, name, config_name):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}_{config_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "latency_ms"])
        for i, lat in enumerate(latencies):
            w.writerow([i, lat])
    return path


# ── Main benchmark ─────────────────────────────────────────────────────────

def main():
    print(f"SO Viewport Sensitivity Benchmark")
    print(f"  Table: {TABLE}")
    print(f"  Fractions: {VIEWPORT_FRACTIONS}")
    print(f"  Trials per fraction: {N_TRIALS}")
    print(f"  Seed: {SEED}")
    print()

    # Load metadata
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    slide_ids = metadata["slide_ids"]
    print(f"  Loaded metadata: {len(slide_ids)} slides")

    # Connect
    conn = psycopg2.connect(DB_DSN)
    print(f"  Connected to database")

    # Warmup
    print(f"  Warming up cache for {TABLE}...")
    warmup_cache(conn, TABLE)
    print(f"  Warmup complete")

    # The SO viewport query: same as Mono but on the slide_only partitioned table
    sql_template = f"""
        SELECT object_id, centroid_x, centroid_y, class_label
        FROM {TABLE}
        WHERE slide_id = %s
          AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))
    """

    results = {}
    for frac in VIEWPORT_FRACTIONS:
        print(f"\n  Viewport fraction: {frac}")
        rng = np.random.RandomState(SEED)
        latencies = []

        for trial in range(N_TRIALS):
            sid = rng.choice(slide_ids)
            m = metadata["metas"][sid]
            w = float(m["image_width"])
            h = float(m["image_height"])
            x0, y0, x1, y1 = random_viewport(w, h, frac, rng)

            _, elapsed = time_query(conn, sql_template, (sid, x0, y0, x1, y1))
            latencies.append(elapsed)

        stats_result = compute_stats(latencies)
        results[str(frac)] = {"SO": stats_result}
        save_raw_latencies(latencies, f"viewport_sens_{frac}", "SO")

        print(f"    SO p50={stats_result['p50']:.1f}ms  "
              f"p95={stats_result['p95']:.1f}ms  "
              f"mean={stats_result['mean']:.1f}ms  "
              f"ci95=+/-{stats_result['ci95_half']:.1f}ms")

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_PATH}")

    conn.close()
    print("  Done.")


if __name__ == "__main__":
    main()
