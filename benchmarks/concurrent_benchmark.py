"""Concurrent client benchmark: HCCI vs GiST throughput under load.

Measures queries/sec at 1, 4, 8, 16 concurrent clients.
Each client runs viewport queries in a tight loop for a fixed duration.

Usage:
    python -m benchmarks.concurrent_benchmark
    python -m benchmarks.concurrent_benchmark --duration 30 --clients 1,4,8,16
"""

from __future__ import annotations

import argparse
import json
import os
import time
import threading
from typing import List, Tuple

import numpy as np
import psycopg2

from spdb import config, hcci
from benchmarks.framework import compute_stats

TABLE = config.TABLE_SLIDE_ONLY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_dims_cache = {}
_dims_lock = threading.Lock()


def get_dims(conn, slide_id: str, metadata: dict = None) -> Tuple[float, float]:
    with _dims_lock:
        if slide_id not in _dims_cache:
            if metadata and slide_id in metadata.get("metas", {}):
                m = metadata["metas"][slide_id]
                _dims_cache[slide_id] = (float(m["image_width"]), float(m["image_height"]))
            else:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT MAX(centroid_x), MAX(centroid_y) FROM {TABLE} WHERE slide_id = %s", (slide_id,))
                    row = cur.fetchone()
                if row and row[0] and row[1]:
                    _dims_cache[slide_id] = (float(row[0]) * 1.05, float(row[1]) * 1.05)
                else:
                    _dims_cache[slide_id] = (100000.0, 100000.0)
    return _dims_cache[slide_id]


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def worker(
    worker_id: int,
    slides: List[str],
    metadata: dict,
    mode: str,  # "hcci" or "gist"
    viewport_frac: float,
    class_label: str,
    duration_sec: float,
    results_out: dict,
):
    """Run queries in a loop for `duration_sec` seconds, recording latencies."""
    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True  # no transaction overhead

    rng = np.random.RandomState(42 + worker_id * 1000)
    p = config.HILBERT_ORDER
    latencies = []

    # Pre-cache dims
    for sid in slides:
        get_dims(conn, sid, metadata)

    deadline = time.monotonic() + duration_sec

    while time.monotonic() < deadline:
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        if mode == "hcci":
            sql, params = hcci.build_hcci_query(
                TABLE, sid, [class_label],
                x0, y0, x1, y1,
                w, h, p, use_direct=True,
            )
        else:
            sql, params = hcci.build_baseline_bbox_query(
                TABLE, sid, [class_label],
                x0, y0, x1, y1,
            )

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cur.fetchall()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    conn.close()

    results_out[worker_id] = {
        "n_queries": len(latencies),
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_concurrent(
    slides: List[str],
    metadata: dict,
    n_clients: int,
    duration_sec: float,
    mode: str,
    viewport_frac: float = 0.05,
    class_label: str = "Tumor",
) -> dict:
    """Run n_clients concurrent workers for duration_sec."""
    results = {}
    threads = []

    for i in range(n_clients):
        t = threading.Thread(
            target=worker,
            args=(i, slides, metadata, mode, viewport_frac, class_label, duration_sec, results),
        )
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.time() - t0

    # Aggregate
    all_lats = []
    total_queries = 0
    for wid, res in sorted(results.items()):
        all_lats.extend(res["latencies"])
        total_queries += res["n_queries"]

    throughput = total_queries / wall_time
    stats = compute_stats(all_lats)

    return {
        "mode": mode,
        "n_clients": n_clients,
        "duration_sec": round(wall_time, 1),
        "total_queries": total_queries,
        "throughput_qps": round(throughput, 1),
        "latency": stats,
        "per_client_queries": {
            str(wid): res["n_queries"] for wid, res in sorted(results.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Concurrent HCCI vs GiST benchmark")
    parser.add_argument("--duration", type=int, default=30, help="Duration per run in seconds")
    parser.add_argument("--clients", type=str, default="1,4,8,16", help="Comma-separated client counts")
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    parser.add_argument("--class-label", type=str, default="Tumor")
    args = parser.parse_args()

    client_counts = [int(c) for c in args.clients.split(",")]

    print("=" * 60)
    print("  Concurrent Client Benchmark: HCCI vs GiST")
    print(f"  Duration: {args.duration}s per config")
    print(f"  Client counts: {client_counts}")
    print("=" * 60)

    # Load metadata
    metadata_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Get slides
    conn = psycopg2.connect(config.dsn())
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        slides = [r[0] for r in cur.fetchall()]
    print(f"  Found {len(slides)} slides in {TABLE}")

    # Pre-cache dims
    for sid in slides:
        get_dims(conn, sid, metadata)
    conn.close()

    all_results = []

    for n in client_counts:
        for mode in ["hcci", "gist"]:
            print(f"\n  Running {mode.upper()} with {n} clients for {args.duration}s...")
            result = run_concurrent(
                slides, metadata, n, args.duration, mode,
                viewport_frac=args.viewport_frac,
                class_label=args.class_label,
            )
            all_results.append(result)
            print(f"    {mode.upper()} @ {n} clients: "
                  f"{result['throughput_qps']:.1f} q/s, "
                  f"p50={result['latency']['p50']:.1f}ms, "
                  f"p95={result['latency']['p95']:.1f}ms, "
                  f"total={result['total_queries']} queries")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'Clients':>8} {'Mode':>6} {'Throughput':>12} {'p50 (ms)':>10} {'p95 (ms)':>10} {'Queries':>8}")
    print(f"  {'-'*62}")
    for r in all_results:
        print(f"  {r['n_clients']:>8} {r['mode'].upper():>6} {r['throughput_qps']:>12.1f} "
              f"{r['latency']['p50']:>10.1f} {r['latency']['p95']:>10.1f} {r['total_queries']:>8}")

    # Save
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_sec": args.duration,
        "viewport_frac": args.viewport_frac,
        "class_label": args.class_label,
        "n_slides": len(slides),
        "results": all_results,
    }

    path = os.path.join(config.RESULTS_DIR, "raw", "concurrent_benchmark.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {path}")


if __name__ == "__main__":
    main()
