"""Cold cache benchmark: HCCI vs GiST with empty PostgreSQL shared_buffers.

Restarts PostgreSQL before each trial to flush all buffer pools.
Paired design: each HCCI/GiST comparison uses the same viewport
under identical cache conditions.

Must be run on a machine where the user has passwordless sudo for
systemctl restart postgresql.

Usage:
    python -m benchmarks.hcci_cold_cache
    python -m benchmarks.hcci_cold_cache --trials 25  # quick run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import psycopg2

from spdb import config, hcci
from benchmarks.framework import compute_stats, save_results, wilcoxon_ranksum

TABLE = config.TABLE_SLIDE_ONLY

RESTART_CMD = ["sudo", "systemctl", "restart", "postgresql@17-main"]
WAIT_AFTER_RESTART = 5
MAX_CONNECT_RETRIES = 10
CONNECT_RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# PostgreSQL restart + reconnect
# ---------------------------------------------------------------------------

def restart_postgresql():
    """Restart PostgreSQL to flush shared_buffers."""
    subprocess.run(RESTART_CMD, check=True, capture_output=True)
    time.sleep(WAIT_AFTER_RESTART)


def connect_with_retry() -> psycopg2.extensions.connection:
    """Connect to PostgreSQL with retry loop after restart."""
    for attempt in range(MAX_CONNECT_RETRIES):
        try:
            conn = psycopg2.connect(config.dsn())
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError:
            if attempt < MAX_CONNECT_RETRIES - 1:
                time.sleep(CONNECT_RETRY_DELAY)
            else:
                raise


def run_cold_query(sql: str, params: tuple) -> tuple[float, dict]:
    """Restart PostgreSQL, connect, run one query with EXPLAIN BUFFERS.

    Returns (latency_ms, buffer_stats).
    """
    restart_postgresql()
    conn = connect_with_retry()

    try:
        # Drop OS page cache too (requires sudo)
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True,
        )
        time.sleep(1)

        with conn.cursor() as cur:
            # Run with EXPLAIN ANALYZE BUFFERS to capture I/O stats
            explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
            t0 = time.perf_counter()
            cur.execute(explain_sql, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            plan = cur.fetchone()[0]

        # Parse buffer stats from plan
        buffers = _parse_plan_buffers(plan)
        buffers["elapsed_ms"] = elapsed_ms

        return elapsed_ms, buffers
    finally:
        conn.close()


def _parse_plan_buffers(plan: list) -> dict:
    """Extract buffer statistics from EXPLAIN JSON output."""
    result = {
        "shared_hit": 0,
        "shared_read": 0,
        "planning_time": 0,
        "execution_time": 0,
    }

    if not plan:
        return result

    top = plan[0] if isinstance(plan, list) else plan
    result["planning_time"] = top.get("Planning Time", 0)
    result["execution_time"] = top.get("Execution Time", 0)

    def _walk(node):
        result["shared_hit"] += node.get("Shared Hit Blocks", 0)
        result["shared_read"] += node.get("Shared Read Blocks", 0)
        for child in node.get("Plans", []):
            _walk(child)

    plan_node = top.get("Plan", top)
    _walk(plan_node)

    return result


# ---------------------------------------------------------------------------
# Slide helpers (copied from hcci_benchmark for self-contained use)
# ---------------------------------------------------------------------------

def _get_all_slides(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        return [r[0] for r in cur.fetchall()]


def _get_slide_dims(conn, slide_id: str) -> tuple[float, float]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(centroid_x), MAX(centroid_y) FROM {TABLE} WHERE slide_id = %s", (slide_id,))
        row = cur.fetchone()
    if row and row[0] and row[1]:
        return float(row[0]) * 1.05, float(row[1]) * 1.05
    return 100000.0, 100000.0


_dims_cache: dict[str, tuple[float, float]] = {}


def get_dims(conn, sid):
    if sid not in _dims_cache:
        _dims_cache[sid] = _get_slide_dims(conn, sid)
    return _dims_cache[sid]


# ---------------------------------------------------------------------------
# Cold cache benchmark
# ---------------------------------------------------------------------------

def run_cold_cache_trials(
    query_id: str,
    class_labels: list[str],
    viewport_frac: float,
    slides: list[str],
    n_trials: int = 50,
    seed: int = config.RANDOM_SEED + 500,
    include_bbox: bool = False,
) -> dict:
    """Run paired cold-cache trials for HCCI vs GiST.

    For each trial:
    1. Generate random viewport
    2. Restart PostgreSQL, run HCCI query, record latency
    3. Restart PostgreSQL, run GiST query (same viewport), record latency
    4. (Optional) Restart PostgreSQL, run bbox query, record latency
    """
    rng = np.random.RandomState(seed)

    print(f"\n{'='*60}")
    print(f"  Cold Cache: Query {query_id} ({class_labels})")
    print(f"  Viewport: {viewport_frac*100:.0f}%  |  {n_trials} paired trials")
    if include_bbox:
        print(f"  Including bbox baseline (3 restarts per trial)")
    print(f"{'='*60}")

    lats_hcci = []
    lats_gist = []
    lats_bbox = []
    bufs_hcci = []
    bufs_gist = []
    bufs_bbox = []

    # We need a persistent connection to look up slide dims
    # but we'll use it only for metadata, not for benchmark queries
    meta_conn = connect_with_retry()
    for sid in slides[:20]:
        get_dims(meta_conn, sid)
    meta_conn.close()

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = _dims_cache.get(sid, (100000.0, 100000.0))

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # --- HCCI cold ---
        hcci_sql, hcci_params = hcci.build_hcci_query(
            TABLE, sid, class_labels,
            x0, y0, x1, y1, w, h,
            config.HILBERT_ORDER, use_direct=True,
        )
        t_hcci, buf_hcci = run_cold_query(hcci_sql, hcci_params)
        lats_hcci.append(t_hcci)
        bufs_hcci.append(buf_hcci)

        # --- GiST cold (same viewport) ---
        if query_id == "E":
            gist_sql, gist_params = hcci.build_baseline_gist_query_all_classes(
                TABLE, sid, x0, y0, x1, y1,
            )
        else:
            gist_sql, gist_params = hcci.build_baseline_gist_query(
                TABLE, sid, class_labels, x0, y0, x1, y1,
            )
        t_gist, buf_gist = run_cold_query(gist_sql, gist_params)
        lats_gist.append(t_gist)
        bufs_gist.append(buf_gist)

        # --- bbox cold (same viewport) ---
        if include_bbox:
            bbox_sql, bbox_params = hcci.build_baseline_bbox_query(
                TABLE, sid, class_labels, x0, y0, x1, y1,
            )
            t_bbox, buf_bbox = run_cold_query(bbox_sql, bbox_params)
            lats_bbox.append(t_bbox)
            bufs_bbox.append(buf_bbox)

        restarts = 3 if include_bbox else 2
        print(f"  Trial {trial+1}/{n_trials}: "
              f"HCCI={t_hcci:.1f}ms (hit={buf_hcci['shared_hit']} read={buf_hcci['shared_read']})  "
              f"GiST={t_gist:.1f}ms (hit={buf_gist['shared_hit']} read={buf_gist['shared_read']})"
              + (f"  bbox={t_bbox:.1f}ms" if include_bbox else ""))

    stats_h = compute_stats(lats_hcci)
    stats_g = compute_stats(lats_gist)

    sp_p50 = stats_g["p50"] / stats_h["p50"] if stats_h["p50"] > 0 else 0
    sp_mean = stats_g["mean"] / stats_h["mean"] if stats_h["mean"] > 0 else 0
    _, wsr_p = wilcoxon_ranksum(lats_hcci, lats_gist)

    # Buffer breakdown averages
    avg_hcci_hit = float(np.mean([b["shared_hit"] for b in bufs_hcci]))
    avg_hcci_read = float(np.mean([b["shared_read"] for b in bufs_hcci]))
    avg_gist_hit = float(np.mean([b["shared_hit"] for b in bufs_gist]))
    avg_gist_read = float(np.mean([b["shared_read"] for b in bufs_gist]))

    hcci_cold_ratio = avg_hcci_read / max(1, avg_hcci_hit + avg_hcci_read)
    gist_cold_ratio = avg_gist_read / max(1, avg_gist_hit + avg_gist_read)

    print(f"\n  {'':>16} {'p50':>8} {'p95':>8} {'mean':>8}")
    print(f"  {'HCCI (cold)':>16} {stats_h['p50']:>8.1f} {stats_h['p95']:>8.1f} {stats_h['mean']:>8.1f}")
    print(f"  {'GiST (cold)':>16} {stats_g['p50']:>8.1f} {stats_g['p95']:>8.1f} {stats_g['mean']:>8.1f}")
    print(f"  Speedup: {sp_p50:.2f}x (p50), {sp_mean:.2f}x (mean)")
    print(f"  Wilcoxon p = {wsr_p:.2e}")
    print(f"\n  Buffer breakdown (cold):")
    print(f"    HCCI: hit={avg_hcci_hit:.0f}  read={avg_hcci_read:.0f}  cold_ratio={hcci_cold_ratio:.1%}")
    print(f"    GiST: hit={avg_gist_hit:.0f}  read={avg_gist_read:.0f}  cold_ratio={gist_cold_ratio:.1%}")

    result = {
        "query_id": query_id,
        "classes": class_labels,
        "viewport_frac": viewport_frac,
        "n_trials": n_trials,
        "hcci": stats_h,
        "gist": stats_g,
        "speedup_p50": round(sp_p50, 3),
        "speedup_mean": round(sp_mean, 3),
        "wilcoxon_p": wsr_p,
        "buffer_breakdown": {
            "hcci_avg_hit": round(avg_hcci_hit, 1),
            "hcci_avg_read": round(avg_hcci_read, 1),
            "hcci_cold_ratio": round(hcci_cold_ratio, 4),
            "gist_avg_hit": round(avg_gist_hit, 1),
            "gist_avg_read": round(avg_gist_read, 1),
            "gist_cold_ratio": round(gist_cold_ratio, 4),
        },
    }

    if include_bbox:
        stats_b = compute_stats(lats_bbox)
        sp_bbox = stats_b["p50"] / stats_h["p50"] if stats_h["p50"] > 0 else 0
        avg_bbox_hit = float(np.mean([b["shared_hit"] for b in bufs_bbox]))
        avg_bbox_read = float(np.mean([b["shared_read"] for b in bufs_bbox]))
        bbox_cold_ratio = avg_bbox_read / max(1, avg_bbox_hit + avg_bbox_read)

        print(f"  {'bbox (cold)':>16} {stats_b['p50']:>8.1f} {stats_b['p95']:>8.1f} {stats_b['mean']:>8.1f}")
        print(f"    bbox: hit={avg_bbox_hit:.0f}  read={avg_bbox_read:.0f}  cold_ratio={bbox_cold_ratio:.1%}")

        result["bbox"] = stats_b
        result["speedup_vs_bbox_p50"] = round(sp_bbox, 3)
        result["buffer_breakdown"]["bbox_avg_hit"] = round(avg_bbox_hit, 1)
        result["buffer_breakdown"]["bbox_avg_read"] = round(avg_bbox_read, 1)
        result["buffer_breakdown"]["bbox_cold_ratio"] = round(bbox_cold_ratio, 4)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cold cache HCCI benchmark")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("  HCCI Cold Cache Benchmark")
    print("  Restarts PostgreSQL between every query")
    print("=" * 60)

    # Verify sudo works for PostgreSQL restart
    print("\n  Verifying sudo access for PostgreSQL restart...")
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "status", "postgresql@17-main"],
            check=True, capture_output=True,
        )
        print("  sudo access: OK")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: sudo check returned: {e}")
        print("  Continuing anyway...")

    # Get slide list (need a warm connection for this)
    conn = connect_with_retry()
    slides = _get_all_slides(conn)
    print(f"  Found {len(slides)} slides")
    for sid in slides[:20]:
        get_dims(conn, sid)
    conn.close()

    t_start = time.time()
    all_results: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table": TABLE,
        "n_slides": len(slides),
        "cache_mode": "cold",
        "methodology": "restart PostgreSQL + drop OS caches before each query",
    }

    # Query A: Tumor (16.4%), 5% viewport — with bbox baseline
    result_a = run_cold_cache_trials(
        "A", ["Tumor"], 0.05, slides,
        n_trials=args.trials, include_bbox=True,
    )
    all_results["query_A"] = result_a

    # Query C: Tumor+Lymphocyte (27.8%), 5% viewport
    result_c = run_cold_cache_trials(
        "C", ["Tumor", "Lymphocyte"], 0.05, slides,
        n_trials=args.trials, include_bbox=False,
    )
    all_results["query_C"] = result_c

    total = time.time() - t_start
    total_restarts = args.trials * 3 + args.trials * 2  # A has 3 (with bbox), C has 2

    print(f"\n{'='*60}")
    print(f"  Cold Cache Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Query A: HCCI={result_a['hcci']['p50']:.1f}ms  GiST={result_a['gist']['p50']:.1f}ms  "
          f"speedup={result_a['speedup_p50']:.2f}x")
    print(f"  Query C: HCCI={result_c['hcci']['p50']:.1f}ms  GiST={result_c['gist']['p50']:.1f}ms  "
          f"speedup={result_c['speedup_p50']:.2f}x")
    print(f"\n  Total: {total_restarts} PostgreSQL restarts in {total:.0f}s ({total/60:.1f}m)")

    path = save_results(all_results, "hcci_cold_cache")
    print(f"  Results saved to {path}")


if __name__ == "__main__":
    main()
