"""Cold-cache partial GiST benchmark: HCCI vs partial GiST with PG restarts.

Restarts PostgreSQL and flushes OS page caches between each query to
guarantee cold-cache conditions.  Paired design: each trial uses the same
random viewport for both HCCI and partial GiST.

Must be run on a machine where the user has passwordless sudo for
systemctl restart postgresql and echo 3 > /proc/sys/vm/drop_caches.

Usage:
    python -m benchmarks.partial_gist_cold_benchmark
    python -m benchmarks.partial_gist_cold_benchmark --trials 50
    python -m benchmarks.partial_gist_cold_benchmark --class-label Lymphocyte
    python -m benchmarks.partial_gist_cold_benchmark --viewport-frac 0.02
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2

from spdb import config, hcci
from benchmarks.framework import (
    compute_stats, save_results, wilcoxon_ranksum,
)

TABLE = config.TABLE_SLIDE_ONLY

# PostgreSQL restart command (Ubuntu with PG 17)
RESTART_CMD = ["sudo", "systemctl", "restart", "postgresql@17-main"]
FLUSH_CMD = ["sudo", "bash", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
WAIT_AFTER_RESTART = 5
MAX_CONNECT_RETRIES = 10
CONNECT_RETRY_DELAY = 2


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load ingestion metadata (slide_ids, object_counts, metas)."""
    path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    with open(path) as f:
        return json.load(f)


def get_slide_dimensions(metadata: dict, slide_id: str) -> Tuple[float, float]:
    """Return (width, height) for a slide from metadata."""
    m = metadata["metas"][slide_id]
    return float(m["image_width"]), float(m["image_height"])


# ---------------------------------------------------------------------------
# Slide dimension helpers (with cache, fall back to DB)
# ---------------------------------------------------------------------------

_slide_dims_cache: Dict[str, Tuple[float, float]] = {}


def get_dims(conn, slide_id: str, metadata: dict = None) -> Tuple[float, float]:
    """Get slide dimensions, preferring metadata, falling back to DB."""
    if slide_id not in _slide_dims_cache:
        if metadata and slide_id in metadata.get("metas", {}):
            _slide_dims_cache[slide_id] = get_slide_dimensions(metadata, slide_id)
        else:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT MAX(centroid_x), MAX(centroid_y) "
                    f"FROM {TABLE} WHERE slide_id = %s",
                    (slide_id,),
                )
                row = cur.fetchone()
            if row and row[0] and row[1]:
                _slide_dims_cache[slide_id] = (
                    float(row[0]) * 1.05,
                    float(row[1]) * 1.05,
                )
            else:
                _slide_dims_cache[slide_id] = (100000.0, 100000.0)
    return _slide_dims_cache[slide_id]


def get_all_slides(conn) -> List[str]:
    """Get all distinct slide_ids from the SO table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# PostgreSQL restart + OS cache flush + reconnect
# ---------------------------------------------------------------------------

def restart_and_flush():
    """Restart PostgreSQL and flush OS page caches."""
    subprocess.run(RESTART_CMD, check=True, capture_output=True)
    time.sleep(WAIT_AFTER_RESTART)
    subprocess.run(FLUSH_CMD, capture_output=True)
    time.sleep(1)


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


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_hcci_q(
    slide_id: str,
    class_label: str,
    viewport: Tuple[float, float, float, float],
    slide_width: float,
    slide_height: float,
) -> Tuple[str, tuple]:
    """Build HCCI query: UNION ALL of composite_key range scans."""
    x0, y0, x1, y1 = viewport
    return hcci.build_hcci_query(
        TABLE, slide_id, [class_label],
        x0, y0, x1, y1,
        slide_width, slide_height,
        config.HILBERT_ORDER,
        use_direct=True,
    )


def build_partial_gist_q(
    slide_id: str,
    class_label: str,
    viewport: Tuple[float, float, float, float],
) -> Tuple[str, tuple]:
    """Build query targeting the partial GiST index for a single class."""
    x0, y0, x1, y1 = viewport
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE slide_id = %s "
        f"  AND class_label = %s "
        f"  AND geom && ST_MakeEnvelope(%s, %s, %s, %s, 0)"
    )
    params = (slide_id, class_label, x0, y0, x1, y1)
    return sql, params


# ---------------------------------------------------------------------------
# Cold-query runner
# ---------------------------------------------------------------------------

def run_cold_query(
    sql: str,
    params: tuple,
) -> Tuple[float, int, dict]:
    """Restart PG, flush caches, connect, run EXPLAIN ANALYZE BUFFERS.

    Returns (latency_ms, n_rows, buffer_stats).
    """
    restart_and_flush()
    conn = connect_with_retry()

    try:
        explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(explain_sql, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            plan = cur.fetchone()[0]

        buffers = _parse_plan_buffers(plan)
        buffers["elapsed_ms"] = elapsed_ms
        n_rows = _extract_actual_rows(plan)

        return elapsed_ms, n_rows, buffers
    finally:
        conn.close()


def run_cold_query_text(
    sql: str,
    params: tuple,
) -> Tuple[float, int, dict, int]:
    """Like run_cold_query but also extracts heap blocks from TEXT EXPLAIN.

    Returns (latency_ms, n_rows, buffer_stats, heap_blocks).
    """
    restart_and_flush()
    conn = connect_with_retry()

    try:
        # JSON plan for buffers
        explain_json_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
        with conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute(explain_json_sql, params)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            plan = cur.fetchone()[0]

        buffers = _parse_plan_buffers(plan)
        buffers["elapsed_ms"] = elapsed_ms
        n_rows = _extract_actual_rows(plan)
        heap_blocks = _extract_heap_fetches(plan)

        return elapsed_ms, n_rows, buffers, heap_blocks
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# EXPLAIN parsing
# ---------------------------------------------------------------------------

def _parse_plan_buffers(plan: list) -> dict:
    """Extract buffer statistics from EXPLAIN JSON output."""
    result = {
        "shared_hit": 0,
        "shared_read": 0,
        "planning_time": 0,
        "execution_time": 0,
        "heap_fetches": 0,
    }
    if not plan:
        return result

    top = plan[0] if isinstance(plan, list) else plan
    result["planning_time"] = top.get("Planning Time", 0)
    result["execution_time"] = top.get("Execution Time", 0)

    def _walk(node):
        result["shared_hit"] += node.get("Shared Hit Blocks", 0)
        result["shared_read"] += node.get("Shared Read Blocks", 0)
        result["heap_fetches"] += node.get("Heap Fetches", 0)
        for child in node.get("Plans", []):
            _walk(child)

    plan_node = top.get("Plan", top)
    _walk(plan_node)
    return result


def _extract_actual_rows(plan: list) -> int:
    """Extract total actual rows from the root plan node."""
    if not plan:
        return 0
    top = plan[0] if isinstance(plan, list) else plan
    root_plan = top.get("Plan", {})
    return root_plan.get("Actual Rows", 0)


def _extract_heap_fetches(plan: list) -> int:
    """Walk EXPLAIN plan and sum Heap Fetches across all nodes."""
    if not plan:
        return 0
    total = 0

    def _walk(node):
        nonlocal total
        total += node.get("Heap Fetches", 0)
        for child in node.get("Plans", []):
            _walk(child)

    top = plan[0] if isinstance(plan, list) else plan
    _walk(top.get("Plan", {}))
    return total


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    metadata: dict,
    slides: List[str],
    class_label: str = "Tumor",
    n_trials: int = 50,
    viewport_frac: float = 0.05,
    seed: int = config.RANDOM_SEED + 700,
) -> dict:
    """Run paired cold-cache trials: HCCI vs partial GiST.

    For each trial:
    1. Generate random slide_id + viewport (deterministic RNG)
    2. Restart PG + flush caches -> run HCCI query
    3. Restart PG + flush caches -> run partial GiST query
    """
    rng = np.random.RandomState(seed)

    class_sel = config.CLASS_DISTRIBUTION.get(class_label, 0.0)

    print(f"\n{'='*65}")
    print(f"  Cold Cache: Partial GiST Benchmark")
    print(f"  Class: {class_label} (selectivity: {class_sel:.1%})")
    print(f"  Viewport fraction: {viewport_frac}")
    print(f"  Trials: {n_trials} (2 PG restarts per trial)")
    print(f"{'='*65}")

    # Pre-cache slide dimensions (need a connection for this)
    conn = connect_with_retry()
    for sid in slides:
        get_dims(conn, sid, metadata)
    conn.close()

    # Trial storage
    trials_data: List[dict] = []
    lats_hcci: List[float] = []
    lats_partial: List[float] = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = _slide_dims_cache.get(sid, (100000.0, 100000.0))

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)
        viewport = (x0, y0, x1, y1)

        # Count HCCI ranges
        hcci_sql, hcci_params = build_hcci_q(sid, class_label, viewport, w, h)
        n_ranges = hcci_sql.count("UNION ALL") + 1

        # --- HCCI cold ---
        t_hcci, nr_hcci, buf_hcci, hf_hcci = run_cold_query_text(
            hcci_sql, hcci_params,
        )
        lats_hcci.append(t_hcci)

        # --- Partial GiST cold (same viewport) ---
        pg_sql, pg_params = build_partial_gist_q(sid, class_label, viewport)
        t_partial, nr_partial, buf_partial, hf_partial = run_cold_query_text(
            pg_sql, pg_params,
        )
        lats_partial.append(t_partial)

        trial_rec = {
            "trial": trial,
            "slide_id": sid,
            "viewport": list(viewport),
            "n_ranges": n_ranges,
            "hcci_ms": round(t_hcci, 3),
            "partial_ms": round(t_partial, 3),
            "hcci_shared_hit": buf_hcci["shared_hit"],
            "hcci_shared_read": buf_hcci["shared_read"],
            "partial_shared_hit": buf_partial["shared_hit"],
            "partial_shared_read": buf_partial["shared_read"],
            "partial_heap_blocks": hf_partial,
            "n_rows_hcci": nr_hcci,
            "n_rows_partial": nr_partial,
        }
        trials_data.append(trial_rec)

        sp = t_partial / t_hcci if t_hcci > 0 else 0
        print(
            f"  [{trial+1:>3}/{n_trials}]  "
            f"HCCI={t_hcci:>7.1f}ms (hit={buf_hcci['shared_hit']:>5} "
            f"read={buf_hcci['shared_read']:>5})  "
            f"Partial={t_partial:>7.1f}ms (hit={buf_partial['shared_hit']:>5} "
            f"read={buf_partial['shared_read']:>5} "
            f"heap={hf_partial:>5})  "
            f"sp={sp:.2f}x  ranges={n_ranges}"
        )

    # --- Statistics ---
    stats_hcci = compute_stats(lats_hcci)
    stats_partial = compute_stats(lats_partial)

    sp_p50 = stats_partial["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0
    sp_mean = stats_partial["mean"] / stats_hcci["mean"] if stats_hcci["mean"] > 0 else 0
    _, wsr_p = wilcoxon_ranksum(lats_hcci, lats_partial)

    # Buffer breakdown averages
    avg_hcci_hit = float(np.mean([t["hcci_shared_hit"] for t in trials_data]))
    avg_hcci_read = float(np.mean([t["hcci_shared_read"] for t in trials_data]))
    avg_partial_hit = float(np.mean([t["partial_shared_hit"] for t in trials_data]))
    avg_partial_read = float(np.mean([t["partial_shared_read"] for t in trials_data]))
    avg_partial_heap = float(np.mean([t["partial_heap_blocks"] for t in trials_data]))
    avg_n_ranges = float(np.mean([t["n_ranges"] for t in trials_data]))
    avg_rows_hcci = float(np.mean([t["n_rows_hcci"] for t in trials_data]))
    avg_rows_partial = float(np.mean([t["n_rows_partial"] for t in trials_data]))

    hcci_cold_ratio = avg_hcci_read / max(1, avg_hcci_hit + avg_hcci_read)
    partial_cold_ratio = avg_partial_read / max(1, avg_partial_hit + avg_partial_read)

    hcci_total_io = avg_hcci_hit + avg_hcci_read
    partial_total_io = avg_partial_hit + avg_partial_read
    io_reduction = (1 - hcci_total_io / max(1, partial_total_io)) * 100

    # --- Print summary ---
    print(f"\n{'='*65}")
    print(f"  Cold Cache Results: {class_label} @ {viewport_frac*100:.0f}% viewport")
    print(f"{'='*65}")
    print(f"\n  {'':>18} {'p50':>8} {'p95':>8} {'mean':>8} {'std':>8} {'CI95':>14}")
    print(f"  {'-'*64}")
    print(
        f"  {'HCCI (cold)':>18} "
        f"{stats_hcci['p50']:>8.1f} {stats_hcci['p95']:>8.1f} "
        f"{stats_hcci['mean']:>8.1f} {stats_hcci['std']:>8.1f} "
        f"[{stats_hcci['ci95_lower']:.1f}, {stats_hcci['ci95_upper']:.1f}]"
    )
    print(
        f"  {'Partial GiST':>18} "
        f"{stats_partial['p50']:>8.1f} {stats_partial['p95']:>8.1f} "
        f"{stats_partial['mean']:>8.1f} {stats_partial['std']:>8.1f} "
        f"[{stats_partial['ci95_lower']:.1f}, {stats_partial['ci95_upper']:.1f}]"
    )
    print(f"\n  Speedup (Partial / HCCI):")
    print(f"    p50:  {sp_p50:.2f}x")
    print(f"    mean: {sp_mean:.2f}x")
    print(f"  Wilcoxon rank-sum p = {wsr_p:.2e}")

    print(f"\n  Buffer breakdown (cold, averages):")
    print(f"    HCCI:    hit={avg_hcci_hit:>7.0f}  read={avg_hcci_read:>7.0f}  "
          f"total={hcci_total_io:>7.0f}  cold%={hcci_cold_ratio:.1%}")
    print(f"    Partial: hit={avg_partial_hit:>7.0f}  read={avg_partial_read:>7.0f}  "
          f"total={partial_total_io:>7.0f}  cold%={partial_cold_ratio:.1%}  "
          f"heap_fetches={avg_partial_heap:.0f}")
    print(f"    I/O reduction (HCCI vs Partial): {io_reduction:.1f}%")

    print(f"\n  Avg rows:   HCCI={avg_rows_hcci:.0f}  Partial={avg_rows_partial:.0f}")
    print(f"  Avg ranges: {avg_n_ranges:.1f}")

    # --- Assemble results ---
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table": TABLE,
        "class_label": class_label,
        "class_selectivity": class_sel,
        "viewport_frac": viewport_frac,
        "n_trials": n_trials,
        "n_slides": len(slides),
        "cache_mode": "cold",
        "methodology": "restart PostgreSQL + drop OS caches before each query",
        "hcci": stats_hcci,
        "partial_gist": stats_partial,
        "speedup_p50": round(sp_p50, 3),
        "speedup_mean": round(sp_mean, 3),
        "wilcoxon_p": wsr_p,
        "buffer_breakdown": {
            "hcci_avg_hit": round(avg_hcci_hit, 1),
            "hcci_avg_read": round(avg_hcci_read, 1),
            "hcci_total_io": round(hcci_total_io, 1),
            "hcci_cold_ratio": round(hcci_cold_ratio, 4),
            "partial_avg_hit": round(avg_partial_hit, 1),
            "partial_avg_read": round(avg_partial_read, 1),
            "partial_total_io": round(partial_total_io, 1),
            "partial_cold_ratio": round(partial_cold_ratio, 4),
            "partial_avg_heap_fetches": round(avg_partial_heap, 1),
            "io_reduction_pct": round(io_reduction, 1),
        },
        "hilbert_ranges": {
            "avg": round(avg_n_ranges, 1),
        },
        "avg_rows": {
            "hcci": round(avg_rows_hcci, 1),
            "partial_gist": round(avg_rows_partial, 1),
        },
        "trials": trials_data,
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cold-cache benchmark: HCCI vs partial GiST"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of paired cold-cache trials (default: 50)",
    )
    parser.add_argument(
        "--viewport-frac", type=float, default=0.05,
        help="Viewport fraction of slide area (default: 0.05)",
    )
    parser.add_argument(
        "--class-label", type=str, default="Tumor",
        help="Class label to benchmark (default: Tumor)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  Partial GiST Cold Cache Benchmark")
    print("  HCCI (index-only) vs Partial GiST (per-class, cold cache)")
    print("  Restarts PostgreSQL + flushes OS caches before every query")
    print("=" * 65)

    # Verify sudo works
    print("\n  Verifying sudo access...")
    try:
        subprocess.run(
            ["sudo", "-n", "systemctl", "status", "postgresql@17-main"],
            check=True, capture_output=True,
        )
        print("  sudo systemctl: OK")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: sudo check returned: {e}")
        print("  Continuing anyway...")

    try:
        subprocess.run(
            ["sudo", "-n", "bash", "-c", "echo test > /dev/null"],
            check=True, capture_output=True,
        )
        print("  sudo bash: OK")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: sudo bash check returned: {e}")
        print("  Continuing anyway...")

    # Get slide list + metadata
    conn = connect_with_retry()
    slides = get_all_slides(conn)
    print(f"  Found {len(slides)} slides in {TABLE}")

    # Load metadata
    print("  Loading metadata...")
    try:
        metadata = load_metadata()
        print(f"  Loaded metadata for {len(metadata.get('slide_ids', []))} slides")
    except FileNotFoundError:
        print("  WARNING: ingest_metadata.json not found, using DB fallback")
        metadata = {}

    # Pre-cache dimensions
    for sid in slides:
        get_dims(conn, sid, metadata)
    conn.close()

    # Verify partial GiST index exists
    conn = connect_with_retry()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s
              AND indexname LIKE 'idx_gist_partial_%%'
        """, (TABLE,))
        existing = [r[0] for r in cur.fetchall()]
    conn.close()

    target_idx = f"idx_gist_partial_{args.class_label.lower()}"
    if target_idx in existing:
        print(f"  Partial GiST index found: {target_idx}")
    else:
        print(f"  WARNING: Partial GiST index '{target_idx}' not found.")
        print(f"  Existing: {existing}")
        print(f"  Run partial_gist_benchmark.py --create-indexes first.")
        print(f"  Continuing anyway (will use standard GiST fallback)...")

    t_start = time.time()

    results = run_benchmark(
        metadata=metadata,
        slides=slides,
        class_label=args.class_label,
        n_trials=args.trials,
        viewport_frac=args.viewport_frac,
    )

    total = time.time() - t_start
    total_restarts = args.trials * 2

    print(f"\n{'='*65}")
    print(f"  Final Summary")
    print(f"{'='*65}")
    print(f"  HCCI p50:    {results['hcci']['p50']:.1f}ms  "
          f"[{results['hcci']['ci95_lower']:.1f}, {results['hcci']['ci95_upper']:.1f}]")
    print(f"  Partial p50: {results['partial_gist']['p50']:.1f}ms  "
          f"[{results['partial_gist']['ci95_lower']:.1f}, "
          f"{results['partial_gist']['ci95_upper']:.1f}]")
    print(f"  Speedup:     {results['speedup_p50']:.2f}x (p50)  "
          f"{results['speedup_mean']:.2f}x (mean)")
    print(f"  Wilcoxon:    p = {results['wilcoxon_p']:.2e}")
    print(f"\n  {total_restarts} PG restarts in {total:.0f}s ({total/60:.1f}m)")

    # Save results
    path = save_results(results, "partial_gist_cold_benchmark")
    print(f"  Results saved to {path}")


if __name__ == "__main__":
    main()
