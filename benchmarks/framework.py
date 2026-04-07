"""Benchmark framework: timing, stats, EXPLAIN BUFFERS parsing, result I/O."""

import json
import os
import time
import csv

import numpy as np
from scipy import stats

from spdb import config


def percentile(data, p):
    return float(np.percentile(data, p))


def compute_stats(latencies_ms):
    """Compute summary statistics for a list of latencies."""
    arr = np.array(latencies_ms, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% confidence interval for the mean
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
        "p50": percentile(arr, 50),
        "p95": percentile(arr, 95),
        "p99": percentile(arr, 99),
        "std": std,
        "cv": std / mean if mean > 0 else 0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_lower": round(mean - ci_half, 3),
        "ci95_upper": round(mean + ci_half, 3),
        "ci95_half": round(ci_half, 3),
    }


def wilcoxon_ranksum(a, b):
    """Two-sided Wilcoxon rank-sum test. Returns (statistic, p-value)."""
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


def save_raw_latencies(latencies, name, config_name):
    """Save raw latency list to CSV."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    path = os.path.join(config.RAW_DIR, f"{name}_{config_name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "latency_ms"])
        for i, lat in enumerate(latencies):
            w.writerow([i, lat])
    return path


def save_results(results, name):
    """Save benchmark results dict to JSON."""
    os.makedirs(config.RAW_DIR, exist_ok=True)
    path = os.path.join(config.RAW_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def load_metadata():
    """Load ingestion metadata (slide_ids, object_counts, etc.)."""
    path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    with open(path) as f:
        return json.load(f)


def get_slide_dimensions(metadata, slide_id):
    """Return (width, height) for a slide from metadata."""
    m = metadata["metas"][slide_id]
    return float(m["image_width"]), float(m["image_height"])


def random_viewport(width, height, frac, rng):
    """Generate a random viewport bounding box covering `frac` of the slide."""
    vw = width * np.sqrt(frac)
    vh = height * np.sqrt(frac)
    x0 = float(rng.uniform(0, max(1, width - vw)))
    y0 = float(rng.uniform(0, max(1, height - vh)))
    return x0, y0, float(x0 + vw), float(y0 + vh)


def random_point(width, height, rng):
    """Generate a random point within slide bounds."""
    return float(rng.uniform(0, width)), float(rng.uniform(0, height))


def time_query(conn, sql, params=None):
    """Execute a query and return (result_rows, elapsed_ms)."""
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        cur.execute(sql, params)
        rows = cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
    return rows, elapsed


def time_query_explain(conn, sql, params=None):
    """Execute EXPLAIN ANALYZE and return (plan_json, exec_time, row_count)."""
    explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}"
    with conn.cursor() as cur:
        t0 = time.perf_counter()
        cur.execute(explain_sql, params)
        plan = cur.fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
    plan_json = plan[0][0] if plan else []
    actual_rows = 0
    if plan_json and len(plan_json) > 0:
        actual_rows = plan_json[0].get("Plan", {}).get("Actual Rows", 0)
    exec_time = 0
    if plan_json and len(plan_json) > 0:
        exec_time = plan_json[0].get("Execution Time", 0)
    return plan_json, exec_time, actual_rows


# ---------------------------------------------------------------------------
# I/O decomposition via EXPLAIN (ANALYZE, BUFFERS)
# ---------------------------------------------------------------------------

def _walk_plan(node, collector):
    """Recursively walk a plan node tree and call collector on each."""
    collector(node)
    for child in node.get("Plans", []):
        _walk_plan(child, collector)


def parse_buffers(plan_json):
    """Extract I/O counters from EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON).

    Returns a dict with aggregate buffer stats across all plan nodes:
      shared_hit, shared_read, shared_dirtied, shared_written,
      planning_time, execution_time, rows_removed_by_filter,
      subplans_removed, partitions_scanned
    """
    if not plan_json or len(plan_json) == 0:
        return {}

    root = plan_json[0]
    result = {
        "planning_time": root.get("Planning Time", 0.0),
        "execution_time": root.get("Execution Time", 0.0),
        "shared_hit": 0,
        "shared_read": 0,
        "shared_dirtied": 0,
        "shared_written": 0,
        "heap_fetches": 0,
        "rows_removed_by_filter": 0,
        "actual_rows": 0,
        "subplans_removed": 0,
        "partitions_scanned": 0,
        "node_types": [],
    }

    def _collect(node):
        sb = node.get("Shared Hit Blocks", 0)
        sr = node.get("Shared Read Blocks", 0)
        result["shared_hit"] += sb
        result["shared_read"] += sr
        result["shared_dirtied"] += node.get("Shared Dirtied Blocks", 0)
        result["shared_written"] += node.get("Shared Written Blocks", 0)
        result["rows_removed_by_filter"] += node.get("Rows Removed by Filter", 0)
        result["heap_fetches"] += node.get("Heap Fetches", 0)
        result["actual_rows"] += node.get("Actual Rows", 0)
        nt = node.get("Node Type", "")
        result["node_types"].append(nt)
        if nt == "Append":
            result["subplans_removed"] += node.get("Subplans Removed", 0)
            result["partitions_scanned"] += len(node.get("Plans", []))

    _walk_plan(root.get("Plan", {}), _collect)

    total_blocks = result["shared_hit"] + result["shared_read"]
    result["hit_ratio"] = (
        result["shared_hit"] / total_blocks if total_blocks > 0 else 1.0
    )
    return result


def time_query_buffers(conn, sql, params=None):
    """EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) -- full I/O decomposition.

    Returns (plan_json, parsed_buffers_dict).
    """
    explain_sql = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}"
    with conn.cursor() as cur:
        cur.execute(explain_sql, params)
        plan = cur.fetchall()
    plan_json = plan[0][0] if plan else []
    return plan_json, parse_buffers(plan_json)


def warmup_cache(conn, table_name, n_passes=3):
    """Run warmup queries to fill PostgreSQL shared buffers."""
    with conn.cursor() as cur:
        for _ in range(n_passes):
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            cur.fetchone()
            cur.execute(
                f"SELECT * FROM {table_name} "
                f"ORDER BY random() LIMIT 1000"
            )
            cur.fetchall()


def print_comparison(results_dict, metric="p50"):
    """Pretty-print comparison across configurations."""
    print(f"\n{'Config':<15} {'p50':>8} {'p95':>8} {'mean':>8} {'std':>8} {'n':>5}")
    print("-" * 55)
    for name, st in results_dict.items():
        print(f"{name:<15} {st['p50']:>8.1f} {st['p95']:>8.1f} "
              f"{st['mean']:>8.1f} {st['std']:>8.1f} {st['n']:>5}")
