#!/usr/bin/env python3
"""Linux benchmark runner with proper cache isolation.

Runs the complete SPDB benchmark suite on Linux with full PostgreSQL
restart and OS cache flush between every configuration.

Usage:
    python deploy/benchmark_linux.py                    # Full suite
    python deploy/benchmark_linux.py --config Mono SPDB # Specific configs
    python deploy/benchmark_linux.py --trials 100       # Fewer trials
    python deploy/benchmark_linux.py --dry-run           # Show plan only
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from spdb import config
from benchmarks.isolation import IsolatedBenchmarkRunner
from benchmarks.framework import compute_stats, save_results


# ---------------------------------------------------------------------------
# Hardware info capture
# ---------------------------------------------------------------------------

def hardware_info():
    """Capture hardware and software metadata for reproducibility."""
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }

    # CPU
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=5)
            for line in result.stdout.split("\n"):
                if "Model name" in line:
                    info["cpu_model"] = line.split(":")[1].strip()
                elif "CPU(s):" in line and "NUMA" not in line and "On-line" not in line:
                    info["cpu_cores"] = int(line.split(":")[1].strip())
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5)
            info["cpu_model"] = result.stdout.strip()
    except Exception:
        pass

    # RAM
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1048576, 1)
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5)
            info["ram_gb"] = round(int(result.stdout.strip()) / (1024**3), 1)
    except Exception:
        pass

    # PostgreSQL version
    try:
        import psycopg2
        conn = psycopg2.connect(config.dsn())
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            info["pg_version"] = cur.fetchone()[0]
            cur.execute("SELECT PostGIS_Version()")
            info["postgis_version"] = cur.fetchone()[0]
        conn.close()
    except Exception:
        pass

    # Disk info
    try:
        if platform.system() == "Linux":
            result = subprocess.run(
                ["lsblk", "-o", "NAME,SIZE,TYPE,ROTA", "-J"],
                capture_output=True, text=True, timeout=5)
            info["disk_info"] = result.stdout.strip()[:500]
    except Exception:
        pass

    return info


# ---------------------------------------------------------------------------
# Isolated benchmark runner
# ---------------------------------------------------------------------------

def run_isolated_benchmarks(configs=None, n_trials=500, dry_run=False):
    """Run complete benchmark suite with proper cache isolation.

    Delegates all restart/flush logic to IsolatedBenchmarkRunner so
    there is a single code-path for cache isolation across the project.
    """
    import psycopg2
    from benchmarks.q1_viewport import run_q1
    from benchmarks.framework import (
        load_metadata, save_raw_latencies, wilcoxon_ranksum,
    )

    hw = hardware_info()
    print(f"\n{'='*60}")
    print(f"  SpatialPathDB Benchmark (Isolated)")
    print(f"  Platform: {hw.get('platform', 'unknown')}")
    print(f"  CPU: {hw.get('cpu_model', 'unknown')}")
    print(f"  RAM: {hw.get('ram_gb', '?')} GB")
    print(f"  PostgreSQL: {hw.get('pg_version', 'unknown')[:40]}")
    print(f"  Trials: {n_trials}")
    print(f"{'='*60}\n")

    if configs is None:
        configs = ["Mono", "SO", "SPDB", "SPDB-Z"]

    config_tables = {
        "Mono": config.TABLE_MONO,
        "SO": config.TABLE_SLIDE_ONLY,
        "SPDB": config.TABLE_SPDB,
        "SPDB-Z": config.TABLE_SPDB_ZORDER,
    }

    valid_configs = [(c, config_tables[c]) for c in configs if c in config_tables]
    skipped = [c for c in configs if c not in config_tables]
    for s in skipped:
        print(f"  SKIP unknown config: {s}")

    if dry_run:
        print("DRY RUN - would execute:")
        for name, _ in valid_configs:
            print(f"  pg_restart + cache_flush -> run Q1 on {name} ({n_trials} trials)")
        return

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]

    runner = IsolatedBenchmarkRunner()

    def _make_bench_fn(table, cfg_name):
        """Factory that returns a zero-arg callable for the runner."""
        def bench():
            conn = psycopg2.connect(config.dsn())
            t0 = time.time()
            lats = run_q1(conn, table, slide_ids, metadata,
                          n_trials=n_trials, viewport_frac=0.05, seed=42)
            elapsed = time.time() - t0
            conn.close()
            stats = compute_stats(lats)
            save_raw_latencies(lats, "q1_isolated_linux", cfg_name)
            print(f"  p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  "
                  f"mean={stats['mean']:.1f}ms  ({elapsed:.0f}s)")
            return {"stats": stats, "latencies": lats}
        return bench

    for name, table in valid_configs:
        runner.add(name, _make_bench_fn(table, name))

    raw_results = runner.run_isolated()

    all_results = {}
    all_latencies = {}
    for name, res in raw_results.items():
        if isinstance(res, dict) and "stats" in res:
            all_results[name] = res["stats"]
            all_latencies[name] = res["latencies"]
        else:
            all_results[name] = res

    stat_tests = {}
    for a, b in [("Mono", "SPDB"), ("SO", "SPDB"), ("Mono", "SO")]:
        if a in all_latencies and b in all_latencies:
            stat, p = wilcoxon_ranksum(all_latencies[a], all_latencies[b])
            stat_tests[f"{a}_vs_{b}"] = {"statistic": stat, "p_value": p}
            print(f"  Wilcoxon {a} vs {b}: p={p:.2e}")

    output = {
        "hardware": hw,
        "n_trials": n_trials,
        "configs": all_results,
        "statistical_tests": stat_tests,
        "isolation": "full_restart_and_cache_flush",
        "isolation_log": runner.get_isolation_log(),
    }
    save_results(output, "q1_viewport_isolated")
    print(f"\n  Results saved to {config.RAW_DIR}/q1_viewport_isolated.json")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPDB Isolated Benchmark Runner")
    parser.add_argument("--config", nargs="+", default=None,
                        help="Configs to benchmark (default: Mono SO SPDB SPDB-Z)")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of trials per config (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without executing")
    args = parser.parse_args()

    run_isolated_benchmarks(
        configs=args.config,
        n_trials=args.trials,
        dry_run=args.dry_run,
    )
