#!/usr/bin/env python3
"""Master benchmark runner for SpatialPathDB.

Usage:
    python run_benchmarks.py              # Run all benchmarks
    python run_benchmarks.py q1           # Run only Q1
    python run_benchmarks.py extended     # Run extended experiments
    python run_benchmarks.py figures      # Generate figures only
    python run_benchmarks.py isolated     # Run cache-isolated Q1 (Linux)
    python run_benchmarks.py duckdb       # Run DuckDB baseline
    python run_benchmarks.py costmodel    # Run cost model analysis
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from spdb import config


def run_q1():
    from benchmarks.q1_viewport import run_q1_all_configs
    print("\n" + "=" * 60)
    print("Q1: Viewport Latency Benchmark")
    print("=" * 60)
    return run_q1_all_configs(n_trials=config.BENCHMARK_TRIALS_Q1)


def run_q2():
    from benchmarks.q2_knn import run_q2_all_configs
    print("\n" + "=" * 60)
    print("Q2: kNN Benchmark (k=50)")
    print("=" * 60)
    return run_q2_all_configs(k=50, n_trials=config.BENCHMARK_TRIALS_Q2)


def run_q3():
    from benchmarks.q3_aggregation import run_q3_all_configs
    print("\n" + "=" * 60)
    print("Q3: Aggregation Benchmark")
    print("=" * 60)
    return run_q3_all_configs(n_trials=config.BENCHMARK_TRIALS_Q3)


def run_q4():
    from benchmarks.q4_spatial_join import run_q4_all_configs
    print("\n" + "=" * 60)
    print("Q4: Spatial Join Benchmark")
    print("=" * 60)
    return run_q4_all_configs(n_trials=config.BENCHMARK_TRIALS_Q4)


def run_concurrent():
    from benchmarks.concurrent import run_concurrent_all
    print("\n" + "=" * 60)
    print("Concurrent Throughput Benchmark")
    print("=" * 60)
    return run_concurrent_all()


def run_extended():
    from benchmarks.extended import (
        viewport_sensitivity, workload_mix, knn_k_sweep,
        storage_overhead, density_analysis,
        hilbert_vs_zorder, cold_cache_benchmark,
        pruning_analysis, hilbert_order_sensitivity,
    )

    print("\n" + "=" * 60)
    print("Extended Experiments")
    print("=" * 60)

    print("\n--- Storage Overhead ---")
    storage_overhead()

    print("\n--- Density Analysis ---")
    density_analysis()

    print("\n--- Viewport Sensitivity ---")
    viewport_sensitivity(n_trials=200)

    print("\n--- kNN k-sweep ---")
    knn_k_sweep(n_trials=200)

    print("\n--- Workload Mix ---")
    workload_mix(n_total=500)

    print("\n--- Hilbert vs Z-order ---")
    hilbert_vs_zorder(n_trials=200)

    print("\n--- Pruning Analysis ---")
    pruning_analysis(n_trials=100)

    print("\n--- Hilbert Order Sensitivity ---")
    hilbert_order_sensitivity(orders=[6, 8, 10, 12], n_trials=200)

    print("\n--- Cold Cache ---")
    cold_cache_benchmark(n_trials=30)


def run_isolated():
    """Run Q1 benchmarks with full cache isolation (Linux only).

    Each configuration gets a full PostgreSQL restart + OS cache flush
    before benchmarking, eliminating cross-config cache contamination.
    """
    try:
        from benchmarks.isolation import IsolatedBenchmarkRunner
    except ImportError as e:
        print(f"ERROR: Cannot import isolation module: {e}")
        print("  Ensure benchmarks/isolation.py exists.")
        return

    import psycopg2
    from benchmarks.q1_viewport import run_q1
    from benchmarks.framework import (
        load_metadata, compute_stats, save_raw_latencies,
        save_results, print_comparison, wilcoxon_ranksum,
    )

    print("\n" + "=" * 60)
    print("Cache-Isolated Q1 Benchmark")
    print("=" * 60)

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    n_trials = config.BENCHMARK_TRIALS_Q1

    configs_to_run = {
        k: v for k, v in config.BENCH_CONFIGS_WITH_ZORDER.items()
        if k in ["Mono", "SO", "SPDB", "SPDB-Z"]
    }

    runner = IsolatedBenchmarkRunner()

    def _make_bench_fn(table, cfg_name):
        """Factory to avoid late-binding closure over loop variables."""
        def bench():
            conn = psycopg2.connect(config.dsn())
            lats = run_q1(conn, table, slide_ids, metadata, n_trials=n_trials)
            conn.close()
            stats = compute_stats(lats)
            save_raw_latencies(lats, "q1_isolated", cfg_name)
            return {"stats": stats, "latencies": lats}
        return bench

    for name, table in configs_to_run.items():
        runner.add(name, _make_bench_fn(table, name))

    raw_results = runner.run_isolated()

    stats_results = {}
    all_latencies = {}
    for name, res in raw_results.items():
        if isinstance(res, dict) and "stats" in res:
            stats_results[name] = res["stats"]
            all_latencies[name] = res["latencies"]
        else:
            stats_results[name] = res

    stat_tests = {}
    for a, b in [("Mono", "SPDB"), ("SO", "SPDB"), ("Mono", "SO")]:
        if a in all_latencies and b in all_latencies:
            stat, p = wilcoxon_ranksum(all_latencies[a], all_latencies[b])
            stat_tests[f"{a}_vs_{b}"] = {"statistic": stat, "p_value": p}
            print(f"  Wilcoxon {a} vs {b}: p={p:.2e}")

    output = {
        "query": "Q1_viewport_isolated",
        "n_trials": n_trials,
        "configs": stats_results,
        "statistical_tests": stat_tests,
        "isolation": "full_restart_and_cache_flush",
        "isolation_log": runner.get_isolation_log(),
    }
    save_results(output, "q1_isolated")
    print_comparison(stats_results)


def run_duckdb():
    """Run DuckDB Spatial baseline comparison."""
    try:
        from benchmarks.duckdb_baseline import compare_pg_vs_duckdb, HAS_DUCKDB
    except ImportError as e:
        print(f"ERROR: Cannot import duckdb_baseline: {e}")
        return

    if not HAS_DUCKDB:
        print("ERROR: duckdb package not installed. Run: pip install duckdb")
        return

    print("\n" + "=" * 60)
    print("DuckDB Spatial Baseline Comparison")
    print("=" * 60)

    results = compare_pg_vs_duckdb(n_trials=200)

    from benchmarks.framework import save_results
    save_results(results, "duckdb_comparison")
    print("DuckDB comparison saved.")


def run_costmodel():
    """Run analytical cost model analysis and validation."""
    try:
        from spdb.cost_model_analytical import full_cost_analysis
    except ImportError as e:
        print(f"ERROR: Cannot import cost_model_analytical: {e}")
        return

    print("\n" + "=" * 60)
    print("Analytical Cost Model Analysis")
    print("=" * 60)

    results = full_cost_analysis(
        n_objects=1_260_000,
        viewport_frac=0.05,
    )

    from benchmarks.framework import save_results
    save_results(results, "cost_model_analysis")

    opt = results.get("optimal", {})
    print(f"  Optimal T*: {opt.get('T_star', 'N/A')}")
    print(f"  Optimal p*: {opt.get('p_star', 'N/A')}")
    print(f"  C_h: {results.get('ch', 'N/A')}")
    print(f"  Predicted pruning: {results.get('pruning_rate', 'N/A'):.1%}")


def run_figures():
    from visualization.figures import generate_all_figures
    from visualization.tables import generate_all_tables
    print("\n" + "=" * 60)
    print("Generating Figures and Tables")
    print("=" * 60)
    generate_all_figures()
    generate_all_tables()


def run_all():
    t0 = time.time()

    run_q1()
    run_q2()
    run_q3()
    run_q4()
    run_concurrent()
    run_extended()
    run_costmodel()
    run_figures()

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete in {total:.0f}s ({total/60:.1f} min)")
    print(f"Results: {config.RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    dispatch = {
        "all": run_all,
        "q1": run_q1,
        "q2": run_q2,
        "q3": run_q3,
        "q4": run_q4,
        "concurrent": run_concurrent,
        "extended": run_extended,
        "isolated": run_isolated,
        "duckdb": run_duckdb,
        "costmodel": run_costmodel,
        "figures": run_figures,
    }

    for arg in args:
        if arg in dispatch:
            dispatch[arg]()
        else:
            print(f"Unknown benchmark: {arg}")
            print(f"Available: {', '.join(dispatch.keys())}")
            sys.exit(1)
