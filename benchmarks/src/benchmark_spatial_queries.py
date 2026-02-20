#!/usr/bin/env python3
"""
Benchmark spatial query performance with and without optimizations.
Tests various query types and measures latency under different conditions.
"""

import argparse
import json
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'spatial-engine'))

from src.utils.db_connection import get_db_connection


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    iterations: int
    latencies_ms: List[float]
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    throughput_qps: float


class SpatialBenchmarkSuite:
    """Benchmarks for spatial query performance."""

    def __init__(self, slide_id: str):
        self.slide_id = slide_id
        self.results: List[BenchmarkResult] = []

    def _execute_timed_query(self, query: str, params: tuple) -> float:
        """Execute a query and return execution time in milliseconds."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                start = time.perf_counter()
                cur.execute(query, params)
                cur.fetchall()
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed

    def _run_benchmark(
        self,
        name: str,
        query: str,
        params: tuple,
        iterations: int = 100
    ) -> BenchmarkResult:
        """Run a benchmark with multiple iterations."""
        print(f"  Running {name}...", end=" ", flush=True)

        latencies = []
        for _ in range(iterations):
            latency = self._execute_timed_query(query, params)
            latencies.append(latency)

        latencies.sort()
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        std = statistics.stdev(latencies) if len(latencies) > 1 else 0

        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            latencies_ms=latencies,
            mean_ms=mean,
            median_ms=median,
            std_ms=std,
            min_ms=min(latencies),
            max_ms=max(latencies),
            p95_ms=latencies[int(len(latencies) * 0.95)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            throughput_qps=1000 / mean if mean > 0 else 0
        )

        print(f"mean={mean:.2f}ms, p95={result.p95_ms:.2f}ms")
        self.results.append(result)
        return result

    def benchmark_bbox_query(self, iterations: int = 100):
        """Benchmark bounding box queries at various viewport sizes."""
        print("\nBenchmarking BBox Queries:")

        # Small viewport (10K x 10K)
        self._run_benchmark(
            "bbox_small",
            """
            SELECT * FROM query_spatial_objects_bbox(
                %s, 0, 0, 10000, 10000, NULL, 0.0, 1000, 0
            )
            """,
            (self.slide_id,),
            iterations
        )

        # Medium viewport (30K x 30K)
        self._run_benchmark(
            "bbox_medium",
            """
            SELECT * FROM query_spatial_objects_bbox(
                %s, 0, 0, 30000, 30000, NULL, 0.0, 1000, 0
            )
            """,
            (self.slide_id,),
            iterations
        )

        # Large viewport (full slide approximation)
        self._run_benchmark(
            "bbox_large",
            """
            SELECT * FROM query_spatial_objects_bbox(
                %s, 0, 0, 100000, 80000, NULL, 0.0, 1000, 0
            )
            """,
            (self.slide_id,),
            iterations
        )

        # With type filter
        self._run_benchmark(
            "bbox_filtered",
            """
            SELECT * FROM query_spatial_objects_bbox(
                %s, 0, 0, 30000, 30000, 'cell', 0.8, 1000, 0
            )
            """,
            (self.slide_id,),
            iterations
        )

    def benchmark_knn_query(self, iterations: int = 100):
        """Benchmark KNN queries with varying K values."""
        print("\nBenchmarking KNN Queries:")

        for k in [1, 5, 10, 50, 100]:
            self._run_benchmark(
                f"knn_k{k}",
                """
                SELECT * FROM query_knn(%s, 50000, 40000, %s, NULL)
                """,
                (self.slide_id, k),
                iterations
            )

    def benchmark_density_grid(self, iterations: int = 50):
        """Benchmark density grid computation."""
        print("\nBenchmarking Density Grid:")

        for grid_size in [256, 512, 1024]:
            self._run_benchmark(
                f"density_grid_{grid_size}",
                """
                SELECT * FROM compute_density_grid(%s, %s, NULL)
                """,
                (self.slide_id, float(grid_size)),
                iterations
            )

    def benchmark_statistics(self, iterations: int = 50):
        """Benchmark statistics computation."""
        print("\nBenchmarking Statistics:")

        self._run_benchmark(
            "slide_statistics",
            """
            SELECT * FROM compute_slide_statistics(%s)
            """,
            (self.slide_id,),
            iterations
        )

    def benchmark_concurrent_queries(
        self,
        n_concurrent: int = 50,
        n_queries: int = 200
    ):
        """Benchmark concurrent query performance."""
        print(f"\nBenchmarking Concurrent Queries ({n_concurrent} threads, {n_queries} queries):")

        query = """
        SELECT * FROM query_spatial_objects_bbox(
            %s, %s, %s, %s, %s, NULL, 0.0, 100, 0
        )
        """

        import random
        rng = random.Random(42)

        def run_query():
            x = rng.uniform(0, 90000)
            y = rng.uniform(0, 70000)
            return self._execute_timed_query(
                query,
                (self.slide_id, x, y, x + 10000, y + 10000)
            )

        latencies = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
            futures = [executor.submit(run_query) for _ in range(n_queries)]
            for future in as_completed(futures):
                latencies.append(future.result())

        total_time = time.perf_counter() - start_time

        latencies.sort()
        result = BenchmarkResult(
            name="concurrent_bbox",
            iterations=n_queries,
            latencies_ms=latencies,
            mean_ms=statistics.mean(latencies),
            median_ms=statistics.median(latencies),
            std_ms=statistics.stdev(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            p95_ms=latencies[int(len(latencies) * 0.95)],
            p99_ms=latencies[int(len(latencies) * 0.99)],
            throughput_qps=n_queries / total_time
        )

        print(f"  mean={result.mean_ms:.2f}ms, p95={result.p95_ms:.2f}ms, throughput={result.throughput_qps:.1f} qps")
        self.results.append(result)

    def benchmark_index_comparison(self, iterations: int = 20):
        """Compare query performance with and without indexes (requires superuser)."""
        print("\nIndex comparison skipped (requires index drop privileges)")
        # In production, this would temporarily disable indexes and compare

    def run_all(self, iterations: int = 100):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("SpatialPathDB Benchmark Suite")
        print("=" * 60)
        print(f"Slide ID: {self.slide_id}")
        print(f"Iterations per test: {iterations}")
        print("=" * 60)

        self.benchmark_bbox_query(iterations)
        self.benchmark_knn_query(iterations)
        self.benchmark_density_grid(iterations // 2)
        self.benchmark_statistics(iterations // 2)
        self.benchmark_concurrent_queries(50, 200)

        return self.results

    def generate_report(self) -> Dict:
        """Generate benchmark report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "slide_id": self.slide_id,
            "results": []
        }

        for result in self.results:
            r = asdict(result)
            # Remove raw latencies for summary
            del r["latencies_ms"]
            report["results"].append(r)

        # Summary statistics
        bbox_results = [r for r in self.results if r.name.startswith("bbox")]
        if bbox_results:
            report["summary"] = {
                "bbox_avg_latency_ms": statistics.mean([r.mean_ms for r in bbox_results]),
                "bbox_p95_latency_ms": statistics.mean([r.p95_ms for r in bbox_results]),
            }

        return report

    def save_report(self, output_path: str):
        """Save report to JSON file."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")


def get_sample_slide_id():
    """Get a slide ID from the database for benchmarking."""
    with get_db_connection(dict_cursor=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM slides LIMIT 1")
            result = cur.fetchone()
            if result:
                return str(result['id'])
    return None


def main():
    parser = argparse.ArgumentParser(description='Run spatial query benchmarks')
    parser.add_argument('--slide-id', help='Slide ID to benchmark')
    parser.add_argument('--iterations', type=int, default=100, help='Iterations per test')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file')
    args = parser.parse_args()

    slide_id = args.slide_id or get_sample_slide_id()
    if not slide_id:
        print("Error: No slides found in database. Run synthetic data generator first.")
        sys.exit(1)

    suite = SpatialBenchmarkSuite(slide_id)
    suite.run_all(args.iterations)
    suite.save_report(args.output)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    report = suite.generate_report()
    for result in report["results"]:
        print(f"{result['name']:25} mean={result['mean_ms']:8.2f}ms  p95={result['p95_ms']:8.2f}ms")


if __name__ == "__main__":
    main()
