"""Planner overhead benchmark: measure PostgreSQL planner time vs. partition count.

Creates a RANGE-partitioned table with B partitions, inserts synthetic rows
with geometry, builds GiST indexes, then runs EXPLAIN (ANALYZE, FORMAT JSON)
to extract Planning Time.  Sweeps across partition counts from 10 to 5000.

Usage:
    python -m benchmarks.planner_overhead
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import psycopg2
from scipy.optimize import curve_fit

from spdb import config
from spdb.config import RAW_DIR
from benchmarks.framework import compute_stats, save_results

# ---------------------------------------------------------------------------
# Partition counts to sweep
# ---------------------------------------------------------------------------

PARTITION_COUNTS = [
    10, 25, 50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000,
]

PLANNING_THRESHOLD_MS = 10.0   # B* threshold: planning dominates execution


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

def _create_partitioned_table(conn, num_partitions: int) -> None:
    """DROP and CREATE a RANGE-partitioned table with *num_partitions* children."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS planner_test CASCADE")

        cur.execute("""
            CREATE TABLE planner_test (
                part_key  INTEGER NOT NULL,
                geom      geometry(Point, 0)
            ) PARTITION BY RANGE (part_key)
        """)

        for i in range(num_partitions):
            lo = i * 1000
            hi = (i + 1) * 1000
            cur.execute(
                f"CREATE TABLE planner_test_p{i} "
                f"PARTITION OF planner_test "
                f"FOR VALUES FROM ({lo}) TO ({hi})"
            )


def _insert_rows(conn, num_partitions: int, rows_per_partition: int = 100) -> None:
    """INSERT synthetic rows: 100 per partition with random geometry points."""
    with conn.cursor() as cur:
        for i in range(num_partitions):
            lo = i * 1000
            hi = (i + 1) * 1000
            cur.execute(
                """
                INSERT INTO planner_test (part_key, geom)
                SELECT
                    floor(random() * (%(hi)s - %(lo)s) + %(lo)s)::int,
                    ST_MakePoint(random() * 1000, random() * 1000)
                FROM generate_series(1, %(n)s)
                """,
                {"lo": lo, "hi": hi, "n": rows_per_partition},
            )


def _create_gist_indexes(conn, num_partitions: int) -> None:
    """CREATE a GiST index on geom for each child partition."""
    with conn.cursor() as cur:
        for i in range(num_partitions):
            cur.execute(
                f"CREATE INDEX planner_test_p{i}_geom_idx "
                f"ON planner_test_p{i} USING gist (geom)"
            )


def _analyze_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("ANALYZE planner_test")


def _drop_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS planner_test CASCADE")


# ---------------------------------------------------------------------------
# Measure planning time
# ---------------------------------------------------------------------------

EXPLAIN_SQL = """
EXPLAIN (ANALYZE, FORMAT JSON)
SELECT * FROM planner_test
WHERE part_key = 500
  AND geom && ST_MakeEnvelope(0, 0, 100, 100, 0)
"""


def _collect_planning_times(conn, n_trials: int) -> list[float]:
    """Run EXPLAIN ANALYZE *n_trials* times and return Planning Time (ms) list."""
    times: list[float] = []
    with conn.cursor() as cur:
        for _ in range(n_trials):
            cur.execute(EXPLAIN_SQL)
            result = cur.fetchall()
            plan_json = result[0][0]
            planning_ms = plan_json[0]["Planning Time"]
            times.append(float(planning_ms))
    return times


# ---------------------------------------------------------------------------
# Curve-fitting models
# ---------------------------------------------------------------------------

def _linear(B, alpha, beta):
    return alpha * B + beta


def _quadratic(B, alpha, beta, gamma):
    return alpha * B + beta * B**2 + gamma


def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_planner_overhead(n_trials: int = 200) -> dict:
    """Sweep partition counts and measure planner overhead.

    Returns the full results dict (also saved to JSON).
    """
    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True

    results: dict = {
        "partition_counts": PARTITION_COUNTS,
        "results": {},
    }

    for B in PARTITION_COUNTS:
        print(f"\n{'='*60}")
        print(f"  Partitions: {B}")
        print(f"{'='*60}")

        t0 = time.perf_counter()

        # -- setup --
        print(f"  Creating {B}-partition table ...", flush=True)
        _create_partitioned_table(conn, B)

        print(f"  Inserting {B * 100:,} rows ...", flush=True)
        _insert_rows(conn, B)

        print(f"  Building GiST indexes on {B} partitions ...", flush=True)
        _create_gist_indexes(conn, B)

        print("  ANALYZE ...", flush=True)
        _analyze_table(conn)

        setup_s = time.perf_counter() - t0
        print(f"  Setup done in {setup_s:.1f}s")

        # -- measure --
        print(f"  Running {n_trials} EXPLAIN ANALYZE trials ...", flush=True)
        plan_times = _collect_planning_times(conn, n_trials)

        stats = compute_stats(plan_times)
        results["results"][B] = {
            "plan_time_p50": stats["p50"],
            "plan_time_p95": stats["p95"],
            "plan_time_mean": stats["mean"],
            "plan_times": plan_times,
        }

        print(f"  Planning time  p50={stats['p50']:.3f} ms  "
              f"p95={stats['p95']:.3f} ms  mean={stats['mean']:.3f} ms")

        # -- cleanup --
        _drop_table(conn)

    conn.close()

    # -----------------------------------------------------------------------
    # Curve fitting
    # -----------------------------------------------------------------------
    Bs = np.array(PARTITION_COUNTS, dtype=np.float64)
    means = np.array(
        [results["results"][B]["plan_time_mean"] for B in PARTITION_COUNTS],
        dtype=np.float64,
    )

    print(f"\n{'='*60}")
    print("  Curve Fitting")
    print(f"{'='*60}")

    # Linear: plan_ms = alpha * B + beta
    try:
        popt_lin, _ = curve_fit(_linear, Bs, means)
        alpha_l, beta_l = popt_lin
        y_pred_lin = _linear(Bs, *popt_lin)
        r2_lin = _r_squared(means, y_pred_lin)
        print(f"  Linear:    plan_ms = {alpha_l:.6f} * B + {beta_l:.4f}  "
              f"(R² = {r2_lin:.4f})")
    except RuntimeError as e:
        print(f"  Linear fit failed: {e}")
        alpha_l, beta_l, r2_lin = None, None, None

    # Quadratic: plan_ms = alpha * B + beta * B^2 + gamma
    try:
        popt_quad, _ = curve_fit(_quadratic, Bs, means)
        alpha_q, beta_q, gamma_q = popt_quad
        y_pred_quad = _quadratic(Bs, *popt_quad)
        r2_quad = _r_squared(means, y_pred_quad)
        print(f"  Quadratic: plan_ms = {alpha_q:.6f} * B + "
              f"{beta_q:.10f} * B² + {gamma_q:.4f}  "
              f"(R² = {r2_quad:.4f})")
    except RuntimeError as e:
        print(f"  Quadratic fit failed: {e}")
        alpha_q, beta_q, gamma_q, r2_quad = None, None, None, None

    # Save model coefficients
    results["models"] = {
        "linear": {
            "alpha": alpha_l,
            "beta": beta_l,
            "r_squared": r2_lin,
        },
        "quadratic": {
            "alpha": alpha_q,
            "beta": beta_q,
            "gamma": gamma_q,
            "r_squared": r2_quad,
        },
    }

    # -----------------------------------------------------------------------
    # Compute B*: partition count where planning overhead exceeds threshold
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  B* (planning > {PLANNING_THRESHOLD_MS} ms)")
    print(f"{'='*60}")

    b_star = None
    for B in PARTITION_COUNTS:
        if results["results"][B]["plan_time_p50"] > PLANNING_THRESHOLD_MS:
            b_star = B
            break

    if b_star is not None:
        print(f"  B* = {b_star}  (first partition count where p50 > "
              f"{PLANNING_THRESHOLD_MS} ms)")
    else:
        print(f"  B* not reached: p50 stays below {PLANNING_THRESHOLD_MS} ms "
              f"even at {PARTITION_COUNTS[-1]} partitions")

        # Extrapolate from linear model if available
        if alpha_l is not None and alpha_l > 0:
            b_star_extrap = (PLANNING_THRESHOLD_MS - beta_l) / alpha_l
            print(f"  Linear extrapolation: B* ≈ {b_star_extrap:.0f}")

    results["b_star"] = b_star
    results["planning_threshold_ms"] = PLANNING_THRESHOLD_MS

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    path = save_results(results, "planner_overhead")
    print(f"\n  Results saved to {path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_planner_overhead(n_trials=200)
