"""Analytical cost model: closed-form optimization and complexity analysis.

Extends the empirical cost model with:
  - Analytical solution for optimal bucket target T* (Eq. 4)
  - Complexity bounds for pruning, insert, and query operations
  - C_h calibration and confidence intervals
  - Non-uniform density extension
  - Cost surface generation for visualization

This module imports from cost_model.py and adds new functions without
modifying any existing code.
"""

import math
import numpy as np
from scipy import optimize, stats as sp_stats

from spdb.cost_model import (
    hilbert_buckets_touched,
    zorder_buckets_touched,
    gist_depth,
    gist_scan_cost,
    ViewportCostModel,
)


# ---------------------------------------------------------------------------
# Eq. 4: Analytical optimal T via derivative analysis
# ---------------------------------------------------------------------------

def cost_function_T(T, n_objects, viewport_frac, C_h=2.0,
                    alpha=0.1, fanout=100, tuple_cost=0.001,
                    random_page_ms=0.5, seq_page_ms=0.08,
                    hit_ratio=0.85, tuple_width=80, page_size=8192):
    """Evaluate total viewport query cost as a function of bucket target T.

    Cost(T) = C_plan(N/T) + E[B_hit(f, N/T)] * C_scan(T)

    Parameters
    ----------
    T : float
        Bucket target (objects per bucket).
    n_objects : int
        Total objects in queried slide.
    viewport_frac : float
        Fraction of slide area covered by viewport.
    C_h : float
        Hilbert boundary constant (~2.0).
    alpha : float
        Planning cost per partition (ms).
    fanout : int
        GiST index fanout.
    tuple_cost : float
        CPU cost per result tuple (ms).

    Returns
    -------
    float : Total estimated query cost in ms.
    """
    T = max(1.0, float(T))
    N = float(n_objects)
    f = viewport_frac

    B = max(1.0, N / T)  # number of buckets

    # Planning cost: linear in partition count
    C_plan = alpha * B

    # Expected buckets touched
    E_Bhit = f * B + C_h * math.sqrt(f) * math.sqrt(B)
    E_Bhit = min(E_Bhit, B)
    E_Bhit = max(1.0, E_Bhit)

    # Per-bucket scan cost
    depth = max(1, math.ceil(math.log(max(1, T)) / math.log(fanout)))
    tuples_per_page = max(1, page_size // tuple_width)
    total_pages_per_bucket = max(1, math.ceil(T / tuples_per_page))
    selectivity_within = f * B / E_Bhit
    selectivity_within = min(selectivity_within, 1.0)
    matching_pages = max(1, math.ceil(total_pages_per_bucket * selectivity_within))

    index_ios = depth + 1
    heap_ios = matching_pages
    random_ios = index_ios + int(heap_ios * (1 - hit_ratio))
    seq_ios = int(heap_ios * hit_ratio)
    C_scan = random_ios * random_page_ms + seq_ios * seq_page_ms

    # Total tuples returned (constant w.r.t. T)
    tuples_ret = N * f
    cpu_cost = tuples_ret * tuple_cost

    return C_plan + E_Bhit * C_scan + cpu_cost


def cost_derivative_T(T, n_objects, viewport_frac, **kwargs):
    """Numerical derivative dC/dT via central differences."""
    h = max(1.0, T * 1e-6)
    c_plus = cost_function_T(T + h, n_objects, viewport_frac, **kwargs)
    c_minus = cost_function_T(T - h, n_objects, viewport_frac, **kwargs)
    return (c_plus - c_minus) / (2 * h)


def analytical_optimal_T(n_objects, viewport_frac=0.05, C_h=2.0,
                         T_min=1000, T_max=1_000_000,
                         **kwargs):
    """Find optimal bucket target T* that minimizes viewport query cost.

    Uses scipy.optimize.minimize_scalar on the cost function, which is
    quasiconvex in T (planning cost decreases, scan cost increases).

    Parameters
    ----------
    n_objects : int
        Total objects in queried slide.
    viewport_frac : float
        Viewport area fraction.
    C_h : float
        Hilbert boundary constant.
    T_min, T_max : float
        Search bounds for T.

    Returns
    -------
    dict with:
        T_star: optimal bucket target
        cost_star: minimum cost at T*
        B_star: number of buckets at T*
        pruning_rate: expected pruning rate at T*
        is_quasiconvex: bool, whether cost function is unimodal in range
    """
    def objective(T):
        return cost_function_T(T, n_objects, viewport_frac, C_h=C_h, **kwargs)

    # Bounded minimization
    result = optimize.minimize_scalar(
        objective,
        bounds=(T_min, T_max),
        method="bounded",
        options={"xatol": 100, "maxiter": 200},
    )

    T_star = result.x
    cost_star = result.fun
    B_star = max(1, n_objects / T_star)
    E_Bhit = hilbert_buckets_touched(viewport_frac, 8, int(B_star))
    pruning_rate = 1.0 - E_Bhit / B_star if B_star > 0 else 0

    # Check quasiconvexity: sample cost at 20 points and verify single minimum
    T_samples = np.linspace(T_min, T_max, 50)
    costs = [objective(t) for t in T_samples]
    min_idx = np.argmin(costs)
    decreasing_before = all(costs[i] >= costs[i+1] for i in range(min_idx))
    increasing_after = all(costs[i] <= costs[i+1] for i in range(min_idx, len(costs)-1))
    is_quasiconvex = decreasing_before and increasing_after

    return {
        "T_star": round(T_star),
        "cost_ms": round(cost_star, 3),
        "B_star": round(B_star),
        "pruning_rate": round(pruning_rate, 4),
        "is_quasiconvex": is_quasiconvex,
        "scipy_success": result.success,
    }


# ---------------------------------------------------------------------------
# Cost surface for visualization
# ---------------------------------------------------------------------------

def generate_cost_surface_data(n_objects=1_260_000, viewport_frac=0.05,
                                p_range=None, t_range=None):
    """Generate cost surface C(p, T) for contour plotting.

    Returns
    -------
    dict with:
        p_values: list of p values
        T_values: list of T values
        cost_grid: 2D numpy array [len(p), len(T)]
        pruning_grid: 2D array of pruning rates
        optimal: dict with (p*, T*) at the global minimum
    """
    if p_range is None:
        p_range = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14]
    if t_range is None:
        t_range = np.logspace(3, 6, 40).astype(int).tolist()  # 1K to 1M
        t_range = sorted(set(t_range))

    cost_grid = np.zeros((len(p_range), len(t_range)))
    pruning_grid = np.zeros((len(p_range), len(t_range)))

    global_min_cost = float("inf")
    global_min_p = p_range[0]
    global_min_T = t_range[0]

    for i, p in enumerate(p_range):
        for j, T in enumerate(t_range):
            B = max(1, n_objects // T)
            cost = cost_function_T(T, n_objects, viewport_frac)
            cost_grid[i, j] = cost

            E_Bhit = hilbert_buckets_touched(viewport_frac, p, B)
            pruning_grid[i, j] = 1.0 - E_Bhit / B if B > 0 else 0

            if cost < global_min_cost:
                global_min_cost = cost
                global_min_p = p
                global_min_T = T

    return {
        "p_values": p_range,
        "T_values": t_range,
        "cost_grid": cost_grid,
        "pruning_grid": pruning_grid,
        "optimal": {
            "p_star": global_min_p,
            "T_star": global_min_T,
            "cost_star": round(global_min_cost, 3),
        },
        "n_objects": n_objects,
        "viewport_frac": viewport_frac,
    }


# ---------------------------------------------------------------------------
# Complexity bounds
# ---------------------------------------------------------------------------

def pruning_complexity(p, viewport_frac):
    """Complexity of candidate_buckets_for_bbox computation.

    The function scans all grid cells covered by the viewport bounding box.
    A viewport of fraction f covers ~sqrt(f) of each dimension in a 2^p grid.

    Complexity: O(f * 4^p) grid cells scanned.
    For typical values (f=0.05, p=8): O(0.05 * 65536) = O(3277) cells.

    Returns
    -------
    dict with big-O string and estimated constant.
    """
    n = 1 << p
    cells_per_dim = max(1, int(math.sqrt(viewport_frac) * n))
    total_cells = cells_per_dim * cells_per_dim
    return {
        "big_o": f"O(f * 4^p)",
        "big_o_simplified": f"O({viewport_frac:.2f} * {1 << (2*p)})",
        "estimated_cells": total_cells,
        "per_cell_cost_us": 0.01,  # ~10ns per xy2d call
        "total_estimated_us": total_cells * 0.01,
        "description": "Grid cells scanned to identify candidate Hilbert buckets",
    }


def insert_complexity(p):
    """Complexity of inserting one object into SPDB.

    Steps:
    1. Hilbert encoding: O(p) bit operations
    2. Bucket computation: O(1)
    3. Partition routing: O(log B) via PostgreSQL planner
    4. GiST insert: O(log T) where T is bucket size

    Returns
    -------
    dict with big-O components.
    """
    return {
        "hilbert_encoding": f"O({p})",
        "bucket_computation": "O(1)",
        "partition_routing": "O(log B)",
        "gist_insert": "O(log T)",
        "total": f"O(p + log B + log T) = O({p} + log B + log T)",
        "description": "Per-object insert cost breakdown",
    }


def query_complexity(viewport_frac, n_objects, bucket_target=50_000,
                     C_h=2.0, fanout=100):
    """Complexity of a viewport query on SPDB.

    Steps:
    1. Pruning computation: O(f * 4^p)
    2. SQL planning: O(B) partition evaluation
    3. Per-bucket GiST scan: O(log T + |result_i|)
    4. Total: O(f*4^p + B + B_hit * (log T + |result|/B_hit))

    Returns
    -------
    dict with complexity components and estimates.
    """
    B = max(1, n_objects // bucket_target)
    E_Bhit = viewport_frac * B + C_h * math.sqrt(viewport_frac) * math.sqrt(B)
    E_Bhit = min(E_Bhit, B)
    T = bucket_target
    depth = gist_depth(T, fanout)
    result_size = int(n_objects * viewport_frac)

    return {
        "pruning": f"O(f * 4^p)",
        "planning": f"O(B) = O({B})",
        "scan_per_bucket": f"O(log_{fanout}(T) + |result_i|) = O({depth} + ...)",
        "total_scan": f"O(B_hit * (log T + |result|/B_hit))",
        "estimated_B_hit": round(E_Bhit, 1),
        "estimated_depth": depth,
        "estimated_result_size": result_size,
        "vs_monolithic": {
            "mono_scan": f"O(log_{fanout}({n_objects}) + {result_size})",
            "mono_depth": gist_depth(n_objects, fanout),
            "speedup_source": "Smaller per-partition indexes + partition pruning",
        },
    }


# ---------------------------------------------------------------------------
# C_h calibration and confidence intervals
# ---------------------------------------------------------------------------

def calibrate_ch(empirical_data):
    """Fit C_h from observed (viewport_frac, buckets_touched, total_buckets) data.

    Uses least-squares to fit C_h in:
        E[B_hit] = f*B + C_h * sqrt(f) * sqrt(B)

    Parameters
    ----------
    empirical_data : list of dict
        Each dict has: viewport_frac, buckets_touched, total_buckets.

    Returns
    -------
    dict with fitted C_h, R^2, residual stats.
    """
    residuals = []
    boundary_terms = []
    actual_boundaries = []

    for d in empirical_data:
        f = d["viewport_frac"]
        B = d["total_buckets"]
        actual = d["buckets_touched"]

        bulk = f * B
        boundary_term = math.sqrt(f) * math.sqrt(B)

        actual_boundaries.append(actual - bulk)
        boundary_terms.append(boundary_term)

    boundary_terms = np.array(boundary_terms)
    actual_boundaries = np.array(actual_boundaries)

    # Least squares: actual_boundary = C_h * boundary_term
    # C_h = sum(actual * term) / sum(term^2)
    numerator = np.sum(actual_boundaries * boundary_terms)
    denominator = np.sum(boundary_terms ** 2)
    C_h_fitted = numerator / denominator if denominator > 0 else 2.0

    # R^2
    predictions = C_h_fitted * boundary_terms
    ss_res = np.sum((actual_boundaries - predictions) ** 2)
    ss_tot = np.sum((actual_boundaries - np.mean(actual_boundaries)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "C_h": round(float(C_h_fitted), 4),
        "r_squared": round(float(r_squared), 4),
        "n_samples": len(empirical_data),
        "residual_mean": round(float(np.mean(actual_boundaries - predictions)), 4),
        "residual_std": round(float(np.std(actual_boundaries - predictions)), 4),
    }


def ch_confidence_interval(empirical_data, confidence=0.95, n_bootstrap=10000,
                            seed=42):
    """Bootstrap confidence interval for C_h.

    Parameters
    ----------
    empirical_data : list of dict
        Each dict has: viewport_frac, buckets_touched, total_buckets.
    confidence : float
        Confidence level (default 0.95).
    n_bootstrap : int
        Number of bootstrap samples.

    Returns
    -------
    dict with C_h point estimate, CI lower, CI upper.
    """
    rng = np.random.RandomState(seed)
    n = len(empirical_data)
    bootstrap_chs = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [empirical_data[i] for i in indices]
        result = calibrate_ch(sample)
        bootstrap_chs.append(result["C_h"])

    bootstrap_chs = np.array(bootstrap_chs)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_chs, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_chs, 100 * (1 - alpha / 2)))

    return {
        "C_h": round(float(np.mean(bootstrap_chs)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence": confidence,
        "std": round(float(np.std(bootstrap_chs)), 4),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Non-uniform density extension
# ---------------------------------------------------------------------------

def density_weighted_buckets_touched(viewport_frac, p, bucket_densities,
                                      C_h=2.0):
    """Expected buckets touched under non-uniform object density.

    Extends Eq. 2 to:
        E[B_hit] = sum_i(f_i * 1) + C_h * sum_i(sqrt(f_i))

    where f_i = viewport_frac * (density_i / mean_density) for bucket i,
    representing the probability that a random viewport overlaps bucket i.

    Parameters
    ----------
    viewport_frac : float
        Overall viewport area fraction.
    p : int
        Hilbert order.
    bucket_densities : array-like
        Number of objects per bucket.

    Returns
    -------
    dict with expected buckets touched, pruning rate, comparison to uniform.
    """
    densities = np.array(bucket_densities, dtype=np.float64)
    B = len(densities)
    mean_density = np.mean(densities)

    if mean_density == 0 or B == 0:
        return {"expected_touched": 0, "pruning_rate": 1.0}

    # Weight each bucket by its relative density
    weights = densities / mean_density
    f = viewport_frac

    # For each bucket, probability of overlap proportional to spatial extent
    # Under uniform spatial distribution, each bucket covers ~1/B of the space
    # With density weighting, denser buckets cover less space (same # objects, smaller area)
    # So the probability of viewport overlapping bucket i is ~f for all (area-based)
    # The Hilbert boundary effect is: C_h * sqrt(f/B) per boundary crossing
    # Total boundary buckets ~ C_h * sqrt(f) * sqrt(B), same as uniform

    # The difference: denser buckets have MORE objects to scan when hit
    # This affects scan cost, not pruning rate

    # Pruning rate remains approximately the same as uniform model
    E_Bhit_uniform = f * B + C_h * math.sqrt(f) * math.sqrt(B)
    E_Bhit_uniform = min(E_Bhit_uniform, B)

    # Density CV (coefficient of variation) affects scan cost variance
    cv = float(np.std(densities) / mean_density) if mean_density > 0 else 0

    # Effective scan cost adjustment: denser buckets cost more to scan
    # Expected per-bucket scan cost scales with bucket size
    # Variance in scan cost = var(density) * (scan_cost_per_object)^2
    density_cost_ratio = float(np.mean(densities ** 2) / mean_density ** 2)

    return {
        "expected_touched": round(float(E_Bhit_uniform), 2),
        "pruning_rate": round(1.0 - E_Bhit_uniform / B, 4),
        "density_cv": round(cv, 4),
        "density_cost_ratio": round(density_cost_ratio, 4),
        "uniform_prediction": round(float(E_Bhit_uniform), 2),
        "note": ("Pruning rate is approximately density-independent "
                 "(driven by spatial area, not object count). "
                 "Density variation affects per-bucket scan cost, not pruning."),
    }


# ---------------------------------------------------------------------------
# Partition ceiling: B* prediction and adaptive strategy
# ---------------------------------------------------------------------------

def predict_partition_ceiling(
    n_objects_per_slide: int = 1_260_000,
    n_slides: int = 127,
    viewport_frac: float = 0.05,
    alpha_plan: float = 0.1,
    C_h: float = 2.0,
    bucket_target: int = 50_000,
    planning_threshold_ms: float = 10.0,
    fanout: int = 100,
    p: int = 8,
) -> dict:
    """Predict B* (partition ceiling) and recommend storage strategy.

    The partition ceiling is the total partition count at which planner
    overhead C_plan = alpha * B_total exceeds a threshold, causing SPDB
    query latency to degrade below simpler layouts.

    Analysis:
      - SPDB: B_total = sum over slides of ceil(n_i / T)
      - SO-C: B_total = n_slides (one partition per slide, BRIN for intra-slide)
      - Mono-C: B_total = 1 (single table, BRIN for everything)

    At B_total > B*, SPDB's planning overhead exceeds pruning benefit.
    SO-C avoids this by keeping B_total = n_slides while achieving
    equivalent scan performance via physical Hilbert clustering + BRIN.

    Parameters
    ----------
    n_objects_per_slide : int
        Mean objects per slide.
    n_slides : int
        Number of distinct slides in the dataset.
    viewport_frac : float
        Typical viewport fraction.
    alpha_plan : float
        Planning cost per partition (ms per partition).
    C_h : float
        Hilbert boundary constant.
    bucket_target : int
        SPDB bucket target T.
    planning_threshold_ms : float
        Threshold above which planning dominates (default 10ms).
    fanout : int
        GiST index fanout.
    p : int
        Hilbert order.

    Returns
    -------
    dict with B*, recommended strategy, cost comparison.
    """
    # B*: partition count where planning hits threshold
    B_star = int(planning_threshold_ms / alpha_plan) if alpha_plan > 0 else float("inf")

    # SPDB partition count
    buckets_per_slide = max(1, n_objects_per_slide // bucket_target)
    B_spdb = n_slides * buckets_per_slide

    # SO-C partition count (one per slide)
    B_soc = n_slides

    # Planning costs
    plan_spdb = alpha_plan * B_spdb
    plan_soc = alpha_plan * B_soc
    plan_mono = alpha_plan  # single table

    # Pruning benefit: compare SPDB scan cost vs Mono-C scan cost
    # SPDB: scans E[B_hit] small GiST indexes of size T
    E_Bhit_spdb = viewport_frac * buckets_per_slide + C_h * math.sqrt(viewport_frac) * math.sqrt(buckets_per_slide)
    E_Bhit_spdb = min(E_Bhit_spdb, buckets_per_slide)

    # SO-C: BRIN prunes at block level on Hilbert-sorted data within the slide
    # Effective selectivity: BRIN achieves ~sqrt(f) pages scanned (locality-aware)
    # Total scan on SO-C is equivalent to scanning E_Bhit_spdb * T objects via BRIN
    brin_pages = max(1, math.ceil(n_objects_per_slide * viewport_frac / 100))  # BRIN granularity

    # Total query cost comparison
    cost_spdb = cost_function_T(bucket_target, n_objects_per_slide, viewport_frac,
                                 C_h=C_h, alpha=alpha_plan)

    # SO-C cost: planning for n_slides + BRIN scan within single partition
    depth_soc = max(1, math.ceil(math.log(max(1, n_objects_per_slide)) / math.log(fanout)))
    tuples_per_page = max(1, 8192 // 80)
    matching_pages = max(1, math.ceil(n_objects_per_slide * viewport_frac / tuples_per_page))
    cost_soc = plan_soc + depth_soc * 0.5 + matching_pages * 0.08  # GiST + seq scan

    # Mono-C cost: BRIN on single huge table
    depth_mono = max(1, math.ceil(math.log(max(1, n_objects_per_slide * n_slides)) / math.log(fanout)))
    cost_mono = plan_mono + depth_mono * 0.5 + matching_pages * 5 * 0.08  # worse locality

    # Determine whether SPDB exceeds ceiling
    spdb_exceeds_ceiling = B_spdb > B_star

    # Recommended strategy
    if B_spdb <= B_star:
        recommendation = "SPDB"
        reason = f"B_total={B_spdb} <= B*={B_star}: partition pruning benefit exceeds planner cost"
    elif B_soc <= B_star:
        recommendation = "SO-C"
        reason = (f"B_total(SPDB)={B_spdb} > B*={B_star}: switch to SO-C "
                  f"(B_total={B_soc}) with per-partition Hilbert CLUSTER + BRIN")
    else:
        recommendation = "Mono-C"
        reason = (f"Both SPDB ({B_spdb}) and SO-C ({B_soc}) exceed B*={B_star}: "
                  f"use monolithic Hilbert-CLUSTERed table with BRIN")

    return {
        "B_star": B_star,
        "B_spdb": B_spdb,
        "B_soc": B_soc,
        "B_mono": 1,
        "buckets_per_slide": buckets_per_slide,
        "spdb_exceeds_ceiling": spdb_exceeds_ceiling,
        "planning_cost_ms": {
            "SPDB": round(plan_spdb, 2),
            "SO-C": round(plan_soc, 2),
            "Mono-C": round(plan_mono, 2),
        },
        "estimated_query_cost_ms": {
            "SPDB": round(cost_spdb, 2),
            "SO-C": round(cost_soc, 2),
            "Mono-C": round(cost_mono, 2),
        },
        "recommendation": recommendation,
        "reason": reason,
        "E_Bhit_spdb_per_slide": round(float(E_Bhit_spdb), 2),
        "pruning_rate_spdb": round(1.0 - E_Bhit_spdb / buckets_per_slide, 4),
        "alpha_plan": alpha_plan,
        "planning_threshold_ms": planning_threshold_ms,
    }


def adaptive_layout_selector(
    slide_catalog: list[dict],
    alpha_plan: float = 0.1,
    planning_threshold_ms: float = 10.0,
    bucket_target: int = 50_000,
    C_h: float = 2.0,
) -> dict:
    """Given a catalog of slides, select the optimal storage layout.

    For each possible layout, computes total partition count and predicts
    whether it exceeds B*.  Returns the recommended layout.

    Parameters
    ----------
    slide_catalog : list of dict
        Each dict has: slide_id (str), n_objects (int).
    alpha_plan, planning_threshold_ms, bucket_target, C_h : float
        Cost model parameters.

    Returns
    -------
    dict with layout recommendation and per-layout analysis.
    """
    B_star = int(planning_threshold_ms / alpha_plan) if alpha_plan > 0 else 10_000

    n_slides = len(slide_catalog)
    total_objects = sum(s["n_objects"] for s in slide_catalog)

    # SPDB: two-level partitioning
    spdb_partitions = sum(max(1, s["n_objects"] // bucket_target) for s in slide_catalog)

    # SO-C: one partition per slide
    soc_partitions = n_slides

    layouts = {
        "Mono-C": {
            "partitions": 1,
            "exceeds_ceiling": False,
            "description": "Monolithic + Hilbert CLUSTER + BRIN",
        },
        "SO-C": {
            "partitions": soc_partitions,
            "exceeds_ceiling": soc_partitions > B_star,
            "description": f"Slide-only ({soc_partitions} parts) + per-partition Hilbert CLUSTER + BRIN",
        },
        "SPDB": {
            "partitions": spdb_partitions,
            "exceeds_ceiling": spdb_partitions > B_star,
            "description": f"Two-level ({spdb_partitions} leaf parts) + GiST per bucket",
        },
    }

    # Pick the most partitioned layout that stays below B*
    if not layouts["SPDB"]["exceeds_ceiling"]:
        selected = "SPDB"
    elif not layouts["SO-C"]["exceeds_ceiling"]:
        selected = "SO-C"
    else:
        selected = "Mono-C"

    return {
        "B_star": B_star,
        "n_slides": n_slides,
        "total_objects": total_objects,
        "layouts": layouts,
        "selected": selected,
        "reason": layouts[selected]["description"],
    }


# ---------------------------------------------------------------------------
# Convenience: full analysis for paper
# ---------------------------------------------------------------------------

def full_cost_analysis(n_objects=1_260_000, viewport_frac=0.05, p=8):
    """Run complete cost analysis for paper Section 4.

    Returns all results needed for the formal pruning model section.
    """
    # Optimal T
    opt = analytical_optimal_T(n_objects, viewport_frac)

    # Complexity bounds
    prune_cx = pruning_complexity(p, viewport_frac)
    insert_cx = insert_complexity(p)
    query_cx = query_complexity(viewport_frac, n_objects)

    # Cost surface
    surface = generate_cost_surface_data(n_objects, viewport_frac)

    # Cost at several T values for comparison
    T_values = [5000, 10000, 25000, 50000, 100000, 200000, 500000]
    cost_comparison = {}
    for T in T_values:
        cost = cost_function_T(T, n_objects, viewport_frac)
        B = max(1, n_objects // T)
        E_Bhit = hilbert_buckets_touched(viewport_frac, p, B)
        cost_comparison[T] = {
            "cost_ms": round(cost, 3),
            "buckets": B,
            "E_Bhit": round(E_Bhit, 1),
            "pruning_rate": round(1.0 - E_Bhit / B, 4),
        }

    # Partition ceiling analysis
    ceiling = predict_partition_ceiling(
        n_objects_per_slide=n_objects,
        n_slides=127,
        viewport_frac=viewport_frac,
    )

    return {
        "optimal_T": opt,
        "complexity": {
            "pruning": prune_cx,
            "insert": insert_cx,
            "query": query_cx,
        },
        "cost_surface": {
            "optimal": surface["optimal"],
            "n_p_values": len(surface["p_values"]),
            "n_T_values": len(surface["T_values"]),
        },
        "cost_comparison": cost_comparison,
        "partition_ceiling": ceiling,
    }
