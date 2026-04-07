"""Budget-Constrained Density-Aware Hilbert Partitioning (BDHP).

Given K slides with N_k objects each, a partition budget B*, and
per-partition executor cost alpha_exec, solve:

    min_{B_k}  sum_k  w_k * C_k(B_k)
    s.t.       sum_k  B_k  <=  B*
               B_k >= 1

Each C_k(B_k) is the workload-weighted query cost for slide k with B_k
Hilbert sub-partitions.  The cost is separable (slide k's cost depends
only on B_k) and quasiconvex in B_k, so the greedy marginal-benefit
allocation is optimal.

Algorithm: heap-based water-filling in O(B* log K).
"""

from __future__ import annotations

import heapq
import json
import math
import os
from typing import Any

from spdb import config

# ---------------------------------------------------------------------------
# Per-slide cost model (workload-weighted)
# ---------------------------------------------------------------------------

# Measured empirically: see benchmarks/planner_overhead.py and Q1 results.
# SPDB (3768 parts) p50=357ms vs SO (127 parts) p50=171ms.
# Overhead = (357-171) / (3768-127) = 0.051 ms/partition.
DEFAULT_ALPHA_EXEC = 0.05  # ms per partition executor startup


def _q1_viewport_cost(N: int, B_k: int, f: float, alpha_exec: float,
                      C_h: float, fanout: int = 100,
                      random_page_ms: float = 0.5,
                      seq_page_ms: float = 0.08,
                      hit_ratio: float = 0.85,
                      tuple_width: int = 80,
                      page_size: int = 8192) -> float:
    """Q1 viewport cost for a single slide with B_k partitions.

    Benefits from more partitions: Hilbert pruning reduces scanned buckets.
    """
    B_k = max(1, B_k)
    T = max(1.0, N / B_k)

    # Executor overhead (the measured bottleneck)
    exec_cost = alpha_exec * B_k

    # Expected buckets touched (Eq. 2)
    E_Bhit = f * B_k + C_h * math.sqrt(f) * math.sqrt(B_k)
    E_Bhit = max(1.0, min(E_Bhit, B_k))

    # Per-bucket GiST scan cost
    depth = max(1, math.ceil(math.log(max(2, T)) / math.log(fanout)))
    tuples_per_page = max(1, page_size // tuple_width)
    total_pages = max(1, math.ceil(T / tuples_per_page))
    selectivity = min(1.0, f * B_k / E_Bhit)
    matching_pages = max(1, math.ceil(total_pages * selectivity))

    random_ios = depth + 1 + int(matching_pages * (1 - hit_ratio))
    seq_ios = int(matching_pages * hit_ratio)
    C_scan = random_ios * random_page_ms + seq_ios * seq_page_ms

    # Result CPU cost (constant w.r.t. B_k)
    cpu = N * f * 0.001

    return exec_cost + E_Bhit * C_scan + cpu


def _q2_knn_cost(N: int, B_k: int, alpha_exec: float,
                 k: int = 50, fanout: int = 100,
                 random_page_ms: float = 0.5) -> float:
    """Q2 kNN cost: MUST scan ALL B_k partitions (no spatial pruning for kNN).

    More partitions = strictly worse.  This is the key trade-off against Q1.
    """
    B_k = max(1, B_k)
    T = max(1.0, N / B_k)

    # Each partition: GiST distance-ordered scan for k nearest
    depth = max(1, math.ceil(math.log(max(2, T)) / math.log(fanout)))
    per_part = depth * random_page_ms + k * 0.01  # index traversal + k tuple fetches

    # Merge: B_k sorted lists of k items
    merge = B_k * k * 0.0001 * math.log2(max(2, B_k))

    return alpha_exec * B_k + B_k * per_part + merge


def _q3_aggregation_cost(N: int, B_k: int, alpha_exec: float,
                         seq_page_ms: float = 0.08,
                         tuple_width: int = 80,
                         page_size: int = 8192) -> float:
    """Q3 aggregation: GROUP BY tile_id, class_label.

    Total scan work is N regardless of B_k.  Only executor startup scales.
    """
    tuples_per_page = max(1, page_size // tuple_width)
    total_pages = max(1, math.ceil(N / tuples_per_page))
    scan_cost = total_pages * seq_page_ms + N * 0.001  # seq scan + CPU

    return alpha_exec * B_k + scan_cost


def _q4_spatial_join_cost(N: int, B_k: int, f: float, alpha_exec: float,
                          C_h: float, fanout: int = 100,
                          random_page_ms: float = 0.5,
                          seq_page_ms: float = 0.08,
                          hit_ratio: float = 0.85) -> float:
    """Q4 spatial join: similar to Q1 but with class filter (smaller selectivity)."""
    # Spatial join uses a smaller effective viewport (2%) with class filter
    f_eff = f * 0.4  # viewport_frac=0.02, ~40% class selectivity
    return _q1_viewport_cost(N, B_k, f_eff, alpha_exec, C_h, fanout,
                             random_page_ms, seq_page_ms, hit_ratio)


def slide_cost(slide: dict, B_k: int,
               viewport_frac: float = 0.05,
               alpha_exec: float = DEFAULT_ALPHA_EXEC,
               C_h: float = 2.0,
               workload: dict[str, float] | None = None,
               k_nn: int = 50) -> float:
    """Workload-weighted total cost for querying slide with B_k partitions.

    Parameters
    ----------
    slide : dict
        Must have 'n_objects' (int).  Optional: 'density_cv' (float).
    B_k : int
        Number of Hilbert sub-partitions for this slide.
    workload : dict
        Query mix weights.  Default: config.WORKLOAD_MIX.

    Returns
    -------
    float : Expected query cost in ms.
    """
    if workload is None:
        workload = config.WORKLOAD_MIX

    N = slide["n_objects"]
    B_k = max(1, int(B_k))

    c_q1 = _q1_viewport_cost(N, B_k, viewport_frac, alpha_exec, C_h)
    c_q2 = _q2_knn_cost(N, B_k, alpha_exec, k=k_nn)
    c_q3 = _q3_aggregation_cost(N, B_k, alpha_exec)
    c_q4 = _q4_spatial_join_cost(N, B_k, viewport_frac, alpha_exec, C_h)

    return (workload["Q1"] * c_q1 + workload["Q2"] * c_q2 +
            workload["Q3"] * c_q3 + workload["Q4"] * c_q4)


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------

def optimal_allocation(
    slides: list[dict],
    B_star: int,
    viewport_frac: float = 0.05,
    alpha_exec: float = DEFAULT_ALPHA_EXEC,
    C_h: float = 2.0,
    workload: dict[str, float] | None = None,
) -> dict[str, int]:
    """Greedy water-filling via max-heap: O(B* log K).

    Start with B_k=1 for all slides (SO baseline), then iteratively
    allocate one partition to the slide with the largest marginal cost
    reduction, until the budget is spent.

    Optimality: C_k(B_k) is quasiconvex and separable across slides,
    so greedy marginal-benefit allocation is optimal for this class of
    separable convex resource-allocation problems.

    Parameters
    ----------
    slides : list of dict
        Each must have 'slide_id' (str) and 'n_objects' (int).
    B_star : int
        Total partition budget.

    Returns
    -------
    dict mapping slide_id -> B_k (allocated partition count).
    """
    K = len(slides)
    if B_star < K:
        raise ValueError(f"B*={B_star} < K={K}: budget insufficient")

    alloc = {s["slide_id"]: 1 for s in slides}
    remaining = B_star - K
    slide_map = {s["slide_id"]: s for s in slides}

    # Build max-heap: (-marginal_benefit, slide_id)
    heap: list[tuple[float, str]] = []
    for s in slides:
        sid = s["slide_id"]
        delta = (slide_cost(s, 1, viewport_frac, alpha_exec, C_h, workload) -
                 slide_cost(s, 2, viewport_frac, alpha_exec, C_h, workload))
        heapq.heappush(heap, (-delta, sid))

    while remaining > 0 and heap:
        neg_delta, sid = heapq.heappop(heap)
        delta = -neg_delta

        if delta <= 0:
            break  # no slide benefits from more partitions

        alloc[sid] += 1
        remaining -= 1

        # Push new marginal benefit for next increment
        s = slide_map[sid]
        new_B = alloc[sid]
        new_delta = (slide_cost(s, new_B, viewport_frac, alpha_exec, C_h, workload) -
                     slide_cost(s, new_B + 1, viewport_frac, alpha_exec, C_h, workload))
        heapq.heappush(heap, (-new_delta, sid))

    return alloc


def uniform_allocation(slides: list[dict], B_star: int) -> dict[str, int]:
    """Uniform: every slide gets floor(B*/K), remainder spread round-robin."""
    K = len(slides)
    base = max(1, B_star // K)
    alloc = {s["slide_id"]: base for s in slides}
    remainder = B_star - base * K
    for i, s in enumerate(slides):
        if i < remainder:
            alloc[s["slide_id"]] += 1
    return alloc


def proportional_allocation(slides: list[dict], B_star: int) -> dict[str, int]:
    """Size-proportional: B_k proportional to N_k, with floor adjustment."""
    total_objects = sum(s["n_objects"] for s in slides)
    K = len(slides)

    # Raw proportional allocation
    raw = {s["slide_id"]: max(1.0, s["n_objects"] / total_objects * B_star)
           for s in slides}

    # Round down, then distribute remainder to largest fractional parts
    alloc = {sid: max(1, int(v)) for sid, v in raw.items()}
    allocated = sum(alloc.values())
    remainder = B_star - allocated

    if remainder > 0:
        # Sort by fractional part descending
        fracs = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]),
                       reverse=True)
        for sid, _ in fracs:
            if remainder <= 0:
                break
            alloc[sid] += 1
            remainder -= 1

    return alloc


# ---------------------------------------------------------------------------
# Comparison and analysis
# ---------------------------------------------------------------------------

def compare_allocations(
    slides: list[dict],
    B_star: int,
    viewport_frac: float = 0.05,
    alpha_exec: float = DEFAULT_ALPHA_EXEC,
    C_h: float = 2.0,
    workload: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare uniform, proportional, and optimal allocations.

    Returns
    -------
    dict with per-strategy allocations, costs, and summary statistics.
    """
    strategies = {
        "SO_baseline": {s["slide_id"]: 1 for s in slides},
        "uniform": uniform_allocation(slides, B_star),
        "proportional": proportional_allocation(slides, B_star),
        "optimal": optimal_allocation(slides, B_star, viewport_frac,
                                      alpha_exec, C_h, workload),
    }

    results = {}
    for name, alloc in strategies.items():
        total_parts = sum(alloc.values())
        per_slide_costs = {}
        for s in slides:
            sid = s["slide_id"]
            b = alloc[sid]
            c = slide_cost(s, b, viewport_frac, alpha_exec, C_h, workload)
            per_slide_costs[sid] = {"B_k": b, "cost_ms": round(c, 3),
                                    "n_objects": s["n_objects"]}

        total_cost = sum(v["cost_ms"] for v in per_slide_costs.values())
        costs_list = [v["cost_ms"] for v in per_slide_costs.values()]
        alloc_list = [v["B_k"] for v in per_slide_costs.values()]

        results[name] = {
            "total_partitions": total_parts,
            "total_cost_ms": round(total_cost, 2),
            "mean_cost_ms": round(total_cost / len(slides), 3),
            "max_slide_cost_ms": round(max(costs_list), 3),
            "alloc_min": min(alloc_list),
            "alloc_max": max(alloc_list),
            "alloc_mean": round(sum(alloc_list) / len(alloc_list), 1),
            "alloc_std": round(
                (sum((b - sum(alloc_list) / len(alloc_list)) ** 2
                     for b in alloc_list) / len(alloc_list)) ** 0.5, 1),
            "per_slide": per_slide_costs,
        }

    # Compute speedups relative to SO baseline
    so_cost = results["SO_baseline"]["total_cost_ms"]
    for name in results:
        results[name]["speedup_vs_SO"] = round(
            so_cost / results[name]["total_cost_ms"], 3) if results[name]["total_cost_ms"] > 0 else 0

    return {
        "B_star": B_star,
        "K": len(slides),
        "alpha_exec": alpha_exec,
        "viewport_frac": viewport_frac,
        "workload": workload or config.WORKLOAD_MIX,
        "strategies": results,
    }


def allocation_summary(comparison: dict) -> str:
    """Human-readable summary of compare_allocations output."""
    lines = [
        f"Budget B* = {comparison['B_star']},  K = {comparison['K']} slides,  "
        f"alpha_exec = {comparison['alpha_exec']} ms/part",
        "",
        f"{'Strategy':<16} {'Parts':>6} {'TotalCost':>10} {'MeanCost':>10} "
        f"{'MaxCost':>10} {'B_min':>5} {'B_max':>5} {'B_mean':>6} {'vs SO':>6}",
        "-" * 82,
    ]
    for name in ["SO_baseline", "uniform", "proportional", "optimal"]:
        r = comparison["strategies"][name]
        lines.append(
            f"{name:<16} {r['total_partitions']:>6} "
            f"{r['total_cost_ms']:>10.1f} {r['mean_cost_ms']:>10.2f} "
            f"{r['max_slide_cost_ms']:>10.2f} "
            f"{r['alloc_min']:>5} {r['alloc_max']:>5} "
            f"{r['alloc_mean']:>6.1f} {r['speedup_vs_SO']:>6.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DDL generation for creating non-uniform partitioned tables
# ---------------------------------------------------------------------------

def generate_partition_ddl(
    alloc: dict[str, int],
    slides: list[dict],
    parent_table: str = "objects_bdhp",
    hilbert_order: int = 8,
    source_table: str = "objects_spdb",
) -> list[str]:
    """Generate SQL DDL for creating a non-uniform partitioned table.

    Creates a two-level partitioned table:
      Level 1: LIST on slide_id  (one partition per slide)
      Level 2: RANGE on hilbert_key (B_k sub-partitions per slide, non-uniform)

    The Hilbert key range boundaries are computed as equal-count quantiles
    (uniform within each slide).  For density-aware boundaries, use
    generate_density_aware_ddl() instead.

    Parameters
    ----------
    alloc : dict
        slide_id -> B_k mapping from the optimizer.
    slides : list of dict
        Each must have 'slide_id' and 'n_objects'.
    parent_table : str
        Name of the new parent table.
    source_table : str
        Existing table to INSERT ... SELECT from.

    Returns
    -------
    list of SQL statements.
    """
    total_cells = 1 << (2 * hilbert_order)
    slide_map = {s["slide_id"]: s for s in slides}

    stmts = []

    # Create parent table (same schema as objects_spdb)
    stmts.append(f"""
        CREATE TABLE IF NOT EXISTS {parent_table} (
            object_id       BIGINT,
            slide_id        TEXT        NOT NULL,
            tile_id         TEXT,
            centroid_x      DOUBLE PRECISION,
            centroid_y      DOUBLE PRECISION,
            class_label     TEXT,
            hilbert_key     BIGINT      NOT NULL,
            geom            GEOMETRY(Point, 0),
            PRIMARY KEY (slide_id, hilbert_key, object_id)
        ) PARTITION BY LIST (slide_id);
    """)

    for sid, B_k in alloc.items():
        safe = sid.replace("-", "_")

        # Level-1: slide partition
        slide_part = f"{parent_table}_{safe}"
        stmts.append(f"""
            CREATE TABLE IF NOT EXISTS {slide_part}
            PARTITION OF {parent_table}
            FOR VALUES IN ('{sid}')
            PARTITION BY RANGE (hilbert_key);
        """)

        # Level-2: B_k sub-partitions with equal Hilbert key ranges
        for j in range(B_k):
            lo = j * total_cells // B_k
            hi = (j + 1) * total_cells // B_k
            child = f"{slide_part}_h{j}"
            stmts.append(f"""
                CREATE TABLE IF NOT EXISTS {child}
                PARTITION OF {slide_part}
                FOR VALUES FROM ({lo}) TO ({hi});
            """)

    # Bulk load from source
    stmts.append(f"""
        INSERT INTO {parent_table}
            (object_id, slide_id, tile_id, centroid_x, centroid_y,
             class_label, hilbert_key, geom)
        SELECT object_id, slide_id, tile_id, centroid_x, centroid_y,
               class_label, hilbert_key, geom
        FROM {source_table};
    """)

    # Build GiST indexes on all leaf partitions
    for sid, B_k in alloc.items():
        safe = sid.replace("-", "_")
        slide_part = f"{parent_table}_{safe}"
        for j in range(B_k):
            child = f"{slide_part}_h{j}"
            stmts.append(
                f"CREATE INDEX IF NOT EXISTS idx_{child}_geom "
                f"ON {child} USING gist (geom);"
            )

    stmts.append(f"ANALYZE {parent_table};")

    return stmts


# ---------------------------------------------------------------------------
# Derive alpha_exec from empirical data
# ---------------------------------------------------------------------------

def estimate_alpha_exec(
    spdb_p50: float, so_p50: float,
    spdb_partitions: int, so_partitions: int,
) -> float:
    """Derive per-partition executor overhead from benchmark deltas.

    alpha_exec = (C_spdb - C_so) / (B_spdb - B_so)
    """
    delta_cost = spdb_p50 - so_p50
    delta_parts = spdb_partitions - so_partitions
    if delta_parts <= 0:
        return DEFAULT_ALPHA_EXEC
    return max(0.001, delta_cost / delta_parts)


def derive_B_star(alpha_exec: float, threshold_ms: float = 50.0) -> int:
    """Partition budget: B* = threshold / alpha_exec."""
    if alpha_exec <= 0:
        return 10_000
    return int(threshold_ms / alpha_exec)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_comparison(comparison: dict, filename: str = "budget_allocation.json"):
    """Save comparison results to results/raw/."""
    out_dir = config.RAW_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    # Strip per-slide detail for compact JSON (keep summary)
    compact = {k: v for k, v in comparison.items() if k != "strategies"}
    compact["strategies"] = {}
    for name, data in comparison["strategies"].items():
        compact["strategies"][name] = {
            k: v for k, v in data.items() if k != "per_slide"
        }
        # Include allocation vector
        compact["strategies"][name]["allocation"] = {
            sid: info["B_k"]
            for sid, info in data["per_slide"].items()
        }

    with open(path, "w") as f:
        json.dump(compact, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_optimizer(B_star: int = 800, alpha_exec: float | None = None):
    """Run the full budget-constrained optimizer pipeline.

    1. Load slide catalog from ingest_metadata.json
    2. Compare uniform, proportional, optimal allocations
    3. Print summary
    4. Save results
    """
    meta_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    slides = []
    for sid, n_obj in meta.get("object_counts", {}).items():
        slides.append({"slide_id": sid, "n_objects": n_obj})

    if not slides:
        raise RuntimeError("No slide data in ingest_metadata.json")

    if alpha_exec is None:
        alpha_exec = DEFAULT_ALPHA_EXEC

    print(f"\n{'='*60}")
    print(f"Budget-Constrained Partition Optimizer")
    print(f"{'='*60}")
    print(f"  Slides: {len(slides)}")
    print(f"  Total objects: {sum(s['n_objects'] for s in slides):,}")
    print(f"  alpha_exec: {alpha_exec} ms/partition")
    print(f"  B*: {B_star}")
    print(f"  Workload: {config.WORKLOAD_MIX}")
    print()

    comparison = compare_allocations(slides, B_star, alpha_exec=alpha_exec)
    print(allocation_summary(comparison))

    path = save_comparison(comparison)
    print(f"\n  Results saved to {path}")

    # Show top-10 slides by optimal allocation
    opt = comparison["strategies"]["optimal"]["per_slide"]
    top = sorted(opt.items(), key=lambda kv: kv[1]["B_k"], reverse=True)[:10]
    print(f"\n  Top-10 slides by optimal B_k:")
    print(f"  {'Slide':<30} {'N_obj':>10} {'B_k':>5} {'Cost':>8}")
    print(f"  {'-'*55}")
    for sid, info in top:
        print(f"  {sid:<30} {info['n_objects']:>10,} {info['B_k']:>5} "
              f"{info['cost_ms']:>8.1f}")

    return comparison


if __name__ == "__main__":
    run_optimizer()
