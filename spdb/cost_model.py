"""Cost model for viewport queries across storage layouts.

Cost(Q) = C_plan(partitions) + SUM_{touched} C_scan(P_i)
C_scan(P) = C_index(depth, selectivity) + C_heap(pages, hit_rate)
"""

import math
import numpy as np
from spdb import hilbert


# ---------------------------------------------------------------------------
# Hilbert / Z-order locality models
# ---------------------------------------------------------------------------

def hilbert_buckets_touched(viewport_frac, p, num_buckets):
    """E[B_hit] = f*B + C_h*sqrt(f)*sqrt(B), C_h ~ 2.0."""
    f = viewport_frac
    B = num_buckets
    C_h = 2.0
    bulk = f * B
    boundary = C_h * math.sqrt(f) * math.sqrt(B)
    return min(bulk + boundary, B)


def zorder_buckets_touched(viewport_frac, p, num_buckets):
    """Same model but C_z ~ 3.2 for Z-order (weaker locality)."""
    f = viewport_frac
    B = num_buckets
    C_z = 3.2
    bulk = f * B
    boundary = C_z * math.sqrt(f) * math.sqrt(B)
    return min(bulk + boundary, B)


def expected_pruning_rate(viewport_frac, p, num_buckets, curve="hilbert"):
    fn = hilbert_buckets_touched if curve == "hilbert" else zorder_buckets_touched
    return 1.0 - fn(viewport_frac, p, num_buckets) / num_buckets


# ---------------------------------------------------------------------------
# GiST index cost model
# ---------------------------------------------------------------------------

def gist_depth(n_tuples, fanout=100):
    """Estimated GiST tree depth for n_tuples with given fanout."""
    if n_tuples <= 0:
        return 1
    return max(1, math.ceil(math.log(max(1, n_tuples)) / math.log(fanout)))


def gist_scan_cost(n_tuples, selectivity, page_size=8192, tuple_width=80):
    """Cost of a GiST index scan within a single table/partition.

    Returns (index_ios, heap_ios, estimated_ms).
    """
    depth = gist_depth(n_tuples)
    tuples_per_page = max(1, page_size // tuple_width)
    total_pages = max(1, math.ceil(n_tuples / tuples_per_page))

    index_ios = depth + 1
    matching_tuples = max(1, int(n_tuples * selectivity))
    matching_pages = max(1, math.ceil(matching_tuples / tuples_per_page))
    heap_ios = min(matching_pages, total_pages)

    return index_ios, heap_ios


def brin_scan_cost(n_tuples, selectivity, pages_per_range=32,
                   page_size=8192, tuple_width=80):
    """Cost of a BRIN index scan on physically sorted (CLUSTER'd) data.

    BRIN indexes store min/max per block range.  On sorted data, a range
    query touches approximately selectivity * total_ranges ranges, with
    one additional range on each side for boundary effects.
    """
    tuples_per_page = max(1, page_size // tuple_width)
    total_pages = max(1, math.ceil(n_tuples / tuples_per_page))
    total_ranges = max(1, math.ceil(total_pages / pages_per_range))

    brin_index_pages = max(1, math.ceil(total_ranges * 20 / page_size))

    matching_ranges = max(1, int(total_ranges * selectivity) + 2)
    heap_ios = min(matching_ranges * pages_per_range, total_pages)

    return brin_index_pages, heap_ios


def seq_scan_cost(n_tuples, page_size=8192, tuple_width=80):
    """Cost of a sequential scan on a table (no index)."""
    tuples_per_page = max(1, page_size // tuple_width)
    return max(1, math.ceil(n_tuples / tuples_per_page))


# ---------------------------------------------------------------------------
# Full query cost model for each configuration
# ---------------------------------------------------------------------------

def _ms_from_ios(index_ios, heap_ios, hit_ratio=0.8,
                 random_page_ms=0.5, seq_page_ms=0.1, tuple_cost_ms=0.001,
                 tuples_returned=0):
    """Convert I/O counts to estimated milliseconds."""
    random_ios = index_ios + int(heap_ios * (1 - hit_ratio))
    seq_ios = int(heap_ios * hit_ratio)
    io_ms = random_ios * random_page_ms + seq_ios * seq_page_ms
    cpu_ms = tuples_returned * tuple_cost_ms
    return io_ms + cpu_ms


class ViewportCostModel:
    """Predicts Q1 viewport latency for all configurations.

    Parameters
    ----------
    n_objects : int
        Total objects in the queried slide.
    image_width, image_height : float
        Slide dimensions in pixels.
    hit_ratio : float
        Estimated shared_buffer hit ratio (0-1).
    """

    def __init__(self, n_objects, image_width, image_height,
                 hit_ratio=0.85,
                 hilbert_order=8, bucket_target=50_000,
                 random_page_ms=0.5, seq_page_ms=0.08,
                 planning_ms_per_partition=0.1,
                 tuple_width=80, page_size=8192):
        self.n = n_objects
        self.w = image_width
        self.h = image_height
        self.hit_ratio = hit_ratio
        self.p = hilbert_order
        self.T = bucket_target
        self.rand_ms = random_page_ms
        self.seq_ms = seq_page_ms
        self.plan_ms = planning_ms_per_partition
        self.tw = tuple_width
        self.ps = page_size

    def _tuples_per_page(self):
        return max(1, self.ps // self.tw)

    def cost_mono(self, viewport_frac):
        """Monolithic: one global GiST, no partitioning."""
        selectivity = viewport_frac
        idx_io, heap_io = gist_scan_cost(self.n, selectivity, self.ps, self.tw)
        tuples_ret = int(self.n * selectivity)
        plan_ms = 0.5  # simple plan
        exec_ms = _ms_from_ios(idx_io, heap_io, self.hit_ratio,
                               self.rand_ms, self.seq_ms, 0.001, tuples_ret)
        return {"planning_ms": plan_ms, "execution_ms": exec_ms,
                "total_ms": plan_ms + exec_ms,
                "index_ios": idx_io, "heap_ios": heap_io,
                "tuples_returned": tuples_ret}

    def cost_mono_clustered(self, viewport_frac):
        """Monolithic CLUSTER'd + BRIN: sorted data, tiny index."""
        selectivity = viewport_frac
        brin_io, heap_io = brin_scan_cost(
            self.n, selectivity, pages_per_range=32,
            page_size=self.ps, tuple_width=self.tw)
        tuples_ret = int(self.n * selectivity)
        plan_ms = 0.5
        exec_ms = _ms_from_ios(brin_io, heap_io, self.hit_ratio,
                               self.rand_ms, self.seq_ms, 0.001, tuples_ret)
        return {"planning_ms": plan_ms, "execution_ms": exec_ms,
                "total_ms": plan_ms + exec_ms,
                "index_ios": brin_io, "heap_ios": heap_io,
                "tuples_returned": tuples_ret}

    def cost_slide_only(self, viewport_frac):
        """Slide-partitioned: one partition scanned, per-partition GiST."""
        n_slide = self.n
        selectivity = viewport_frac
        idx_io, heap_io = gist_scan_cost(n_slide, selectivity, self.ps, self.tw)
        tuples_ret = int(n_slide * selectivity)
        plan_ms = 1.0
        exec_ms = _ms_from_ios(idx_io, heap_io, self.hit_ratio,
                               self.rand_ms, self.seq_ms, 0.001, tuples_ret)
        return {"planning_ms": plan_ms, "execution_ms": exec_ms,
                "total_ms": plan_ms + exec_ms,
                "index_ios": idx_io, "heap_ios": heap_io,
                "tuples_returned": tuples_ret}

    def cost_slide_only_clustered(self, viewport_frac):
        """Slide-partitioned, each partition CLUSTER'd + BRIN."""
        n_slide = self.n
        selectivity = viewport_frac
        brin_io, heap_io = brin_scan_cost(
            n_slide, selectivity, pages_per_range=32,
            page_size=self.ps, tuple_width=self.tw)
        tuples_ret = int(n_slide * selectivity)
        plan_ms = 1.0
        exec_ms = _ms_from_ios(brin_io, heap_io, self.hit_ratio,
                               self.rand_ms, self.seq_ms, 0.001, tuples_ret)
        return {"planning_ms": plan_ms, "execution_ms": exec_ms,
                "total_ms": plan_ms + exec_ms,
                "index_ios": brin_io, "heap_ios": heap_io,
                "tuples_returned": tuples_ret}

    def cost_spdb(self, viewport_frac, curve="hilbert"):
        """SPDB: two-level partitioned, per-bucket GiST."""
        num_buckets = max(1, self.n // self.T)
        fn = hilbert_buckets_touched if curve == "hilbert" else zorder_buckets_touched
        b_hit = fn(viewport_frac, self.p, num_buckets)
        b_hit = max(1, b_hit)

        tuples_per_bucket = self.n / num_buckets
        selectivity_within = viewport_frac * num_buckets / b_hit
        selectivity_within = min(selectivity_within, 1.0)

        total_idx_io = 0
        total_heap_io = 0
        for _ in range(int(math.ceil(b_hit))):
            idx_io, heap_io = gist_scan_cost(
                int(tuples_per_bucket), selectivity_within, self.ps, self.tw)
            total_idx_io += idx_io
            total_heap_io += heap_io

        tuples_ret = int(self.n * viewport_frac)
        plan_ms = self.plan_ms * num_buckets
        exec_ms = _ms_from_ios(total_idx_io, total_heap_io, self.hit_ratio,
                               self.rand_ms, self.seq_ms, 0.001, tuples_ret)
        return {"planning_ms": plan_ms, "execution_ms": exec_ms,
                "total_ms": plan_ms + exec_ms,
                "index_ios": total_idx_io, "heap_ios": total_heap_io,
                "tuples_returned": tuples_ret,
                "buckets_touched": b_hit,
                "total_buckets": num_buckets,
                "pruning_rate": 1 - b_hit / num_buckets}

    def predict_all(self, viewport_frac):
        """Predict costs for all configurations at a given viewport fraction."""
        return {
            "Mono": self.cost_mono(viewport_frac),
            "Mono-C": self.cost_mono_clustered(viewport_frac),
            "SO": self.cost_slide_only(viewport_frac),
            "SO-C": self.cost_slide_only_clustered(viewport_frac),
            "SPDB": self.cost_spdb(viewport_frac, curve="hilbert"),
            "SPDB-Z": self.cost_spdb(viewport_frac, curve="zorder"),
        }


# ---------------------------------------------------------------------------
# Optimal parameter selection
# ---------------------------------------------------------------------------

def optimal_hilbert_order(n_objects, viewport_frac=0.05, bucket_target=50_000,
                          p_range=range(4, 16)):
    """Find the Hilbert order p that minimizes predicted SPDB viewport cost."""
    best_p = 8
    best_cost = float("inf")

    for p in p_range:
        num_buckets = max(1, n_objects // bucket_target)
        b_hit = hilbert_buckets_touched(viewport_frac, p, num_buckets)

        tuples_per_bucket = n_objects / num_buckets
        depth = gist_depth(int(tuples_per_bucket))
        scan_cost = b_hit * (depth + tuples_per_bucket * viewport_frac * 0.001)
        plan_cost = 0.1 * num_buckets
        total = scan_cost + plan_cost

        if total < best_cost:
            best_cost = total
            best_p = p

    return best_p


def optimal_bucket_target(n_objects, viewport_frac=0.05, p=8,
                          t_range=None):
    """Find the bucket target T that minimizes predicted viewport cost."""
    if t_range is None:
        t_range = [5000, 10000, 25000, 50000, 100000, 200000, 500000]

    best_t = 50000
    best_cost = float("inf")

    for T in t_range:
        num_buckets = max(1, n_objects // T)
        b_hit = hilbert_buckets_touched(viewport_frac, p, num_buckets)
        tuples_per_bucket = n_objects / max(1, num_buckets)
        depth = gist_depth(int(tuples_per_bucket))

        scan_cost = b_hit * (depth + tuples_per_bucket * viewport_frac * 0.001)
        plan_cost = 0.1 * num_buckets
        total = scan_cost + plan_cost

        if total < best_cost:
            best_cost = total
            best_t = T

    return best_t


# ---------------------------------------------------------------------------
# Model validation against empirical data
# ---------------------------------------------------------------------------

def validate_against_buffers(empirical_trials, model_params):
    """Compare model predictions against observed EXPLAIN BUFFERS data.

    Parameters
    ----------
    empirical_trials : list of dict
        Each dict has: config, viewport_frac, planning_time, execution_time,
        shared_hit, shared_read, actual_rows.
    model_params : dict
        Keys: n_objects, image_width, image_height, hit_ratio.

    Returns
    -------
    dict with per-config error metrics (MAE, RMSE, R^2 for total latency).
    """
    model = ViewportCostModel(**model_params)
    results = {}

    by_config = {}
    for t in empirical_trials:
        cfg = t["config"]
        by_config.setdefault(cfg, []).append(t)

    for cfg, trials in by_config.items():
        predicted = []
        observed = []
        for t in trials:
            vf = t["viewport_frac"]
            if cfg == "Mono":
                pred = model.cost_mono(vf)
            elif cfg == "Mono-C":
                pred = model.cost_mono_clustered(vf)
            elif cfg == "SO":
                pred = model.cost_slide_only(vf)
            elif cfg == "SO-C":
                pred = model.cost_slide_only_clustered(vf)
            elif cfg == "SPDB":
                pred = model.cost_spdb(vf, "hilbert")
            elif cfg == "SPDB-Z":
                pred = model.cost_spdb(vf, "zorder")
            else:
                continue
            predicted.append(pred["total_ms"])
            observed.append(t["planning_time"] + t["execution_time"])

        predicted = np.array(predicted)
        observed = np.array(observed)
        if len(predicted) == 0:
            continue

        mae = float(np.mean(np.abs(predicted - observed)))
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        ss_res = float(np.sum((observed - predicted) ** 2))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        results[cfg] = {
            "mae_ms": mae,
            "rmse_ms": rmse,
            "r_squared": r2,
            "n_trials": len(predicted),
            "mean_predicted": float(np.mean(predicted)),
            "mean_observed": float(np.mean(observed)),
        }

    return results


def generate_design_space_table(n_objects=1_260_000, image_width=100_000,
                                 image_height=100_000,
                                 viewport_fracs=None):
    """Generate a table showing predicted cost for every config x viewport size.

    This is the core "design space exploration" table for the paper.
    """
    if viewport_fracs is None:
        viewport_fracs = [0.01, 0.02, 0.05, 0.10, 0.20]

    model = ViewportCostModel(n_objects, image_width, image_height)
    table = {}

    for vf in viewport_fracs:
        predictions = model.predict_all(vf)
        table[vf] = {}
        for cfg, pred in predictions.items():
            table[vf][cfg] = {
                "total_ms": round(pred["total_ms"], 1),
                "planning_ms": round(pred["planning_ms"], 2),
                "execution_ms": round(pred["execution_ms"], 1),
                "index_ios": pred["index_ios"],
                "heap_ios": pred["heap_ios"],
            }
            if "pruning_rate" in pred:
                table[vf][cfg]["pruning_rate"] = round(pred["pruning_rate"], 3)
                table[vf][cfg]["buckets_touched"] = round(pred["buckets_touched"], 1)

    return table
