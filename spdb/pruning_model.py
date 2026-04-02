"""Pruning model for Hilbert-bucketed partitions.  See paper Section 4."""

import numpy as np
from spdb import hilbert


def expected_buckets_touched(viewport_frac, p, num_buckets):
    """Expected buckets touched by a viewport of area fraction f.

    E[B_hit] = f*B + C_h*sqrt(f)*sqrt(B), where C_h ~ 2.0 for Hilbert.
    See paper Section 4.1 for derivation.
    """
    f = viewport_frac
    B = num_buckets
    C = 2.0  # empirical boundary crossing constant for Hilbert

    bulk_buckets = f * B
    boundary_buckets = C * np.sqrt(f) * np.sqrt(B)
    expected = bulk_buckets + boundary_buckets
    return min(expected, B)


def expected_pruning_rate(viewport_frac, p, num_buckets):
    """Expected fraction of buckets pruned (not touched)."""
    e_hit = expected_buckets_touched(viewport_frac, p, num_buckets)
    return 1.0 - e_hit / num_buckets


def optimal_p(viewport_frac, n_objects, bucket_target=50_000, p_range=range(4, 16)):
    """Find p that minimizes expected scan cost (see paper Eq. 3)."""
    best_p = None
    best_cost = float("inf")

    for p in p_range:
        num_buckets = max(1, n_objects // bucket_target)
        e_hit = expected_buckets_touched(viewport_frac, p, num_buckets)
        tuples_scanned = e_hit * (n_objects / num_buckets)
        planning_cost = 0.01 * num_buckets  # ms per partition for planner
        total_cost = tuples_scanned + planning_cost * 1000
        if total_cost < best_cost:
            best_cost = total_cost
            best_p = p

    return best_p


def validate_model(empirical_data, p, num_buckets):
    """Compare model predictions against empirical measurements."""
    predictions = []
    actuals = []
    for d in empirical_data:
        pred = expected_buckets_touched(d["viewport_frac"], p, d["total_buckets"])
        predictions.append(pred)
        actuals.append(d["buckets_touched"])

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    r2 = 1.0 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r_squared": float(r2),
        "n_samples": len(actuals),
        "mean_predicted": float(np.mean(predictions)),
        "mean_actual": float(np.mean(actuals)),
    }


def generate_prediction_table(p_values, viewport_fracs, n_objects=1_260_000,
                               bucket_target=50_000):
    """Predicted pruning rates for different (p, f) combinations."""
    num_buckets = max(1, n_objects // bucket_target)
    table = {}
    for p in p_values:
        table[p] = {}
        for f in viewport_fracs:
            e_hit = expected_buckets_touched(f, p, num_buckets)
            pruning = 1.0 - e_hit / num_buckets
            table[p][f] = {
                "expected_buckets": round(e_hit, 1),
                "pruning_rate": round(pruning, 3),
                "total_buckets": num_buckets,
            }
    return table
