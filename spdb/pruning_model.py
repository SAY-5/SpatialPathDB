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


# ---------------------------------------------------------------------------
# Density-dependent boundary crossing constant  C_h(cv)
# ---------------------------------------------------------------------------
# The global model uses C = 2.0.  In practice, slides with non-uniform object
# density cause more Hilbert-curve boundary crossings.  We parameterise this
# as C_h(cv) = C_base + alpha * cv, where cv = tile_count_std / tile_count_mean
# is the coefficient of variation of per-tile object density.
# C_base ~ 1.5 is the uniform-density limit; alpha is fitted from data.
# ---------------------------------------------------------------------------

_GLOBAL_C = 2.03  # fallback when no density data is available


def density_adjusted_ch(density_cv):
    """Return density-adjusted boundary crossing constant C_h.

    Parameters
    ----------
    density_cv : float or None
        Coefficient of variation (std / mean) of per-tile object counts
        within a slide.  If *None*, fall back to global C = 2.03.

    Returns
    -------
    float
        Adjusted C_h value.

    Model: C_h(cv) = C_base + alpha * cv
        - C_base ~ 1.5 (uniform-density limit)
        - alpha fitted by :func:`calibrate_density_ch`
        - Falls back to global C = 2.03 when *density_cv* is None.
    """
    if density_cv is None:
        return _GLOBAL_C
    # Default coefficients (overridden once calibrate_density_ch is called)
    C_base = 1.5
    alpha = 0.75
    return C_base + alpha * density_cv


def calibrate_density_ch(density_data, pruning_data):
    """Fit the linear relationship C_h(cv) = C_base + alpha * density_cv.

    Parameters
    ----------
    density_data : dict
        Per-slide density statistics keyed by slide_id, as produced by
        ``density_analysis.json``.  Each entry must contain
        ``tile_count_mean`` and ``tile_count_std``.
    pruning_data : dict
        Pruning benchmark output (``pruning_analysis.json``).  Must contain
        a ``trials`` list where each trial has ``slide_id``,
        ``total_buckets``, and ``candidate_buckets``.

    Returns
    -------
    dict
        ``C_base``, ``alpha``, ``r_squared``, ``n_slides``, and
        ``per_slide`` details.

    Notes
    -----
    For each trial we back-solve the empirical C_h from:

        B_hit = f * B + C_h * sqrt(f) * sqrt(B)
        =>  C_h = (B_hit - f * B) / (sqrt(f) * sqrt(B))

    The viewport fraction *f* is taken from the pruning data's top-level
    ``viewport_frac`` field, defaulting to 0.05.
    """
    from scipy.stats import linregress

    f = pruning_data.get("viewport_frac", 0.05)
    sqrt_f = np.sqrt(f)

    # --- Aggregate per-slide: average empirical C_h and density CV ----------
    slide_ch_samples = {}  # slide_id -> list of empirical C_h values
    for trial in pruning_data.get("trials", []):
        sid = trial["slide_id"]
        B = trial["total_buckets"]
        B_hit = trial["candidate_buckets"]
        if B <= 0 or sqrt_f == 0:
            continue
        denom = sqrt_f * np.sqrt(B)
        if denom == 0:
            continue
        ch_emp = (B_hit - f * B) / denom
        slide_ch_samples.setdefault(sid, []).append(ch_emp)

    # Build paired arrays: density_cv vs mean empirical C_h per slide
    cvs = []
    chs = []
    per_slide = []
    for sid, ch_list in sorted(slide_ch_samples.items()):
        if sid not in density_data:
            continue
        d = density_data[sid]
        mean_count = d.get("tile_count_mean", 0)
        std_count = d.get("tile_count_std", 0)
        if mean_count <= 0:
            continue
        cv = std_count / mean_count
        mean_ch = float(np.mean(ch_list))

        cvs.append(cv)
        chs.append(mean_ch)
        per_slide.append({
            "slide_id": sid,
            "density_cv": round(cv, 4),
            "empirical_ch": round(mean_ch, 4),
            "n_trials": len(ch_list),
        })

    cvs = np.array(cvs)
    chs = np.array(chs)

    if len(cvs) < 2:
        return {
            "C_base": _GLOBAL_C,
            "alpha": 0.0,
            "r_squared": 0.0,
            "n_slides": len(cvs),
            "per_slide": per_slide,
        }

    slope, intercept, r_value, p_value, std_err = linregress(cvs, chs)

    return {
        "C_base": float(intercept),
        "alpha": float(slope),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
        "n_slides": len(cvs),
        "per_slide": per_slide,
    }


def expected_buckets_touched_density(viewport_frac, p, num_buckets, density_cv):
    """Expected buckets touched using density-adjusted C_h.

    Same formula as :func:`expected_buckets_touched` but replaces the global
    C = 2.0 with ``density_adjusted_ch(density_cv)``.

    Parameters
    ----------
    viewport_frac : float
        Fraction of the image area covered by the viewport.
    p : int
        Hilbert order (unused in formula but kept for API symmetry).
    num_buckets : int
        Total number of Hilbert buckets for this slide.
    density_cv : float or None
        Coefficient of variation of per-tile object density.

    Returns
    -------
    float
        Expected number of buckets touched.
    """
    f = viewport_frac
    B = num_buckets
    C = density_adjusted_ch(density_cv)

    bulk_buckets = f * B
    boundary_buckets = C * np.sqrt(f) * np.sqrt(B)
    expected = bulk_buckets + boundary_buckets
    return min(expected, B)


def validate_density_model(density_data, pruning_data):
    """Compare global vs density-adjusted C_h pruning predictions.

    Parameters
    ----------
    density_data : dict
        Per-slide density statistics (``density_analysis.json``).
    pruning_data : dict
        Pruning benchmark output (``pruning_analysis.json``).

    Returns
    -------
    dict
        RMSE for global model, RMSE for density-adjusted model, improvement
        percentage, and per-trial comparison details.
    """
    # First calibrate to get fitted coefficients
    cal = calibrate_density_ch(density_data, pruning_data)
    C_base = cal["C_base"]
    alpha = cal["alpha"]

    f = pruning_data.get("viewport_frac", 0.05)

    actuals = []
    pred_global = []
    pred_density = []
    per_trial = []

    for trial in pruning_data.get("trials", []):
        sid = trial["slide_id"]
        B = trial["total_buckets"]
        B_hit_actual = trial["candidate_buckets"]

        if B <= 0:
            continue

        # Global model prediction (C = 2.03)
        global_hit = f * B + _GLOBAL_C * np.sqrt(f) * np.sqrt(B)
        global_hit = min(global_hit, B)

        # Density-adjusted prediction
        if sid in density_data:
            d = density_data[sid]
            mean_count = d.get("tile_count_mean", 0)
            std_count = d.get("tile_count_std", 0)
            cv = std_count / mean_count if mean_count > 0 else None
        else:
            cv = None

        if cv is not None:
            ch_adj = C_base + alpha * cv
        else:
            ch_adj = _GLOBAL_C
        density_hit = f * B + ch_adj * np.sqrt(f) * np.sqrt(B)
        density_hit = min(density_hit, B)

        actuals.append(B_hit_actual)
        pred_global.append(global_hit)
        pred_density.append(density_hit)

        # Pruning rates for comparison
        actual_pr = 1.0 - B_hit_actual / B
        global_pr = 1.0 - global_hit / B
        density_pr = 1.0 - density_hit / B

        per_trial.append({
            "slide_id": sid,
            "total_buckets": B,
            "actual_buckets_hit": B_hit_actual,
            "global_predicted": round(global_hit, 2),
            "density_predicted": round(density_hit, 2),
            "actual_pruning_rate": round(actual_pr, 4),
            "global_pruning_rate": round(global_pr, 4),
            "density_pruning_rate": round(density_pr, 4),
        })

    actuals = np.array(actuals, dtype=np.float64)
    pred_global = np.array(pred_global, dtype=np.float64)
    pred_density = np.array(pred_density, dtype=np.float64)

    rmse_global = float(np.sqrt(np.mean((actuals - pred_global) ** 2)))
    rmse_density = float(np.sqrt(np.mean((actuals - pred_density) ** 2)))
    mae_global = float(np.mean(np.abs(actuals - pred_global)))
    mae_density = float(np.mean(np.abs(actuals - pred_density)))

    if rmse_global > 0:
        improvement_pct = (rmse_global - rmse_density) / rmse_global * 100.0
    else:
        improvement_pct = 0.0

    return {
        "rmse_global": round(rmse_global, 4),
        "rmse_density": round(rmse_density, 4),
        "improvement_pct": round(improvement_pct, 2),
        "mae_global": round(mae_global, 4),
        "mae_density": round(mae_density, 4),
        "calibration": cal,
        "n_trials": len(actuals),
        "per_trial": per_trial,
    }
