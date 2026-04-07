"""Learned SFC baseline: CDF-equalized Hilbert vs standard Hilbert.

Implements a data-adaptive SFC by:
1. Computing empirical CDFs on x and y coordinates per slide
2. Mapping (x,y) -> (CDF_x(x), CDF_y(y)) in [0,1]^2
3. Applying Hilbert encoding on the equalized coordinates

This simulates the core idea behind LISA/Flood-style learned indexes:
equalize density so the spatial grid has uniform occupancy, reducing
false positives on non-uniform data.

Usage:
    python -m benchmarks.learned_sfc_benchmark --trials 200
    python -m benchmarks.learned_sfc_benchmark --trials 200 --dataset taxi
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple, Dict

import numpy as np
import psycopg2

from spdb import config, hcci, hilbert
from benchmarks.framework import compute_stats, time_query, wilcoxon_ranksum

TABLE = config.TABLE_SLIDE_ONLY

# ---------------------------------------------------------------------------
# CDF computation and learned key encoding
# ---------------------------------------------------------------------------

class LearnedSFC:
    """CDF-equalized Hilbert encoding (per-slide learned mapping)."""

    def __init__(self, p: int = 8, n_bins: int = 1024):
        self.p = p
        self.n_bins = n_bins
        self.cdfs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.train_time: Dict[str, float] = {}

    def train(self, conn, slide_id: str, table: str = TABLE):
        """Learn CDFs from data. Returns training time in seconds."""
        t0 = time.time()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT centroid_x, centroid_y FROM {table} WHERE slide_id = %s",
                (slide_id,)
            )
            rows = cur.fetchall()

        xs = np.array([float(r[0]) for r in rows], dtype=np.float64)
        ys = np.array([float(r[1]) for r in rows], dtype=np.float64)

        # Compute empirical CDF using sorted values + interpolation points
        x_sorted = np.sort(xs)
        y_sorted = np.sort(ys)

        # Subsample for CDF (use n_bins quantile points)
        x_quantiles = np.percentile(xs, np.linspace(0, 100, self.n_bins + 1))
        y_quantiles = np.percentile(ys, np.linspace(0, 100, self.n_bins + 1))
        x_cdf_vals = np.linspace(0, 1, self.n_bins + 1)
        y_cdf_vals = np.linspace(0, 1, self.n_bins + 1)

        self.cdfs[slide_id] = (x_quantiles, x_cdf_vals, y_quantiles, y_cdf_vals)

        elapsed = time.time() - t0
        self.train_time[slide_id] = elapsed
        return elapsed

    def transform(self, slide_id: str, xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map (x,y) -> (CDF_x(x), CDF_y(y)) in [0,1]^2."""
        x_q, x_c, y_q, y_c = self.cdfs[slide_id]
        cx = np.interp(xs, x_q, x_c)
        cy = np.interp(ys, y_q, y_c)
        return cx, cy

    def encode(self, slide_id: str, xs: np.ndarray, ys: np.ndarray,
               class_labels: List[str]) -> np.ndarray:
        """Compute learned composite keys: CDF-equalized Hilbert + class prefix."""
        cx, cy = self.transform(slide_id, xs, ys)

        n = 1 << self.p
        gx = np.clip((cx * n).astype(np.int64), 0, n - 1)
        gy = np.clip((cy * n).astype(np.int64), 0, n - 1)
        h_keys = hilbert.encode_batch(gx, gy, self.p)

        class_enums = np.array(
            [hcci.CLASS_ENUM.get(c, 0) for c in class_labels],
            dtype=np.int64
        )
        composite = (class_enums << hcci.COMPOSITE_SHIFT) | h_keys
        return composite

    def ranges_for_viewport(self, slide_id: str,
                            x0: float, y0: float, x1: float, y1: float,
                            slide_width: float, slide_height: float) -> List[Tuple[int, int]]:
        """Compute Hilbert ranges in CDF-equalized space for a viewport."""
        # Transform viewport corners to CDF space
        cx0, cy0 = self.transform(slide_id, np.array([x0]), np.array([y0]))
        cx1, cy1 = self.transform(slide_id, np.array([x1]), np.array([y1]))

        # Now compute Hilbert ranges in [0,1]^2 space
        return hcci.hilbert_ranges_direct(
            float(cx0[0]), float(cy0[0]),
            float(cx1[0]), float(cy1[0]),
            1.0, 1.0,  # CDF space is [0,1]^2
            self.p,
            max_ranges=64,
            x_origin=0.0,
            y_origin=0.0,
        )


# ---------------------------------------------------------------------------
# Setup: create learned_composite_key column and index
# ---------------------------------------------------------------------------

def setup_learned_column(conn, learned: LearnedSFC, metadata=None):
    """Create learned_composite_key column, train CDFs, populate keys, build index."""
    from psycopg2.extras import execute_values

    # Add column
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN IF NOT EXISTS learned_composite_key BIGINT")
    conn.commit()

    # Get slides
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        slides = [r[0] for r in cur.fetchall()]

    print(f"  Training CDFs and populating learned keys for {len(slides)} slides...")
    total_train_time = 0
    total_updated = 0

    for i, sid in enumerate(slides):
        # Train CDF
        train_t = learned.train(conn, sid)
        total_train_time += train_t

        # Read data
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, centroid_x, centroid_y, class_label FROM {TABLE} WHERE slide_id = %s",
                (sid,)
            )
            rows = cur.fetchall()

        if not rows:
            continue

        ids = np.array([r[0] for r in rows], dtype=np.int64)
        xs = np.array([float(r[1]) for r in rows], dtype=np.float64)
        ys = np.array([float(r[2]) for r in rows], dtype=np.float64)
        classes = [r[3] for r in rows]

        # Compute learned composite keys
        composite = learned.encode(sid, xs, ys, classes)

        # Batch update
        values = [(int(ck), int(oid)) for ck, oid in zip(composite, ids)]
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _learned_tmp (ck BIGINT, oid BIGINT)")
            cur.execute("TRUNCATE _learned_tmp")
            execute_values(cur, "INSERT INTO _learned_tmp (ck, oid) VALUES %s", values, page_size=10000)
            cur.execute(f"""
                UPDATE {TABLE} t
                SET learned_composite_key = z.ck
                FROM _learned_tmp z
                WHERE t.id = z.oid
            """)
            total_updated += cur.rowcount
        conn.commit()

        if (i + 1) % 20 == 0 or i == len(slides) - 1:
            print(f"    Slide {i+1}/{len(slides)}: {total_updated:,} rows, "
                  f"train={total_train_time:.1f}s")

    print(f"  Total training time: {total_train_time:.1f}s for {total_updated:,} rows")

    # Create covering index
    print("  Creating idx_learned_covering index...")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_learned_covering ON {TABLE}
            (slide_id, learned_composite_key)
            INCLUDE (centroid_x, centroid_y, class_label, area)
        """)
        cur.execute(f"ANALYZE {TABLE}")
    conn.autocommit = False
    print("  Done.")

    return total_train_time


# ---------------------------------------------------------------------------
# Query builder for learned SFC
# ---------------------------------------------------------------------------

def build_learned_query(
    table: str, slide_id: str, class_label: str,
    x0: float, y0: float, x1: float, y1: float,
    learned: LearnedSFC,
    slide_width: float, slide_height: float,
) -> Tuple[str, tuple, int]:
    """Build UNION ALL query using learned SFC ranges."""
    enum_val = hcci.CLASS_ENUM.get(class_label, 0)
    prefix = enum_val << hcci.COMPOSITE_SHIFT

    ranges = learned.ranges_for_viewport(
        slide_id, x0, y0, x1, y1, slide_width, slide_height
    )

    branches = []
    params = []
    for lo, hi in ranges:
        ck_lo = prefix | lo
        ck_hi = prefix | (hi - 1)
        branches.append(
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table} "
            f"WHERE slide_id = %s AND learned_composite_key >= %s AND learned_composite_key <= %s"
        )
        params.extend([slide_id, ck_lo, ck_hi])

    sql = " UNION ALL ".join(branches)
    return sql, tuple(params), len(ranges)


# ---------------------------------------------------------------------------
# Staleness test: measure FP degradation after inserts without retraining
# ---------------------------------------------------------------------------

def test_staleness(conn, learned: LearnedSFC, slide_id: str,
                   n_inserts: int = 10000, seed: int = 42):
    """Insert new rows without retraining the CDF, measure FP increase."""
    rng = np.random.RandomState(seed)
    w, h = get_dims(conn, slide_id)

    # Generate random new points (simulating new data arriving)
    new_xs = rng.uniform(0, w, n_inserts).astype(np.float64)
    new_ys = rng.uniform(0, h, n_inserts).astype(np.float64)
    new_classes = rng.choice(['Epithelial', 'Stromal', 'Tumor', 'Lymphocyte'], n_inserts)

    # Encode with STALE CDF (not retrained on new data)
    stale_keys = learned.encode(slide_id, new_xs, new_ys, list(new_classes))

    # Retrain and encode with fresh CDF
    # (We can't easily retrain without inserting, so instead measure how far
    #  the new points' CDF values deviate from uniform)
    cx, cy = learned.transform(slide_id, new_xs, new_ys)

    # Points outside the original data range will clip to 0 or 1
    clipped = np.sum((cx <= 0.001) | (cx >= 0.999) | (cy <= 0.001) | (cy >= 0.999))
    clip_frac = clipped / n_inserts

    # Measure CDF uniformity (should be ~uniform if data is from same distribution)
    # KS test against uniform
    from scipy import stats as sp_stats
    ks_x, p_x = sp_stats.kstest(cx, 'uniform')
    ks_y, p_y = sp_stats.kstest(cy, 'uniform')

    return {
        'n_inserts': n_inserts,
        'clipped_fraction': round(float(clip_frac), 4),
        'ks_stat_x': round(float(ks_x), 4),
        'ks_stat_y': round(float(ks_y), 4),
        'ks_p_x': float(p_x),
        'ks_p_y': float(p_y),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_dims_cache = {}

def get_dims(conn, slide_id, metadata=None):
    if slide_id not in _dims_cache:
        if metadata and slide_id in metadata.get("metas", {}):
            m = metadata["metas"][slide_id]
            _dims_cache[slide_id] = (float(m["image_width"]), float(m["image_height"]))
        else:
            with conn.cursor() as cur:
                cur.execute(f"SELECT MAX(centroid_x), MAX(centroid_y) FROM {TABLE} WHERE slide_id = %s", (slide_id,))
                row = cur.fetchone()
            if row and row[0] and row[1]:
                _dims_cache[slide_id] = (float(row[0]) * 1.05, float(row[1]) * 1.05)
            else:
                _dims_cache[slide_id] = (100000.0, 100000.0)
    return _dims_cache[slide_id]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_comparison(
    conn, learned: LearnedSFC, slides: List[str], metadata,
    n_trials: int = 200, viewport_frac: float = 0.05,
    class_label: str = "Tumor", seed: int = 42,
) -> dict:
    """Compare standard Hilbert HCCI vs learned SFC HCCI."""
    rng = np.random.RandomState(seed)
    p = config.HILBERT_ORDER

    lats_hilbert = []
    lats_learned = []
    ranges_hilbert = []
    ranges_learned = []
    fp_hilbert = []
    fp_learned = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # Standard Hilbert HCCI
        h_sql, h_params = hcci.build_hcci_query(
            TABLE, sid, [class_label],
            x0, y0, x1, y1, w, h, p, use_direct=True,
        )
        h_rows, t_h = time_query(conn, h_sql, h_params)
        lats_hilbert.append(t_h)

        # Count ranges from standard Hilbert
        h_ranges = hcci.hilbert_ranges_direct(x0, y0, x1, y1, w, h, p, max_ranges=64)
        ranges_hilbert.append(len(h_ranges))

        # Learned SFC HCCI
        l_sql, l_params, l_n = build_learned_query(
            TABLE, sid, class_label,
            x0, y0, x1, y1,
            learned, w, h,
        )
        l_rows, t_l = time_query(conn, l_sql, l_params)
        lats_learned.append(t_l)
        ranges_learned.append(l_n)

        # Exact count for FP
        exact_sql = (
            f"SELECT centroid_x, centroid_y FROM {TABLE} "
            f"WHERE slide_id = %s AND class_label = %s "
            f"AND centroid_x >= %s AND centroid_x <= %s "
            f"AND centroid_y >= %s AND centroid_y <= %s"
        )
        with conn.cursor() as cur:
            cur.execute(exact_sql, (sid, class_label, x0, x1, y0, y1))
            exact_count = cur.rowcount

        if len(h_rows) > 0 and exact_count > 0:
            fp_hilbert.append((len(h_rows) - exact_count) / len(h_rows))
        else:
            fp_hilbert.append(0.0)
        if len(l_rows) > 0 and exact_count > 0:
            fp_learned.append((len(l_rows) - exact_count) / len(l_rows))
        else:
            fp_learned.append(0.0)

        if (trial + 1) % 50 == 0:
            print(f"    Trial {trial+1}/{n_trials}: "
                  f"Hilbert p50={np.median(lats_hilbert):.2f}ms ({np.mean(ranges_hilbert):.0f} ranges, "
                  f"FP={np.mean(fp_hilbert)*100:.1f}%)  "
                  f"Learned p50={np.median(lats_learned):.2f}ms ({np.mean(ranges_learned):.0f} ranges, "
                  f"FP={np.mean(fp_learned)*100:.1f}%)")

    stats_h = compute_stats(lats_hilbert)
    stats_l = compute_stats(lats_learned)
    _, p_val = wilcoxon_ranksum(lats_hilbert, lats_learned)

    return {
        'n_trials': n_trials,
        'hilbert': {
            'latency': stats_h,
            'avg_ranges': round(float(np.mean(ranges_hilbert)), 1),
            'fp_mean': round(float(np.mean(fp_hilbert)), 4),
            'fp_p95': round(float(np.percentile(fp_hilbert, 95)), 4),
        },
        'learned': {
            'latency': stats_l,
            'avg_ranges': round(float(np.mean(ranges_learned)), 1),
            'fp_mean': round(float(np.mean(fp_learned)), 4),
            'fp_p95': round(float(np.percentile(fp_learned, 95)), 4),
        },
        'learned_vs_hilbert_p50': round(stats_l['p50'] / stats_h['p50'], 3) if stats_h['p50'] > 0 else 0,
        'fp_reduction': round((np.mean(fp_hilbert) - np.mean(fp_learned)) / np.mean(fp_hilbert) * 100, 1) if np.mean(fp_hilbert) > 0 else 0,
        'wilcoxon_p': p_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Learned SFC vs standard Hilbert HCCI")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    parser.add_argument("--class-label", type=str, default="Tumor")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--staleness-test", action="store_true",
                        help="Run staleness degradation test")
    args = parser.parse_args()

    print("=" * 60)
    print("  Learned SFC Baseline: CDF-Equalized Hilbert vs Standard Hilbert")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())

    # Load metadata
    metadata_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Get slides
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        slides = [r[0] for r in cur.fetchall()]
    print(f"  Found {len(slides)} slides")

    # Pre-cache dims
    for sid in slides:
        get_dims(conn, sid, metadata)

    learned = LearnedSFC(p=config.HILBERT_ORDER, n_bins=1024)

    if not args.skip_setup:
        print("\n  Phase 1: Training CDFs and building learned index...")
        train_time = setup_learned_column(conn, learned, metadata)
        print(f"  Total CDF training time: {train_time:.1f}s")
        if args.setup_only:
            conn.close()
            return
    else:
        # Still need to train CDFs for query-time range computation
        print("  Training CDFs (query-time only, no index rebuild)...")
        for sid in slides:
            learned.train(conn, sid)
        print(f"  Trained {len(slides)} CDFs in {sum(learned.train_time.values()):.1f}s")

    conn.autocommit = False

    # Warmup
    print("  Warming up...")
    sid = slides[0]
    w, h = get_dims(conn, sid, metadata)
    for _ in range(10):
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {TABLE} WHERE slide_id = %s "
                f"AND composite_key >= 0 AND composite_key <= 1000 LIMIT 10",
                (sid,),
            )
            cur.fetchall()
            cur.execute(
                f"SELECT * FROM {TABLE} WHERE slide_id = %s "
                f"AND learned_composite_key >= 0 AND learned_composite_key <= 1000 LIMIT 10",
                (sid,),
            )
            cur.fetchall()

    # Run comparison
    print(f"\n  Phase 2: Running {args.trials} trial comparison...")
    t0 = time.time()
    results = run_comparison(
        conn, learned, slides, metadata,
        n_trials=args.trials,
        viewport_frac=args.viewport_frac,
        class_label=args.class_label,
    )
    total_time = time.time() - t0

    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    results['total_time_sec'] = round(total_time, 1)
    results['total_train_time_sec'] = round(sum(learned.train_time.values()), 1)
    results['n_slides'] = len(slides)

    # Staleness test
    if args.staleness_test:
        print("\n  Phase 3: Staleness degradation test...")
        stale_results = {}
        for sid in slides[:5]:  # Test on first 5 slides
            stale = test_staleness(conn, learned, sid)
            stale_results[sid] = stale
        results['staleness'] = stale_results

    # Summary
    print(f"\n  {'':>12} {'p50':>8} {'mean':>8} {'p95':>8} {'ranges':>8} {'FP%':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Hilbert':>12} {results['hilbert']['latency']['p50']:>8.2f} "
          f"{results['hilbert']['latency']['mean']:>8.2f} "
          f"{results['hilbert']['latency']['p95']:>8.1f} "
          f"{results['hilbert']['avg_ranges']:>8.1f} "
          f"{results['hilbert']['fp_mean']*100:>7.1f}%")
    print(f"  {'Learned':>12} {results['learned']['latency']['p50']:>8.2f} "
          f"{results['learned']['latency']['mean']:>8.2f} "
          f"{results['learned']['latency']['p95']:>8.1f} "
          f"{results['learned']['avg_ranges']:>8.1f} "
          f"{results['learned']['fp_mean']*100:>7.1f}%")
    print(f"\n  Learned/Hilbert p50 ratio: {results['learned_vs_hilbert_p50']:.3f}x")
    print(f"  FP reduction: {results['fp_reduction']:.1f}%")
    print(f"  Training time: {results['total_train_time_sec']:.1f}s "
          f"({results['total_train_time_sec']/len(slides):.2f}s/slide)")

    # Save
    path = os.path.join(config.RESULTS_DIR, 'raw', 'learned_sfc_benchmark.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {path}")

    conn.close()


if __name__ == "__main__":
    main()
