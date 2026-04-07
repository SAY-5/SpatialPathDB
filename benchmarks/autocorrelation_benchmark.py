"""Spatial autocorrelation sensitivity: HCCI speedup under varying class-space correlation.

Tests three correlation levels:
1. Random: class assigned uniformly at random (current labels)
2. Voronoi: class assigned by nearest Voronoi seed (realistic tumor-like boundaries)
3. Quadrant: class assigned by spatial quadrant (extreme correlation)

For each level, re-labels the data, rebuilds composite keys, and runs the
standard viewport benchmark.

Usage:
    python -m benchmarks.autocorrelation_benchmark --trials 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import psycopg2

from spdb import config, hcci, hilbert
from benchmarks.framework import compute_stats, time_query, wilcoxon_ranksum

TABLE = config.TABLE_SLIDE_ONLY
CLASSES = ['Epithelial', 'Stromal', 'Tumor', 'Lymphocyte']

# ---------------------------------------------------------------------------
# Label assignment strategies
# ---------------------------------------------------------------------------

def assign_random(xs, ys, rng):
    """Uniform random class assignment (baseline)."""
    return rng.choice(CLASSES, size=len(xs))


def assign_voronoi(xs, ys, rng, n_seeds=10):
    """Assign class by nearest Voronoi seed (realistic spatial correlation)."""
    # Random seed points
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    seed_x = rng.uniform(xmin, xmax, n_seeds)
    seed_y = rng.uniform(ymin, ymax, n_seeds)
    seed_class = [CLASSES[i % len(CLASSES)] for i in range(n_seeds)]

    # Assign each point to nearest seed
    labels = []
    # Vectorized: compute distances to all seeds
    # For large arrays, process in chunks
    chunk = 50000
    result = np.empty(len(xs), dtype=object)
    for start in range(0, len(xs), chunk):
        end = min(start + chunk, len(xs))
        x_chunk = xs[start:end, np.newaxis]  # (chunk, 1)
        y_chunk = ys[start:end, np.newaxis]
        # Distance to each seed
        dx = x_chunk - seed_x[np.newaxis, :]  # (chunk, n_seeds)
        dy = y_chunk - seed_y[np.newaxis, :]
        dist = dx ** 2 + dy ** 2
        nearest = np.argmin(dist, axis=1)
        for i, idx in enumerate(nearest):
            result[start + i] = seed_class[idx]
    return result


def assign_quadrant(xs, ys):
    """Assign class by spatial quadrant (extreme correlation)."""
    xmid = (xs.min() + xs.max()) / 2
    ymid = (ys.min() + ys.max()) / 2
    labels = np.where(
        (xs < xmid) & (ys < ymid), CLASSES[0],
        np.where(
            (xs >= xmid) & (ys < ymid), CLASSES[1],
            np.where(
                (xs < xmid) & (ys >= ymid), CLASSES[2],
                CLASSES[3]
            )
        )
    )
    return labels


# ---------------------------------------------------------------------------
# Relabeling and key recomputation
# ---------------------------------------------------------------------------

HILBERT_MASK = (1 << hcci.COMPOSITE_SHIFT) - 1  # lower 48 bits = hilbert_key


def relabel_random(conn, slide_id: str):
    """Random class assignment — two-pass: set composite_key, then match class_label."""
    with conn.cursor() as cur:
        # Pass 1: randomize the class bits in composite_key
        cur.execute(f"""
            UPDATE {TABLE}
            SET composite_key = (floor(random()*4)::bigint << {hcci.COMPOSITE_SHIFT})
                                | (composite_key & {HILBERT_MASK}::bigint)
            WHERE slide_id = %s
        """, (slide_id,))
        # Pass 2: set class_label to match the class bits in composite_key
        cur.execute(f"""
            UPDATE {TABLE}
            SET class_label = (ARRAY['Epithelial','Stromal','Tumor','Lymphocyte'])[
                1 + ((composite_key >> {hcci.COMPOSITE_SHIFT}) & 3)::int]
            WHERE slide_id = %s
        """, (slide_id,))
        updated = cur.rowcount
    conn.commit()
    return updated


def relabel_quadrant(conn, slide_id: str, slide_width, slide_height):
    """Quadrant class assignment — single UPDATE, no self-join."""
    xmid = slide_width / 2
    ymid = slide_height / 2
    enum_expr = f"""CASE
        WHEN centroid_x < %s AND centroid_y < %s THEN 0
        WHEN centroid_x >= %s AND centroid_y < %s THEN 1
        WHEN centroid_x < %s AND centroid_y >= %s THEN 2
        ELSE 3
    END"""
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {TABLE}
            SET composite_key = (({enum_expr})::bigint << {hcci.COMPOSITE_SHIFT})
                                | (composite_key & {HILBERT_MASK}::bigint),
                class_label = CASE
                    WHEN centroid_x < %s AND centroid_y < %s THEN 'Epithelial'
                    WHEN centroid_x >= %s AND centroid_y < %s THEN 'Stromal'
                    WHEN centroid_x < %s AND centroid_y >= %s THEN 'Tumor'
                    ELSE 'Lymphocyte'
                END
            WHERE slide_id = %s
        """, (xmid, ymid, xmid, ymid, xmid, ymid,
              xmid, ymid, xmid, ymid, xmid, ymid,
              slide_id))
        updated = cur.rowcount
    conn.commit()
    return updated


def relabel_voronoi(conn, slide_id: str, rng, slide_width, slide_height, n_seeds=10):
    """Voronoi class assignment — Python computes nearest seed, SQL updates via ctid."""
    seed_x = rng.uniform(0, slide_width, n_seeds)
    seed_y = rng.uniform(0, slide_height, n_seeds)
    seed_class = [CLASSES[i % len(CLASSES)] for i in range(n_seeds)]
    seed_enum = [hcci.CLASS_ENUM[c] for c in seed_class]

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT ctid, centroid_x, centroid_y FROM {TABLE} WHERE slide_id = %s",
            (slide_id,)
        )
        rows = cur.fetchall()

    if not rows:
        return 0

    ctids = [r[0] for r in rows]
    xs = np.array([float(r[1]) for r in rows], dtype=np.float64)
    ys = np.array([float(r[2]) for r in rows], dtype=np.float64)

    # Vectorized nearest-seed computation
    chunk = 50000
    new_enums = np.empty(len(xs), dtype=np.int64)
    new_labels = np.empty(len(xs), dtype=object)
    for start in range(0, len(xs), chunk):
        end = min(start + chunk, len(xs))
        dx = xs[start:end, np.newaxis] - seed_x[np.newaxis, :]
        dy = ys[start:end, np.newaxis] - seed_y[np.newaxis, :]
        dist = dx ** 2 + dy ** 2
        nearest = np.argmin(dist, axis=1)
        for i, idx in enumerate(nearest):
            new_labels[start + i] = seed_class[idx]
            new_enums[start + i] = seed_enum[idx]

    # Compute new composite keys: replace class bits, keep hilbert bits
    from psycopg2.extras import execute_values
    values = [(str(lbl), int(enum), str(ct))
              for lbl, enum, ct in zip(new_labels, new_enums, ctids)]

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS _relabel_tmp (lbl TEXT, enum INT, rid TID)")
        cur.execute("TRUNCATE _relabel_tmp")
        execute_values(
            cur,
            "INSERT INTO _relabel_tmp (lbl, enum, rid) VALUES %s",
            values,
            page_size=10000,
        )
        # slide_id filter ensures we only match within one partition (ctid is unique per partition)
        cur.execute(f"""
            UPDATE {TABLE} t
            SET class_label = z.lbl,
                composite_key = (z.enum::bigint << {hcci.COMPOSITE_SHIFT})
                                | (t.composite_key & {HILBERT_MASK}::bigint)
            FROM _relabel_tmp z
            WHERE t.ctid = z.rid AND t.slide_id = %s
        """, (slide_id,))
        updated = cur.rowcount
    conn.commit()
    return updated


# ---------------------------------------------------------------------------
# Benchmark runner
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


def run_benchmark(conn, slides, metadata, n_trials, viewport_frac, class_label, seed=42):
    """Run HCCI vs GiST benchmark and return stats."""
    rng = np.random.RandomState(seed)
    p = config.HILBERT_ORDER

    lats_hcci = []
    lats_gist = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)

        # HCCI
        h_sql, h_params = hcci.build_hcci_query(
            TABLE, sid, [class_label],
            x0, y0, x1, y1, w, h, p, use_direct=True,
        )
        _, t_h = time_query(conn, h_sql, h_params)
        lats_hcci.append(t_h)

        # GiST
        g_sql, g_params = hcci.build_baseline_bbox_query(
            TABLE, sid, [class_label],
            x0, y0, x1, y1,
        )
        _, t_g = time_query(conn, g_sql, g_params)
        lats_gist.append(t_g)

        if (trial + 1) % 50 == 0:
            print(f'    Trial {trial+1}/{n_trials}: '
                  f'HCCI p50={np.median(lats_hcci):.2f}ms, '
                  f'GiST p50={np.median(lats_gist):.2f}ms, '
                  f'speedup={np.median(lats_gist)/np.median(lats_hcci):.1f}x')

    stats_h = compute_stats(lats_hcci)
    stats_g = compute_stats(lats_gist)
    _, p_val = wilcoxon_ranksum(lats_hcci, lats_gist)

    return {
        'hcci': stats_h,
        'gist': stats_g,
        'speedup_p50': round(stats_g['p50'] / stats_h['p50'], 2) if stats_h['p50'] > 0 else 0,
        'speedup_mean': round(stats_g['mean'] / stats_h['mean'], 2) if stats_h['mean'] > 0 else 0,
        'wilcoxon_p': p_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Autocorrelation sensitivity benchmark")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--viewport-frac", type=float, default=0.05)
    parser.add_argument("--class-label", type=str, default="Tumor")
    parser.add_argument("--voronoi-seeds", type=int, default=10)
    parser.add_argument("--skip-relabel", action="store_true",
                        help="Skip relabeling, just run benchmark on current labels")
    parser.add_argument("--level", type=str, default=None,
                        help="Run only one level: random, voronoi, quadrant")
    args = parser.parse_args()

    print("=" * 60)
    print("  Autocorrelation Sensitivity: HCCI Speedup vs Class Correlation")
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

    level_names = ['random', 'voronoi', 'quadrant']
    if args.level:
        level_names = [args.level]

    all_results = {}

    for level_name in level_names:
        print(f"\n{'='*60}")
        print(f"  Level: {level_name.upper()}")
        print(f"{'='*60}")

        if not args.skip_relabel:
            # Drop ALL indexes to speed up bulk UPDATE (avoid per-row index maintenance)
            print("  Dropping indexes for fast relabeling...")
            conn.commit()  # end any open transaction before autocommit
            conn.autocommit = True
            with conn.cursor() as cur:
                for idx in ['idx_hcci_covering', 'idx_gist_partial_epithelial',
                            'idx_gist_partial_stromal', 'idx_gist_partial_tumor',
                            'idx_gist_partial_lymphocyte', 'idx_zorder_covering']:
                    cur.execute(f"DROP INDEX IF EXISTS {idx}")
                    print(f"    Dropped {idx}")
            conn.autocommit = False

            print(f"  Relabeling {len(slides)} slides ({level_name})...")
            rng = np.random.RandomState(42)
            total_updated = 0
            t0 = time.time()
            for i, sid in enumerate(slides):
                w, h = get_dims(conn, sid, metadata)
                if level_name == 'random':
                    updated = relabel_random(conn, sid)
                elif level_name == 'voronoi':
                    updated = relabel_voronoi(conn, sid, rng, w, h, n_seeds=args.voronoi_seeds)
                elif level_name == 'quadrant':
                    updated = relabel_quadrant(conn, sid, w, h)
                else:
                    raise ValueError(f"Unknown level: {level_name}")
                total_updated += updated
                if (i + 1) % 20 == 0 or i == len(slides) - 1:
                    elapsed = time.time() - t0
                    print(f"    Slide {i+1}/{len(slides)}: {total_updated:,} rows relabeled ({elapsed:.0f}s)")
            relabel_time = time.time() - t0
            print(f"  Relabeled {total_updated:,} rows in {relabel_time:.0f}s")

            # Recreate only the indexes needed for the benchmark:
            # HCCI covering index + GiST partial for the benchmark class
            print("  Recreating benchmark indexes...")
            conn.commit()
            conn.autocommit = True
            with conn.cursor() as cur:
                t_idx = time.time()
                print("    Creating idx_hcci_covering...")
                cur.execute(f"""CREATE INDEX idx_hcci_covering ON {TABLE}
                    USING btree (slide_id, composite_key)
                    INCLUDE (centroid_x, centroid_y, class_label, area)""")
                print(f"    idx_hcci_covering done ({time.time()-t_idx:.0f}s)")

                cls = args.class_label  # e.g., 'Tumor'
                idx_name = f"idx_gist_partial_{cls.lower()}"
                t_idx = time.time()
                print(f"    Creating {idx_name}...")
                cur.execute(f"""CREATE INDEX {idx_name} ON {TABLE}
                    USING gist (geom) WHERE (class_label = '{cls}')""")
                print(f"    {idx_name} done ({time.time()-t_idx:.0f}s)")
            conn.autocommit = False

            # ANALYZE after relabeling
            print("  ANALYZE...")
            if not conn.autocommit:
                conn.commit()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f"ANALYZE {TABLE}")
            conn.autocommit = False

        # Warmup
        print("  Warming up (10 queries)...")
        rng_warmup = np.random.RandomState(99)
        for _ in range(10):
            sid = rng_warmup.choice(slides)
            w, h = get_dims(conn, sid, metadata)
            sql, params = hcci.build_hcci_query(
                TABLE, sid, [args.class_label],
                0, 0, w * 0.2, h * 0.2,
                w, h, config.HILBERT_ORDER, use_direct=True,
            )
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cur.fetchall()

        # Run benchmark
        print(f"  Running {args.trials} trials...")
        result = run_benchmark(
            conn, slides, metadata,
            n_trials=args.trials,
            viewport_frac=args.viewport_frac,
            class_label=args.class_label,
        )
        result['level'] = level_name
        result['n_trials'] = args.trials
        all_results[level_name] = result

        print(f"\n  {level_name.upper()}: HCCI p50={result['hcci']['p50']:.2f}ms, "
              f"GiST p50={result['gist']['p50']:.2f}ms, "
              f"speedup={result['speedup_p50']:.1f}x (p50), "
              f"{result['speedup_mean']:.1f}x (mean)")

        # Save incrementally after each level to avoid data loss
        incr_path = os.path.join(config.RESULTS_DIR, 'raw', f'autocorrelation_{level_name}.json')
        os.makedirs(os.path.dirname(incr_path), exist_ok=True)
        with open(incr_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'level': level_name,
                'viewport_frac': args.viewport_frac,
                'class_label': args.class_label,
                'n_trials': args.trials,
                'n_slides': len(slides),
                'result': result,
            }, f, indent=2)
        print(f"  Saved to {incr_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  {'Level':>12} {'HCCI p50':>10} {'GiST p50':>10} {'Speedup p50':>12} {'Speedup mean':>13}")
    print(f"  {'-'*58}")
    for name, r in all_results.items():
        print(f"  {name:>12} {r['hcci']['p50']:>10.2f} {r['gist']['p50']:>10.2f} "
              f"{r['speedup_p50']:>12.1f}x {r['speedup_mean']:>12.1f}x")

    # Save
    output = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'viewport_frac': args.viewport_frac,
        'class_label': args.class_label,
        'n_trials': args.trials,
        'voronoi_seeds': args.voronoi_seeds,
        'n_slides': len(slides),
        'results': all_results,
    }
    path = os.path.join(config.RESULTS_DIR, 'raw', 'autocorrelation_benchmark.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {path}")

    # Rebuild all indexes to leave DB in clean state
    print("\n  Rebuilding all indexes to restore DB state...")
    if not conn.autocommit:
        conn.commit()
    conn.autocommit = True
    with conn.cursor() as cur:
        for idx in ['idx_hcci_covering', 'idx_gist_partial_epithelial',
                    'idx_gist_partial_stromal', 'idx_gist_partial_tumor',
                    'idx_gist_partial_lymphocyte', 'idx_zorder_covering']:
            cur.execute(f"DROP INDEX IF EXISTS {idx}")

        t_idx = time.time()
        print("    Creating idx_hcci_covering...")
        cur.execute(f"""CREATE INDEX idx_hcci_covering ON {TABLE}
            USING btree (slide_id, composite_key)
            INCLUDE (centroid_x, centroid_y, class_label, area)""")
        print(f"    done ({time.time()-t_idx:.0f}s)")

        for cls in CLASSES:
            idx_name = f"idx_gist_partial_{cls.lower()}"
            t_idx = time.time()
            print(f"    Creating {idx_name}...")
            cur.execute(f"""CREATE INDEX {idx_name} ON {TABLE}
                USING gist (geom) WHERE (class_label = '{cls}')""")
            print(f"    done ({time.time()-t_idx:.0f}s)")

        t_idx = time.time()
        print("    Creating idx_zorder_covering...")
        cur.execute(f"""CREATE INDEX idx_zorder_covering ON {TABLE}
            USING btree (slide_id, zorder_composite_key)
            INCLUDE (centroid_x, centroid_y, class_label, area)""")
        print(f"    done ({time.time()-t_idx:.0f}s)")
    conn.autocommit = False

    conn.close()


if __name__ == "__main__":
    main()
