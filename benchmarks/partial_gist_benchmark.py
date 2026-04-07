"""Partial GiST benchmark: compare HCCI vs partial GiST indexes vs standard GiST.

A partial GiST index is a class-specific spatial index:
    CREATE INDEX idx_gist_partial_tumor
    ON objects_slide_only USING gist(geom)
    WHERE class_label = 'Tumor';

This is the natural DBA response to class-filtered spatial queries: one
smaller GiST per class, so the planner can skip irrelevant rows earlier.
HCCI should still win because partial GiST requires heap fetches while
HCCI uses index-only scans.

Usage:
    python -m benchmarks.partial_gist_benchmark
    python -m benchmarks.partial_gist_benchmark --trials 100 --viewport-frac 0.05
    python -m benchmarks.partial_gist_benchmark --class-label Lymphocyte
    python -m benchmarks.partial_gist_benchmark --create-indexes
    python -m benchmarks.partial_gist_benchmark --drop-indexes
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2

from spdb import config, hcci
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results,
    time_query, time_query_buffers, parse_buffers,
    wilcoxon_ranksum, print_comparison,
)

TABLE = config.TABLE_SLIDE_ONLY

ALL_CLASSES = config.CLASS_LABELS  # ["Epithelial", "Stromal", "Tumor", "Lymphocyte"]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_metadata() -> dict:
    """Load ingestion metadata (slide_ids, object_counts, metas)."""
    path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
    with open(path) as f:
        return json.load(f)


def get_slide_dimensions(metadata: dict, slide_id: str) -> Tuple[float, float]:
    """Return (width, height) for a slide from metadata."""
    m = metadata["metas"][slide_id]
    return float(m["image_width"]), float(m["image_height"])


# ---------------------------------------------------------------------------
# Slide dimension helpers (with cache, fall back to DB if not in metadata)
# ---------------------------------------------------------------------------

_slide_dims_cache: Dict[str, Tuple[float, float]] = {}


def get_dims(conn, slide_id: str, metadata: dict = None) -> Tuple[float, float]:
    """Get slide dimensions, preferring metadata, falling back to DB."""
    if slide_id not in _slide_dims_cache:
        if metadata and slide_id in metadata.get("metas", {}):
            _slide_dims_cache[slide_id] = get_slide_dimensions(metadata, slide_id)
        else:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT MAX(centroid_x), MAX(centroid_y)
                    FROM {TABLE}
                    WHERE slide_id = %s
                """, (slide_id,))
                row = cur.fetchone()
            if row and row[0] and row[1]:
                _slide_dims_cache[slide_id] = (float(row[0]) * 1.05, float(row[1]) * 1.05)
            else:
                _slide_dims_cache[slide_id] = (100000.0, 100000.0)
    return _slide_dims_cache[slide_id]


def get_all_slides(conn) -> List[str]:
    """Get all distinct slide_ids from the SO table."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT slide_id FROM {TABLE}")
        return [r[0] for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Partial GiST index management
# ---------------------------------------------------------------------------

def create_partial_gist_indexes(conn):
    """Create one partial GiST index per class on the SO table."""
    print("\n  Creating partial GiST indexes...")
    with conn.cursor() as cur:
        for cls in ALL_CLASSES:
            idx_name = f"idx_gist_partial_{cls.lower()}"
            sql = (
                f"CREATE INDEX IF NOT EXISTS {idx_name} "
                f"ON {TABLE} USING gist(geom) "
                f"WHERE class_label = '{cls}'"
            )
            print(f"    {idx_name} ...", end=" ", flush=True)
            t0 = time.time()
            cur.execute(sql)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
        cur.execute(f"ANALYZE {TABLE}")
        print("    ANALYZE done")
    conn.commit()


def drop_partial_gist_indexes(conn):
    """Drop all partial GiST indexes."""
    print("\n  Dropping partial GiST indexes...")
    with conn.cursor() as cur:
        for cls in ALL_CLASSES:
            idx_name = f"idx_gist_partial_{cls.lower()}"
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
            print(f"    Dropped {idx_name}")
        cur.execute(f"ANALYZE {TABLE}")
        print("    ANALYZE done")
    conn.commit()


def check_partial_gist_indexes(conn) -> List[str]:
    """Return names of existing partial GiST indexes."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s
              AND indexname LIKE 'idx_gist_partial_%%'
        """, (TABLE,))
        return [r[0] for r in cur.fetchall()]


def get_index_sizes(conn) -> Dict[str, str]:
    """Get sizes of partial GiST indexes, HCCI covering index, and standard GiST."""
    sizes = {}
    with conn.cursor() as cur:
        # Partial GiST indexes
        for cls in ALL_CLASSES:
            idx_name = f"idx_gist_partial_{cls.lower()}"
            try:
                cur.execute(
                    "SELECT pg_size_pretty(pg_relation_size(%s))",
                    (idx_name,)
                )
                row = cur.fetchone()
                if row:
                    sizes[idx_name] = row[0]
            except Exception:
                conn.rollback()

        # HCCI covering index
        cur.execute("""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes
            WHERE tablename = %s
              AND indexdef LIKE '%%composite_key%%'
            LIMIT 1
        """, (TABLE,))
        row = cur.fetchone()
        if row:
            sizes[row[0]] = row[1]

        # Standard GiST index
        cur.execute("""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes
            WHERE tablename = %s
              AND indexdef LIKE '%%gist%%'
              AND indexname NOT LIKE 'idx_gist_partial_%%'
            LIMIT 1
        """, (TABLE,))
        row = cur.fetchone()
        if row:
            sizes[row[0]] = row[1]

    return sizes


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_hcci_query(
    slide_id: str,
    class_label: str,
    viewport: Tuple[float, float, float, float],
    slide_width: float,
    slide_height: float,
) -> Tuple[str, tuple]:
    """Build HCCI query: UNION ALL of composite_key range scans (index-only)."""
    x0, y0, x1, y1 = viewport
    return hcci.build_hcci_query(
        TABLE, slide_id, [class_label],
        x0, y0, x1, y1,
        slide_width, slide_height,
        config.HILBERT_ORDER,
        use_direct=True,
    )


def build_standard_gist_query(
    slide_id: str,
    class_label: str,
    viewport: Tuple[float, float, float, float],
) -> Tuple[str, tuple]:
    """Build standard GiST query: spatial scan + class_label post-filter."""
    x0, y0, x1, y1 = viewport
    return hcci.build_baseline_bbox_query(
        TABLE, slide_id, [class_label],
        x0, y0, x1, y1,
    )


def build_partial_gist_query(
    slide_id: str,
    class_label: str,
    viewport: Tuple[float, float, float, float],
) -> Tuple[str, tuple]:
    """Build query targeting the partial GiST index for a single class.

    Uses && (bbox overlap) for spatial filter on a class-specific partial
    GiST index. PostgreSQL automatically selects the partial index
    idx_gist_partial_{class} because the WHERE class_label = ? predicate
    matches the index's WHERE clause.

    For a single class, the class_label = %s predicate (not IN) gives the
    planner the clearest signal to use the partial index.
    """
    x0, y0, x1, y1 = viewport
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {TABLE} "
        f"WHERE slide_id = %s "
        f"  AND class_label = %s "
        f"  AND geom && ST_MakeEnvelope(%s, %s, %s, %s, 0)"
    )
    params = (slide_id, class_label, x0, y0, x1, y1)
    return sql, params


# ---------------------------------------------------------------------------
# EXPLAIN BUFFERS parsing: extract heap_fetches
# ---------------------------------------------------------------------------

def extract_heap_fetches(plan_json) -> int:
    """Walk EXPLAIN plan and sum Heap Fetches across all nodes."""
    if not plan_json or len(plan_json) == 0:
        return 0
    total = 0

    def _walk(node):
        nonlocal total
        total += node.get("Heap Fetches", 0)
        for child in node.get("Plans", []):
            _walk(child)

    root = plan_json[0]
    _walk(root.get("Plan", {}))
    return total


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    conn,
    metadata: dict,
    slides: List[str],
    class_label: str = "Tumor",
    n_trials: int = 200,
    viewport_frac: float = 0.05,
    io_trials: int = 30,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Run HCCI vs standard GiST vs partial GiST for viewport queries."""
    rng = np.random.RandomState(seed)

    class_sel = config.CLASS_DISTRIBUTION.get(class_label, 0.0)

    print(f"\n{'='*60}")
    print(f"  Partial GiST Benchmark")
    print(f"  Class: {class_label} (selectivity: {class_sel:.1%})")
    print(f"  Viewport fraction: {viewport_frac}")
    print(f"  Trials: {n_trials} (latency) + {io_trials} (I/O decomposition)")
    print(f"{'='*60}")

    # --- Phase 1: Latency measurement ---
    print(f"\n  Phase 1: Latency measurement ({n_trials} trials)...")
    lats_hcci: List[float] = []
    lats_gist: List[float] = []
    lats_partial: List[float] = []
    row_counts_hcci: List[int] = []
    row_counts_gist: List[int] = []
    row_counts_partial: List[int] = []
    range_counts: List[int] = []

    for trial in range(n_trials):
        sid = rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(rng.uniform(0, max(1, w - vw)))
        y0 = float(rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)
        viewport = (x0, y0, x1, y1)

        # 1) HCCI (UNION ALL of composite_key range scans)
        hcci_sql, hcci_params = build_hcci_query(
            sid, class_label, viewport, w, h,
        )
        hcci_rows, t_hcci = time_query(conn, hcci_sql, hcci_params)
        lats_hcci.append(t_hcci)
        row_counts_hcci.append(len(hcci_rows))

        # Count UNION ALL branches (= number of Hilbert ranges)
        n_ranges = hcci_sql.count("UNION ALL") + 1
        range_counts.append(n_ranges)

        # 2) Standard GiST (bbox overlap + class_label IN filter)
        gist_sql, gist_params = build_standard_gist_query(
            sid, class_label, viewport,
        )
        gist_rows, t_gist = time_query(conn, gist_sql, gist_params)
        lats_gist.append(t_gist)
        row_counts_gist.append(len(gist_rows))

        # 3) Partial GiST (class-specific spatial index)
        pg_sql, pg_params = build_partial_gist_query(
            sid, class_label, viewport,
        )
        pg_rows, t_partial = time_query(conn, pg_sql, pg_params)
        lats_partial.append(t_partial)
        row_counts_partial.append(len(pg_rows))

        if (trial + 1) % 50 == 0:
            print(f"    Trial {trial+1}/{n_trials}: "
                  f"HCCI p50={np.median(lats_hcci):.1f}ms  "
                  f"GiST p50={np.median(lats_gist):.1f}ms  "
                  f"Partial p50={np.median(lats_partial):.1f}ms")

    stats_hcci = compute_stats(lats_hcci)
    stats_gist = compute_stats(lats_gist)
    stats_partial = compute_stats(lats_partial)

    # Speedup ratios (GiST / X = how much faster X is than GiST)
    sp_hcci_vs_gist = stats_gist["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0
    sp_hcci_vs_partial = stats_partial["p50"] / stats_hcci["p50"] if stats_hcci["p50"] > 0 else 0
    sp_partial_vs_gist = stats_gist["p50"] / stats_partial["p50"] if stats_partial["p50"] > 0 else 0

    # Statistical significance
    _, wsr_hcci_gist = wilcoxon_ranksum(lats_hcci, lats_gist)
    _, wsr_hcci_partial = wilcoxon_ranksum(lats_hcci, lats_partial)
    _, wsr_partial_gist = wilcoxon_ranksum(lats_partial, lats_gist)

    # Save raw latencies
    tag = f"partial_gist_{class_label.lower()}"
    save_raw_latencies(lats_hcci, tag, "HCCI")
    save_raw_latencies(lats_gist, tag, "GiST")
    save_raw_latencies(lats_partial, tag, "PartialGiST")

    print(f"\n  Latency Results:")
    print(f"  {'':>16} {'p50':>8} {'p95':>8} {'mean':>8} {'std':>8} {'rows':>8}")
    print(f"  {'-'*58}")
    print(f"  {'HCCI':>16} {stats_hcci['p50']:>8.1f} {stats_hcci['p95']:>8.1f} "
          f"{stats_hcci['mean']:>8.1f} {stats_hcci['std']:>8.1f} "
          f"{np.mean(row_counts_hcci):>8.0f}")
    print(f"  {'GiST':>16} {stats_gist['p50']:>8.1f} {stats_gist['p95']:>8.1f} "
          f"{stats_gist['mean']:>8.1f} {stats_gist['std']:>8.1f} "
          f"{np.mean(row_counts_gist):>8.0f}")
    print(f"  {'Partial GiST':>16} {stats_partial['p50']:>8.1f} {stats_partial['p95']:>8.1f} "
          f"{stats_partial['mean']:>8.1f} {stats_partial['std']:>8.1f} "
          f"{np.mean(row_counts_partial):>8.0f}")
    print(f"\n  Speedup ratios (p50):")
    print(f"    HCCI vs GiST:         {sp_hcci_vs_gist:.2f}x  (p={wsr_hcci_gist:.2e})")
    print(f"    HCCI vs Partial GiST: {sp_hcci_vs_partial:.2f}x  (p={wsr_hcci_partial:.2e})")
    print(f"    Partial vs GiST:      {sp_partial_vs_gist:.2f}x  (p={wsr_partial_gist:.2e})")
    print(f"    Avg Hilbert ranges:   {np.mean(range_counts):.1f}")

    # --- Phase 2: I/O decomposition (EXPLAIN ANALYZE BUFFERS) ---
    print(f"\n  Phase 2: I/O decomposition ({io_trials} trials)...")
    io_rng = np.random.RandomState(seed + 999)

    hcci_buffers: List[dict] = []
    gist_buffers: List[dict] = []
    partial_buffers: List[dict] = []
    hcci_heap_fetches: List[int] = []
    gist_heap_fetches: List[int] = []
    partial_heap_fetches: List[int] = []

    for trial in range(io_trials):
        sid = io_rng.choice(slides)
        w, h = get_dims(conn, sid, metadata)

        vw = w * float(np.sqrt(viewport_frac))
        vh = h * float(np.sqrt(viewport_frac))
        x0 = float(io_rng.uniform(0, max(1, w - vw)))
        y0 = float(io_rng.uniform(0, max(1, h - vh)))
        x1 = float(x0 + vw)
        y1 = float(y0 + vh)
        viewport = (x0, y0, x1, y1)

        # HCCI
        hcci_sql, hcci_p = build_hcci_query(sid, class_label, viewport, w, h)
        hcci_plan, hb = time_query_buffers(conn, hcci_sql, hcci_p)
        hcci_buffers.append(hb)
        hcci_heap_fetches.append(extract_heap_fetches(hcci_plan))

        # Standard GiST
        gist_sql, gist_p = build_standard_gist_query(sid, class_label, viewport)
        gist_plan, gb = time_query_buffers(conn, gist_sql, gist_p)
        gist_buffers.append(gb)
        gist_heap_fetches.append(extract_heap_fetches(gist_plan))

        # Partial GiST
        pg_sql, pg_p = build_partial_gist_query(sid, class_label, viewport)
        pg_plan, pb = time_query_buffers(conn, pg_sql, pg_p)
        partial_buffers.append(pb)
        partial_heap_fetches.append(extract_heap_fetches(pg_plan))

    def _avg_buffers(buf_list: List[dict]) -> dict:
        keys = ["shared_hit", "shared_read", "heap_fetches",
                "rows_removed_by_filter", "actual_rows",
                "planning_time", "execution_time"]
        return {
            k: round(float(np.mean([b.get(k, 0) for b in buf_list])), 1)
            for k in keys
        }

    hcci_io = _avg_buffers(hcci_buffers)
    gist_io = _avg_buffers(gist_buffers)
    partial_io = _avg_buffers(partial_buffers)

    # Compute heap_fetches from EXPLAIN output (more reliable than parsed buffers)
    hcci_io["heap_fetches_explain"] = round(float(np.mean(hcci_heap_fetches)), 1)
    gist_io["heap_fetches_explain"] = round(float(np.mean(gist_heap_fetches)), 1)
    partial_io["heap_fetches_explain"] = round(float(np.mean(partial_heap_fetches)), 1)

    # Check scan types
    hcci_node_types = set()
    for b in hcci_buffers:
        hcci_node_types.update(b.get("node_types", []))
    partial_node_types = set()
    for b in partial_buffers:
        partial_node_types.update(b.get("node_types", []))
    gist_node_types = set()
    for b in gist_buffers:
        gist_node_types.update(b.get("node_types", []))

    hcci_index_only = "Index Only Scan" in hcci_node_types
    partial_has_heap = partial_io["heap_fetches_explain"] > 0

    # Buffer reduction percentages
    gist_total_blocks = gist_io["shared_hit"] + gist_io["shared_read"]
    hcci_total_blocks = hcci_io["shared_hit"] + hcci_io["shared_read"]
    partial_total_blocks = partial_io["shared_hit"] + partial_io["shared_read"]

    buf_reduction_hcci_vs_gist = (
        (1 - hcci_total_blocks / max(1, gist_total_blocks)) * 100
    )
    buf_reduction_partial_vs_gist = (
        (1 - partial_total_blocks / max(1, gist_total_blocks)) * 100
    )
    buf_reduction_hcci_vs_partial = (
        (1 - hcci_total_blocks / max(1, partial_total_blocks)) * 100
    )

    print(f"\n  I/O Decomposition Results:")
    print(f"  {'':>16} {'shared_hit':>11} {'shared_read':>12} {'heap_fetch':>11} "
          f"{'rows_filt':>10} {'exec_ms':>8}")
    print(f"  {'-'*70}")
    for label, io in [("HCCI", hcci_io), ("GiST", gist_io), ("Partial GiST", partial_io)]:
        print(f"  {label:>16} {io['shared_hit']:>11.0f} {io['shared_read']:>12.0f} "
              f"{io['heap_fetches_explain']:>11.0f} "
              f"{io['rows_removed_by_filter']:>10.0f} "
              f"{io['execution_time']:>8.1f}")

    print(f"\n  HCCI index-only scan:         {'YES' if hcci_index_only else 'NO'}")
    print(f"  HCCI heap fetches:            {hcci_io['heap_fetches_explain']:.0f} "
          f"{'(VERIFIED: zero heap access)' if hcci_io['heap_fetches_explain'] < 1 else '(WARNING: heap fetches -- run VACUUM)'}")
    print(f"  Partial GiST heap fetches:    {partial_io['heap_fetches_explain']:.0f} "
          f"{'(still requires heap access)' if partial_has_heap else '(unexpected: zero heap fetches)'}")
    print(f"\n  Buffer reduction HCCI vs GiST:    {buf_reduction_hcci_vs_gist:.1f}%")
    print(f"  Buffer reduction Partial vs GiST: {buf_reduction_partial_vs_gist:.1f}%")
    print(f"  Buffer reduction HCCI vs Partial: {buf_reduction_hcci_vs_partial:.1f}%")

    # --- Assemble results ---
    results = {
        "class_label": class_label,
        "class_selectivity": class_sel,
        "viewport_frac": viewport_frac,
        "n_trials": n_trials,
        "io_trials": io_trials,
        "table": TABLE,
        "hilbert_order": config.HILBERT_ORDER,
        "n_slides": len(slides),
        "avg_rows": {
            "hcci": round(float(np.mean(row_counts_hcci)), 1),
            "gist": round(float(np.mean(row_counts_gist)), 1),
            "partial_gist": round(float(np.mean(row_counts_partial)), 1),
        },
        "latency": {
            "hcci": stats_hcci,
            "gist": stats_gist,
            "partial_gist": stats_partial,
        },
        "speedup": {
            "hcci_vs_gist_p50": round(sp_hcci_vs_gist, 3),
            "hcci_vs_partial_p50": round(sp_hcci_vs_partial, 3),
            "partial_vs_gist_p50": round(sp_partial_vs_gist, 3),
            "hcci_vs_gist_mean": round(
                stats_gist["mean"] / stats_hcci["mean"], 3
            ) if stats_hcci["mean"] > 0 else 0,
            "hcci_vs_partial_mean": round(
                stats_partial["mean"] / stats_hcci["mean"], 3
            ) if stats_hcci["mean"] > 0 else 0,
            "partial_vs_gist_mean": round(
                stats_gist["mean"] / stats_partial["mean"], 3
            ) if stats_partial["mean"] > 0 else 0,
        },
        "significance": {
            "wilcoxon_hcci_vs_gist": wsr_hcci_gist,
            "wilcoxon_hcci_vs_partial": wsr_hcci_partial,
            "wilcoxon_partial_vs_gist": wsr_partial_gist,
        },
        "io_decomposition": {
            "hcci": hcci_io,
            "gist": gist_io,
            "partial_gist": partial_io,
            "hcci_index_only": hcci_index_only,
            "partial_gist_has_heap_fetches": partial_has_heap,
            "hcci_node_types": sorted(hcci_node_types),
            "gist_node_types": sorted(gist_node_types),
            "partial_gist_node_types": sorted(partial_node_types),
            "buffer_reduction_hcci_vs_gist_pct": round(buf_reduction_hcci_vs_gist, 1),
            "buffer_reduction_partial_vs_gist_pct": round(buf_reduction_partial_vs_gist, 1),
            "buffer_reduction_hcci_vs_partial_pct": round(buf_reduction_hcci_vs_partial, 1),
        },
        "hilbert_ranges": {
            "avg": round(float(np.mean(range_counts)), 1),
            "max": int(np.max(range_counts)),
            "min": int(np.min(range_counts)),
        },
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Partial GiST benchmark: HCCI vs partial GiST vs standard GiST"
    )
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of latency trials (default: 200)")
    parser.add_argument("--viewport-frac", type=float, default=0.05,
                        help="Viewport fraction of slide area (default: 0.05)")
    parser.add_argument("--class-label", type=str, default="Tumor",
                        help="Class label to benchmark (default: Tumor)")
    parser.add_argument("--io-trials", type=int, default=30,
                        help="Number of I/O decomposition trials (default: 30)")
    parser.add_argument("--create-indexes", action="store_true",
                        help="Create partial GiST indexes and exit")
    parser.add_argument("--drop-indexes", action="store_true",
                        help="Drop partial GiST indexes and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("  Partial GiST Benchmark")
    print("  HCCI (index-only) vs Partial GiST (per-class) vs Standard GiST")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())

    # --- Index management modes ---
    if args.create_indexes:
        conn.autocommit = True
        create_partial_gist_indexes(conn)
        existing = check_partial_gist_indexes(conn)
        print(f"\n  Partial GiST indexes: {existing}")
        sizes = get_index_sizes(conn)
        if sizes:
            print("  Index sizes:")
            for name, size in sorted(sizes.items()):
                print(f"    {name}: {size}")
        conn.close()
        return

    if args.drop_indexes:
        conn.autocommit = True
        drop_partial_gist_indexes(conn)
        print("  Done.")
        conn.close()
        return

    # --- Benchmark mode ---
    conn.autocommit = False

    # Load metadata
    print("\n  Loading metadata...")
    metadata = load_metadata()
    print(f"  Loaded metadata for {len(metadata.get('slide_ids', []))} slides")
    print(f"  Total objects: {metadata.get('total_objects', 'unknown'):,}")

    # Get slide list
    slides = get_all_slides(conn)
    print(f"  Found {len(slides)} slides in {TABLE}")

    # Pre-cache dimensions
    print("  Pre-caching slide dimensions...")
    for sid in slides:
        get_dims(conn, sid, metadata)

    # Verify partial GiST indexes exist
    existing = check_partial_gist_indexes(conn)
    target_idx = f"idx_gist_partial_{args.class_label.lower()}"
    if target_idx not in existing:
        print(f"\n  WARNING: Partial GiST index '{target_idx}' not found.")
        print(f"  Existing: {existing}")
        print(f"  Run with --create-indexes first, or results will use standard GiST.")
        print(f"  Continuing anyway...\n")
    else:
        print(f"  Partial GiST index found: {target_idx}")

    # Report index sizes
    sizes = get_index_sizes(conn)
    if sizes:
        print("\n  Index sizes:")
        for name, size in sorted(sizes.items()):
            print(f"    {name}: {size}")

    # Row count
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total_rows = cur.fetchone()[0]
    print(f"\n  Table {TABLE}: {total_rows:,} rows")

    # Warmup
    print("  Warming up...")
    for _ in range(10):
        sid = slides[0]
        w, h = get_dims(conn, sid, metadata)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {TABLE} WHERE slide_id = %s "
                f"AND geom && ST_MakeEnvelope(0, 0, %s, %s, 0) LIMIT 100",
                (sid, w * 0.1, h * 0.1),
            )
            cur.fetchall()

    t_start = time.time()

    # Run benchmark
    bench_results = run_benchmark(
        conn, metadata, slides,
        class_label=args.class_label,
        n_trials=args.trials,
        viewport_frac=args.viewport_frac,
        io_trials=args.io_trials,
    )

    total = time.time() - t_start

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    sp = bench_results["speedup"]
    print(f"  Class:          {args.class_label} "
          f"({bench_results['class_selectivity']:.1%} selectivity)")
    print(f"  Viewport:       {args.viewport_frac}")
    print(f"  Trials:         {args.trials}")
    print(f"")
    print(f"  HCCI vs GiST:          {sp['hcci_vs_gist_p50']:.2f}x (p50)")
    print(f"  HCCI vs Partial GiST:  {sp['hcci_vs_partial_p50']:.2f}x (p50)")
    print(f"  Partial vs GiST:       {sp['partial_vs_gist_p50']:.2f}x (p50)")
    io = bench_results["io_decomposition"]
    print(f"")
    print(f"  HCCI heap fetches:     {io['hcci']['heap_fetches_explain']:.0f}")
    print(f"  Partial heap fetches:  {io['partial_gist']['heap_fetches_explain']:.0f}")
    print(f"  GiST heap fetches:     {io['gist']['heap_fetches_explain']:.0f}")
    print(f"")
    print(f"  Buffer reduction HCCI vs GiST:    {io['buffer_reduction_hcci_vs_gist_pct']:.1f}%")
    print(f"  Buffer reduction HCCI vs Partial:  {io['buffer_reduction_hcci_vs_partial_pct']:.1f}%")
    print(f"  Total time: {total:.0f}s ({total/60:.1f}m)")

    # Save results
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_time_sec": round(total, 1),
        "total_rows": total_rows,
        "index_sizes": sizes,
        "benchmark": bench_results,
    }

    path = save_results(all_results, "partial_gist_benchmark")
    print(f"\n  Results saved to {path}")

    conn.close()


if __name__ == "__main__":
    main()
