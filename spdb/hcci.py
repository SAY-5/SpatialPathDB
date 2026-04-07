"""Hilbert-Composite Covering Indexes (HCCI).

Fuses spatial location and categorical attributes into a single B-tree
covering index, enabling multi-attribute spatial queries via index-only
scans with zero heap access and analytically bounded false positives.

Composite key layout (BIGINT, 64 bits):
    [bits 63..48] class_label enum (0-15)
    [bits 47..0]  Hilbert key (up to 2^48 values)

Within each class segment, objects are ordered by Hilbert curve, so a
B-tree range scan on composite_key retrieves all objects of a given class
in a spatial region as a contiguous leaf-page scan.

The covering index:
    CREATE INDEX ... ON table (slide_id, composite_key)
    INCLUDE (centroid_x, centroid_y, class_label, area)

enables index-only scans: PostgreSQL reads B-tree leaf pages (which
contain the INCLUDEd columns) and never touches the heap.  The I/O
reduction vs GiST is proportional to class selectivity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from spdb import config, hilbert

# ---------------------------------------------------------------------------
# Class label enum
# ---------------------------------------------------------------------------

CLASS_ENUM: Dict[str, int] = {
    "Epithelial": 0,
    "Stromal": 1,
    "Tumor": 2,
    "Lymphocyte": 3,
}

CLASS_ENUM_REVERSE: Dict[int, str] = {v: k for k, v in CLASS_ENUM.items()}

COMPOSITE_SHIFT = 48


def class_to_enum(label: str, class_enum: Optional[Dict[str, int]] = None) -> int:
    enum = class_enum if class_enum is not None else CLASS_ENUM
    return enum.get(label, len(enum))


# ---------------------------------------------------------------------------
# Composite key encoding
# ---------------------------------------------------------------------------

def compute_composite_key(class_label: str, hilbert_key: int,
                          class_enum: Optional[Dict[str, int]] = None) -> int:
    """Encode (class_label, hilbert_key) into a single BIGINT.

    Upper 16 bits: class enum.  Lower 48 bits: Hilbert key.
    """
    return (class_to_enum(class_label, class_enum) << COMPOSITE_SHIFT) | (hilbert_key & 0xFFFFFFFFFFFF)


def decode_composite_key(ck: int) -> Tuple[int, int]:
    """Decode composite key back to (class_enum, hilbert_key)."""
    cls = (ck >> COMPOSITE_SHIFT) & 0xFFFF
    hk = ck & 0xFFFFFFFFFFFF
    return cls, hk


# ---------------------------------------------------------------------------
# Hilbert key ranges for a viewport
# ---------------------------------------------------------------------------

def _hilbert_key_ranges(
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    p: int, num_buckets: int,
) -> List[Tuple[int, int]]:
    """Compute contiguous Hilbert key ranges covering a viewport (bucket-based).

    Uses candidate_buckets_for_bbox to find touched buckets, converts
    bucket IDs to Hilbert key ranges, and merges adjacent ranges.

    WARNING: coarse when num_buckets is small (e.g., 2-5 for small slides).
    Prefer hilbert_ranges_direct() for HCCI queries.

    Returns list of (h_lo, h_hi) where h_hi is EXCLUSIVE.
    """
    total_cells = 1 << (2 * p)
    bucket_ids = hilbert.candidate_buckets_for_bbox(
        x0, y0, x1, y1,
        slide_width, slide_height,
        p, num_buckets,
    )
    if not bucket_ids:
        return [(0, total_cells)]

    ranges: List[Tuple[int, int]] = []
    for b in sorted(bucket_ids):
        lo = b * total_cells // num_buckets
        hi = (b + 1) * total_cells // num_buckets
        if ranges and ranges[-1][1] >= lo:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
        else:
            ranges.append((lo, hi))
    return ranges


def hilbert_ranges_direct(
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    p: int,
    max_ranges: int = 64,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> List[Tuple[int, int]]:
    """Compute exact Hilbert key ranges for a viewport, bypassing buckets.

    Enumerates all Hilbert grid cells covered by the viewport, computes
    their Hilbert indices via vectorized encode_batch, sorts, and merges
    into contiguous ranges.  If the result exceeds *max_ranges*, the
    smallest inter-range gaps are filled until the count is within limit.

    FP rate is determined solely by Hilbert cell boundary effects — no
    bucket discretization error.

    x_origin, y_origin: coordinate system origin. For pathology data this
    is (0, 0); for WGS84 data this is (x_min, y_min) of the bounding box.

    Returns list of (h_lo, h_hi) where h_hi is EXCLUSIVE.
    """
    n = 1 << p
    gx0 = max(0, int((x0 - x_origin) * n / slide_width))
    gx1 = min(n - 1, int((x1 - x_origin) * n / slide_width))
    gy0 = max(0, int((y0 - y_origin) * n / slide_height))
    gy1 = min(n - 1, int((y1 - y_origin) * n / slide_height))

    if gx0 > gx1 or gy0 > gy1:
        return [(0, 1 << (2 * p))]

    # Enumerate all grid cells in the viewport
    gxs = np.arange(gx0, gx1 + 1, dtype=np.int64)
    gys = np.arange(gy0, gy1 + 1, dtype=np.int64)
    gx_mesh, gy_mesh = np.meshgrid(gxs, gys)
    gx_flat = gx_mesh.ravel()
    gy_flat = gy_mesh.ravel()

    # Vectorized Hilbert encoding
    h_indices = hilbert.encode_batch(gx_flat, gy_flat, p)
    h_sorted = np.sort(h_indices)

    if len(h_sorted) == 0:
        return [(0, 1 << (2 * p))]

    # Merge contiguous indices using numpy diff
    diffs = np.diff(h_sorted)
    gap_idx = np.where(diffs > 1)[0]

    starts = np.empty(len(gap_idx) + 1, dtype=np.int64)
    ends = np.empty(len(gap_idx) + 1, dtype=np.int64)
    starts[0] = h_sorted[0]
    ends[-1] = h_sorted[-1] + 1  # exclusive
    if len(gap_idx) > 0:
        ends[:-1] = h_sorted[gap_idx] + 1
        starts[1:] = h_sorted[gap_idx + 1]

    ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]

    # If too many ranges, merge smallest gaps to stay under max_ranges
    while len(ranges) > max_ranges:
        # Find the smallest gap between consecutive ranges
        gaps = [(ranges[i + 1][0] - ranges[i][1], i) for i in range(len(ranges) - 1)]
        gaps.sort()
        _, merge_i = gaps[0]
        # Merge ranges[merge_i] and ranges[merge_i + 1]
        merged = (ranges[merge_i][0], ranges[merge_i + 1][1])
        ranges = ranges[:merge_i] + [merged] + ranges[merge_i + 2:]

    return ranges


def hilbert_ranges_at_order(
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    query_order: int,
    index_order: int = 8,
    max_ranges: int = 64,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> Tuple[List[Tuple[int, int]], int]:
    """Compute Hilbert ranges at query_order, mapped to index_order space.

    When query_order < index_order (e.g., p=6 querying p=8 index):
      Each p=6 cell contains 2^(2*(8-6)) = 16 p=8 cells.
      Enumerate p=6 cells in viewport, expand each to all child p=8 cells,
      merge into ranges.  Result: coarser ranges, fewer of them, higher FP.

    When query_order > index_order (e.g., p=10 querying p=8 index):
      Each p=10 cell maps to a parent p=8 cell by dividing coordinates.
      Deduplicate and merge.  Result: identical to index_order (can't
      exceed index resolution).

    When query_order == index_order:
      Same as hilbert_ranges_direct.

    Returns (ranges, n_cells_enumerated) where ranges is list of (h_lo, h_hi)
    with h_hi EXCLUSIVE, in index_order Hilbert space.
    """
    if query_order == index_order:
        ranges = hilbert_ranges_direct(
            x0, y0, x1, y1, slide_width, slide_height,
            index_order, max_ranges, x_origin, y_origin,
        )
        # Count cells at this order
        n_q = 1 << query_order
        gx0 = max(0, int((x0 - x_origin) * n_q / slide_width))
        gx1 = min(n_q - 1, int((x1 - x_origin) * n_q / slide_width))
        gy0 = max(0, int((y0 - y_origin) * n_q / slide_height))
        gy1 = min(n_q - 1, int((y1 - y_origin) * n_q / slide_height))
        n_cells = max(0, (gx1 - gx0 + 1) * (gy1 - gy0 + 1))
        return ranges, n_cells

    n_q = 1 << query_order
    n_i = 1 << index_order

    # Compute viewport grid cells at query_order
    gx0 = max(0, int((x0 - x_origin) * n_q / slide_width))
    gx1 = min(n_q - 1, int((x1 - x_origin) * n_q / slide_width))
    gy0 = max(0, int((y0 - y_origin) * n_q / slide_height))
    gy1 = min(n_q - 1, int((y1 - y_origin) * n_q / slide_height))

    if gx0 > gx1 or gy0 > gy1:
        return [(0, 1 << (2 * index_order))], 0

    n_query_cells = (gx1 - gx0 + 1) * (gy1 - gy0 + 1)

    if query_order < index_order:
        # Expand each query-order cell to all its child cells at index_order
        scale = 1 << (index_order - query_order)
        all_index_cells = []
        for qx in range(gx0, gx1 + 1):
            for qy in range(gy0, gy1 + 1):
                # This query cell maps to scale x scale child cells
                ix_base = qx * scale
                iy_base = qy * scale
                for dx in range(scale):
                    ix = ix_base + dx
                    if ix >= n_i:
                        continue
                    for dy in range(scale):
                        iy = iy_base + dy
                        if iy >= n_i:
                            continue
                        all_index_cells.append((ix, iy))

        if not all_index_cells:
            return [(0, 1 << (2 * index_order))], n_query_cells

        # Vectorized Hilbert encoding at index_order
        ixs = np.array([c[0] for c in all_index_cells], dtype=np.int64)
        iys = np.array([c[1] for c in all_index_cells], dtype=np.int64)
        h_indices = hilbert.encode_batch(ixs, iys, index_order)

    else:
        # query_order > index_order: map to parent cells at index_order
        scale = 1 << (query_order - index_order)
        parent_cells = set()
        for qx in range(gx0, gx1 + 1):
            for qy in range(gy0, gy1 + 1):
                ix = min(qx // scale, n_i - 1)
                iy = min(qy // scale, n_i - 1)
                parent_cells.add((ix, iy))

        if not parent_cells:
            return [(0, 1 << (2 * index_order))], n_query_cells

        cells_list = list(parent_cells)
        ixs = np.array([c[0] for c in cells_list], dtype=np.int64)
        iys = np.array([c[1] for c in cells_list], dtype=np.int64)
        h_indices = hilbert.encode_batch(ixs, iys, index_order)

    # Sort and merge into ranges
    h_sorted = np.sort(h_indices)
    h_sorted = np.unique(h_sorted)

    if len(h_sorted) == 0:
        return [(0, 1 << (2 * index_order))], n_query_cells

    diffs = np.diff(h_sorted)
    gap_idx = np.where(diffs > 1)[0]

    starts = np.empty(len(gap_idx) + 1, dtype=np.int64)
    ends = np.empty(len(gap_idx) + 1, dtype=np.int64)
    starts[0] = h_sorted[0]
    ends[-1] = h_sorted[-1] + 1
    if len(gap_idx) > 0:
        ends[:-1] = h_sorted[gap_idx] + 1
        starts[1:] = h_sorted[gap_idx + 1]

    ranges = [(int(s), int(e)) for s, e in zip(starts, ends)]

    # Merge smallest gaps if too many ranges
    while len(ranges) > max_ranges:
        gaps = [(ranges[i + 1][0] - ranges[i][1], i) for i in range(len(ranges) - 1)]
        gaps.sort()
        _, merge_i = gaps[0]
        merged = (ranges[merge_i][0], ranges[merge_i + 1][1])
        ranges = ranges[:merge_i] + [merged] + ranges[merge_i + 2:]

    return ranges, n_query_cells


# ---------------------------------------------------------------------------
# Composite key ranges for a viewport + class filter
# ---------------------------------------------------------------------------

def composite_key_ranges_for_viewport(
    class_labels: List[str],
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    hilbert_order: int = config.HILBERT_ORDER,
    num_buckets: int = 30,
    use_direct: bool = True,
    class_enum: Optional[Dict[str, int]] = None,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> List[Tuple[int, int]]:
    """Build composite key ranges for a phenotype-filtered viewport query.

    For each requested class label, generates composite key ranges by
    OR-ing the class enum into the upper bits of each Hilbert range.

    When use_direct=True (default), computes exact Hilbert ranges from
    viewport coordinates without bucket discretization.  When False,
    uses the legacy bucket-based approach for SPDB compatibility.

    Returns list of (composite_lo, composite_hi) tuples, hi EXCLUSIVE.
    """
    if use_direct:
        h_ranges = hilbert_ranges_direct(
            x0, y0, x1, y1,
            slide_width, slide_height,
            hilbert_order,
            x_origin=x_origin,
            y_origin=y_origin,
        )
    else:
        h_ranges = _hilbert_key_ranges(
            x0, y0, x1, y1,
            slide_width, slide_height,
            hilbert_order, num_buckets,
        )
    ck_ranges: List[Tuple[int, int]] = []
    for label in class_labels:
        enum_val = class_to_enum(label, class_enum)
        prefix = enum_val << COMPOSITE_SHIFT
        for h_lo, h_hi in h_ranges:
            ck_ranges.append((prefix | h_lo, prefix | h_hi))
    return ck_ranges


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------

def build_hcci_query(
    table_name: str,
    slide_id: str,
    class_labels: List[str],
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    hilbert_order: int = config.HILBERT_ORDER,
    num_buckets: int = 30,
    use_direct: bool = True,
    class_enum: Optional[Dict[str, int]] = None,
    id_column: str = "slide_id",
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> Tuple[str, tuple]:
    """Build an HCCI query: pure B-tree index-only scan, no ST_Intersects.

    Uses UNION ALL (not OR) so each range gets its own Index Only Scan.
    OR clauses trigger PostgreSQL's BitmapOr path which forces a Bitmap
    Heap Scan, defeating the covering index.  UNION ALL preserves the
    Index Only Scan on every branch with Heap Fetches: 0.

    The SELECT columns must match the covering index's INCLUDE list so
    PostgreSQL uses an index-only scan.

    Returns (sql_string, params_tuple).
    """
    ck_ranges = composite_key_ranges_for_viewport(
        class_labels, x0, y0, x1, y1,
        slide_width, slide_height,
        hilbert_order, num_buckets,
        use_direct=use_direct,
        class_enum=class_enum,
        x_origin=x_origin,
        y_origin=y_origin,
    )
    if not ck_ranges:
        sql = (
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE {id_column} = %s AND FALSE"
        )
        return sql, (slide_id,)

    # Single range: no need for UNION ALL
    if len(ck_ranges) == 1:
        lo, hi = ck_ranges[0]
        sql = (
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE {id_column} = %s "
            f"AND composite_key >= %s AND composite_key < %s"
        )
        return sql, (slide_id, lo, hi)

    # Multiple ranges: UNION ALL so each branch gets Index Only Scan
    branches = []
    params: list = []
    for lo, hi in ck_ranges:
        branches.append(
            f"SELECT centroid_x, centroid_y, class_label, area "
            f"FROM {table_name} "
            f"WHERE {id_column} = %s "
            f"AND composite_key >= %s AND composite_key < %s"
        )
        params.extend([slide_id, lo, hi])

    sql = " UNION ALL ".join(branches)
    return sql, tuple(params)


def build_baseline_gist_query(
    table_name: str,
    slide_id: str,
    class_labels: List[str],
    x0: float, y0: float, x1: float, y1: float,
    id_column: str = "slide_id",
    srid: int = 0,
) -> Tuple[str, tuple]:
    """Build the GiST baseline query: spatial scan + class_label post-filter.

    Uses ST_Intersects for exact spatial filtering and IN (...) for class.
    """
    placeholders = ", ".join(["%s"] * len(class_labels))
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {table_name} "
        f"WHERE {id_column} = %s "
        f"  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, {srid})) "
        f"  AND class_label IN ({placeholders})"
    )
    params = (slide_id, x0, y0, x1, y1) + tuple(class_labels)
    return sql, params


def build_baseline_gist_query_all_classes(
    table_name: str,
    slide_id: str,
    x0: float, y0: float, x1: float, y1: float,
    id_column: str = "slide_id",
    srid: int = 0,
) -> Tuple[str, tuple]:
    """GiST baseline without class filter (for control comparison)."""
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {table_name} "
        f"WHERE {id_column} = %s "
        f"  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, {srid}))"
    )
    return sql, (slide_id, x0, y0, x1, y1)


def build_baseline_bbox_query(
    table_name: str,
    slide_id: str,
    class_labels: List[str],
    x0: float, y0: float, x1: float, y1: float,
    id_column: str = "slide_id",
    srid: int = 0,
) -> Tuple[str, tuple]:
    """GiST baseline using && (bbox overlap) instead of ST_Intersects.

    Isolates heap I/O cost from geometry computation cost.
    The && operator uses bounding-box overlap only — no geometry
    deserialization or computational geometry.
    """
    placeholders = ", ".join(["%s"] * len(class_labels))
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {table_name} "
        f"WHERE {id_column} = %s "
        f"  AND geom && ST_MakeEnvelope(%s, %s, %s, %s, {srid}) "
        f"  AND class_label IN ({placeholders})"
    )
    params = (slide_id, x0, y0, x1, y1) + tuple(class_labels)
    return sql, params


def build_baseline_bbox_query_all_classes(
    table_name: str,
    slide_id: str,
    x0: float, y0: float, x1: float, y1: float,
    id_column: str = "slide_id",
    srid: int = 0,
) -> Tuple[str, tuple]:
    """GiST baseline using && (bbox overlap) without class filter."""
    sql = (
        f"SELECT centroid_x, centroid_y, class_label, area "
        f"FROM {table_name} "
        f"WHERE {id_column} = %s "
        f"  AND geom && ST_MakeEnvelope(%s, %s, %s, %s, {srid})"
    )
    return sql, (slide_id, x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# False positive analysis
# ---------------------------------------------------------------------------

def false_positive_rate(viewport_frac: float, hilbert_order: int) -> float:
    """Theoretical upper bound on FP rate from Hilbert cell boundaries.

    A viewport of fraction f at order p covers ~sqrt(f)*2^p cells per
    dimension.  Boundary cells (perimeter) may return objects outside
    the exact viewport.

    FP_rate = perimeter_cells / total_cells
            = 4 * sqrt(f) * 2^p / (f * 4^p)
            = 4 / (sqrt(f) * 2^p)

    At p=8, f=0.05: FP_rate ~ 4 / (0.2236 * 256) ~ 7.0%
    At p=8, f=0.01: FP_rate ~ 4 / (0.1 * 256) ~ 15.6%
    At p=10, f=0.05: FP_rate ~ 4 / (0.2236 * 1024) ~ 1.7%
    """
    n = 1 << hilbert_order
    return 4.0 / (math.sqrt(viewport_frac) * n)


def measure_false_positive_rate(
    conn,
    table_name: str,
    slide_id: str,
    class_labels: List[str],
    x0: float, y0: float, x1: float, y1: float,
    slide_width: float, slide_height: float,
    hilbert_order: int = config.HILBERT_ORDER,
    num_buckets: int = 30,
    use_direct: bool = True,
    id_column: str = "slide_id",
    class_enum: Optional[Dict[str, int]] = None,
    srid: int = 0,
    x_origin: float = 0.0,
    y_origin: float = 0.0,
) -> dict:
    """Empirically measure FP rate by comparing HCCI vs exact results.

    Runs the HCCI query (no ST_Intersects) and the exact GiST query
    (with ST_Intersects) on the same viewport, and counts how many
    HCCI results fall outside the exact viewport.
    """
    from benchmarks.framework import time_query

    hcci_sql, hcci_params = build_hcci_query(
        table_name, slide_id, class_labels,
        x0, y0, x1, y1, slide_width, slide_height,
        hilbert_order, num_buckets,
        use_direct=use_direct,
        class_enum=class_enum,
        id_column=id_column,
        x_origin=x_origin,
        y_origin=y_origin,
    )
    exact_sql, exact_params = build_baseline_gist_query(
        table_name, slide_id, class_labels,
        x0, y0, x1, y1,
        id_column=id_column,
        srid=srid,
    )

    hcci_rows, _ = time_query(conn, hcci_sql, hcci_params)
    exact_rows, _ = time_query(conn, exact_sql, exact_params)

    n_hcci = len(hcci_rows)
    n_exact = len(exact_rows)
    n_fp = max(0, n_hcci - n_exact)
    fp_rate = n_fp / n_hcci if n_hcci > 0 else 0.0

    return {
        "hcci_count": n_hcci,
        "exact_count": n_exact,
        "false_positives": n_fp,
        "fp_rate": round(fp_rate, 4),
        "theoretical_bound": round(
            false_positive_rate(
                max(0.001, (x1 - x0) * (y1 - y0) / (slide_width * slide_height)),
                hilbert_order,
            ), 4
        ),
    }


# ---------------------------------------------------------------------------
# Cost model: HCCI vs GiST
# ---------------------------------------------------------------------------

def hcci_cost_model(
    n_objects: int,
    viewport_frac: float,
    class_selectivity: float,
    hilbert_order: int = 8,
    index_tuple_bytes: int = 48,
    heap_tuple_bytes: int = 120,
    page_size: int = 8192,
    random_page_ms: float = 0.1,
    seq_page_ms: float = 0.01,
    cpu_per_tuple_ms: float = 0.001,
    cpu_filter_ms: float = 0.0005,
) -> dict:
    """Predict I/O cost for HCCI vs GiST on a phenotype-filtered viewport.

    GiST path: GiST index scan (all classes in viewport) -> heap fetch
               -> CPU filter on class_label.
    HCCI path: B-tree index-only scan (only target class in viewport).
               No heap access. No CPU filter.

    Returns cost breakdown and predicted speedup.
    """
    viewport_objects = n_objects * viewport_frac
    matching_objects = viewport_objects * class_selectivity

    # GiST: reads all viewport objects from heap, filters to matching
    gist_heap_pages = math.ceil(viewport_objects * heap_tuple_bytes / page_size)
    gist_index_depth = max(1, math.ceil(math.log(max(1, n_objects)) / math.log(100)))
    gist_index_pages = gist_index_depth + 1
    gist_io_ms = gist_index_pages * random_page_ms + gist_heap_pages * seq_page_ms
    gist_cpu_ms = viewport_objects * cpu_per_tuple_ms + viewport_objects * cpu_filter_ms
    gist_total = gist_io_ms + gist_cpu_ms

    # HCCI: index-only scan, reads only matching class's leaf pages
    hcci_leaf_pages = math.ceil(matching_objects * index_tuple_bytes / page_size)
    hcci_index_depth = max(1, math.ceil(
        math.log(max(1, n_objects * class_selectivity)) / math.log(100)
    ))
    hcci_io_ms = hcci_index_depth * random_page_ms + hcci_leaf_pages * seq_page_ms
    hcci_cpu_ms = matching_objects * cpu_per_tuple_ms
    hcci_total = hcci_io_ms + hcci_cpu_ms

    speedup = gist_total / hcci_total if hcci_total > 0 else float("inf")

    # Crossover: class_selectivity at which HCCI = GiST
    # Approximate: HCCI wins when selectivity < heap_tuple_bytes / index_tuple_bytes
    crossover_selectivity = heap_tuple_bytes / (heap_tuple_bytes + index_tuple_bytes)

    return {
        "gist": {
            "viewport_objects": round(viewport_objects),
            "matching_objects": round(matching_objects),
            "heap_pages": gist_heap_pages,
            "index_pages": gist_index_pages,
            "io_ms": round(gist_io_ms, 3),
            "cpu_ms": round(gist_cpu_ms, 3),
            "total_ms": round(gist_total, 3),
        },
        "hcci": {
            "leaf_pages": hcci_leaf_pages,
            "index_depth": hcci_index_depth,
            "io_ms": round(hcci_io_ms, 3),
            "cpu_ms": round(hcci_cpu_ms, 3),
            "total_ms": round(hcci_total, 3),
        },
        "speedup": round(speedup, 2),
        "class_selectivity": class_selectivity,
        "crossover_selectivity": round(crossover_selectivity, 3),
    }


# ---------------------------------------------------------------------------
# DDL helpers
# ---------------------------------------------------------------------------

def composite_key_update_sql() -> str:
    """SQL to populate the composite_key column."""
    return """
        UPDATE {table} SET composite_key = (
            CASE class_label
                WHEN 'Epithelial'  THEN 0::bigint
                WHEN 'Stromal'    THEN 1::bigint
                WHEN 'Tumor'      THEN 2::bigint
                WHEN 'Lymphocyte' THEN 3::bigint
                ELSE 4::bigint
            END << 48
        ) | hilbert_key
    """


def covering_index_ddl(table_name: str, index_name: str = "idx_hcci_covering") -> str:
    """DDL for the covering B-tree index."""
    return (
        f"CREATE INDEX {index_name} ON {table_name} "
        f"(slide_id, composite_key) "
        f"INCLUDE (centroid_x, centroid_y, class_label, area)"
    )
