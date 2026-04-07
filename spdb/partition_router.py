"""Application-level partition routing for SPDB.

Core systems contribution: bypasses the PostgreSQL query planner's O(B_total)
constraint-exclusion overhead by computing candidate Hilbert buckets in Python
and issuing queries directly against child partition tables via UNION ALL.

When querying the parent ``objects_spdb`` table, the planner must evaluate
CHECK constraints on every one of the ~4,000 child partitions to determine
which to scan -- even when the WHERE clause can only match a handful.  This
constraint-exclusion pass dominates planning time once partition counts exceed
a few hundred (see paper Section 5.3).

``PartitionRouter`` eliminates that bottleneck:

1.  At init, it introspects ``pg_inherits`` / ``pg_class`` to build a
    mapping from ``(slide_id, bucket_id)`` to physical child-table names.
2.  At query time, ``candidate_buckets_for_bbox()`` (or a kNN expansion
    strategy) identifies the O(B_hit) relevant buckets in microseconds.
3.  A UNION ALL query targeting only those child tables is assembled and
    returned.  The planner sees *only* the targeted partitions, so planning
    is O(B_hit) instead of O(B_total).

Typical B_hit is 1--6 for a 5% viewport; B_total can exceed 4,000.
"""

from __future__ import annotations

import re
import time
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2

from spdb import config, hilbert


# ---------------------------------------------------------------------------
# Partition map construction
# ---------------------------------------------------------------------------

def _safe_slide_id(slide_id: str) -> str:
    """Apply the same sanitisation used by schema.py when naming partitions."""
    return slide_id.replace("-", "_").replace(".", "_")


def _build_partition_map(
    conn,
    parent_table: str = config.TABLE_SPDB,
) -> Dict[str, Dict[int, str]]:
    """Discover the two-level partition hierarchy from the catalogue.

    Queries ``pg_inherits`` twice: once for Level-1 (slide) partitions, then
    for each slide's Level-2 (Hilbert bucket) children.  Child table names
    follow the convention ``{parent}_{safe_slide_id}_h{bucket_id}``.

    Returns
    -------
    partition_map : dict
        ``{slide_id: {bucket_id: child_table_name}}``
        where *slide_id* is the original (un-sanitised) identifier and
        *bucket_id* is the integer Hilbert bucket index.
    """
    # Pattern to extract bucket id from leaf partition name.
    # Leaf names end with _h<int>, e.g.  objects_spdb_tcga_2f_a9ko_01z_00_dx1_h3
    bucket_re = re.compile(r"_h(\d+)$")

    # ------------------------------------------------------------------
    # Step 1: find Level-1 slide partitions (direct children of parent)
    # ------------------------------------------------------------------
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.relname
              FROM pg_inherits  i
              JOIN pg_class     c ON c.oid = i.inhrelid
              JOIN pg_class     p ON p.oid = i.inhparent
             WHERE p.relname = %s
        """, (parent_table,))
        level1_names = [row[0] for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Step 2: for each Level-1 partition, find its Level-2 bucket children
    # ------------------------------------------------------------------
    partition_map: Dict[str, Dict[int, str]] = {}
    # Build a reverse lookup: sanitised prefix -> original slide_id.
    # We will populate it lazily from the Level-2 query.

    for l1_name in level1_names:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.relname
                  FROM pg_inherits  i
                  JOIN pg_class     c ON c.oid = i.inhrelid
                  JOIN pg_class     p ON p.oid = i.inhparent
                 WHERE p.relname = %s
            """, (l1_name,))
            l2_names = [row[0] for row in cur.fetchall()]

        if not l2_names:
            # Level-1 partition with no children -- not a two-level table.
            continue

        # Recover the original slide_id from the Level-1 partition's CHECK
        # constraint (the VALUES IN ('slide_id') clause).
        slide_id = _recover_slide_id(conn, l1_name)
        if slide_id is None:
            # Fallback: strip the parent prefix to obtain the safe slug,
            # but we cannot recover the original form.  Use the safe form.
            prefix = parent_table + "_"
            slide_id = l1_name[len(prefix):] if l1_name.startswith(prefix) else l1_name

        bucket_dict: Dict[int, str] = {}
        for l2_name in l2_names:
            m = bucket_re.search(l2_name)
            if m:
                bucket_id = int(m.group(1))
                bucket_dict[bucket_id] = l2_name
        partition_map[slide_id] = bucket_dict

    return partition_map


def _recover_slide_id(conn, partition_name: str) -> Optional[str]:
    """Extract the original slide_id from a LIST partition's CHECK constraint.

    The constraint looks like ``CHECK ((slide_id = 'TCGA-2F-A9KO-01Z-00-DX1'::text))``
    or equivalently ``FOR VALUES IN ('...')``.  We parse it from pg_catalog.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pg_get_expr(c.relpartbound, c.oid)
              FROM pg_class c
             WHERE c.relname = %s
        """, (partition_name,))
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    # Expression looks like: FOR VALUES IN ('TCGA-2F-A9KO-01Z-00-DX1')
    expr = row[0]
    m = re.search(r"IN\s*\('([^']+)'\)", expr, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# PartitionRouter
# ---------------------------------------------------------------------------

class PartitionRouter:
    """Application-level partition routing for SPDB queries.

    Instead of querying the parent SPDB table (which forces the PostgreSQL
    planner to evaluate all ~4,000 partition constraints), we compute
    candidate Hilbert buckets in Python and build UNION ALL queries that
    target only the relevant child tables.

    Parameters
    ----------
    conn : psycopg2 connection
        Active database connection (must stay open for the router's lifetime).
    parent_table : str
        Name of the top-level partitioned table (default ``objects_spdb``).
    hilbert_order : int
        Hilbert curve order *p* -- grid is 2^p x 2^p (default 8).
    bucket_target : int
        Target number of objects per Hilbert bucket (default 50,000).
        Used to compute ``num_buckets`` for a slide when not known a priori.
    """

    def __init__(
        self,
        conn,
        parent_table: str = config.TABLE_SPDB,
        hilbert_order: int = config.HILBERT_ORDER,
        bucket_target: int = config.BUCKET_TARGET,
    ):
        self.conn = conn
        self.parent_table = parent_table
        self.hilbert_order = hilbert_order
        self.bucket_target = bucket_target

        # {slide_id: {bucket_id: child_table_name}}
        self.partition_map = _build_partition_map(conn, parent_table)

        # Reverse index: child_table_name -> (slide_id, bucket_id)
        self.reverse_map: Dict[str, Tuple[str, int]] = {}
        for sid, buckets in self.partition_map.items():
            for bid, tname in buckets.items():
                self.reverse_map[tname] = (sid, bid)

        # Cache num_buckets per slide for fast lookup.
        self.num_buckets: Dict[str, int] = {
            sid: len(buckets) for sid, buckets in self.partition_map.items()
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_slide_partitions(self, slide_id: str) -> List[str]:
        """Return all child partition table names for *slide_id*.

        Returns an empty list if the slide is unknown to the router.
        """
        buckets = self.partition_map.get(slide_id, {})
        return [buckets[b] for b in sorted(buckets)]

    def _resolve_buckets(
        self,
        slide_id: str,
        bucket_ids: List[int],
    ) -> List[str]:
        """Map bucket IDs to child table names, skipping any that are missing."""
        buckets = self.partition_map.get(slide_id, {})
        return [buckets[b] for b in bucket_ids if b in buckets]

    def _num_buckets_for_slide(self, slide_id: str) -> int:
        """Return the number of Hilbert buckets for a slide.

        Falls back to 1 if the slide is not in the partition map.
        """
        return self.num_buckets.get(slide_id, 1)

    # ------------------------------------------------------------------
    # Q1: Viewport (spatial range) query
    # ------------------------------------------------------------------

    def route_viewport(
        self,
        slide_id: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        slide_width: float,
        slide_height: float,
    ) -> Tuple[str, Tuple]:
        """Build a routed viewport query targeting only relevant partitions.

        Instead of::

            SELECT ... FROM objects_spdb
             WHERE slide_id = %s AND ST_Intersects(geom, ...)

        which forces the planner to examine *all* partitions, we emit::

            SELECT ... FROM child_1 WHERE ST_Intersects(geom, ...)
            UNION ALL
            SELECT ... FROM child_2 WHERE ST_Intersects(geom, ...)

        The planner sees O(B_hit) tables instead of O(B_total).

        Parameters
        ----------
        slide_id : str
            Whole-slide image identifier.
        x0, y0, x1, y1 : float
            Viewport bounding box in slide coordinates.
        slide_width, slide_height : float
            Full slide dimensions (needed for Hilbert normalisation).

        Returns
        -------
        sql : str
            The UNION ALL SQL string with ``%s`` parameter placeholders.
        params : tuple
            Bind parameters ``(x0, y0, x1, y1)`` repeated once per child.
        """
        nb = self._num_buckets_for_slide(slide_id)
        bucket_ids = hilbert.candidate_buckets_for_bbox(
            x0, y0, x1, y1,
            slide_width, slide_height,
            self.hilbert_order, nb,
        )
        child_tables = self._resolve_buckets(slide_id, bucket_ids)

        if not child_tables:
            # Fallback: query the parent table if no partitions matched.
            sql = (
                f"SELECT object_id, centroid_x, centroid_y, class_label "
                f"FROM {self.parent_table} "
                f"WHERE slide_id = %s "
                f"  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))"
            )
            return sql, (slide_id, x0, y0, x1, y1)

        parts = []
        params: list = []
        for tbl in child_tables:
            parts.append(
                f"SELECT object_id, centroid_x, centroid_y, class_label "
                f"FROM {tbl} "
                f"WHERE ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))"
            )
            params.extend([x0, y0, x1, y1])

        sql = "\nUNION ALL\n".join(parts)
        return sql, tuple(params)

    # ------------------------------------------------------------------
    # Q2: k-Nearest-Neighbour query
    # ------------------------------------------------------------------

    def route_knn(
        self,
        slide_id: str,
        qx: float,
        qy: float,
        k: int,
        slide_width: float,
        slide_height: float,
    ) -> Tuple[str, Tuple]:
        """Build a routed k-nearest-neighbour query.

        Strategy (prototype): query *all* partitions for this slide via
        UNION ALL, each with ``ORDER BY geom <-> query_point``, then wrap
        in an outer ``ORDER BY ... LIMIT k``.

        This still eliminates the global planner scan: the planner only sees
        B_slide partitions (typically 5--40) instead of B_total (~4,000).

        A production implementation would use an expanding-ring strategy:
        start with the bucket containing the query point, check if the k-th
        distance could be beaten by adjacent buckets, and expand only as
        needed.

        Parameters
        ----------
        slide_id : str
            Whole-slide image identifier.
        qx, qy : float
            Query point coordinates.
        k : int
            Number of nearest neighbours to return.
        slide_width, slide_height : float
            Full slide dimensions.

        Returns
        -------
        sql : str
            The wrapped UNION ALL SQL string.
        params : tuple
            Bind parameters.
        """
        child_tables = self._get_slide_partitions(slide_id)

        if not child_tables:
            # Fallback: query the parent table.
            sql = (
                f"SELECT object_id, centroid_x, centroid_y, class_label, "
                f"       geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) AS dist "
                f"FROM {self.parent_table} "
                f"WHERE slide_id = %s "
                f"ORDER BY dist LIMIT %s"
            )
            return sql, (qx, qy, slide_id, k)

        parts = []
        params: list = []
        for tbl in child_tables:
            parts.append(
                f"SELECT object_id, centroid_x, centroid_y, class_label, "
                f"       geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 0) AS dist "
                f"FROM {tbl}"
            )
            params.extend([qx, qy])

        inner = "\nUNION ALL\n".join(parts)
        sql = f"SELECT * FROM (\n{inner}\n) AS knn_union ORDER BY dist LIMIT %s"
        params.append(k)
        return sql, tuple(params)

    # ------------------------------------------------------------------
    # Q3: Aggregation (tile-level class counts)
    # ------------------------------------------------------------------

    def route_aggregation(
        self,
        slide_id: str,
    ) -> Tuple[str, Tuple]:
        """Build a routed aggregation query across all partitions for a slide.

        Each child partition computes its own partial aggregation::

            SELECT tile_id, class_label, COUNT(*) AS cnt FROM child_i
            GROUP BY tile_id, class_label

        The results are merged in an outer query::

            SELECT tile_id, class_label, SUM(cnt)
            FROM ( ... UNION ALL ... ) t
            GROUP BY tile_id, class_label

        Returns
        -------
        sql : str
            Aggregation SQL.
        params : tuple
            Empty tuple (no bind parameters needed for per-slide aggregation;
            slide routing is implicit in which child tables we select).
        """
        child_tables = self._get_slide_partitions(slide_id)

        if not child_tables:
            sql = (
                f"SELECT tile_id, class_label, COUNT(*) AS cnt "
                f"FROM {self.parent_table} "
                f"WHERE slide_id = %s "
                f"GROUP BY tile_id, class_label"
            )
            return sql, (slide_id,)

        parts = []
        for tbl in child_tables:
            parts.append(
                f"SELECT tile_id, class_label, COUNT(*) AS cnt "
                f"FROM {tbl} "
                f"GROUP BY tile_id, class_label"
            )

        inner = "\nUNION ALL\n".join(parts)
        sql = (
            f"SELECT tile_id, class_label, SUM(cnt) AS cnt "
            f"FROM (\n{inner}\n) AS agg_union "
            f"GROUP BY tile_id, class_label"
        )
        return sql, ()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return summary statistics about the partition map."""
        slide_count = len(self.partition_map)
        bucket_counts = [len(b) for b in self.partition_map.values()]
        total_partitions = sum(bucket_counts)
        return {
            "slides": slide_count,
            "total_leaf_partitions": total_partitions,
            "buckets_per_slide_mean": (
                statistics.mean(bucket_counts) if bucket_counts else 0
            ),
            "buckets_per_slide_min": min(bucket_counts) if bucket_counts else 0,
            "buckets_per_slide_max": max(bucket_counts) if bucket_counts else 0,
        }


# ---------------------------------------------------------------------------
# Benchmark: routed (SPDB-R) vs. native (SPDB) viewport queries
# ---------------------------------------------------------------------------

def benchmark_routed_vs_native(
    conn,
    n_trials: int = 500,
    viewport_frac: float = config.VIEWPORT_FRACTION,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """Compare planning + execution cost of routed vs. native SPDB queries.

    For each trial we:
      1. Pick a random slide and viewport.
      2. Run the *native* query (via parent table -- planner sees all partitions).
      3. Run the *routed* query (UNION ALL on child tables -- planner sees B_hit).
      4. Record wall-clock latency for each.

    The difference isolates the planner overhead caused by partition
    constraint-exclusion over the full tree.

    Returns
    -------
    results : dict
        Contains per-trial latencies, summary statistics, and the speedup
        ratio for reporting in the paper.
    """
    from benchmarks.framework import (
        load_metadata, get_slide_dimensions, random_viewport,
        time_query, warmup_cache, compute_stats,
    )

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    rng = np.random.RandomState(seed)

    router = PartitionRouter(conn)
    parent_table = config.TABLE_SPDB

    # Warm both paths.
    warmup_cache(conn, parent_table)

    latencies_native: list[float] = []
    latencies_routed: list[float] = []
    buckets_hit: list[int] = []

    for _ in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)

        # --- Native: query the parent table with Hilbert key predicates ---
        n_obj = metadata["object_counts"].get(sid, 1_000_000)
        num_buckets = max(1, n_obj // config.BUCKET_TARGET)
        bucket_ids = hilbert.candidate_buckets_for_bbox(
            x0, y0, x1, y1, w, h, config.HILBERT_ORDER, num_buckets,
        )
        total_cells = 1 << (2 * config.HILBERT_ORDER)
        # Merge adjacent buckets into contiguous key ranges.
        ranges = []
        for b in sorted(bucket_ids):
            lo = b * total_cells // num_buckets
            hi = (b + 1) * total_cells // num_buckets
            if ranges and ranges[-1][1] == lo:
                ranges[-1] = (ranges[-1][0], hi)
            else:
                ranges.append((lo, hi))
        hk_clauses = " OR ".join(
            f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
            for lo, hi in ranges
        )
        native_sql = (
            f"SELECT object_id, centroid_x, centroid_y, class_label "
            f"FROM {parent_table} "
            f"WHERE slide_id = %s "
            f"  AND ({hk_clauses}) "
            f"  AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 0))"
        )
        _, t_native = time_query(conn, native_sql, (sid, x0, y0, x1, y1))
        latencies_native.append(t_native)

        # --- Routed: UNION ALL on child partitions ---
        routed_sql, routed_params = router.route_viewport(
            sid, x0, y0, x1, y1, w, h,
        )
        _, t_routed = time_query(conn, routed_sql, routed_params)
        latencies_routed.append(t_routed)

        buckets_hit.append(len(bucket_ids))

    stats_native = compute_stats(latencies_native)
    stats_routed = compute_stats(latencies_routed)

    speedup_p50 = stats_native["p50"] / stats_routed["p50"] if stats_routed["p50"] > 0 else float("inf")
    speedup_mean = stats_native["mean"] / stats_routed["mean"] if stats_routed["mean"] > 0 else float("inf")

    results = {
        "n_trials": n_trials,
        "viewport_frac": viewport_frac,
        "native": stats_native,
        "routed": stats_routed,
        "speedup_p50": round(speedup_p50, 2),
        "speedup_mean": round(speedup_mean, 2),
        "mean_buckets_hit": round(float(np.mean(buckets_hit)), 2),
        "router_stats": router.stats(),
    }

    print(f"\n{'='*60}")
    print(f"  Partition Router Benchmark  ({n_trials} trials, {viewport_frac:.0%} viewport)")
    print(f"{'='*60}")
    print(f"  {'':18s} {'p50':>8s} {'p95':>8s} {'mean':>8s}")
    print(f"  {'Native (SPDB)':18s} {stats_native['p50']:>8.2f} {stats_native['p95']:>8.2f} {stats_native['mean']:>8.2f} ms")
    print(f"  {'Routed (SPDB-R)':18s} {stats_routed['p50']:>8.2f} {stats_routed['p95']:>8.2f} {stats_routed['mean']:>8.2f} ms")
    print(f"  Speedup (p50):  {speedup_p50:.2f}x")
    print(f"  Speedup (mean): {speedup_mean:.2f}x")
    print(f"  Avg buckets hit: {np.mean(buckets_hit):.1f} / {router.stats()['total_leaf_partitions']}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conn = psycopg2.connect(config.dsn())
    try:
        results = benchmark_routed_vs_native(conn)
    finally:
        conn.close()
