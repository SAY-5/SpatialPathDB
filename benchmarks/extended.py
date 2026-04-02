"""Extended experiments: viewport sensitivity, workload mix, kNN k-sweep,
storage overhead, density analysis, Hilbert sensitivity, Hilbert vs Z-order,
cold cache, pruning analysis.
"""

import io
import os
import time
import json
import numpy as np
import pandas as pd
import psycopg2

from spdb import config, hilbert, zorder, schema
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    get_slide_dimensions, random_viewport, random_point, time_query,
    warmup_cache, print_comparison,
)
from benchmarks.q1_viewport import run_q1
from benchmarks.q2_knn import run_q2


STORAGE_TABLES = {
    "Mono": config.TABLE_MONO,
    "Mono-T": config.TABLE_MONO_TUNED,
    "SO": config.TABLE_SLIDE_ONLY,
    "SPDB": config.TABLE_SPDB,
    "SPDB-Z": config.TABLE_SPDB_ZORDER,
}


# ---------- Viewport Size Sensitivity ----------

def viewport_sensitivity(n_trials=200, seed=42):
    """Q1 at varying viewport fractions."""
    fracs = config.VIEWPORT_FRACTIONS_SWEEP
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    results = {}
    for frac in fracs:
        print(f"\n  Viewport fraction: {frac}")
        lats_spdb = run_q1(conn, config.TABLE_SPDB, slide_ids, metadata,
                           n_trials=n_trials, viewport_frac=frac, seed=seed)
        lats_mono = run_q1(conn, config.TABLE_MONO, slide_ids, metadata,
                           n_trials=n_trials, viewport_frac=frac, seed=seed)
        results[str(frac)] = {
            "SPDB": compute_stats(lats_spdb),
            "Mono": compute_stats(lats_mono),
        }
        save_raw_latencies(lats_spdb, f"viewport_sens_{frac}", "SPDB")
        save_raw_latencies(lats_mono, f"viewport_sens_{frac}", "Mono")
        print(f"    SPDB p50={results[str(frac)]['SPDB']['p50']:.1f}ms  "
              f"Mono p50={results[str(frac)]['Mono']['p50']:.1f}ms")

    save_results(results, "viewport_sensitivity")
    conn.close()
    return results


# ---------- Workload Mix ----------

def workload_mix(n_total=500, seed=42):
    """Mixed workload: 70% Q1, 15% Q2, 10% Q3, 5% Q4."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())
    rng = np.random.RandomState(seed)

    warmup_cache(conn, config.TABLE_SPDB)

    mix = config.WORKLOAD_MIX
    query_types = rng.choice(
        list(mix.keys()), size=n_total,
        p=list(mix.values()),
    )

    latencies = {"Q1": [], "Q2": [], "Q3": [], "Q4": [], "all": []}
    t_start = time.time()

    for qt in query_types:
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)

        if qt == "Q1":
            x0, y0, x1, y1 = random_viewport(w, h, 0.05, rng)
            n_obj = metadata["object_counts"].get(sid, 1_000_000)
            num_buckets = max(1, n_obj // config.BUCKET_TARGET)
            bids = hilbert.candidate_buckets_for_bbox(
                x0, y0, x1, y1, w, h, config.HILBERT_ORDER, num_buckets
            )
            tc = 1 << (2 * config.HILBERT_ORDER)
            hkr = []
            for b in sorted(bids):
                lo = b * tc // num_buckets
                hi_ = (b + 1) * tc // num_buckets
                if hkr and hkr[-1][1] == lo:
                    hkr[-1] = (hkr[-1][0], hi_)
                else:
                    hkr.append((lo, hi_))
            hk = " OR ".join(f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
                             for lo, hi in hkr)
            sql = (f"SELECT object_id FROM {config.TABLE_SPDB} "
                   f"WHERE slide_id = %s AND ({hk}) AND "
                   f"ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,0))")
            _, elapsed = time_query(conn, sql, (sid, x0, y0, x1, y1))
        elif qt == "Q2":
            qx, qy = random_point(w, h, rng)
            sql = (f"SELECT object_id, geom <-> ST_SetSRID(ST_MakePoint(%s,%s),0) AS d "
                   f"FROM {config.TABLE_SPDB} WHERE slide_id = %s "
                   f"ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s,%s),0) LIMIT 50")
            _, elapsed = time_query(conn, sql, (qx, qy, sid, qx, qy))
        elif qt == "Q3":
            sql = (f"SELECT tile_id, COUNT(*) FROM {config.TABLE_SPDB} "
                   f"WHERE slide_id = %s GROUP BY tile_id")
            _, elapsed = time_query(conn, sql, (sid,))
        else:  # Q4
            x0, y0, x1, y1 = random_viewport(w, h, 0.02, rng)
            sql = (f"SELECT object_id FROM {config.TABLE_SPDB} "
                   f"WHERE slide_id = %s AND "
                   f"ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,0)) "
                   f"AND class_label IN ('Tumor','Lymphocyte') LIMIT 500")
            _, elapsed = time_query(conn, sql, (sid, x0, y0, x1, y1))

        latencies[qt].append(elapsed)
        latencies["all"].append(elapsed)

    total_time = time.time() - t_start
    results = {
        "n_total": n_total,
        "total_time_sec": total_time,
        "overall_qps": n_total / total_time,
    }
    for qt in ["Q1", "Q2", "Q3", "Q4", "all"]:
        if latencies[qt]:
            results[qt] = compute_stats(latencies[qt])
            results[qt]["count"] = len(latencies[qt])

    save_results(results, "workload_mix")
    print(f"\n  Mixed workload: {n_total} queries in {total_time:.1f}s "
          f"({n_total/total_time:.1f} QPS)")
    for qt in ["Q1", "Q2", "Q3", "Q4"]:
        if qt in results and isinstance(results[qt], dict):
            print(f"    {qt}: n={results[qt]['count']}  p50={results[qt]['p50']:.1f}ms")

    conn.close()
    return results


# ---------- kNN k-sweep ----------

def knn_k_sweep(n_trials=200, seed=42):
    """Run Q2 with varying k values."""
    ks = config.KNN_K_SWEEP
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    results = {}
    for k in ks:
        print(f"  k={k}...")
        lats, rings = run_q2(conn, config.TABLE_SPDB, slide_ids, metadata,
                             k=k, n_trials=n_trials, seed=seed)
        stats = compute_stats(lats)
        stats["k"] = k
        results[k] = stats
        save_raw_latencies(lats, f"knn_k{k}", "SPDB")
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    save_results(results, "knn_k_sweep")
    conn.close()
    return results


# ---------- Storage Overhead ----------

def _recursive_partition_size(conn, table_name):
    """Sum sizes across all leaf partitions via recursive CTE.

    pg_total_relation_size on a partitioned parent returns 0 because the
    parent has no heap storage.  We recursively walk pg_inherits to find
    every leaf and sum their sizes.
    """
    with conn.cursor() as cur:
        cur.execute("""
            WITH RECURSIVE all_parts AS (
                SELECT inhrelid
                FROM pg_inherits
                WHERE inhparent = %s::regclass
                UNION ALL
                SELECT i.inhrelid
                FROM pg_inherits i
                JOIN all_parts a ON i.inhparent = a.inhrelid
            )
            SELECT COALESCE(SUM(pg_total_relation_size(inhrelid)), 0)::bigint AS total,
                   COALESCE(SUM(pg_table_size(inhrelid)), 0)::bigint        AS tbl,
                   COALESCE(SUM(pg_indexes_size(inhrelid)), 0)::bigint      AS idx,
                   COUNT(*)::int                                             AS n_leaves
            FROM all_parts
        """, (table_name,))
        return cur.fetchone()   # (total, tbl, idx, n_leaves)


def storage_overhead():
    """Measure table sizes, index sizes for all configs.

    For partitioned tables, sums sizes across all leaf partitions.
    """
    conn = psycopg2.connect(config.dsn())
    results = {}

    for name, tbl in STORAGE_TABLES.items():
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                row_count = cur.fetchone()[0]

                cur.execute("""
                    SELECT COUNT(*) FROM pg_inherits
                    WHERE inhparent = %s::regclass
                """, (tbl,))
                n_direct_parts = cur.fetchone()[0]

            if n_direct_parts > 0:
                total_b, table_b, index_b, n_leaves = _recursive_partition_size(conn, tbl)
            else:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT pg_total_relation_size(%s),
                               pg_table_size(%s),
                               pg_indexes_size(%s)
                    """, (tbl, tbl, tbl))
                    total_b, table_b, index_b = cur.fetchone()
                n_leaves = 0

            results[name] = {
                "row_count": row_count,
                "total_bytes": total_b,
                "total_mb": round(total_b / 1048576, 1),
                "table_mb": round(table_b / 1048576, 1),
                "index_mb": round(index_b / 1048576, 1),
                "partitions": n_direct_parts,
                "leaf_partitions": n_leaves,
                "bytes_per_row": round(total_b / max(1, row_count), 1),
            }
            print(f"  {name}: {results[name]['total_mb']:.0f} MB total "
                  f"({results[name]['table_mb']:.0f} table + "
                  f"{results[name]['index_mb']:.0f} idx), "
                  f"{n_direct_parts} L1 parts, {n_leaves} leaves, "
                  f"{row_count:,} rows")
        except Exception as e:
            print(f"  {name}: SKIP ({e})")
            conn.rollback()

    save_results(results, "storage_overhead")
    conn.close()
    return results


# ---------- Density Analysis ----------

def density_analysis():
    """Per-slide density distribution analysis."""
    metadata = load_metadata()
    conn = psycopg2.connect(config.dsn())
    results = {}

    for sid in metadata["slide_ids"]:
        w, h = get_slide_dimensions(metadata, sid)
        n_objects = metadata["object_counts"][sid]
        area_px = w * h
        density = n_objects / area_px * 1e6

        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT tile_id, COUNT(*) FROM {config.TABLE_SPDB}
                WHERE slide_id = %s GROUP BY tile_id
            """, (sid,))
            tile_counts = [r[1] for r in cur.fetchall()]

        results[sid] = {
            "n_objects": n_objects,
            "image_width": w,
            "image_height": h,
            "density_per_mpx": round(density, 2),
            "n_tiles": len(tile_counts),
            "tile_count_mean": float(np.mean(tile_counts)) if tile_counts else 0,
            "tile_count_std": float(np.std(tile_counts)) if tile_counts else 0,
            "tile_count_max": int(np.max(tile_counts)) if tile_counts else 0,
            "tile_count_min": int(np.min(tile_counts)) if tile_counts else 0,
        }

    save_results(results, "density_analysis")
    print(f"  Analyzed {len(results)} slides")
    densities = [v["density_per_mpx"] for v in results.values()]
    print(f"  Density range: {min(densities):.1f} - {max(densities):.1f} per Mpx")
    conn.close()
    return results


# ---------- Hilbert vs Z-order ----------

def hilbert_vs_zorder(n_trials=200, seed=42):
    """Controlled comparison: SPDB Hilbert vs SPDB Z-order."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())

    results = {}
    for name, table in [("Hilbert", config.TABLE_SPDB), ("Z-order", config.TABLE_SPDB_ZORDER)]:
        print(f"  Running Q1 on {name}...")
        lats = run_q1(conn, table, slide_ids, metadata,
                      n_trials=n_trials, viewport_frac=0.05, seed=seed)
        stats = compute_stats(lats)
        results[name] = stats
        save_raw_latencies(lats, "hilbert_vs_zorder", name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    save_results(results, "hilbert_vs_zorder")
    conn.close()
    return results


# ---------- Cold Cache ----------

def cold_cache_benchmark(n_trials=30, seed=42):
    """Cold cache Q1 benchmark using fresh connections."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    rng = np.random.RandomState(seed)

    configs = {
        "Mono": config.TABLE_MONO,
        "SO": config.TABLE_SLIDE_ONLY,
        "SPDB": config.TABLE_SPDB,
    }

    can_purge = os.system("sudo -n purge 2>/dev/null") == 0

    results = {}
    for cfg_name, table in configs.items():
        print(f"  Cold-cache Q1 on {cfg_name}...")
        latencies = []
        is_spdb = table in (config.TABLE_SPDB, config.TABLE_SPDB_ZORDER)

        for trial in range(n_trials):
            if can_purge:
                os.system("sudo purge 2>/dev/null")
                time.sleep(0.5)

            conn = psycopg2.connect(config.dsn())
            sid = rng.choice(slide_ids)
            w, h = get_slide_dimensions(metadata, sid)
            x0, y0, x1, y1 = random_viewport(w, h, 0.05, rng)

            if is_spdb:
                n_obj = metadata["object_counts"].get(sid, 1_000_000)
                num_buckets = max(1, n_obj // config.BUCKET_TARGET)
                bucket_ids = hilbert.candidate_buckets_for_bbox(
                    x0, y0, x1, y1, w, h, config.HILBERT_ORDER, num_buckets
                )
                total_cells = 1 << (2 * config.HILBERT_ORDER)
                hk_ranges = []
                for b in sorted(bucket_ids):
                    lo = b * total_cells // num_buckets
                    hi_ = (b + 1) * total_cells // num_buckets
                    if hk_ranges and hk_ranges[-1][1] == lo:
                        hk_ranges[-1] = (hk_ranges[-1][0], hi_)
                    else:
                        hk_ranges.append((lo, hi_))
                hk_clause = " OR ".join(
                    f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
                    for lo, hi in hk_ranges
                )
                sql = (f"SELECT object_id FROM {table} "
                       f"WHERE slide_id = %s AND ({hk_clause}) AND "
                       f"ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,0))")
            else:
                sql = (f"SELECT object_id FROM {table} "
                       f"WHERE slide_id = %s AND "
                       f"ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,0))")
            _, elapsed = time_query(conn, sql, (sid, x0, y0, x1, y1))
            latencies.append(elapsed)
            conn.close()

        stats = compute_stats(latencies)
        stats["can_purge"] = can_purge
        results[cfg_name] = stats
        save_raw_latencies(latencies, "cold_cache", cfg_name)
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    save_results(results, "cold_cache")
    return results


# ---------- Pruning Analysis ----------

def _count_partitions_from_plan(plan_json):
    """Walk the EXPLAIN JSON tree to extract partition pruning stats.

    Finds the Append node and counts its children (= partitions scanned).
    Also reads "Subplans Removed" (= partitions pruned).
    """
    scanned = 0
    removed = 0

    def _walk(node):
        nonlocal scanned, removed
        nt = node.get("Node Type", "")
        if nt == "Append":
            removed += node.get("Subplans Removed", 0)
            scanned += len(node.get("Plans", []))
        for child in node.get("Plans", []):
            _walk(child)

    if plan_json and len(plan_json) > 0:
        _walk(plan_json[0].get("Plan", {}))
    return scanned, removed


def pruning_analysis(n_trials=100, seed=42):
    """Analyze partition pruning via EXPLAIN ANALYZE on SPDB."""
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    conn = psycopg2.connect(config.dsn())
    rng = np.random.RandomState(seed)

    results = []
    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, 0.05, rng)

        n_obj = metadata["object_counts"][sid]
        n_total_buckets = max(1, n_obj // config.BUCKET_TARGET)
        bucket_ids = hilbert.candidate_buckets_for_bbox(
            x0, y0, x1, y1, w, h, config.HILBERT_ORDER, n_total_buckets
        )
        total_cells = 1 << (2 * config.HILBERT_ORDER)
        hk_ranges = []
        for b in sorted(bucket_ids):
            lo = b * total_cells // n_total_buckets
            hi = (b + 1) * total_cells // n_total_buckets
            if hk_ranges and hk_ranges[-1][1] == lo:
                hk_ranges[-1] = (hk_ranges[-1][0], hi)
            else:
                hk_ranges.append((lo, hi))
        hk_clause = " OR ".join(
            f"(hilbert_key >= {lo} AND hilbert_key < {hi})"
            for lo, hi in hk_ranges
        )

        with conn.cursor() as cur:
            cur.execute(f"""
                EXPLAIN (ANALYZE, FORMAT JSON)
                SELECT object_id FROM {config.TABLE_SPDB}
                WHERE slide_id = %s
                  AND ({hk_clause})
                  AND ST_Intersects(geom, ST_MakeEnvelope(%s,%s,%s,%s,0))
            """, (sid, x0, y0, x1, y1))
            plan = cur.fetchone()[0]

        scanned, pruned_reported = _count_partitions_from_plan(plan)
        actual_pruned = max(0, n_total_buckets - scanned)
        pruning_rate = actual_pruned / max(1, n_total_buckets)

        results.append({
            "slide_id": sid,
            "total_buckets": n_total_buckets,
            "partitions_scanned": scanned,
            "partitions_pruned": actual_pruned,
            "pruning_rate": pruning_rate,
            "candidate_buckets": len(bucket_ids),
        })

    avg_pruning = np.mean([r["pruning_rate"] for r in results])
    avg_scanned = np.mean([r["partitions_scanned"] for r in results])
    avg_pruned = np.mean([r["partitions_pruned"] for r in results])
    avg_candidates = np.mean([r["candidate_buckets"] for r in results])

    output = {
        "n_trials": n_trials,
        "avg_pruning_rate": float(avg_pruning),
        "avg_scanned_partitions": float(avg_scanned),
        "avg_pruned_partitions": float(avg_pruned),
        "avg_candidate_buckets": float(avg_candidates),
        "trials": results,
    }
    save_results(output, "pruning_analysis")
    print(f"  Average pruning rate: {avg_pruning:.1%}")
    print(f"  Average scanned: {avg_scanned:.1f} / avg total: {np.mean([r['total_buckets'] for r in results]):.0f}")
    print(f"  Average candidate buckets: {avg_candidates:.1f}")
    conn.close()
    return output


# ---------- Hilbert Order Sensitivity ----------

def _copy_rows_to_table(conn, table_name, rows_df, chunk_size=200_000):
    """COPY a DataFrame into table using the COPY protocol."""
    cols = [
        "object_id", "slide_id", "geom", "centroid_x", "centroid_y",
        "class_label", "tile_id", "hilbert_key", "zorder_key",
        "area", "perimeter", "confidence", "pipeline_id",
    ]
    n = len(rows_df)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = rows_df.iloc[start:end]
        lines = []
        for _, r in chunk.iterrows():
            wkt = f"POINT({r['centroid_x']:.2f} {r['centroid_y']:.2f})"
            lines.append(
                f"\\N\t{r['slide_id']}\t{wkt}\t{r['centroid_x']:.2f}\t"
                f"{r['centroid_y']:.2f}\t{r['class_label']}\t{r['tile_id']}\t"
                f"{int(r['hilbert_key'])}\t{int(r['zorder_key'])}\t"
                f"{r['area']:.2f}\t{r['perimeter']:.2f}\t"
                f"{r['confidence']:.1f}\t{r['pipeline_id']}"
            )
        buf = io.StringIO("\n".join(lines) + "\n")
        with conn.cursor() as cur:
            cur.copy_from(buf, table_name, columns=cols)
        conn.commit()


def hilbert_order_sensitivity(orders=None, n_trials=200, seed=42):
    """Create SPDB variants with different Hilbert orders and benchmark Q1.

    For each order p, creates a partitioned table objects_spdb_h{p}, populates
    it by reading data from objects_mono and recomputing Hilbert keys, then
    runs Q1 viewport benchmarks.
    """
    if orders is None:
        orders = config.HILBERT_ORDERS_SWEEP
    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]
    results = {}

    conn = psycopg2.connect(config.dsn())
    conn.autocommit = False

    for p_order in orders:
        table_name = f"objects_spdb_h{p_order}"

        if p_order == config.HILBERT_ORDER:
            print(f"\n  p={p_order} (baseline): using existing {config.TABLE_SPDB}")
            lats = run_q1(conn, config.TABLE_SPDB, slide_ids, metadata,
                          n_trials=n_trials, viewport_frac=0.05, seed=seed,
                          hilbert_order=p_order)
            stats = compute_stats(lats)
            results[str(p_order)] = stats
            save_raw_latencies(lats, f"hilbert_p{p_order}", "SPDB")
            print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")
            continue

        print(f"\n  p={p_order}: creating {table_name}...")

        conn.rollback()
        conn.close()
        conn = psycopg2.connect(config.dsn())
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE {table_name} (
                    object_id       BIGINT,
                    slide_id        TEXT        NOT NULL,
                    geom            GEOMETRY(Point, 0) NOT NULL,
                    centroid_x      DOUBLE PRECISION NOT NULL,
                    centroid_y      DOUBLE PRECISION NOT NULL,
                    class_label     TEXT        NOT NULL,
                    tile_id         TEXT,
                    hilbert_key     BIGINT      NOT NULL DEFAULT 0,
                    zorder_key      BIGINT      NOT NULL DEFAULT 0,
                    area            DOUBLE PRECISION,
                    perimeter       DOUBLE PRECISION,
                    confidence      DOUBLE PRECISION DEFAULT 1.0,
                    pipeline_id     TEXT
                ) PARTITION BY LIST (slide_id)
            """)
        conn.commit()

        for sid in slide_ids:
            n_objects = metadata["object_counts"][sid]
            num_buckets = max(1, n_objects // config.BUCKET_TARGET)
            safe = sid.replace("-", "_").replace(".", "_")
            slide_part = f"{table_name}_{safe}"

            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {slide_part}
                    PARTITION OF {table_name}
                    FOR VALUES IN ('{sid}')
                    PARTITION BY RANGE (hilbert_key)
                """)
            conn.commit()

            total_cells = 1 << (2 * p_order)
            for b in range(num_buckets):
                lo = b * total_cells // num_buckets
                hi = (b + 1) * total_cells // num_buckets
                sub_name = f"{slide_part}_h{b}"
                with conn.cursor() as cur:
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {sub_name}
                        PARTITION OF {slide_part}
                        FOR VALUES FROM ({lo}) TO ({hi})
                    """)
                conn.commit()

        print(f"    Partitions created. Populating from objects_mono...")

        for sid in slide_ids:
            w = float(metadata["metas"][sid]["image_width"])
            h = float(metadata["metas"][sid]["image_height"])

            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT slide_id, centroid_x, centroid_y, class_label,
                           tile_id, zorder_key, area, perimeter, confidence, pipeline_id
                    FROM {config.TABLE_MONO}
                    WHERE slide_id = %s
                """, (sid,))
                rows = cur.fetchall()

            if not rows:
                continue

            cols_names = ["slide_id", "centroid_x", "centroid_y", "class_label",
                          "tile_id", "zorder_key", "area", "perimeter",
                          "confidence", "pipeline_id"]
            df = pd.DataFrame(rows, columns=cols_names)

            cx = df["centroid_x"].values.astype(np.float64)
            cy = df["centroid_y"].values.astype(np.float64)
            gx, gy = hilbert.normalize_coords(cx, cy, w, h, p_order)
            df["hilbert_key"] = hilbert.encode_batch(gx, gy, p_order)

            _copy_rows_to_table(conn, table_name, df)
            print(f"    {sid}: {len(df):,} rows inserted")

        print(f"    Building indexes for {table_name}...")
        for sid in slide_ids:
            n_objects = metadata["object_counts"][sid]
            num_buckets = max(1, n_objects // config.BUCKET_TARGET)
            safe = sid.replace("-", "_").replace(".", "_")
            slide_part = f"{table_name}_{safe}"
            for b in range(num_buckets):
                sub = f"{slide_part}_h{b}"
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{sub}_geom ON {sub} USING gist (geom)")
                        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{sub}_hkey ON {sub} USING btree (slide_id, hilbert_key)")
                    conn.commit()
                except Exception:
                    conn.rollback()

        with conn.cursor() as cur:
            cur.execute(f"ANALYZE {table_name}")
        conn.commit()

        print(f"    Running Q1 on {table_name} (p={p_order})...")
        lats = run_q1(conn, table_name, slide_ids, metadata,
                      n_trials=n_trials, viewport_frac=0.05, seed=seed,
                      hilbert_order=p_order)
        stats = compute_stats(lats)
        results[str(p_order)] = stats
        save_raw_latencies(lats, f"hilbert_p{p_order}", "SPDB")
        print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms")

    save_results(results, "hilbert_order_sensitivity")
    conn.close()
    return results
