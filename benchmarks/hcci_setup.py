"""HCCI setup: add composite_key column, populate per-partition, build covering index.

Prepares the 195M-row objects_slide_only table for HCCI queries by:
  1. ALTER TABLE ADD COLUMN composite_key BIGINT
  2. UPDATE per-partition to avoid a single massive transaction
  3. CREATE covering B-tree index with INCLUDE columns
  4. VACUUM for visibility map (critical for index-only scans)
  5. ANALYZE

Usage:
    python -m benchmarks.hcci_setup
    python -m benchmarks.hcci_setup --index-only   # skip column + update
    python -m benchmarks.hcci_setup --vacuum-only   # just vacuum + analyze
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import psycopg2

from spdb import config
from spdb.hcci import composite_key_update_sql, covering_index_ddl

TABLE = config.TABLE_SLIDE_ONLY
INDEX_NAME = "idx_hcci_covering"


def _get_partitions(conn) -> list[tuple[str, str]]:
    """Discover child partitions of the SO table.

    Returns list of (partition_name, slide_id).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.relname,
                   pg_get_expr(c.relpartbound, c.oid) AS bound_expr
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = %s
            ORDER BY c.relname
        """, (TABLE,))
        raw = cur.fetchall()

    partitions = []
    for relname, bound_expr in raw:
        m = re.search(r"IN\s*\('([^']+)'\)", bound_expr, re.IGNORECASE)
        slide_id = m.group(1) if m else relname
        partitions.append((relname, slide_id))
    return partitions


def step1_add_column(conn):
    """Add composite_key BIGINT column (and area if missing)."""
    print("\n[Step 1] Adding required columns...")
    t0 = time.time()
    with conn.cursor() as cur:
        # Check if area column exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'area'
        """, (TABLE,))
        has_area = cur.fetchone() is not None

        if not has_area:
            print("  Adding area column (DOUBLE PRECISION)...")
            cur.execute(f"""
                ALTER TABLE {TABLE}
                ADD COLUMN IF NOT EXISTS area DOUBLE PRECISION DEFAULT 0.0
            """)
            conn.commit()
            print("  Populating area from geometry (ST_Area)...")
            # For point geometries, area is 0; but if polygons exist, compute it.
            # For nuclei centroids (points), set a synthetic area from the data
            # or leave as 0.  The covering index just needs the column to exist.
            cur.execute(f"""
                UPDATE {TABLE} SET area = COALESCE(ST_Area(geom), 0.0)
                WHERE area IS NULL OR area = 0
            """)
            conn.commit()
            print(f"  area column added and populated")

        print("  Adding composite_key column...")
        cur.execute(f"""
            ALTER TABLE {TABLE}
            ADD COLUMN IF NOT EXISTS composite_key BIGINT
        """)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def step2_populate(conn):
    """Populate composite_key per-partition."""
    print("\n[Step 2] Populating composite_key per-partition...")
    partitions = _get_partitions(conn)
    total = len(partitions)
    print(f"  Found {total} partitions")

    update_tmpl = composite_key_update_sql()
    total_rows = 0
    t0_all = time.time()

    for i, (part_name, slide_id) in enumerate(partitions, 1):
        t0 = time.time()
        sql = update_tmpl.format(table=part_name)
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.rowcount
        conn.commit()
        total_rows += rows
        elapsed = time.time() - t0
        print(f"  [{i:>3}/{total}] {slide_id:<30} {rows:>10,} rows  {elapsed:.1f}s")

    elapsed_all = time.time() - t0_all
    print(f"\n  Total: {total_rows:,} rows in {elapsed_all:.0f}s "
          f"({total_rows / elapsed_all:,.0f} rows/s)")


def step3_create_index(conn):
    """Create the covering B-tree index."""
    print(f"\n[Step 3] Creating covering index {INDEX_NAME}...")
    print("  This will take a long time on 195M rows. Be patient.")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        cur.execute(f"SET maintenance_work_mem = '8GB'")
        ddl = covering_index_ddl(TABLE, INDEX_NAME)
        print(f"  DDL: {ddl}")
        cur.execute(ddl)
    conn.commit()
    elapsed = time.time() - t0
    print(f"  Index created in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


def step4_vacuum(conn):
    """VACUUM the table for visibility map (index-only scan requirement)."""
    print(f"\n[Step 4] VACUUM {TABLE}...")
    print("  Required for index-only scans (visibility map must be current).")
    t0 = time.time()
    old_autocommit = conn.autocommit
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"VACUUM {TABLE}")
    conn.autocommit = old_autocommit
    elapsed = time.time() - t0
    print(f"  VACUUM done in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


def step5_analyze(conn):
    """ANALYZE for up-to-date planner statistics."""
    print(f"\n[Step 5] ANALYZE {TABLE}...")
    t0 = time.time()
    old_autocommit = conn.autocommit
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"ANALYZE {TABLE}")
    conn.autocommit = old_autocommit
    print(f"  Done in {time.time() - t0:.1f}s")


def verify(conn):
    """Quick sanity check: sample composite_key values and check index."""
    print("\n[Verify] Checking setup...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label,
                   composite_key,
                   (composite_key >> 48) AS class_enum,
                   (composite_key & x'FFFFFFFFFFFF'::bigint) AS hilbert_part,
                   hilbert_key
            FROM {TABLE}
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("  Sample rows:")
        print(f"  {'class_label':<14} {'composite_key':>20} {'enum':>5} "
              f"{'hilbert_part':>14} {'hilbert_key':>12}")
        for r in rows:
            print(f"  {r[0]:<14} {r[1]:>20} {r[2]:>5} {r[3]:>14} {r[4]:>12}")

        cur.execute("""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes
            WHERE tablename = %s AND indexname LIKE '%%hcci%%'
        """, (TABLE,))
        idx_rows = cur.fetchall()
        if idx_rows:
            print(f"\n  HCCI indexes found:")
            for name, size in idx_rows:
                print(f"    {name}: {size}")
        else:
            cur.execute("""
                SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
                FROM pg_indexes
                WHERE indexdef LIKE '%composite_key%'
                LIMIT 10
            """)
            idx_rows = cur.fetchall()
            if idx_rows:
                print(f"\n  Composite key indexes:")
                for name, size in idx_rows:
                    print(f"    {name}: {size}")

        # Test index-only scan
        cur.execute(f"""
            EXPLAIN (FORMAT TEXT)
            SELECT centroid_x, centroid_y, class_label, area
            FROM {TABLE}
            WHERE slide_id = (SELECT slide_id FROM {TABLE} LIMIT 1)
              AND composite_key >= 0 AND composite_key < 1000
        """)
        plan_lines = [r[0] for r in cur.fetchall()]
        is_index_only = any("Index Only Scan" in l for l in plan_lines)
        print(f"\n  Index-only scan detected: {'YES' if is_index_only else 'NO'}")
        if not is_index_only:
            print("  WARNING: PostgreSQL is NOT using index-only scan.")
            print("  Plan:")
            for l in plan_lines[:10]:
                print(f"    {l}")


def main():
    parser = argparse.ArgumentParser(description="HCCI setup for objects_slide_only")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip column addition and population, just build index")
    parser.add_argument("--vacuum-only", action="store_true",
                        help="Just run VACUUM and ANALYZE")
    parser.add_argument("--verify-only", action="store_true",
                        help="Just run verification checks")
    args = parser.parse_args()

    print("=" * 60)
    print("  HCCI Setup: Hilbert-Composite Covering Index")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())

    t_start = time.time()

    if args.verify_only:
        verify(conn)
    elif args.vacuum_only:
        step4_vacuum(conn)
        step5_analyze(conn)
        verify(conn)
    elif args.index_only:
        step3_create_index(conn)
        step4_vacuum(conn)
        step5_analyze(conn)
        verify(conn)
    else:
        step1_add_column(conn)
        step2_populate(conn)
        step3_create_index(conn)
        step4_vacuum(conn)
        step5_analyze(conn)
        verify(conn)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  HCCI setup complete in {total:.0f}s ({total / 60:.1f}m)")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
