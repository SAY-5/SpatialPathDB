"""Create SO-C (Slide-Only Clustered) at 195M scale.

Copies data from objects_slide_only (127 partitions, 195M rows) into a new
objects_so_clustered_195m table, CLUSTERs each partition on GiST (Hilbert sort),
and adds BRIN indexes on hilbert_key.

Usage:
    python -m benchmarks.create_soc_195m
"""

from __future__ import annotations

import time
import psycopg2
from spdb import config


def create_soc_195m():
    conn = psycopg2.connect(config.dsn())
    conn.autocommit = True

    with conn.cursor() as cur:
        # Drop existing
        print("Dropping existing objects_so_clustered_195m...")
        cur.execute("DROP TABLE IF EXISTS objects_so_clustered_195m CASCADE")

        # Create parent
        print("Creating parent table...")
        cur.execute("""
            CREATE TABLE objects_so_clustered_195m (
                LIKE objects_slide_only INCLUDING DEFAULTS
            ) PARTITION BY LIST (slide_id)
        """)

        # Get all slide partitions from objects_slide_only
        cur.execute("""
            SELECT c.relname,
                   pg_get_expr(c.relpartbound, c.oid) AS bound_expr
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'objects_slide_only'
            ORDER BY c.relname
        """)
        raw_parts = cur.fetchall()

        # Parse slide_id from "FOR VALUES IN ('TCGA-...')"
        import re
        partitions = []
        for relname, bound_expr in raw_parts:
            m = re.search(r"IN \('([^']+)'\)", bound_expr)
            if m:
                partitions.append((relname, m.group(1)))
            else:
                print(f"  WARNING: could not parse slide_id from {relname}: {bound_expr}")
        print(f"Found {len(partitions)} slide partitions to process")

    total = len(partitions)
    ppr = config.BRIN_PAGES_PER_RANGE

    for i, (src_part, slide_id) in enumerate(partitions, 1):
        t0 = time.time()
        safe = slide_id.replace("-", "_").replace(".", "_").lower()
        dst_part = f"objects_so_clustered_195m_{safe}"

        print(f"\n[{i}/{total}] {slide_id} -> {dst_part}")

        with conn.cursor() as cur:
            # Create partition
            cur.execute(
                f"CREATE TABLE {dst_part} PARTITION OF objects_so_clustered_195m "
                f"FOR VALUES IN (%s)",
                (slide_id,),
            )

            # Copy data
            print(f"  Copying from {src_part}...")
            cur.execute(f"INSERT INTO {dst_part} SELECT * FROM {src_part}")
            row_count = cur.rowcount
            print(f"  Copied {row_count:,} rows")

            # GiST index
            print(f"  Creating GiST index...")
            cur.execute(
                f"CREATE INDEX idx_{dst_part}_geom ON {dst_part} USING gist (geom)"
            )

            # CLUSTER (Hilbert sort via PostGIS 3.6)
            print(f"  CLUSTERing by Hilbert order...")
            cur.execute(f"SET maintenance_work_mem = '4GB'")
            cur.execute(f"CLUSTER {dst_part} USING idx_{dst_part}_geom")

            # BRIN on hilbert_key
            print(f"  Creating BRIN index...")
            cur.execute(
                f"CREATE INDEX idx_{dst_part}_hbrin ON {dst_part} "
                f"USING brin (hilbert_key) WITH (pages_per_range={ppr})"
            )

            # Disable autovacuum
            cur.execute(
                f"ALTER TABLE {dst_part} SET (autovacuum_enabled = false)"
            )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

    # Final ANALYZE
    print("\nRunning ANALYZE on objects_so_clustered_195m...")
    with conn.cursor() as cur:
        cur.execute("ANALYZE objects_so_clustered_195m")

    print("\nDONE — SO-C 195M table ready.")
    conn.close()


if __name__ == "__main__":
    create_soc_195m()
