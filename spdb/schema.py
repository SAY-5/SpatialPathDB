"""Schema creation for all database configurations.

Seven configurations spanning the full storage layout design space:
  1. Monolithic (Mono)       -- single table, global GiST
  2. Monolithic Tuned        -- same schema, DBA-tuned PG settings
  3. Monolithic Clustered    -- single table, CLUSTER'd on Hilbert via GiST, + BRIN
  4. Slide-Only (SO)         -- LIST partitioned by slide_id, per-partition GiST
  5. Slide-Only Clustered    -- LIST partitioned, each partition CLUSTER'd + BRIN
  6. SPDB                    -- LIST(slide_id) + RANGE(hilbert_key), hybrid indexes
  7. SPDB-Zorder             -- same as SPDB but Z-order keys
"""

import psycopg2
from spdb import config

COLUMNS = """
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
"""


def _exec(conn, sql):
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _exec_many(conn, statements):
    with conn.cursor() as cur:
        for sql in statements:
            cur.execute(sql)
    conn.commit()


def drop_all(conn):
    """Drop all experiment tables."""
    for t in config.ALL_TABLES:
        try:
            _exec(conn, f"DROP TABLE IF EXISTS {t} CASCADE;")
        except Exception:
            conn.rollback()


# ---------- Monolithic ----------

def create_monolithic(conn, table_name=None):
    tbl = table_name or config.TABLE_MONO
    _exec_many(conn, [
        f"DROP TABLE IF EXISTS {tbl} CASCADE;",
        f"""CREATE TABLE {tbl} (
            {COLUMNS}
        );""",
    ])
    return tbl


def index_monolithic(conn, table_name=None):
    tbl = table_name or config.TABLE_MONO
    _exec_many(conn, [
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_geom ON {tbl} USING gist (geom);",
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_slide ON {tbl} USING btree (slide_id);",
    ])


# ---------- Monolithic Tuned ----------

def create_monolithic_tuned(conn):
    return create_monolithic(conn, config.TABLE_MONO_TUNED)


def index_monolithic_tuned(conn):
    index_monolithic(conn, config.TABLE_MONO_TUNED)


# ---------- Monolithic Clustered (CLUSTER on GiST + BRIN on hilbert_key) ----------

def create_monolithic_clustered(conn):
    return create_monolithic(conn, config.TABLE_MONO_CLUSTERED)


def index_monolithic_clustered(conn):
    """Build GiST, CLUSTER on it (PostGIS 3+ uses Hilbert sort), then add BRIN."""
    tbl = config.TABLE_MONO_CLUSTERED
    _exec_many(conn, [
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_geom ON {tbl} USING gist (geom);",
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_slide ON {tbl} USING btree (slide_id);",
    ])
    _exec(conn, f"CLUSTER {tbl} USING idx_{tbl}_geom;")
    ppr = config.BRIN_PAGES_PER_RANGE
    _exec_many(conn, [
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_hbrin ON {tbl} "
        f"USING brin (hilbert_key) WITH (pages_per_range={ppr});",
        f"CREATE INDEX IF NOT EXISTS idx_{tbl}_geom_brin ON {tbl} "
        f"USING brin (centroid_x, centroid_y) WITH (pages_per_range={ppr});",
    ])


# ---------- Slide-Only ----------

def create_slide_only(conn, table_name=None):
    tbl = table_name or config.TABLE_SLIDE_ONLY
    _exec_many(conn, [
        f"DROP TABLE IF EXISTS {tbl} CASCADE;",
        f"""CREATE TABLE {tbl} (
            {COLUMNS}
        ) PARTITION BY LIST (slide_id);""",
    ])
    return tbl


def add_slide_partition_so(conn, slide_id, table_name=None):
    tbl = table_name or config.TABLE_SLIDE_ONLY
    safe = slide_id.replace("-", "_").replace(".", "_")
    part_name = f"{tbl}_{safe}"
    _exec(conn, f"""
        CREATE TABLE IF NOT EXISTS {part_name}
        PARTITION OF {tbl}
        FOR VALUES IN ('{slide_id}');
    """)
    return part_name


def index_slide_only(conn, slide_ids, table_name=None):
    tbl = table_name or config.TABLE_SLIDE_ONLY
    for sid in slide_ids:
        safe = sid.replace("-", "_").replace(".", "_")
        part = f"{tbl}_{safe}"
        _exec_many(conn, [
            f"CREATE INDEX IF NOT EXISTS idx_{part}_geom ON {part} USING gist (geom);",
            f"CREATE INDEX IF NOT EXISTS idx_{part}_slide ON {part} USING btree (slide_id);",
        ])


# ---------- Slide-Only Clustered ----------

def create_slide_only_clustered(conn):
    return create_slide_only(conn, config.TABLE_SLIDE_ONLY_CLUSTERED)


def add_slide_partition_soc(conn, slide_id):
    return add_slide_partition_so(conn, slide_id, config.TABLE_SLIDE_ONLY_CLUSTERED)


def index_slide_only_clustered(conn, slide_ids):
    """Per-partition: GiST + CLUSTER + BRIN on hilbert_key."""
    tbl = config.TABLE_SLIDE_ONLY_CLUSTERED
    ppr = config.BRIN_PAGES_PER_RANGE
    for sid in slide_ids:
        safe = sid.replace("-", "_").replace(".", "_")
        part = f"{tbl}_{safe}"
        _exec_many(conn, [
            f"CREATE INDEX IF NOT EXISTS idx_{part}_geom ON {part} USING gist (geom);",
            f"CREATE INDEX IF NOT EXISTS idx_{part}_slide ON {part} USING btree (slide_id);",
        ])
        try:
            _exec(conn, f"CLUSTER {part} USING idx_{part}_geom;")
        except Exception:
            conn.rollback()
        _exec_many(conn, [
            f"CREATE INDEX IF NOT EXISTS idx_{part}_hbrin ON {part} "
            f"USING brin (hilbert_key) WITH (pages_per_range={ppr});",
        ])


# ---------- SPDB (two-level) ----------

def create_spdb(conn, table_name=None):
    tbl = table_name or config.TABLE_SPDB
    _exec_many(conn, [
        f"DROP TABLE IF EXISTS {tbl} CASCADE;",
        f"""CREATE TABLE {tbl} (
            {COLUMNS}
        ) PARTITION BY LIST (slide_id);""",
    ])
    return tbl


def add_slide_hilbert_partitions(conn, slide_id, num_buckets,
                                  table_name=None, key_col="hilbert_key"):
    """Create Level-1 slide partition and Level-2 Hilbert/Z-order sub-partitions."""
    tbl = table_name or config.TABLE_SPDB
    safe = slide_id.replace("-", "_").replace(".", "_")
    slide_part = f"{tbl}_{safe}"

    _exec(conn, f"""
        CREATE TABLE IF NOT EXISTS {slide_part}
        PARTITION OF {tbl}
        FOR VALUES IN ('{slide_id}')
        PARTITION BY RANGE ({key_col});
    """)

    total_cells = 1 << (2 * config.HILBERT_ORDER)
    for b in range(num_buckets):
        lo = b * total_cells // num_buckets
        hi = (b + 1) * total_cells // num_buckets
        sub_name = f"{slide_part}_h{b}"
        _exec(conn, f"""
            CREATE TABLE IF NOT EXISTS {sub_name}
            PARTITION OF {slide_part}
            FOR VALUES FROM ({lo}) TO ({hi});
        """)

    return slide_part


def index_spdb(conn, slide_ids, num_buckets, table_name=None, key_col="hilbert_key"):
    tbl = table_name or config.TABLE_SPDB
    for sid in slide_ids:
        safe = sid.replace("-", "_").replace(".", "_")
        slide_part = f"{tbl}_{safe}"
        for b in range(num_buckets):
            sub = f"{slide_part}_h{b}"
            _exec_many(conn, [
                f"CREATE INDEX IF NOT EXISTS idx_{sub}_geom ON {sub} USING gist (geom);",
                f"CREATE INDEX IF NOT EXISTS idx_{sub}_hkey ON {sub} USING btree (slide_id, {key_col});",
                f"CREATE INDEX IF NOT EXISTS idx_{sub}_cls  ON {sub} USING btree (slide_id, class_label);",
                f"CREATE INDEX IF NOT EXISTS idx_{sub}_tile ON {sub} USING btree (tile_id);",
            ])


# ---------- SPDB Z-order ----------

def create_spdb_zorder(conn):
    return create_spdb(conn, config.TABLE_SPDB_ZORDER)


def add_slide_zorder_partitions(conn, slide_id, num_buckets):
    return add_slide_hilbert_partitions(
        conn, slide_id, num_buckets,
        table_name=config.TABLE_SPDB_ZORDER,
        key_col="zorder_key",
    )


def index_spdb_zorder(conn, slide_ids, num_buckets):
    index_spdb(conn, slide_ids, num_buckets,
               table_name=config.TABLE_SPDB_ZORDER,
               key_col="zorder_key")


# ---------- Utility ----------

def analyze_all(conn):
    for t in config.ALL_TABLES:
        try:
            _exec(conn, f"ANALYZE {t};")
        except Exception:
            conn.rollback()


def table_size_bytes(conn, table_name):
    """Return (total_bytes, table_bytes, index_bytes) for a table."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT pg_total_relation_size(%s),
                   pg_table_size(%s),
                   pg_indexes_size(%s)
        """, (table_name, table_name, table_name))
        return cur.fetchone()


def get_connection():
    return psycopg2.connect(config.dsn())
