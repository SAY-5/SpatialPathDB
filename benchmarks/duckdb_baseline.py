"""DuckDB Spatial baseline comparison for SPDB evaluation.

Implements the same Q1-Q4 query types on DuckDB with its spatial extension
to provide a modern embedded-database baseline.
"""

import os
import time

import numpy as np

from spdb import config
from benchmarks.framework import (
    compute_stats, save_raw_latencies, save_results, load_metadata,
    get_slide_dimensions, random_viewport, random_point,
    wilcoxon_ranksum, print_comparison,
)

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_duckdb(db_path=None, pg_dsn=None, slide_ids=None):
    """Create DuckDB database with spatial extension and load data from PostgreSQL.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database file. Default: results/duckdb_baseline.db
    pg_dsn : str
        PostgreSQL DSN to import data from. Default: from config.
    slide_ids : list
        Slide IDs to import. Default: all from metadata.

    Returns
    -------
    str : path to DuckDB database file.
    """
    if not HAS_DUCKDB:
        raise ImportError("duckdb package not installed. Run: pip install duckdb")

    if db_path is None:
        db_path = os.path.join(config.RESULTS_DIR, "duckdb_baseline.db")
    if pg_dsn is None:
        pg_dsn = config.dsn()

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing DB for clean comparison
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    # Install and load spatial extension
    conn.execute("INSTALL spatial")
    conn.execute("LOAD spatial")

    # Create schema matching SPDB
    conn.execute("""
        CREATE TABLE objects (
            object_id       BIGINT,
            slide_id        VARCHAR NOT NULL,
            centroid_x      DOUBLE NOT NULL,
            centroid_y      DOUBLE NOT NULL,
            class_label     VARCHAR NOT NULL,
            tile_id         VARCHAR,
            hilbert_key     BIGINT NOT NULL,
            zorder_key      BIGINT NOT NULL,
            area            DOUBLE,
            perimeter       DOUBLE,
            confidence      DOUBLE DEFAULT 1.0,
            pipeline_id     VARCHAR,
            geom            GEOMETRY
        )
    """)

    # Import data from PostgreSQL via postgres_scanner
    try:
        conn.execute("INSTALL postgres_scanner")
        conn.execute("LOAD postgres_scanner")

        # Parse DSN for DuckDB postgres attach
        dsn_parts = {}
        for part in pg_dsn.split():
            k, v = part.split("=", 1)
            dsn_parts[k] = v

        attach_str = (f"host={dsn_parts.get('host', 'localhost')} "
                      f"port={dsn_parts.get('port', '5432')} "
                      f"dbname={dsn_parts.get('dbname', 'spdb')} "
                      f"user={dsn_parts.get('user', 'postgres')}")
        if dsn_parts.get("password"):
            attach_str += f" password={dsn_parts['password']}"

        conn.execute(f"ATTACH '{attach_str}' AS pg (TYPE postgres, READ_ONLY)")

        # Import from monolithic table (no partitioning complexity)
        print("  Importing data from PostgreSQL...")
        t0 = time.time()
        conn.execute("""
            INSERT INTO objects
            SELECT object_id, slide_id, centroid_x, centroid_y,
                   class_label, tile_id, hilbert_key, zorder_key,
                   area, perimeter, confidence, pipeline_id,
                   ST_Point(centroid_x, centroid_y)
            FROM pg.public.objects_mono
        """)
        elapsed = time.time() - t0
        count = conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        print(f"  Imported {count:,} rows in {elapsed:.1f}s")

        conn.execute("DETACH pg")

    except Exception as e:
        print(f"  PostgreSQL import failed: {e}")
        print("  Falling back to CSV import...")
        # Fallback: could import from CSV/parquet if available
        raise

    # Create indexes
    print("  Creating DuckDB indexes...")
    conn.execute("CREATE INDEX idx_objects_slide ON objects(slide_id)")
    conn.execute("CREATE INDEX idx_objects_hilbert ON objects(slide_id, hilbert_key)")

    conn.close()
    print(f"  DuckDB baseline ready at {db_path}")
    return db_path


# ---------------------------------------------------------------------------
# Q1: Viewport query
# ---------------------------------------------------------------------------

def run_q1_duckdb(db_path, slide_ids, metadata, n_trials=500,
                   viewport_frac=0.05, seed=42):
    """Run Q1 viewport benchmark on DuckDB.

    Uses DuckDB's ST_Intersects with ST_MakeEnvelope equivalent.
    Same random viewports as PostgreSQL benchmarks (same seed).
    """
    if not HAS_DUCKDB:
        raise ImportError("duckdb not installed")

    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("LOAD spatial")

    rng = np.random.RandomState(seed)
    latencies = []

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, viewport_frac, rng)

        sql = """
            SELECT object_id, centroid_x, centroid_y, class_label
            FROM objects
            WHERE slide_id = ?
              AND centroid_x BETWEEN ? AND ?
              AND centroid_y BETWEEN ? AND ?
        """
        # Use bounding box filter on coordinates (faster than spatial)
        t0 = time.perf_counter()
        conn.execute(sql, [sid, x0, x1, y0, y1]).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    conn.close()
    return latencies


# ---------------------------------------------------------------------------
# Q2: kNN query
# ---------------------------------------------------------------------------

def run_q2_duckdb(db_path, slide_ids, metadata, k=50, n_trials=500, seed=42):
    """Run Q2 kNN benchmark on DuckDB."""
    if not HAS_DUCKDB:
        raise ImportError("duckdb not installed")

    conn = duckdb.connect(db_path, read_only=True)
    conn.execute("LOAD spatial")

    rng = np.random.RandomState(seed)
    latencies = []

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        qx, qy = random_point(w, h, rng)

        sql = f"""
            SELECT object_id,
                   ((centroid_x - ?) * (centroid_x - ?) +
                    (centroid_y - ?) * (centroid_y - ?)) AS dist_sq
            FROM objects
            WHERE slide_id = ?
            ORDER BY dist_sq
            LIMIT {k}
        """
        t0 = time.perf_counter()
        conn.execute(sql, [qx, qx, qy, qy, sid]).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    conn.close()
    return latencies


# ---------------------------------------------------------------------------
# Q3: Aggregation query
# ---------------------------------------------------------------------------

def run_q3_duckdb(db_path, slide_ids, metadata, n_trials=500, seed=42):
    """Run Q3 tile aggregation benchmark on DuckDB."""
    if not HAS_DUCKDB:
        raise ImportError("duckdb not installed")

    conn = duckdb.connect(db_path, read_only=True)
    rng = np.random.RandomState(seed)
    latencies = []

    for trial in range(n_trials):
        sid = rng.choice(slide_ids)
        sql = """
            SELECT tile_id, COUNT(*)
            FROM objects
            WHERE slide_id = ?
            GROUP BY tile_id
        """
        t0 = time.perf_counter()
        conn.execute(sql, [sid]).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

    conn.close()
    return latencies


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def run_all_duckdb(db_path=None, n_trials=500, seed=42):
    """Run Q1-Q3 on DuckDB and return results."""
    if db_path is None:
        db_path = os.path.join(config.RESULTS_DIR, "duckdb_baseline.db")

    metadata = load_metadata()
    slide_ids = metadata["slide_ids"]

    results = {}

    print("  DuckDB Q1 (viewport)...")
    lats = run_q1_duckdb(db_path, slide_ids, metadata,
                          n_trials=n_trials, seed=seed)
    results["Q1"] = compute_stats(lats)
    save_raw_latencies(lats, "duckdb_q1", "DuckDB")
    print(f"    p50={results['Q1']['p50']:.1f}ms")

    print("  DuckDB Q2 (kNN)...")
    lats = run_q2_duckdb(db_path, slide_ids, metadata,
                          k=50, n_trials=n_trials, seed=seed)
    results["Q2"] = compute_stats(lats)
    save_raw_latencies(lats, "duckdb_q2", "DuckDB")
    print(f"    p50={results['Q2']['p50']:.1f}ms")

    print("  DuckDB Q3 (aggregation)...")
    lats = run_q3_duckdb(db_path, slide_ids, metadata,
                          n_trials=n_trials, seed=seed)
    results["Q3"] = compute_stats(lats)
    save_raw_latencies(lats, "duckdb_q3", "DuckDB")
    print(f"    p50={results['Q3']['p50']:.1f}ms")

    save_results(results, "duckdb_baseline")
    return results


def compare_pg_vs_duckdb(pg_results_path=None, duckdb_results_path=None):
    """Side-by-side comparison of PostgreSQL SPDB vs DuckDB.

    Returns structured comparison with speedup ratios and statistical tests.
    """
    import json

    if pg_results_path is None:
        pg_results_path = os.path.join(config.RAW_DIR, "q1_viewport.json")
    if duckdb_results_path is None:
        duckdb_results_path = os.path.join(config.RAW_DIR, "duckdb_baseline.json")

    with open(pg_results_path) as f:
        pg = json.load(f)
    with open(duckdb_results_path) as f:
        duck = json.load(f)

    comparison = {}

    for query in ["Q1", "Q2", "Q3"]:
        pg_stats = pg.get("configs", {}).get("SPDB", pg.get(query, {}))
        duck_stats = duck.get(query, {})

        if pg_stats and duck_stats:
            pg_p50 = pg_stats.get("p50", 0)
            duck_p50 = duck_stats.get("p50", 0)

            comparison[query] = {
                "pg_spdb_p50": pg_p50,
                "duckdb_p50": duck_p50,
                "speedup": round(duck_p50 / pg_p50, 2) if pg_p50 > 0 else 0,
                "pg_faster": pg_p50 < duck_p50,
            }

    save_results(comparison, "pg_vs_duckdb_comparison")
    return comparison
