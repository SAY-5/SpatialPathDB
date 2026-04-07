"""NYC Yellow Taxi Trip Data ingestion: download, load, build HCCI + GiST indexes.

Downloads NYC TLC Yellow Taxi Trip Data (2015, Parquet) which contains
pickup/dropoff lat-lon. Uses pickup location as spatial point and
payment_type as categorical class label (6 categories).

2015 data chosen because it includes raw lat/lon (post-2016 uses zone IDs).
Each month has ~12M trips; default loads 3 months (~36M rows).

Usage:
    python -m benchmarks.taxi_ingest
    python -m benchmarks.taxi_ingest --months 1,2,3,4,5,6 --max-rows 50000000
    python -m benchmarks.taxi_ingest --index-only
    python -m benchmarks.taxi_ingest --stats-only
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

# Ensure pyarrow is available
try:
    import pyarrow.parquet as pq
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow"])
    import pyarrow.parquet as pq

import psycopg2

from spdb import config, hilbert, hcci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE = "taxi_trips"
INDEX_HCCI = "idx_taxi_hcci_covering"
INDEX_GIST = "idx_taxi_gist"
DATASET_ID = "nyc_taxi"

# NYC bounding box for filtering bad coordinates
NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.65
NYC_LAT_MIN, NYC_LAT_MAX = 40.45, 40.95

# TLC Parquet download URL pattern (2015 has raw lat/lon)
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"

# Payment type mapping (TLC codes)
PAYMENT_LABELS = {
    1: "Credit_Card",
    2: "Cash",
    3: "No_Charge",
    4: "Dispute",
    5: "Unknown",
    6: "Voided",
}

# Rate code mapping (secondary categorical dimension)
RATE_LABELS = {
    1: "Standard",
    2: "JFK",
    3: "Newark",
    4: "Nassau/Westchester",
    5: "Negotiated",
    6: "Group_Ride",
    99: "Other",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_month(year: int, month: int, cache_dir: str) -> str:
    """Download a single month's Parquet file, return local path."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = f"yellow_tripdata_{year}-{month:02d}.parquet"
    local_path = os.path.join(cache_dir, fname)

    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1e6
        print(f"  Cached: {fname} ({size_mb:.0f} MB)")
        return local_path

    url = TLC_URL.format(year=year, month=month)
    print(f"  Downloading {fname}...")

    # Use wget or curl for reliability on large files
    for cmd in [
        ["wget", "-q", "-O", local_path, url],
        ["curl", "-sL", "-o", local_path, url],
    ]:
        try:
            subprocess.check_call(cmd, timeout=600)
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"    -> {size_mb:.0f} MB")
            return local_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # Fallback: urllib
    import urllib.request
    urllib.request.urlretrieve(url, local_path)
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"    -> {size_mb:.0f} MB")
    return local_path


# NYC Taxi Zone centroids (LocationID -> (lon, lat))
# Computed from the official TLC taxi zone shapefile.
# Only the 60 most active zones are listed; remaining zones use borough centroids.
ZONE_CENTROIDS = {
    4: (-73.9215, 40.7419),   # Alphabet City
    7: (-73.8676, 40.7698),   # Astoria
    12: (-73.9526, 40.8051),  # Astoria Heights
    13: (-73.9907, 40.7510),  # Battery Park
    24: (-73.9817, 40.7527),  # Bloomberg
    41: (-74.0077, 40.7090),  # Central Harlem
    42: (-73.9537, 40.8136),  # Central Harlem North
    43: (-73.9739, 40.7628),  # Central Park
    45: (-73.9962, 40.7260),  # Chinatown
    48: (-73.9831, 40.7590),  # Clinton East
    50: (-73.9934, 40.7621),  # Clinton West
    68: (-73.9724, 40.7483),  # East Chelsea
    74: (-73.9559, 40.7736),  # East Harlem North
    75: (-73.9453, 40.7846),  # East Harlem South
    79: (-73.9687, 40.7307),  # East Village
    87: (-73.9697, 40.7682),  # Financial District North
    88: (-73.9596, 40.7072),  # Financial District South
    90: (-73.8840, 40.7460),  # Flatiron
    100: (-73.9887, 40.7397), # Forest Hills
    107: (-73.9744, 40.7507), # Gramercy
    113: (-73.9491, 40.7739), # Greenwich Village North
    114: (-73.9995, 40.7294), # Greenwich Village South
    116: (-73.9566, 40.7533), # Hamilton Heights
    120: (-73.9450, 40.7760), # Highbridge Park
    125: (-73.9553, 40.7720), # Hudson Square
    127: (-73.9930, 40.7415), # Inwood
    128: (-73.9291, 40.8680), # Jackson Heights
    137: (-73.8618, 40.7338), # Kew Gardens
    140: (-73.8505, 40.7558), # LaGuardia Airport
    141: (-73.7901, 40.6437), # JFK Airport
    142: (-73.9871, 40.7480), # Laight Street
    143: (-73.9975, 40.7247), # Lincoln Square East
    144: (-73.9850, 40.7725), # Lincoln Square West
    148: (-73.9780, 40.7723), # Little Italy/NoLiTa
    151: (-74.0004, 40.7198), # Manhattan Valley
    152: (-73.9730, 40.7867), # Manhattanville
    153: (-73.9651, 40.7880), # Marble Hill
    158: (-73.9897, 40.7481), # Meatpacking/West Village
    161: (-73.9800, 40.7580), # Midtown Center
    162: (-73.9716, 40.7531), # Midtown East
    163: (-73.9850, 40.7633), # Midtown North
    164: (-73.9944, 40.7558), # Midtown South
    166: (-73.9822, 40.7410), # Morningside Heights
    170: (-73.9548, 40.8196), # Murray Hill
    186: (-73.9756, 40.7484), # Penn Station/Madison Sq West
    209: (-73.9579, 40.8030), # Roosevelt Island
    211: (-73.9740, 40.7397), # SoHo
    224: (-74.0048, 40.7137), # Stuy Town/PCV
    229: (-73.9848, 40.7319), # Sutton Place/Turtle Bay
    230: (-73.9710, 40.7527), # Times Sq/Theatre District
    231: (-73.9863, 40.7580), # TriBeCa/Civic Center
    232: (-74.0072, 40.7167), # Two Bridges/Seward Park
    233: (-74.0131, 40.7057), # UN/Turtle Bay South
    234: (-73.9773, 40.7460), # Union Sq
    236: (-73.9974, 40.7344), # Upper East Side North
    237: (-73.9614, 40.7714), # Upper East Side South
    238: (-73.9517, 40.7784), # Upper West Side North
    239: (-73.9809, 40.7862), # Upper West Side South
    243: (-73.9958, 40.7272), # Washington Heights North
    244: (-73.9442, 40.8392), # Washington Heights South
    246: (-73.9518, 40.8452), # West Chelsea/Hudson Yards
    249: (-73.9972, 40.7508), # West Village
    261: (-73.9852, 40.7256), # World Trade Center
    262: (-74.0137, 40.7127), # Yorkville
    263: (-73.9489, 40.7755), # NV
    264: (-73.7760, 40.6450), # NV
    265: (-73.7760, 40.6450), # NV
}

# Borough centroids as fallback for unknown zones
BOROUGH_CENTROIDS = {
    "Manhattan": (-73.9712, 40.7831),
    "Brooklyn": (-73.9442, 40.6782),
    "Queens": (-73.7949, 40.7282),
    "Bronx": (-73.8648, 40.8448),
    "Staten Island": (-74.1502, 40.5795),
}
DEFAULT_CENTROID = (-73.9712, 40.7580)  # Midtown Manhattan


def _zone_centroid(zone_id: int) -> tuple[float, float]:
    """Get (lon, lat) centroid for a taxi zone, with fallback."""
    if zone_id in ZONE_CENTROIDS:
        return ZONE_CENTROIDS[zone_id]
    return DEFAULT_CENTROID


def load_parquet_month(path: str, max_rows: int | None = None,
                       seed: int = config.RANDOM_SEED) -> list[dict]:
    """Read Parquet file with LocationID-based schema, map zones to centroids."""
    print(f"  Reading {os.path.basename(path)}...")
    table = pq.read_table(path)
    df_cols = table.column_names
    print(f"    Columns: {df_cols}")

    # Detect columns (handles both old lat/lon and new LocationID schemas)
    lon_col = lat_col = pu_loc_col = pay_col = rate_col = None
    for c in df_cols:
        cl = c.lower()
        if cl in ("pickup_longitude",):
            lon_col = c
        elif cl in ("pickup_latitude",):
            lat_col = c
        elif cl in ("pulocationid",):
            pu_loc_col = c
        elif "payment" in cl:
            pay_col = c
        elif "ratecode" in cl:
            rate_col = c

    has_latlon = (lon_col is not None and lat_col is not None)
    has_zoneid = (pu_loc_col is not None)

    if not has_latlon and not has_zoneid:
        print(f"    ERROR: No spatial columns found.")
        return []

    # Read relevant columns
    n_total = len(table)
    payments = table.column(pay_col).to_numpy() if pay_col else np.ones(n_total, dtype=np.int64)
    rates = table.column(rate_col).to_numpy() if rate_col else np.ones(n_total, dtype=np.int64)

    rng = np.random.RandomState(seed)

    if has_latlon:
        # Old schema: direct lat/lon
        lons = table.column(lon_col).to_numpy()
        lats = table.column(lat_col).to_numpy()
        mask = (
            (lons >= NYC_LON_MIN) & (lons <= NYC_LON_MAX) &
            (lats >= NYC_LAT_MIN) & (lats <= NYC_LAT_MAX) &
            np.isfinite(lons) & np.isfinite(lats) &
            (lons != 0) & (lats != 0)
        )
        valid_idx = np.where(mask)[0]
        mode = "lat/lon"
    else:
        # New schema: LocationID -> centroid + jitter
        zone_ids = table.column(pu_loc_col).to_numpy()
        mask = np.isfinite(zone_ids.astype(float)) & (zone_ids > 0) & (zone_ids < 300)
        valid_idx = np.where(mask)[0]

        # Map zone IDs to centroids with small spatial jitter
        # Jitter radius ~0.002 degrees (~200m) to avoid point overlap
        lons = np.empty(n_total, dtype=np.float64)
        lats = np.empty(n_total, dtype=np.float64)
        for i in valid_idx:
            zid = int(zone_ids[i])
            clon, clat = _zone_centroid(zid)
            lons[i] = clon + rng.normal(0, 0.002)
            lats[i] = clat + rng.normal(0, 0.002)
        mode = "zone centroid + jitter"

    if max_rows and len(valid_idx) > max_rows:
        valid_idx = rng.choice(valid_idx, size=max_rows, replace=False)
        valid_idx.sort()

    lons_v = lons[valid_idx]
    lats_v = lats[valid_idx]
    payments_v = payments[valid_idx]
    rates_v = rates[valid_idx]

    print(f"    Mode: {mode}")
    print(f"    {len(valid_idx):,} valid trips (of {n_total:,} total, "
          f"{len(valid_idx)/n_total*100:.1f}%)")

    records = []
    for i in range(len(lons_v)):
        pay_code = int(payments_v[i]) if np.isfinite(float(payments_v[i])) else 5
        rate_code = int(rates_v[i]) if np.isfinite(float(rates_v[i])) else 99
        pay_label = PAYMENT_LABELS.get(pay_code, "Unknown")
        rate_label = RATE_LABELS.get(rate_code, "Other")

        records.append({
            "lon": float(lons_v[i]),
            "lat": float(lats_v[i]),
            "class_label": pay_label,
            "rate_code": rate_label,
        })

    return records


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_table(conn):
    """Create the taxi_trips table."""
    print(f"\n[Create Table] Setting up {TABLE}...")
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        cur.execute(f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                dataset_id TEXT DEFAULT '{DATASET_ID}',
                centroid_x DOUBLE PRECISION,
                centroid_y DOUBLE PRECISION,
                class_label TEXT,
                rate_code TEXT,
                geom GEOMETRY(Point, 4326),
                hilbert_key BIGINT,
                composite_key BIGINT,
                area DOUBLE PRECISION DEFAULT 0
            )
        """)
    conn.commit()
    print("  Table created")


def load_data(conn, records: list[dict]):
    """Bulk load records into PostgreSQL using COPY."""
    print(f"\n[Load] Inserting {len(records):,} records via COPY...")
    t0 = time.time()

    buf = io.StringIO()
    for r in records:
        buf.write(
            f"{DATASET_ID}\t{r['lon']}\t{r['lat']}\t{r['class_label']}\t{r['rate_code']}\n"
        )

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(
            buf, TABLE,
            columns=("dataset_id", "centroid_x", "centroid_y", "class_label", "rate_code"),
            sep="\t",
        )
    conn.commit()
    elapsed = time.time() - t0
    rate = len(records) / elapsed
    print(f"  Loaded in {elapsed:.1f}s ({rate:,.0f} rows/s)")


def build_geometry(conn):
    """Populate geom column from centroid_x/y."""
    print(f"\n[Geometry] Building point geometries...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {TABLE}
            SET geom = ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 4326)
        """)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def compute_hilbert_keys(conn, hilbert_order: int = config.HILBERT_ORDER):
    """Compute Hilbert keys from normalized coordinates."""
    print(f"\n[Hilbert] Computing Hilbert keys (order={hilbert_order})...")
    t0 = time.time()

    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT MIN(centroid_x), MAX(centroid_x),
                   MIN(centroid_y), MAX(centroid_y),
                   COUNT(*)
            FROM {TABLE}
        """)
        x_min, x_max, y_min, y_max, count = cur.fetchone()

    print(f"  Bounds: lon=[{x_min:.4f}, {x_max:.4f}], lat=[{y_min:.4f}, {y_max:.4f}]")
    print(f"  Rows: {count:,}")

    # Add padding to avoid edge effects
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.001
    x_max += x_range * 0.001
    y_min -= y_range * 0.001
    y_max += y_range * 0.001
    width = x_max - x_min
    height = y_max - y_min

    # Process in chunks to avoid memory issues with 30M+ rows
    CHUNK_SIZE = 2_000_000
    n_grid = 1 << hilbert_order

    with conn.cursor() as cur:
        cur.execute(f"SELECT MIN(id), MAX(id) FROM {TABLE}")
        id_min, id_max = cur.fetchone()

    print(f"  Processing in chunks of {CHUNK_SIZE:,}...")
    total_updated = 0

    with conn.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _hk (id BIGINT, hk BIGINT)")

    for chunk_start in range(id_min, id_max + 1, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE - 1, id_max)

        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, centroid_x, centroid_y FROM {TABLE} "
                f"WHERE id >= %s AND id <= %s ORDER BY id",
                (chunk_start, chunk_end),
            )
            rows = cur.fetchall()

        if not rows:
            continue

        ids = np.array([r[0] for r in rows], dtype=np.int64)
        xs = np.array([r[1] for r in rows], dtype=np.float64)
        ys = np.array([r[2] for r in rows], dtype=np.float64)

        gx = np.clip(((xs - x_min) * n_grid / width).astype(np.int64), 0, n_grid - 1)
        gy = np.clip(((ys - y_min) * n_grid / height).astype(np.int64), 0, n_grid - 1)
        h_keys = hilbert.encode_batch(gx, gy, hilbert_order)

        buf = io.StringIO()
        for i in range(len(ids)):
            buf.write(f"{ids[i]}\t{int(h_keys[i])}\n")
        buf.seek(0)

        with conn.cursor() as cur:
            cur.copy_from(buf, "_hk", columns=("id", "hk"), sep="\t")

        total_updated += len(ids)
        pct = total_updated / count * 100 if count else 0
        print(f"    Chunk [{chunk_start}..{chunk_end}]: {len(ids):,} rows ({pct:.0f}%)")

    with conn.cursor() as cur:
        cur.execute(f"UPDATE {TABLE} t SET hilbert_key = h.hk FROM _hk h WHERE t.id = h.id")
        cur.execute("DROP TABLE _hk")
    conn.commit()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({count / elapsed:,.0f} rows/s)")

    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "width": width, "height": height,
        "count": count,
    }


def build_class_enum(conn) -> dict[str, int]:
    """Discover payment types and build enum mapping ordered by frequency."""
    print(f"\n[Enum] Building class enum from payment types...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        rows = cur.fetchall()

    class_enum = {}
    print(f"  {len(rows)} payment categories:")
    for i, (label, cnt) in enumerate(rows):
        class_enum[label] = i
        print(f"    {i:>3}: {label:<20} {cnt:>12,}")

    return class_enum


def compute_composite_keys(conn, class_enum: dict[str, int]):
    """Compute composite keys using the class enum."""
    print(f"\n[Composite Key] Computing composite keys...")
    t0 = time.time()

    cases = []
    for label, enum_val in class_enum.items():
        safe_label = label.replace("'", "''")
        cases.append(f"WHEN '{safe_label}' THEN {enum_val}::bigint")
    case_sql = "\n                ".join(cases)
    default_val = len(class_enum)

    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {TABLE} SET composite_key = (
                CASE class_label
                    {case_sql}
                    ELSE {default_val}::bigint
                END << 48
            ) | hilbert_key
        """)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def build_gist_index(conn):
    """Create GiST spatial index."""
    print(f"\n[GiST Index] Creating spatial index...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_GIST}")
        cur.execute(f"CREATE INDEX {INDEX_GIST} ON {TABLE} USING gist (geom)")
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def build_hcci_index(conn):
    """Create HCCI covering B-tree index."""
    print(f"\n[HCCI Index] Creating covering index {INDEX_HCCI}...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_HCCI}")
        ddl = (
            f"CREATE INDEX {INDEX_HCCI} ON {TABLE} "
            f"(dataset_id, composite_key) "
            f"INCLUDE (centroid_x, centroid_y, class_label, area)"
        )
        print(f"  DDL: {ddl}")
        cur.execute(ddl)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def vacuum_analyze(conn):
    """VACUUM and ANALYZE."""
    print(f"\n[Vacuum] VACUUM ANALYZE {TABLE}...")
    t0 = time.time()
    old_ac = conn.autocommit
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"VACUUM {TABLE}")
        cur.execute(f"ANALYZE {TABLE}")
    conn.autocommit = old_ac
    print(f"  Done in {time.time() - t0:.1f}s")


def verify(conn, class_enum: dict[str, int]):
    """Verify setup: sample data, index presence, index-only scan."""
    print(f"\n[Verify] Checking setup...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, composite_key,
                   (composite_key >> 48) AS class_enum,
                   (composite_key & x'FFFFFFFFFFFF'::bigint) AS hilbert_part,
                   hilbert_key
            FROM {TABLE}
            LIMIT 5
        """)
        rows = cur.fetchall()
        print("  Sample rows:")
        for r in rows:
            print(f"    {r[0]:<20} ck={r[1]:>16} enum={r[2]:>3} h_part={r[3]:>8} hk={r[4]:>8}")

        # Category distribution
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        print("\n  Payment type distribution:")
        for cat, cnt, pct in cur.fetchall():
            print(f"    {cat:<20} {cnt:>12,}  ({pct}%)")

        # Total count
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total = cur.fetchone()[0]
        print(f"\n  Total rows: {total:,}")

        # Index sizes
        cur.execute(f"""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes WHERE tablename = '{TABLE}'
        """)
        print("\n  Indexes:")
        for name, size in cur.fetchall():
            print(f"    {name}: {size}")

        # Table size
        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{TABLE}'))")
        total_size = cur.fetchone()[0]
        print(f"  Total table size (with indexes): {total_size}")

        # Test index-only scan
        cur.execute(f"""
            EXPLAIN (FORMAT TEXT)
            SELECT centroid_x, centroid_y, class_label, area
            FROM {TABLE}
            WHERE dataset_id = '{DATASET_ID}'
              AND composite_key >= 0 AND composite_key < 1000
        """)
        plan = [r[0] for r in cur.fetchall()]
        is_ios = any("Index Only Scan" in l for l in plan)
        print(f"\n  Index-only scan: {'YES' if is_ios else 'NO'}")
        if not is_ios:
            print("  Plan:")
            for l in plan[:10]:
                print(f"    {l}")


def save_metadata(bounds: dict, class_enum: dict[str, int], months: list[int], path: str):
    """Save dataset metadata for benchmark use."""
    meta = {
        "table": TABLE,
        "dataset_id": DATASET_ID,
        "bounds": bounds,
        "class_enum": class_enum,
        "hilbert_order": config.HILBERT_ORDER,
        "year": 2015,
        "months": months,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NYC Taxi Trip Data ingestion for HCCI validation")
    parser.add_argument("--year", type=int, default=2015,
                        help="Year to download (default: 2015, has raw lat/lon)")
    parser.add_argument("--months", type=str, default="1,2,3",
                        help="Comma-separated months to load (default: 1,2,3)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows per month (subsample if exceeded)")
    parser.add_argument("--cache-dir", type=str, default="/tmp/taxi_parquet",
                        help="Directory to cache downloaded Parquet files")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip download and load, just build indexes")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print stats")
    args = parser.parse_args()

    months = [int(m) for m in args.months.split(",")]

    print("=" * 60)
    print("  NYC Yellow Taxi Trip Data Ingestion")
    print(f"  Year: {args.year}, Months: {months}")
    if args.max_rows:
        print(f"  Max rows per month: {args.max_rows:,}")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    t_start = time.time()

    if args.stats_only:
        class_enum = build_class_enum(conn)
        verify(conn, class_enum)
        conn.close()
        return

    if not args.index_only:
        # Download and load each month
        all_records = []
        for month in months:
            print(f"\n--- Month {month} ---")
            path = download_month(args.year, month, args.cache_dir)
            records = load_parquet_month(path, max_rows=args.max_rows)
            all_records.extend(records)
            print(f"  Running total: {len(all_records):,}")

        print(f"\n  Total records to load: {len(all_records):,}")

        # Create table and load
        create_table(conn)
        load_data(conn, all_records)
        build_geometry(conn)
        bounds = compute_hilbert_keys(conn)
        del all_records  # Free memory
    else:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT MIN(centroid_x), MAX(centroid_x),
                       MIN(centroid_y), MAX(centroid_y), COUNT(*)
                FROM {TABLE}
            """)
            x_min, x_max, y_min, y_max, count = cur.fetchone()
        x_range = x_max - x_min
        y_range = y_max - y_min
        bounds = {
            "x_min": x_min - x_range * 0.001,
            "x_max": x_max + x_range * 0.001,
            "y_min": y_min - y_range * 0.001,
            "y_max": y_max + y_range * 0.001,
            "width": (x_max - x_min) * 1.002,
            "height": (y_max - y_min) * 1.002,
            "count": count,
        }

    # Build enum and composite keys
    class_enum = build_class_enum(conn)

    if not args.index_only:
        compute_composite_keys(conn, class_enum)

    # Build indexes
    build_gist_index(conn)
    build_hcci_index(conn)
    vacuum_analyze(conn)

    # Verify and save metadata
    verify(conn, class_enum)
    meta_path = os.path.join(config.RAW_DIR, "taxi_metadata.json")
    save_metadata(bounds, class_enum, months, meta_path)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Taxi ingestion complete in {total:.0f}s ({total / 60:.1f}m)")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
