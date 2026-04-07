#!/usr/bin/env python3
"""
osm_large_benchmark.py – Download 10M+ building footprints from OpenStreetMap
and benchmark three PostgreSQL/PostGIS table configurations.

Designed for an AWS r5.4xlarge Ubuntu instance with:
  - PostgreSQL + PostGIS already configured
  - Database: spdb, User: postgres, Host: localhost
  - 128 GB RAM (shared_buffers tuned accordingly)

Usage:
    python3 osm_large_benchmark.py [--download-only] [--benchmark-only] [--cold-only]

The script is fully self-contained.  All helpers are defined inline.
"""

# ---------------------------------------------------------------------------
# 0.  Install runtime dependencies (idempotent)
# ---------------------------------------------------------------------------
import subprocess, sys

def _ensure_packages():
    required = {
        "psycopg2": "psycopg2-binary",
        "requests": "requests",
        "shapely":  "shapely",
    }
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"[setup] Installing {pip_name} …")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
            )

_ensure_packages()

# ---------------------------------------------------------------------------
# 1.  Imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import struct
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import requests
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# 2.  Constants & configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR  = SCRIPT_DIR / "cache" / "osm_buildings"
RESULTS_DIR = SCRIPT_DIR / "results" / "raw"

DB_NAME = "spdb"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Max buildings per single Overpass request – split into grid tiles if above
MAX_BUILDINGS_PER_REQUEST = 500_000

# Hilbert curve order (2^10 = 1024 on each axis → ~1M cells)
HILBERT_ORDER = 10
HILBERT_N = 1 << HILBERT_ORDER  # 1024

# Number of sub-partitions per city for SPDB config
SPDB_SUB_PARTITIONS = 30

# Benchmark parameters
VIEWPORT_QUERIES  = 500
KNN_QUERIES       = 500
COLD_QUERIES      = 200
KNN_K             = 50
VIEWPORT_F_VALUES = [0.05, 0.01]  # 5% and 1%

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 3.  City definitions  (name, city_id, south, west, north, east)
# ---------------------------------------------------------------------------
CITIES: List[Dict[str, Any]] = [
    {"name": "New York City",  "city_id": 1,  "bbox": (40.49, -74.26, 40.92, -73.70)},
    {"name": "Los Angeles",    "city_id": 2,  "bbox": (33.70, -118.67, 34.34, -118.16)},
    {"name": "Chicago",        "city_id": 3,  "bbox": (41.64, -87.94, 42.02, -87.52)},
    {"name": "Houston",        "city_id": 4,  "bbox": (29.52, -95.79, 30.11, -95.01)},
    {"name": "Phoenix",        "city_id": 5,  "bbox": (33.29, -112.33, 33.75, -111.93)},
    {"name": "Philadelphia",   "city_id": 6,  "bbox": (39.87, -75.28, 40.14, -74.96)},
    {"name": "San Antonio",    "city_id": 7,  "bbox": (29.25, -98.72, 29.65, -98.35)},
    {"name": "San Diego",      "city_id": 8,  "bbox": (32.53, -117.28, 32.96, -116.93)},
    {"name": "Dallas",         "city_id": 9,  "bbox": (32.62, -96.99, 33.02, -96.56)},
    {"name": "San Jose",       "city_id": 10, "bbox": (37.13, -122.05, 37.47, -121.73)},
    {"name": "Austin",         "city_id": 11, "bbox": (30.10, -97.94, 30.52, -97.57)},
    {"name": "Jacksonville",   "city_id": 12, "bbox": (30.10, -81.77, 30.59, -81.39)},
    {"name": "Fort Worth",     "city_id": 13, "bbox": (32.55, -97.53, 32.96, -97.15)},
    {"name": "Columbus",       "city_id": 14, "bbox": (39.81, -83.13, 40.13, -82.77)},
    {"name": "Charlotte",      "city_id": 15, "bbox": (35.05, -80.95, 35.39, -80.70)},
    {"name": "Indianapolis",   "city_id": 16, "bbox": (39.63, -86.33, 39.93, -85.97)},
    {"name": "San Francisco",  "city_id": 17, "bbox": (37.70, -122.52, 37.81, -122.35)},
    {"name": "Seattle",        "city_id": 18, "bbox": (47.49, -122.44, 47.74, -122.24)},
    {"name": "Denver",         "city_id": 19, "bbox": (39.61, -105.11, 39.81, -104.87)},
    {"name": "Washington DC",  "city_id": 20, "bbox": (38.79, -77.12, 38.99, -76.91)},
    {"name": "Boston",         "city_id": 21, "bbox": (42.23, -71.19, 42.40, -70.99)},
    {"name": "Nashville",      "city_id": 22, "bbox": (35.97, -86.97, 36.30, -86.62)},
    {"name": "Detroit",        "city_id": 23, "bbox": (42.26, -83.29, 42.45, -82.91)},
    {"name": "Portland",       "city_id": 24, "bbox": (45.43, -122.84, 45.59, -122.57)},
    {"name": "Memphis",        "city_id": 25, "bbox": (34.99, -90.15, 35.23, -89.81)},
    {"name": "Atlanta",        "city_id": 26, "bbox": (33.65, -84.55, 33.89, -84.29)},
    {"name": "Miami",          "city_id": 27, "bbox": (25.71, -80.32, 25.86, -80.13)},
]

# ---------------------------------------------------------------------------
# 4.  Hilbert curve helpers  (xy2d, d2xy for order-N curve)
# ---------------------------------------------------------------------------

def _rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """Rotate/flip a quadrant."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def xy2d(order: int, x: int, y: int) -> int:
    """Convert (x, y) in [0, 2^order) to Hilbert distance d."""
    n = 1 << order
    d = 0
    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(s, x, y, rx, ry)
        s >>= 1
    return d


def lonlat_to_hilbert(lon: float, lat: float, order: int = HILBERT_ORDER) -> int:
    """Map WGS-84 (lon, lat) → Hilbert key in [0, 4^order)."""
    n = 1 << order
    # Normalise to [0, 1)
    nx = (lon + 180.0) / 360.0
    ny = (lat + 90.0) / 180.0
    # Clamp to valid grid cell
    ix = max(0, min(n - 1, int(nx * n)))
    iy = max(0, min(n - 1, int(ny * n)))
    return xy2d(order, ix, iy)


# ---------------------------------------------------------------------------
# 5.  Overpass download helpers
# ---------------------------------------------------------------------------

def _bbox_area_deg2(s: float, w: float, n: float, e: float) -> float:
    return (n - s) * (e - w)


def _estimate_buildings(s: float, w: float, n: float, e: float) -> float:
    """Very rough estimate of building count for a US-city bbox."""
    area = _bbox_area_deg2(s, w, n, e)
    # Rough US average: ~5000 buildings per square degree (varies hugely)
    return area * 50_000


def split_bbox_into_tiles(
    s: float, w: float, n: float, e: float,
    max_buildings: int = MAX_BUILDINGS_PER_REQUEST,
) -> List[Tuple[float, float, float, float]]:
    """
    Split a bounding box into a grid of tiles so that each tile
    is estimated to contain fewer than max_buildings.
    """
    est = _estimate_buildings(s, w, n, e)
    if est <= max_buildings:
        return [(s, w, n, e)]

    # Determine grid size
    n_splits = max(2, math.ceil(math.sqrt(est / max_buildings)))
    lat_step = (n - s) / n_splits
    lon_step = (e - w) / n_splits

    tiles = []
    for i in range(n_splits):
        for j in range(n_splits):
            ts = s + i * lat_step
            tn = s + (i + 1) * lat_step
            tw = w + j * lon_step
            te = w + (j + 1) * lon_step
            tiles.append((ts, tw, tn, te))
    return tiles


def _cache_path(city_name: str, tile_idx: int) -> Path:
    safe = city_name.replace(" ", "_").lower()
    return CACHE_DIR / f"{safe}_tile{tile_idx:03d}.json"


def _overpass_query_buildings(
    s: float, w: float, n: float, e: float,
    timeout: int = 900,
) -> str:
    """Return Overpass QL that fetches building=* centroids in bbox."""
    return f"""
[out:json][timeout:{timeout}];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out center;
"""


def _fetch_overpass(query: str, max_retries: int = 5) -> dict:
    """Send query to Overpass with round-robin servers and exponential backoff."""
    servers = list(OVERPASS_SERVERS)
    random.shuffle(servers)

    for attempt in range(max_retries):
        server = servers[attempt % len(servers)]
        wait = min(2 ** attempt * 10, 300)  # 10, 20, 40, 80, 160 …
        try:
            print(f"    [overpass] attempt {attempt+1}/{max_retries}  server={server}")
            resp = requests.post(
                server,
                data={"data": query},
                timeout=960,
                headers={"User-Agent": "SpatialPathDB-Benchmark/1.0"},
            )
            if resp.status_code == 429 or resp.status_code == 504:
                print(f"    [overpass] HTTP {resp.status_code} – backing off {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"    [overpass] error: {exc} – backing off {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Overpass query failed after {max_retries} retries")


def _extract_centroids(data: dict) -> List[Tuple[float, float]]:
    """Extract (lon, lat) centroids from Overpass JSON response."""
    centroids = []
    for el in data.get("elements", []):
        if "center" in el:
            centroids.append((el["center"]["lon"], el["center"]["lat"]))
        elif "lat" in el and "lon" in el:
            centroids.append((el["lon"], el["lat"]))
    return centroids


def download_city(city: Dict[str, Any]) -> List[Tuple[float, float]]:
    """
    Download building centroids for one city.  Caches to JSON files so
    re-runs skip completed tiles.
    """
    name = city["name"]
    s, w, n, e = city["bbox"]
    tiles = split_bbox_into_tiles(s, w, n, e)
    print(f"  [{name}] bbox=({s},{w},{n},{e})  tiles={len(tiles)}")

    all_centroids: List[Tuple[float, float]] = []
    for idx, (ts, tw, tn, te) in enumerate(tiles):
        cp = _cache_path(name, idx)
        if cp.exists():
            with open(cp) as f:
                cached = json.load(f)
            all_centroids.extend([(c[0], c[1]) for c in cached])
            print(f"    tile {idx}: {len(cached)} buildings (cached)")
            continue

        try:
            query = _overpass_query_buildings(ts, tw, tn, te)
            data = _fetch_overpass(query)
            centroids = _extract_centroids(data)

            # Cache
            with open(cp, "w") as f:
                json.dump(centroids, f)
            all_centroids.extend(centroids)
            print(f"    tile {idx}: {len(centroids)} buildings (downloaded)")
            # Polite pause between tiles
            time.sleep(5)
        except Exception as exc:
            print(f"    tile {idx}: SKIPPED ({exc})")
            continue

    print(f"  [{name}] total buildings: {len(all_centroids)}")
    return all_centroids


def download_all_cities() -> Dict[int, List[Tuple[float, float]]]:
    """Download buildings for every city.  Returns {city_id: [(lon,lat), …]}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    city_buildings: Dict[int, List[Tuple[float, float]]] = {}
    total = 0
    for city in CITIES:
        try:
            centroids = download_city(city)
            city_buildings[city["city_id"]] = centroids
            total += len(centroids)
            print(f"  Running total: {total:,} buildings\n")
        except Exception as exc:
            print(f"  [{city['name']}] SKIPPED due to error: {exc}\n")
            continue

    print(f"\n=== Download complete: {total:,} buildings across {len(city_buildings)} cities ===\n")
    return city_buildings


# ---------------------------------------------------------------------------
# 6.  Database helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, host=DB_HOST, port=DB_PORT
    )


def exec_sql(conn, sql: str, params=None, fetch: bool = False):
    """Execute SQL.  Returns rows if fetch=True."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        if fetch:
            return cur.fetchall()
    conn.commit()


def ensure_postgis(conn):
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS postgis;")
    print("[db] PostGIS extension ensured")


# ---------------------------------------------------------------------------
# 7.  Table creation – three configurations
# ---------------------------------------------------------------------------

def _drop_table(conn, name: str):
    exec_sql(conn, f"DROP TABLE IF EXISTS {name} CASCADE;")


# ---- 7a. Monolithic table with GiST index ----------------------------------

def create_monolithic(conn):
    _drop_table(conn, "osm_large_mono")
    exec_sql(conn, """
        CREATE TABLE osm_large_mono (
            id          BIGSERIAL PRIMARY KEY,
            city_id     INTEGER   NOT NULL,
            lon         DOUBLE PRECISION NOT NULL,
            lat         DOUBLE PRECISION NOT NULL,
            hilbert_key BIGINT    NOT NULL,
            geom        GEOMETRY(Point, 4326) NOT NULL
        );
    """)
    print("[db] created osm_large_mono")


def index_monolithic(conn):
    print("[db] creating GiST index on osm_large_mono …")
    exec_sql(conn, """
        CREATE INDEX idx_mono_geom ON osm_large_mono USING gist (geom);
    """)
    exec_sql(conn, "ANALYZE osm_large_mono;")
    print("[db] GiST index + ANALYZE done on osm_large_mono")


# ---- 7b. LIST partitioned by city (slide-ordered) --------------------------

def create_so_partitioned(conn):
    _drop_table(conn, "osm_large_so")
    exec_sql(conn, """
        CREATE TABLE osm_large_so (
            id          BIGSERIAL,
            city_id     INTEGER   NOT NULL,
            lon         DOUBLE PRECISION NOT NULL,
            lat         DOUBLE PRECISION NOT NULL,
            hilbert_key BIGINT    NOT NULL,
            geom        GEOMETRY(Point, 4326) NOT NULL
        ) PARTITION BY LIST (city_id);
    """)
    for city in CITIES:
        cid = city["city_id"]
        exec_sql(conn, f"""
            CREATE TABLE osm_large_so_c{cid}
                PARTITION OF osm_large_so
                FOR VALUES IN ({cid});
        """)
    print(f"[db] created osm_large_so with {len(CITIES)} list partitions")


def index_so_partitioned(conn):
    print("[db] creating GiST indexes on osm_large_so partitions …")
    for city in CITIES:
        cid = city["city_id"]
        exec_sql(conn, f"""
            CREATE INDEX idx_so_c{cid}_geom
                ON osm_large_so_c{cid} USING gist (geom);
        """)
    exec_sql(conn, "ANALYZE osm_large_so;")
    print("[db] GiST indexes + ANALYZE done on osm_large_so")


# ---- 7c. SPDB: LIST by city + RANGE by hilbert_key -------------------------

def _hilbert_range_bounds(n_parts: int, order: int = HILBERT_ORDER) -> List[int]:
    """Return n_parts+1 boundary values splitting [0, 4^order) evenly."""
    max_key = 4 ** order
    return [int(i * max_key / n_parts) for i in range(n_parts + 1)]


def create_spdb_partitioned(conn):
    _drop_table(conn, "osm_large_spdb")
    exec_sql(conn, """
        CREATE TABLE osm_large_spdb (
            id          BIGSERIAL,
            city_id     INTEGER   NOT NULL,
            lon         DOUBLE PRECISION NOT NULL,
            lat         DOUBLE PRECISION NOT NULL,
            hilbert_key BIGINT    NOT NULL,
            geom        GEOMETRY(Point, 4326) NOT NULL
        ) PARTITION BY LIST (city_id);
    """)

    bounds = _hilbert_range_bounds(SPDB_SUB_PARTITIONS)

    for city in CITIES:
        cid = city["city_id"]
        # Intermediate partition for this city (range on hilbert_key)
        exec_sql(conn, f"""
            CREATE TABLE osm_large_spdb_c{cid}
                PARTITION OF osm_large_spdb
                FOR VALUES IN ({cid})
                PARTITION BY RANGE (hilbert_key);
        """)
        for pidx in range(SPDB_SUB_PARTITIONS):
            lo = bounds[pidx]
            hi = bounds[pidx + 1]
            # Use MAXVALUE for the last sub-partition to catch any overflow
            if pidx == SPDB_SUB_PARTITIONS - 1:
                exec_sql(conn, f"""
                    CREATE TABLE osm_large_spdb_c{cid}_h{pidx}
                        PARTITION OF osm_large_spdb_c{cid}
                        FOR VALUES FROM ({lo}) TO (MAXVALUE);
                """)
            else:
                exec_sql(conn, f"""
                    CREATE TABLE osm_large_spdb_c{cid}_h{pidx}
                        PARTITION OF osm_large_spdb_c{cid}
                        FOR VALUES FROM ({lo}) TO ({hi});
                """)

    total_parts = len(CITIES) * SPDB_SUB_PARTITIONS
    print(f"[db] created osm_large_spdb with {len(CITIES)} cities × "
          f"{SPDB_SUB_PARTITIONS} hilbert sub-partitions = {total_parts} leaves")


def index_spdb_partitioned(conn):
    print("[db] creating GiST indexes on osm_large_spdb sub-partitions …")
    for city in CITIES:
        cid = city["city_id"]
        for pidx in range(SPDB_SUB_PARTITIONS):
            exec_sql(conn, f"""
                CREATE INDEX idx_spdb_c{cid}_h{pidx}_geom
                    ON osm_large_spdb_c{cid}_h{pidx} USING gist (geom);
            """)
    exec_sql(conn, "ANALYZE osm_large_spdb;")
    print("[db] GiST indexes + ANALYZE done on osm_large_spdb")


# ---------------------------------------------------------------------------
# 8.  Data loading
# ---------------------------------------------------------------------------

BATCH_SIZE = 50_000  # rows per COPY batch


def _generate_rows(city_buildings: Dict[int, List[Tuple[float, float]]]):
    """Yield (city_id, lon, lat, hilbert_key, wkt) tuples."""
    for city_id, centroids in city_buildings.items():
        for lon, lat in centroids:
            hk = lonlat_to_hilbert(lon, lat)
            wkt = f"SRID=4326;POINT({lon} {lat})"
            yield (city_id, lon, lat, hk, wkt)


def load_table(conn, table_name: str, city_buildings: Dict[int, List[Tuple[float, float]]]):
    """Bulk-load rows into *table_name* using COPY."""
    from io import StringIO

    total = sum(len(v) for v in city_buildings.values())
    print(f"[load] inserting {total:,} rows into {table_name} …")

    loaded = 0
    buf = StringIO()
    for city_id, lon, lat, hk, wkt in _generate_rows(city_buildings):
        buf.write(f"{city_id}\t{lon}\t{lat}\t{hk}\t{wkt}\n")
        loaded += 1
        if loaded % BATCH_SIZE == 0:
            buf.seek(0)
            with conn.cursor() as cur:
                cur.copy_from(
                    buf, table_name,
                    columns=("city_id", "lon", "lat", "hilbert_key", "geom"),
                    sep="\t",
                )
            conn.commit()
            buf = StringIO()
            if loaded % 500_000 == 0:
                print(f"    {loaded:,} / {total:,} ({100*loaded/total:.1f}%)")

    # Flush remainder
    if buf.tell() > 0:
        buf.seek(0)
        with conn.cursor() as cur:
            cur.copy_from(
                buf, table_name,
                columns=("city_id", "lon", "lat", "hilbert_key", "geom"),
                sep="\t",
            )
        conn.commit()

    print(f"[load] {loaded:,} rows loaded into {table_name}")


# ---------------------------------------------------------------------------
# 9.  Query generators
# ---------------------------------------------------------------------------

def _city_bbox(city_id: int) -> Tuple[float, float, float, float]:
    for c in CITIES:
        if c["city_id"] == city_id:
            return c["bbox"]
    raise ValueError(f"Unknown city_id {city_id}")


def _city_name(city_id: int) -> str:
    for c in CITIES:
        if c["city_id"] == city_id:
            return c["name"]
    raise ValueError(f"Unknown city_id {city_id}")


def generate_viewport_queries(
    city_buildings: Dict[int, List[Tuple[float, float]]],
    n: int,
    f: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Generate n viewport (range) queries.
    Each viewport covers approximately f fraction of a city's bounding box area.
    """
    queries = []
    city_ids = list(city_buildings.keys())
    for _ in range(n):
        cid = rng.choice(city_ids)
        s, w, n_lat, e = _city_bbox(cid)
        lat_span = n_lat - s
        lon_span = e - w
        # viewport side length = sqrt(f) * full side
        vp_lat = lat_span * math.sqrt(f)
        vp_lon = lon_span * math.sqrt(f)
        # random origin within the remaining area
        qs = rng.uniform(s, n_lat - vp_lat)
        qw = rng.uniform(w, e - vp_lon)
        qn = qs + vp_lat
        qe = qw + vp_lon
        queries.append({
            "city_id": cid,
            "bbox": (qs, qw, qn, qe),
            "f": f,
        })
    return queries


def generate_knn_queries(
    city_buildings: Dict[int, List[Tuple[float, float]]],
    n: int,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate n kNN queries centred on random existing buildings."""
    queries = []
    city_ids = [cid for cid in city_buildings.keys() if len(city_buildings[cid]) > 0]
    for _ in range(n):
        cid = rng.choice(city_ids)
        pts = city_buildings[cid]
        lon, lat = rng.choice(pts)
        queries.append({
            "city_id": cid,
            "lon": lon,
            "lat": lat,
            "k": k,
        })
    return queries


# ---------------------------------------------------------------------------
# 10. Benchmark runners
# ---------------------------------------------------------------------------

def _viewport_sql(table: str, bbox: tuple, city_id: int, use_city_filter: bool) -> str:
    s, w, n, e = bbox
    env = f"ST_MakeEnvelope({w},{s},{e},{n},4326)"
    where = f"ST_Intersects(geom, {env})"
    if use_city_filter:
        where = f"city_id = {city_id} AND " + where
    return f"SELECT count(*) FROM {table} WHERE {where};"


def _knn_sql(table: str, lon: float, lat: float, k: int, city_id: int,
             use_city_filter: bool) -> str:
    pt = f"ST_SetSRID(ST_MakePoint({lon},{lat}),4326)"
    order = f"geom <-> {pt}"
    where = f"city_id = {city_id}" if use_city_filter else "TRUE"
    return (f"SELECT id, geom <-> {pt} AS dist FROM {table} "
            f"WHERE {where} ORDER BY {order} LIMIT {k};")


def run_viewport_benchmark(
    conn,
    table: str,
    queries: List[Dict[str, Any]],
    use_city_filter: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Run viewport queries, return timing statistics."""
    times = []
    rows_returned = []
    for i, q in enumerate(queries):
        sql = _viewport_sql(table, q["bbox"], q["city_id"], use_city_filter)
        t0 = time.perf_counter()
        result = exec_sql(conn, sql, fetch=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
        rows_returned.append(result[0][0] if result else 0)
        if (i + 1) % 100 == 0:
            print(f"    [{label}] {i+1}/{len(queries)}  "
                  f"median={statistics.median(times):.1f}ms")

    return {
        "label": label,
        "table": table,
        "query_count": len(queries),
        "median_ms": round(statistics.median(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "p95_ms": round(sorted(times)[int(0.95 * len(times))], 2),
        "p99_ms": round(sorted(times)[int(0.99 * len(times))], 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "total_rows": sum(rows_returned),
        "avg_rows": round(statistics.mean(rows_returned), 1),
    }


def run_knn_benchmark(
    conn,
    table: str,
    queries: List[Dict[str, Any]],
    use_city_filter: bool = False,
    label: str = "",
) -> Dict[str, Any]:
    """Run kNN queries, return timing statistics."""
    times = []
    for i, q in enumerate(queries):
        sql = _knn_sql(table, q["lon"], q["lat"], q["k"], q["city_id"], use_city_filter)
        t0 = time.perf_counter()
        exec_sql(conn, sql, fetch=True)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        if (i + 1) % 100 == 0:
            print(f"    [{label}] {i+1}/{len(queries)}  "
                  f"median={statistics.median(times):.1f}ms")

    return {
        "label": label,
        "table": table,
        "query_count": len(queries),
        "k": queries[0]["k"],
        "median_ms": round(statistics.median(times), 2),
        "mean_ms": round(statistics.mean(times), 2),
        "p95_ms": round(sorted(times)[int(0.95 * len(times))], 2),
        "p99_ms": round(sorted(times)[int(0.99 * len(times))], 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
    }


# ---------------------------------------------------------------------------
# 11. Cold-cache helpers  (flush OS page cache + restart PostgreSQL)
# ---------------------------------------------------------------------------

def flush_caches():
    """
    Drop OS page cache and restart PostgreSQL.
    Requires sudo privileges (typical on a benchmark EC2 instance).
    """
    print("[cold] syncing and dropping OS page cache …")
    try:
        subprocess.run(["sync"], check=True)
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[cold] WARNING: could not drop caches: {e}")

    print("[cold] restarting PostgreSQL …")
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "postgresql"],
            check=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        print(f"[cold] WARNING: could not restart PostgreSQL: {e}")

    # Wait for PostgreSQL to become available
    for attempt in range(30):
        try:
            c = get_conn()
            c.close()
            print("[cold] PostgreSQL is back up")
            return
        except Exception:
            time.sleep(2)
    print("[cold] WARNING: PostgreSQL did not come back within 60s")


# ---------------------------------------------------------------------------
# 12. Table size reporter
# ---------------------------------------------------------------------------

def report_table_sizes(conn) -> Dict[str, Any]:
    """Report on-disk sizes for each table configuration."""
    sizes = {}
    for tbl in ["osm_large_mono", "osm_large_so", "osm_large_spdb"]:
        try:
            rows = exec_sql(conn, f"""
                SELECT pg_total_relation_size('{tbl}')::bigint,
                       pg_table_size('{tbl}')::bigint,
                       pg_indexes_size('{tbl}')::bigint,
                       (SELECT count(*) FROM {tbl})
            """, fetch=True)
            total, table, indexes, cnt = rows[0]
            sizes[tbl] = {
                "total_bytes": total,
                "total_human": f"{total / (1024**3):.2f} GB",
                "table_bytes": table,
                "index_bytes": indexes,
                "row_count": cnt,
            }
            print(f"  {tbl}: {cnt:,} rows, "
                  f"table={table/(1024**3):.2f}GB, "
                  f"index={indexes/(1024**3):.2f}GB, "
                  f"total={total/(1024**3):.2f}GB")
        except Exception as e:
            print(f"  {tbl}: ERROR – {e}")
            sizes[tbl] = {"error": str(e)}
    return sizes


# ---------------------------------------------------------------------------
# 13. Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OSM Large Building Benchmark")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download data, skip DB operations")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Skip download+load, run benchmarks on existing tables")
    parser.add_argument("--cold-only", action="store_true",
                        help="Run only the cold-cache benchmark")
    args = parser.parse_args()

    rng = random.Random(RANDOM_SEED)
    results: Dict[str, Any] = {
        "meta": {
            "script": "osm_large_benchmark.py",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cities": len(CITIES),
            "hilbert_order": HILBERT_ORDER,
            "spdb_sub_partitions": SPDB_SUB_PARTITIONS,
        },
        "download": {},
        "table_sizes": {},
        "warm_viewport": [],
        "warm_knn": [],
        "cold_viewport": [],
    }

    # ------------------------------------------------------------------
    # Phase 1: Download
    # ------------------------------------------------------------------
    if not args.benchmark_only and not args.cold_only:
        print("\n" + "=" * 70)
        print("PHASE 1: Downloading building footprints from OpenStreetMap")
        print("=" * 70 + "\n")

        t0 = time.time()
        city_buildings = download_all_cities()
        dl_time = time.time() - t0

        total = sum(len(v) for v in city_buildings.values())
        results["download"] = {
            "total_buildings": total,
            "download_time_s": round(dl_time, 1),
            "per_city": {
                _city_name(cid): len(pts)
                for cid, pts in city_buildings.items()
            },
        }

        if args.download_only:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out = RESULTS_DIR / "osm_large_buildings.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDownload results saved to {out}")
            return
    else:
        # Load from cache
        print("\n[info] Loading buildings from cache …")
        city_buildings = {}
        for city in CITIES:
            centroids = []
            tiles = split_bbox_into_tiles(*city["bbox"])
            for idx in range(len(tiles)):
                cp = _cache_path(city["name"], idx)
                if cp.exists():
                    with open(cp) as f:
                        centroids.extend(json.load(f))
            city_buildings[city["city_id"]] = [(c[0], c[1]) for c in centroids]
        total = sum(len(v) for v in city_buildings.values())
        print(f"[info] Loaded {total:,} buildings from cache\n")

    # ------------------------------------------------------------------
    # Phase 2: Create tables and load data
    # ------------------------------------------------------------------
    if not args.benchmark_only and not args.cold_only:
        print("\n" + "=" * 70)
        print("PHASE 2: Creating PostgreSQL tables and loading data")
        print("=" * 70 + "\n")

        conn = get_conn()
        conn.autocommit = True
        ensure_postgis(conn)

        # 2a. Monolithic
        print("\n--- Monolithic table ---")
        create_monolithic(conn)
        load_table(conn, "osm_large_mono", city_buildings)
        index_monolithic(conn)

        # 2b. LIST by city
        print("\n--- LIST-partitioned by city ---")
        create_so_partitioned(conn)
        load_table(conn, "osm_large_so", city_buildings)
        index_so_partitioned(conn)

        # 2c. SPDB: LIST by city + RANGE by hilbert_key
        print("\n--- SPDB: LIST × RANGE ---")
        create_spdb_partitioned(conn)
        load_table(conn, "osm_large_spdb", city_buildings)
        index_spdb_partitioned(conn)

        # Report sizes
        print("\n--- Table sizes ---")
        results["table_sizes"] = report_table_sizes(conn)
        conn.close()

    # ------------------------------------------------------------------
    # Phase 3: Warm-cache benchmarks
    # ------------------------------------------------------------------
    if not args.cold_only:
        print("\n" + "=" * 70)
        print("PHASE 3: Warm-cache benchmarks")
        print("=" * 70 + "\n")

        conn = get_conn()
        conn.autocommit = True

        # Pre-warm: run a quick scan so PG buffers are populated
        print("[warm] pre-warming caches …")
        for tbl in ["osm_large_mono", "osm_large_so", "osm_large_spdb"]:
            try:
                exec_sql(conn, f"SELECT count(*) FROM {tbl};", fetch=True)
            except Exception:
                pass

        # 3a. Viewport queries
        configs = [
            ("osm_large_mono",  False, "mono"),
            ("osm_large_so",    True,  "so"),
            ("osm_large_spdb",  True,  "spdb"),
        ]
        for f_val in VIEWPORT_F_VALUES:
            print(f"\n  --- Viewport f={f_val*100:.0f}% ---")
            vp_queries = generate_viewport_queries(
                city_buildings, VIEWPORT_QUERIES, f_val, rng
            )
            for table, use_city, tag in configs:
                label = f"viewport_f{f_val}_{tag}"
                print(f"\n  Running {label} ({VIEWPORT_QUERIES} queries) …")
                try:
                    res = run_viewport_benchmark(
                        conn, table, vp_queries, use_city, label
                    )
                    res["f"] = f_val
                    results["warm_viewport"].append(res)
                    print(f"  → median={res['median_ms']:.1f}ms  "
                          f"p95={res['p95_ms']:.1f}ms  "
                          f"p99={res['p99_ms']:.1f}ms")
                except Exception as e:
                    print(f"  ERROR on {label}: {e}")
                    traceback.print_exc()

        # 3b. kNN queries
        print(f"\n  --- kNN k={KNN_K} ---")
        knn_queries = generate_knn_queries(
            city_buildings, KNN_QUERIES, KNN_K, rng
        )
        for table, use_city, tag in configs:
            label = f"knn_k{KNN_K}_{tag}"
            print(f"\n  Running {label} ({KNN_QUERIES} queries) …")
            try:
                res = run_knn_benchmark(
                    conn, table, knn_queries, use_city, label
                )
                results["warm_knn"].append(res)
                print(f"  → median={res['median_ms']:.1f}ms  "
                      f"p95={res['p95_ms']:.1f}ms  "
                      f"p99={res['p99_ms']:.1f}ms")
            except Exception as e:
                print(f"  ERROR on {label}: {e}")
                traceback.print_exc()

        conn.close()

        # Save intermediate results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "osm_large_buildings.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[save] Intermediate results → {out}")

    # ------------------------------------------------------------------
    # Phase 4: Cold-cache benchmarks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 4: Cold-cache benchmarks (viewport f=5%)")
    print("=" * 70 + "\n")

    cold_vp_queries = generate_viewport_queries(
        city_buildings, COLD_QUERIES, 0.05, rng
    )

    configs_cold = [
        ("osm_large_mono",  False, "mono"),
        ("osm_large_so",    True,  "so"),
        ("osm_large_spdb",  True,  "spdb"),
    ]

    for table, use_city, tag in configs_cold:
        label = f"cold_viewport_f0.05_{tag}"
        print(f"\n  [{label}] flushing caches …")
        flush_caches()
        time.sleep(5)  # extra settle time

        print(f"  [{label}] running {COLD_QUERIES} queries …")
        try:
            conn = get_conn()
            conn.autocommit = True
            res = run_viewport_benchmark(
                conn, table, cold_vp_queries, use_city, label
            )
            res["f"] = 0.05
            res["cold"] = True
            results["cold_viewport"].append(res)
            print(f"  → median={res['median_ms']:.1f}ms  "
                  f"p95={res['p95_ms']:.1f}ms  "
                  f"p99={res['p99_ms']:.1f}ms")
            conn.close()
        except Exception as e:
            print(f"  ERROR on {label}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Phase 5: Save final results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 5: Saving results")
    print("=" * 70 + "\n")

    results["meta"]["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "osm_large_buildings.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = sum(len(v) for v in city_buildings.values())
    print(f"  Buildings: {total:,}")
    print(f"  Cities:    {len(CITIES)}")
    print()

    for section_name, section_key in [
        ("Warm Viewport", "warm_viewport"),
        ("Warm kNN", "warm_knn"),
        ("Cold Viewport", "cold_viewport"),
    ]:
        entries = results.get(section_key, [])
        if entries:
            print(f"  {section_name}:")
            for e in entries:
                print(f"    {e['label']:40s}  "
                      f"median={e['median_ms']:8.1f}ms  "
                      f"p95={e['p95_ms']:8.1f}ms  "
                      f"p99={e['p99_ms']:8.1f}ms")
            print()

    print("Done.\n")


if __name__ == "__main__":
    main()
