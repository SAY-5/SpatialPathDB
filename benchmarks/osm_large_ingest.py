"""Large-scale OSM POI ingestion: 10M+ points from 20+ US metro areas.

Downloads building centroids and amenity/shop nodes from OpenStreetMap
via Overpass API for major US metropolitan areas. Combines into a single
table with HCCI + GiST indexes for cross-domain validation at scale.

Target: 10-20M points across diverse geographic regions.
Class label = category (amenity:restaurant, shop:supermarket, building:yes, etc.)

Usage:
    python -m benchmarks.osm_large_ingest
    python -m benchmarks.osm_large_ingest --resume         # continue from cache
    python -m benchmarks.osm_large_ingest --index-only     # rebuild indexes
    python -m benchmarks.osm_large_ingest --stats-only     # print stats
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE = "osm_large"
INDEX_HCCI = "idx_osm_large_hcci_covering"
INDEX_GIST = "idx_osm_large_gist"
DATASET_ID = "osm_us"

CACHE_DIR = "results/raw/osm_large_cache"

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]

# Major US metro areas — bbox: (south, west, north, east)
# Each metro covers a large area to maximize POI count
METRO_AREAS = {
    "nyc":          (40.477, -74.259, 40.918, -73.700),
    "los_angeles":  (33.700, -118.670, 34.337, -117.648),
    "chicago":      (41.644, -87.940, 42.023, -87.524),
    "houston":      (29.523, -95.790, 30.111, -95.015),
    "phoenix":      (33.290, -112.325, 33.720, -111.789),
    "philadelphia": (39.867, -75.280, 40.138, -74.955),
    "san_antonio":  (29.217, -98.725, 29.605, -98.275),
    "san_diego":    (32.534, -117.282, 32.964, -116.908),
    "dallas":       (32.619, -97.000, 33.017, -96.463),
    "san_jose":     (37.124, -122.046, 37.469, -121.589),
    "austin":       (30.099, -97.938, 30.517, -97.561),
    "jacksonville": (30.103, -81.880, 30.586, -81.350),
    "san_francisco":(37.707, -122.517, 37.813, -122.356),
    "columbus":     (39.862, -83.125, 40.128, -82.771),
    "indianapolis": (39.632, -86.327, 39.928, -85.937),
    "charlotte":    (35.041, -80.978, 35.393, -80.660),
    "seattle":      (47.490, -122.436, 47.734, -122.236),
    "denver":       (39.614, -105.110, 39.914, -104.600),
    "washington_dc":(38.792, -77.120, 38.996, -76.909),
    "boston":        (42.227, -71.191, 42.400, -70.923),
    "nashville":    (35.975, -86.970, 36.276, -86.588),
    "detroit":      (42.255, -83.288, 42.450, -82.910),
    "portland":     (45.432, -122.836, 45.654, -122.472),
    "las_vegas":    (35.974, -115.375, 36.325, -115.015),
    "miami":        (25.709, -80.439, 25.855, -80.131),
    "atlanta":      (33.647, -84.551, 33.887, -84.290),
    "minneapolis":  (44.890, -93.329, 45.051, -93.193),
}

# Tags to query per metro area
TAG_QUERIES = [
    ("building", "way"),     # largest count per metro
    ("amenity", "node"),     # restaurants, schools, hospitals
    ("shop", "node"),        # stores
    ("tourism", "node"),     # attractions
    ("leisure", "node"),     # parks, playgrounds
    ("office", "node"),      # offices
]


# ---------------------------------------------------------------------------
# Overpass download
# ---------------------------------------------------------------------------

def _query_overpass(tag_key: str, elem_type: str, bbox: tuple,
                    metro_name: str, server_idx: int = 0) -> list[dict]:
    """Query Overpass API for elements with given tag in bbox."""
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"

    if elem_type == "way":
        query = f'[out:json][timeout:300][bbox:{bbox_str}];way["{tag_key}"];out center;'
    else:
        query = f'[out:json][timeout:300][bbox:{bbox_str}];node["{tag_key}"];out;'

    data = urllib.parse.urlencode({"data": query}).encode()

    for attempt in range(len(OVERPASS_SERVERS)):
        server = OVERPASS_SERVERS[(server_idx + attempt) % len(OVERPASS_SERVERS)]
        try:
            req = urllib.request.Request(server, data=data)
            req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode())
            return result.get("elements", [])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception) as exc:
            if attempt < len(OVERPASS_SERVERS) - 1:
                time.sleep(5)
            else:
                print(f"      FAILED {metro_name}/{tag_key}: {exc}")
                return []
    return []


def download_metro(metro_name: str, bbox: tuple, server_idx: int = 0) -> list[dict]:
    """Download all POI types for a single metro area."""
    cache_path = os.path.join(CACHE_DIR, f"{metro_name}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            records = json.load(f)
        print(f"  {metro_name}: {len(records):,} (cached)")
        return [tuple(r) if isinstance(r, list) else r for r in records]

    print(f"  {metro_name}: downloading...")
    all_records = []
    seen_ids = set()

    for tag_key, elem_type in TAG_QUERIES:
        elements = _query_overpass(tag_key, elem_type, bbox, metro_name, server_idx)
        time.sleep(3)  # Rate limit courtesy

        for el in elements:
            osm_id = el.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)

            tags = el.get("tags", {})
            if elem_type == "way":
                center = el.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")
            else:
                lat = el.get("lat")
                lon = el.get("lon")

            if lat is None or lon is None:
                continue

            tag_val = tags.get(tag_key, "yes")
            category = f"{tag_key}:{tag_val}"
            name = tags.get("name", "")

            all_records.append({
                "osm_id": osm_id,
                "lon": float(lon),
                "lat": float(lat),
                "category": category,
                "name": name,
                "metro": metro_name,
            })

        count = sum(1 for _ in elements)
        print(f"    {tag_key}: {count:,}")

    # Cache to disk
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(all_records, f)

    print(f"    Total: {len(all_records):,}")
    return all_records


def download_all_metros(metro_list: dict) -> list[dict]:
    """Download POIs from all metro areas."""
    print(f"\n[Download] Fetching POIs from {len(metro_list)} metro areas...")
    t0 = time.time()

    all_records = []
    for idx, (name, bbox) in enumerate(metro_list.items()):
        records = download_metro(name, bbox, server_idx=idx % len(OVERPASS_SERVERS))
        all_records.extend(records)
        print(f"  Running total: {len(all_records):,}")

        # Rate limit between metros
        if not os.path.exists(os.path.join(CACHE_DIR, f"{name}.json")):
            time.sleep(5)

    elapsed = time.time() - t0
    print(f"\n  Downloaded {len(all_records):,} total POIs in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    return all_records


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_table(conn):
    """Create the osm_large table."""
    print(f"\n[Create Table] Setting up {TABLE}...")
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        cur.execute(f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                osm_id BIGINT,
                dataset_id TEXT DEFAULT '{DATASET_ID}',
                metro TEXT,
                centroid_x DOUBLE PRECISION,
                centroid_y DOUBLE PRECISION,
                class_label TEXT,
                name TEXT,
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

    def _sanitize(s):
        """Sanitize string for PostgreSQL COPY (tab-separated)."""
        if not s:
            return ""
        return s.replace("\\", "").replace("\t", " ").replace("\n", " ").replace("\r", " ")

    buf = io.StringIO()
    for r in records:
        name = _sanitize(r.get("name", ""))
        metro = _sanitize(r.get("metro", "unknown"))
        category = _sanitize(r.get("category", "unknown"))
        buf.write(
            f"{r['osm_id']}\t{DATASET_ID}\t{metro}\t"
            f"{r['lon']}\t{r['lat']}\t{category}\t{name}\n"
        )

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(
            buf, TABLE,
            columns=("osm_id", "dataset_id", "metro", "centroid_x", "centroid_y",
                     "class_label", "name"),
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

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.001
    x_max += x_range * 0.001
    y_min -= y_range * 0.001
    y_max += y_range * 0.001
    width = x_max - x_min
    height = y_max - y_min

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
    """Discover categories and build enum mapping ordered by frequency."""
    print(f"\n[Enum] Building category enum...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        rows = cur.fetchall()

    class_enum = {}
    print(f"  Found {len(rows)} unique categories:")
    for i, (cat, cnt) in enumerate(rows):
        class_enum[cat] = i
        if i < 20:
            print(f"    {i:>3}: {cat:<35} {cnt:>10,}")
    if len(rows) > 20:
        print(f"    ... and {len(rows) - 20} more categories")

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
    """Verify setup."""
    print(f"\n[Verify] Checking setup...")
    with conn.cursor() as cur:
        # Total count
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total = cur.fetchone()[0]
        print(f"  Total rows: {total:,}")

        # Per-metro counts
        cur.execute(f"""
            SELECT metro, COUNT(*) as cnt
            FROM {TABLE}
            GROUP BY metro
            ORDER BY cnt DESC
        """)
        print("\n  Per-metro counts:")
        for metro, cnt in cur.fetchall():
            print(f"    {metro:<20} {cnt:>10,}")

        # Top 15 categories
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
            LIMIT 15
        """)
        print("\n  Top 15 categories:")
        for cat, cnt, pct in cur.fetchall():
            print(f"    {cat:<35} {cnt:>10,}  ({pct}%)")

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
            for l in plan[:10]:
                print(f"    {l}")

        # Sample rows
        cur.execute(f"""
            SELECT class_label, composite_key,
                   (composite_key >> 48) AS class_enum,
                   (composite_key & x'FFFFFFFFFFFF'::bigint) AS hilbert_part,
                   hilbert_key, metro
            FROM {TABLE}
            LIMIT 5
        """)
        print("\n  Sample rows:")
        for r in cur.fetchall():
            print(f"    {r[0]:<30} ck={r[1]:>16} enum={r[2]:>3} hk={r[4]:>8} metro={r[5]}")


def save_metadata(bounds: dict, class_enum: dict[str, int], path: str):
    """Save dataset metadata for benchmark use."""
    meta = {
        "table": TABLE,
        "dataset_id": DATASET_ID,
        "bounds": bounds,
        "class_enum": class_enum,
        "hilbert_order": config.HILBERT_ORDER,
        "metros": list(METRO_AREAS.keys()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Large-scale OSM POI ingestion for HCCI cross-domain validation"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Use cached downloads, skip already-downloaded metros")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip download and load, just rebuild indexes")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print stats for existing table")
    parser.add_argument("--metros", type=str, default=None,
                        help="Comma-separated metro names to include (default: all)")
    args = parser.parse_args()

    # Select metro areas
    if args.metros:
        metro_names = [m.strip() for m in args.metros.split(",")]
        metros = {k: v for k, v in METRO_AREAS.items() if k in metro_names}
        if not metros:
            print(f"ERROR: No matching metros. Available: {list(METRO_AREAS.keys())}")
            sys.exit(1)
    else:
        metros = METRO_AREAS

    print("=" * 60)
    print("  Large-Scale OSM POI Ingestion")
    print(f"  Metro areas: {len(metros)}")
    print(f"  Target: 10M+ POIs across US")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    t_start = time.time()

    if args.stats_only:
        class_enum = build_class_enum(conn)
        verify(conn, class_enum)
        conn.close()
        return

    if not args.index_only:
        all_records = download_all_metros(metros)
        create_table(conn)
        load_data(conn, all_records)
        build_geometry(conn)
        bounds = compute_hilbert_keys(conn)
        del all_records
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

    class_enum = build_class_enum(conn)

    if not args.index_only:
        compute_composite_keys(conn, class_enum)

    build_gist_index(conn)
    build_hcci_index(conn)
    vacuum_analyze(conn)

    verify(conn, class_enum)
    meta_path = os.path.join(config.RAW_DIR, "osm_large_metadata.json")
    save_metadata(bounds, class_enum, meta_path)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  OSM large ingestion complete in {total:.0f}s ({total / 60:.1f}m)")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
