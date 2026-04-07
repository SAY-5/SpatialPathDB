"""OSM POI ingestion: download NYC POIs, load into PostgreSQL, build HCCI index.

Downloads point-of-interest data from OpenStreetMap via Overpass API for
New York City, loads into a PostGIS table with HCCI covering index.

Validates HCCI on urban geospatial data with 50+ category types and
very different selectivity distributions from pathology.

Usage:
    python -m benchmarks.osm_ingest
    python -m benchmarks.osm_ingest --index-only   # skip download + load
    python -m benchmarks.osm_ingest --stats-only    # just print stats
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE = "osm_pois"
INDEX_NAME = "idx_osm_hcci_covering"
DATASET_ID = "nyc"

# NYC bounding box (WGS84)
NYC_BBOX = {
    "south": 40.4774,
    "west": -74.2591,
    "north": 40.9176,
    "east": -73.7004,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Tags to query — each produces a category prefix
TAG_QUERIES = [
    ("amenity", "node"),
    ("shop", "node"),
    ("tourism", "node"),
    ("leisure", "node"),
    ("office", "node"),
    ("craft", "node"),
    ("healthcare", "node"),
    ("building", "way"),  # ways need 'out center' for centroids
]


# ---------------------------------------------------------------------------
# Overpass API download
# ---------------------------------------------------------------------------

def _overpass_query(tag_key: str, element_type: str) -> list[dict]:
    """Query Overpass API for elements with given tag in NYC bbox."""
    bbox = f"{NYC_BBOX['south']},{NYC_BBOX['west']},{NYC_BBOX['north']},{NYC_BBOX['east']}"

    if element_type == "way":
        query = f'[out:json][timeout:300][bbox:{bbox}];way["{tag_key}"];out center;'
    else:
        query = f'[out:json][timeout:300][bbox:{bbox}];node["{tag_key}"];out;'

    print(f"    Querying {tag_key} ({element_type})...")
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data)
    req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode())
            elements = result.get("elements", [])
            print(f"      -> {len(elements):,} elements")
            return elements
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            print(f"      Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
    return []


def download_osm_pois() -> list[dict]:
    """Download all POIs from Overpass API and normalize to records."""
    print("\n[Download] Fetching NYC POIs from Overpass API...")
    t0 = time.time()

    all_records = []
    seen_ids = set()

    for tag_key, elem_type in TAG_QUERIES:
        elements = _overpass_query(tag_key, elem_type)
        time.sleep(2)  # Rate limit courtesy

        for el in elements:
            osm_id = el.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)

            tags = el.get("tags", {})

            # Get coordinates
            if elem_type == "way":
                center = el.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")
            else:
                lat = el.get("lat")
                lon = el.get("lon")

            if lat is None or lon is None:
                continue

            # Determine category: tag_key + ":" + tag_value
            tag_val = tags.get(tag_key, "yes")
            category = f"{tag_key}:{tag_val}"

            name = tags.get("name", "")

            all_records.append({
                "osm_id": osm_id,
                "lon": float(lon),
                "lat": float(lat),
                "category": category,
                "name": name,
            })

    elapsed = time.time() - t0
    print(f"\n  Downloaded {len(all_records):,} unique POIs in {elapsed:.0f}s")
    return all_records


def save_pois_json(records: list[dict], path: str):
    """Save POIs to JSON for caching."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"  Saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_pois_json(path: str) -> list[dict]:
    """Load cached POIs from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_table(conn):
    """Create the osm_pois table."""
    print("\n[Create Table] Setting up osm_pois...")
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        cur.execute(f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                osm_id BIGINT,
                dataset_id TEXT DEFAULT '{DATASET_ID}',
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
    """Bulk load POI records into PostgreSQL using COPY."""
    print(f"\n[Load] Inserting {len(records):,} records...")
    t0 = time.time()

    buf = io.StringIO()
    for r in records:
        # Escape name for COPY (tab-separated)
        name = r["name"].replace("\t", " ").replace("\n", " ").replace("\\", "\\\\")
        buf.write(f"{r['osm_id']}\t{DATASET_ID}\t{r['lon']}\t{r['lat']}\t{r['category']}\t{name}\n")

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(buf, TABLE,
                       columns=("osm_id", "dataset_id", "centroid_x", "centroid_y",
                                "class_label", "name"),
                       sep="\t")
    conn.commit()
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s ({len(records) / elapsed:,.0f} rows/s)")


def build_geometry(conn):
    """Populate geom column from centroid_x/y."""
    print("\n[Geometry] Building point geometries...")
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

    # Get coordinate bounds
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT MIN(centroid_x), MAX(centroid_x),
                   MIN(centroid_y), MAX(centroid_y),
                   COUNT(*)
            FROM {TABLE}
        """)
        x_min, x_max, y_min, y_max, count = cur.fetchone()

    print(f"  Bounds: x=[{x_min:.4f}, {x_max:.4f}], y=[{y_min:.4f}, {y_max:.4f}]")
    print(f"  Rows: {count:,}")

    # Add small padding to avoid edge effects
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.001
    x_max += x_range * 0.001
    y_min -= y_range * 0.001
    y_max += y_range * 0.001
    width = x_max - x_min
    height = y_max - y_min

    # Fetch all coordinates
    with conn.cursor() as cur:
        cur.execute(f"SELECT id, centroid_x, centroid_y FROM {TABLE} ORDER BY id")
        rows = cur.fetchall()

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    xs = np.array([r[1] for r in rows], dtype=np.float64)
    ys = np.array([r[2] for r in rows], dtype=np.float64)

    # Normalize to grid
    n = 1 << hilbert_order
    gx = np.clip(((xs - x_min) * n / width).astype(np.int64), 0, n - 1)
    gy = np.clip(((ys - y_min) * n / height).astype(np.int64), 0, n - 1)

    # Vectorized Hilbert encoding
    h_keys = hilbert.encode_batch(gx, gy, hilbert_order)

    # Batch update
    print("  Updating hilbert_key column...")
    buf = io.StringIO()
    for i in range(len(ids)):
        buf.write(f"{ids[i]}\t{int(h_keys[i])}\n")
    buf.seek(0)

    with conn.cursor() as cur:
        cur.execute(f"CREATE TEMP TABLE _hk (id BIGINT, hk BIGINT)")
        cur.copy_from(buf, "_hk", columns=("id", "hk"), sep="\t")
        cur.execute(f"UPDATE {TABLE} t SET hilbert_key = h.hk FROM _hk h WHERE t.id = h.id")
        cur.execute("DROP TABLE _hk")
    conn.commit()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Return bounds for later use
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "width": width, "height": height,
        "count": count,
    }


def build_class_enum(conn) -> dict[str, int]:
    """Discover categories and build enum mapping ordered by frequency."""
    print("\n[Enum] Building category enum...")
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
        if i < 15:
            print(f"    {i:>3}: {cat:<35} {cnt:>8,}")
    if len(rows) > 15:
        print(f"    ... and {len(rows) - 15} more categories")

    return class_enum


def compute_composite_keys(conn, class_enum: dict[str, int]):
    """Compute composite keys using the class enum."""
    print("\n[Composite Key] Computing composite keys...")
    t0 = time.time()

    # Build SQL CASE statement from enum
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
        cur.execute(f"CREATE INDEX idx_osm_gist ON {TABLE} USING gist (geom)")
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def build_hcci_index(conn):
    """Create HCCI covering B-tree index."""
    print(f"\n[HCCI Index] Creating covering index {INDEX_NAME}...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        ddl = (
            f"CREATE INDEX {INDEX_NAME} ON {TABLE} "
            f"(dataset_id, composite_key) "
            f"INCLUDE (centroid_x, centroid_y, class_label, area)"
        )
        print(f"  DDL: {ddl}")
        cur.execute(ddl)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def vacuum_analyze(conn):
    """VACUUM and ANALYZE for visibility map + planner stats."""
    print(f"\n[Vacuum] VACUUM {TABLE}...")
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
    print("\n[Verify] Checking setup...")
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
            print(f"    {r[0]:<30} ck={r[1]:>16} enum={r[2]:>3} h_part={r[3]:>8} hk={r[4]:>8}")

        # Category distribution
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
            LIMIT 10
        """)
        print("\n  Top 10 categories:")
        for cat, cnt, pct in cur.fetchall():
            print(f"    {cat:<35} {cnt:>8,}  ({pct}%)")

        # Index sizes
        cur.execute(f"""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes WHERE tablename = '{TABLE}'
        """)
        print("\n  Indexes:")
        for name, size in cur.fetchall():
            print(f"    {name}: {size}")

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


def save_metadata(bounds: dict, class_enum: dict[str, int], path: str):
    """Save dataset metadata (bounds, enum) for benchmark use."""
    meta = {
        "table": TABLE,
        "dataset_id": DATASET_ID,
        "bounds": bounds,
        "class_enum": class_enum,
        "hilbert_order": config.HILBERT_ORDER,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Metadata saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OSM POI ingestion for HCCI validation")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip download and load, just build indexes")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print category stats")
    parser.add_argument("--cache", type=str, default="results/raw/osm_pois_cache.json",
                        help="Path to cache downloaded POIs")
    args = parser.parse_args()

    print("=" * 60)
    print("  OSM POI Ingestion for HCCI Validation")
    print("  Dataset: New York City")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    t_start = time.time()

    if args.stats_only:
        class_enum = build_class_enum(conn)
        verify(conn, class_enum)
        conn.close()
        return

    if not args.index_only:
        # Download or load from cache
        if os.path.exists(args.cache):
            print(f"\n  Loading from cache: {args.cache}")
            records = load_pois_json(args.cache)
        else:
            records = download_osm_pois()
            save_pois_json(records, args.cache)

        # Create table and load
        create_table(conn)
        load_data(conn, records)
        build_geometry(conn)
        bounds = compute_hilbert_keys(conn)
    else:
        # Get bounds from existing data
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
    meta_path = os.path.join(config.RAW_DIR, "osm_metadata.json")
    save_metadata(bounds, class_enum, meta_path)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  OSM ingestion complete in {total:.0f}s ({total / 60:.1f}m)")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
