#!/usr/bin/env python3
"""OSM Buildings benchmark — non-pathology generalizability test.

Downloads Manhattan building footprints from OpenStreetMap,
creates Mono/SO/SPDB tables, and runs viewport benchmarks.
"""
import json, os, time, random, statistics, math, subprocess, sys

# Ensure requests is available
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extras import execute_values

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "raw")
os.makedirs(RESULTS_DIR, exist_ok=True)

DB_PARAMS = dict(dbname="spdb", user="postgres", host="localhost")

# ---------- Hilbert helpers ----------
def xy2d(n, x, y):
    """Convert (x,y) to Hilbert distance for order n (n must be power of 2)."""
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        # rotate
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        s //= 2
    return d

def hilbert_key(lon, lat, p=8, lon_min=-74.03, lon_max=-73.90, lat_min=40.70, lat_max=40.88):
    """Compute Hilbert key for a geographic coordinate."""
    n = 1 << p
    x = int((lon - lon_min) / (lon_max - lon_min) * (n - 1))
    y = int((lat - lat_min) / (lat_max - lat_min) * (n - 1))
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return xy2d(n, x, y)

# ---------- Download Manhattan buildings ----------
def download_manhattan_buildings():
    """Download Manhattan building centroids from Overpass API (with file cache)."""
    cache_path = os.path.join(os.path.dirname(__file__), "osm_manhattan_cache.json")
    if os.path.exists(cache_path):
        print("Loading cached Manhattan buildings...")
        with open(cache_path) as f:
            buildings = json.load(f)
        print(f"  Loaded {len(buildings)} building centroids from cache")
        return [tuple(b) for b in buildings]

    print("Downloading Manhattan buildings from Overpass API...")
    query = """
    [out:json][timeout:300];
    way["building"](40.700,-74.020,40.882,-73.907);
    out center;
    """
    for attempt in range(3):
        try:
            server = "https://overpass-api.de/api/interpreter"
            if attempt == 1:
                server = "https://overpass.kumi.systems/api/interpreter"
            elif attempt == 2:
                server = "https://overpass.openstreetmap.ru/api/interpreter"
            print(f"  Attempt {attempt+1} via {server.split('/')[2]}...")
            resp = requests.post(server, data={"data": query}, timeout=360)
            resp.raise_for_status()
            break
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt == 2:
                raise
            time.sleep(10)
    data = resp.json()

    buildings = []
    for el in data.get("elements", []):
        if "center" in el:
            buildings.append((el["center"]["lon"], el["center"]["lat"]))

    with open(cache_path, "w") as f:
        json.dump(buildings, f)
    print(f"  Downloaded {len(buildings)} building centroids (cached)")
    return buildings

def assign_districts(buildings):
    """Assign buildings to districts based on latitude bands (proxy for neighborhoods)."""
    lats = [b[1] for b in buildings]
    lat_min, lat_max = min(lats), max(lats)
    n_districts = 10
    band = (lat_max - lat_min) / n_districts

    result = []
    for lon, lat in buildings:
        district = min(int((lat - lat_min) / band), n_districts - 1)
        hkey = hilbert_key(lon, lat)
        result.append((lon, lat, f"district_{district:02d}", hkey))

    return result

# ---------- PostgreSQL setup ----------
def setup_tables(conn, buildings_data):
    """Create Mono, SO, and SPDB tables for OSM buildings."""
    cur = conn.cursor()

    districts = sorted(set(b[2] for b in buildings_data))
    n_buildings = len(buildings_data)
    print(f"  {n_buildings} buildings across {len(districts)} districts")

    # --- Mono ---
    cur.execute("DROP TABLE IF EXISTS osm_mono CASCADE")
    cur.execute("""
        CREATE TABLE osm_mono (
            id SERIAL,
            district TEXT NOT NULL,
            hilbert_key INT NOT NULL,
            geom geometry(Point, 4326) NOT NULL
        )
    """)

    values = [(b[2], b[3], f"SRID=4326;POINT({b[0]} {b[1]})") for b in buildings_data]
    execute_values(cur,
        "INSERT INTO osm_mono (district, hilbert_key, geom) VALUES %s",
        values, template="(%s, %s, ST_GeomFromEWKT(%s))", page_size=5000)
    cur.execute("CREATE INDEX idx_osm_mono_geom ON osm_mono USING gist(geom)")
    cur.execute("ANALYZE osm_mono")
    conn.commit()
    print(f"  osm_mono: {n_buildings} rows, GiST indexed")

    # --- SO (partitioned by district) ---
    cur.execute("DROP TABLE IF EXISTS osm_so CASCADE")
    cur.execute("""
        CREATE TABLE osm_so (
            id SERIAL,
            district TEXT NOT NULL,
            hilbert_key INT NOT NULL,
            geom geometry(Point, 4326) NOT NULL
        ) PARTITION BY LIST (district)
    """)
    for d in districts:
        safe = d.replace("-", "_")
        cur.execute(f"CREATE TABLE osm_so_{safe} PARTITION OF osm_so FOR VALUES IN ('{d}')")

    execute_values(cur,
        "INSERT INTO osm_so (district, hilbert_key, geom) VALUES %s",
        values, template="(%s, %s, ST_GeomFromEWKT(%s))", page_size=5000)
    for d in districts:
        safe = d.replace("-", "_")
        cur.execute(f"CREATE INDEX idx_osm_so_{safe}_geom ON osm_so_{safe} USING gist(geom)")
    cur.execute("ANALYZE osm_so")
    conn.commit()
    print(f"  osm_so: {n_buildings} rows, {len(districts)} partitions")

    # --- SPDB (LIST by district, RANGE by hilbert_key) ---
    cur.execute("DROP TABLE IF EXISTS osm_spdb CASCADE")
    cur.execute("""
        CREATE TABLE osm_spdb (
            id SERIAL,
            district TEXT NOT NULL,
            hilbert_key INT NOT NULL,
            geom geometry(Point, 4326) NOT NULL
        ) PARTITION BY LIST (district)
    """)

    bucket_target = max(n_buildings // (len(districts) * 30), 100)  # ~30 buckets per district
    max_hkey = max(b[3] for b in buildings_data)
    n_buckets = max(max_hkey // bucket_target, 1)

    for d in districts:
        safe = d.replace("-", "_")
        cur.execute(f"""
            CREATE TABLE osm_spdb_{safe} PARTITION OF osm_spdb
            FOR VALUES IN ('{d}') PARTITION BY RANGE (hilbert_key)
        """)
        for i in range(n_buckets):
            lo = i * bucket_target
            hi = (i + 1) * bucket_target if i < n_buckets - 1 else max_hkey + 1
            cur.execute(f"""
                CREATE TABLE osm_spdb_{safe}_h{i}
                PARTITION OF osm_spdb_{safe}
                FOR VALUES FROM ({lo}) TO ({hi})
            """)
            cur.execute(f"CREATE INDEX idx_osm_spdb_{safe}_h{i}_geom ON osm_spdb_{safe}_h{i} USING gist(geom)")

    execute_values(cur,
        "INSERT INTO osm_spdb (district, hilbert_key, geom) VALUES %s",
        values, template="(%s, %s, ST_GeomFromEWKT(%s))", page_size=5000)
    cur.execute("ANALYZE osm_spdb")
    conn.commit()

    total_parts = n_buckets * len(districts)
    print(f"  osm_spdb: {n_buildings} rows, {len(districts)} L1 x {n_buckets} L2 = {total_parts} leaf partitions")
    return districts, bucket_target, n_buckets

# ---------- Benchmark ----------
def run_viewport_benchmark(conn, table, districts, n_trials=200, f=0.05):
    """Run random viewport queries on a table."""
    cur = conn.cursor()

    # Get extent
    cur.execute(f"SELECT ST_Extent(geom) FROM {table}")
    extent = cur.fetchone()[0]  # BOX(lon_min lat_min, lon_max lat_max)
    # Parse BOX string
    parts = extent.replace("BOX(", "").replace(")", "").split(",")
    lon_min, lat_min = map(float, parts[0].strip().split())
    lon_max, lat_max = map(float, parts[1].strip().split())

    dx = lon_max - lon_min
    dy = lat_max - lat_min
    side = math.sqrt(f)
    wx = dx * side
    wy = dy * side

    random.seed(42)
    latencies = []
    for _ in range(n_trials):
        x0 = random.uniform(lon_min, lon_max - wx)
        y0 = random.uniform(lat_min, lat_max - wy)
        x1 = x0 + wx
        y1 = y0 + wy

        sql = f"""
            SELECT COUNT(*) FROM {table}
            WHERE geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 4326)
        """

        # Add Hilbert pruning for SPDB
        if "spdb" in table:
            hlo = hilbert_key(x0, y0)
            hhi = hilbert_key(x1, y1)
            hmin, hmax = min(hlo, hhi), max(hlo, hhi)
            # Widen range for Hilbert coverage
            hrange = hmax - hmin
            hmin = max(0, hmin - hrange)
            hmax = hmax + hrange
            sql = f"""
                SELECT COUNT(*) FROM {table}
                WHERE hilbert_key BETWEEN {hmin} AND {hmax}
                AND geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 4326)
            """

        t0 = time.perf_counter()
        cur.execute(sql)
        cur.fetchone()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "n": n_trials,
        "median": latencies[n_trials // 2],
        "p50": latencies[n_trials // 2],
        "p95": latencies[int(n_trials * 0.95)],
        "mean": statistics.mean(latencies),
        "std": statistics.stdev(latencies),
    }

def main():
    print("=== OSM Buildings Benchmark ===\n")

    # Download
    buildings = download_manhattan_buildings()
    if len(buildings) < 100:
        print("ERROR: Too few buildings downloaded. Aborting.")
        sys.exit(1)

    # Assign districts and Hilbert keys
    buildings_data = assign_districts(buildings)

    # Setup PostgreSQL
    conn = psycopg2.connect(**DB_PARAMS)
    print("\nSetting up tables...")
    districts, bucket_target, n_buckets = setup_tables(conn, buildings_data)

    # Run benchmarks
    print("\nRunning viewport benchmarks (200 trials, f=5%)...\n")
    results = {}
    for config, table in [("Mono", "osm_mono"), ("SO", "osm_so"), ("SPDB", "osm_spdb")]:
        print(f"  {config}...", end=" ", flush=True)
        r = run_viewport_benchmark(conn, table, districts)
        results[config] = r
        print(f"p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms")

    # Compute speedups
    mono_p50 = results["Mono"]["p50"]
    for config in results:
        results[config]["speedup_vs_mono"] = round(mono_p50 / results[config]["p50"], 1) if results[config]["p50"] > 0 else 0

    # Save
    meta = {
        "dataset": "OSM Manhattan Buildings",
        "n_buildings": len(buildings),
        "n_districts": len(districts),
        "n_buckets_per_district": n_buckets,
        "bucket_target": bucket_target,
        "configs": results,
    }

    out_path = os.path.join(RESULTS_DIR, "osm_buildings.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n=== Summary ({len(buildings)} Manhattan buildings) ===")
    print(f"  Mono:  p50={results['Mono']['p50']:.1f}ms  ({results['Mono']['speedup_vs_mono']}x)")
    print(f"  SO:    p50={results['SO']['p50']:.1f}ms  ({results['SO']['speedup_vs_mono']}x)")
    print(f"  SPDB:  p50={results['SPDB']['p50']:.1f}ms  ({results['SPDB']['speedup_vs_mono']}x)")

    conn.close()

if __name__ == "__main__":
    main()
