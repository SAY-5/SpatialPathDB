#!/usr/bin/env bash
# ===========================================================================
# run_remaining.sh  --  Run ALL remaining benchmarks on the AWS instance
#
# Designed to run ON the AWS machine (ubuntu@44.200.185.165).
# Idempotent: checks for existing result files before re-running.
# Survives SSH disconnects via nohup.
#
# Usage (from the AWS instance):
#   nohup bash /home/ubuntu/SpatialPathDB_CLEAN/deploy/run_remaining.sh \
#         > /tmp/remaining_benchmarks.log 2>&1 &
#
# Or from local Mac:
#   ssh -i spdb-benchmark.pem ubuntu@44.200.185.165 \
#     "nohup bash ~/SpatialPathDB_CLEAN/deploy/run_remaining.sh \
#       > /tmp/remaining_benchmarks.log 2>&1 &"
# ===========================================================================
set -euo pipefail

PROJ_DIR="/home/ubuntu/SpatialPathDB_CLEAN"
RESULTS_DIR="${PROJ_DIR}/results"
RAW_DIR="${RESULTS_DIR}/raw"
LOG="/tmp/remaining_benchmarks.log"

cd "${PROJ_DIR}"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"

mkdir -p "${RAW_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ts() { date "+%Y-%m-%d %H:%M:%S"; }

banner() {
    echo ""
    echo "=================================================================="
    echo "  $(ts)  $1"
    echo "=================================================================="
}

check_result() {
    # Usage: check_result <file_path> <description>
    if [[ -f "$1" ]] && [[ -s "$1" ]]; then
        echo "  SKIP: $2 -- result already exists at $1"
        return 0
    fi
    return 1
}

SUMMARY=""
add_summary() {
    SUMMARY="${SUMMARY}\n  $1"
}

# ===========================================================================
# STEP 1: DuckDB Baseline
# ===========================================================================
banner "STEP 1/4: DuckDB Baseline"

DUCKDB_RESULT="${RAW_DIR}/duckdb_baseline.json"

if check_result "${DUCKDB_RESULT}" "DuckDB baseline"; then
    add_summary "DuckDB baseline: SKIPPED (already exists)"
else
    echo "  Installing duckdb Python package..."
    pip3 install --quiet duckdb 2>&1 || pip3 install duckdb 2>&1

    echo "  Running DuckDB benchmark (setup + Q1-Q3 x200 trials + comparison)..."
    set +e
    python3 -c "
import sys, os
sys.path.insert(0, '${PROJ_DIR}')
os.chdir('${PROJ_DIR}')

from benchmarks.duckdb_baseline import setup_duckdb, run_all_duckdb, compare_pg_vs_duckdb

print('  Setting up DuckDB (importing data from PostgreSQL)...')
setup_duckdb()

print('  Running DuckDB Q1-Q3 benchmarks (200 trials each)...')
run_all_duckdb(n_trials=200)

print('  Computing PG vs DuckDB comparison...')
try:
    compare_pg_vs_duckdb()
except FileNotFoundError as e:
    print(f'  WARNING: Could not compute comparison (missing PG results): {e}')
    print('  DuckDB results saved independently.')

print('  DuckDB baseline complete.')
"
    STEP_RC=$?
    set -e
    if [[ ${STEP_RC} -eq 0 ]]; then
        add_summary "DuckDB baseline: DONE"
    else
        add_summary "DuckDB baseline: FAILED (exit code ${STEP_RC}, see log)"
    fi
fi

# ===========================================================================
# STEP 2: SO Viewport Sensitivity
# ===========================================================================
banner "STEP 2/4: SO Viewport Sensitivity"

SO_VP_RESULT="${RAW_DIR}/viewport_sensitivity_so.json"

if check_result "${SO_VP_RESULT}" "SO viewport sensitivity"; then
    add_summary "SO viewport sensitivity: SKIPPED (already exists)"
else
    echo "  Running SO viewport sensitivity (fracs=[0.01,0.02,0.05,0.10,0.20], 200 trials each)..."
    set +e
    python3 << 'PYEOF'
import sys, os, json, time
sys.path.insert(0, '/home/ubuntu/SpatialPathDB_CLEAN')
os.chdir('/home/ubuntu/SpatialPathDB_CLEAN')

import numpy as np
import psycopg2
from scipy import stats as sp_stats

from spdb import config

# --- Load metadata ---
meta_path = os.path.join(config.RESULTS_DIR, "ingest_metadata.json")
with open(meta_path) as f:
    metadata = json.load(f)

slide_ids = metadata["slide_ids"]
print(f"  Loaded metadata: {len(slide_ids)} slides")

# --- Compute stats helper (mirrors benchmarks.framework.compute_stats) ---
def compute_stats(latencies_ms):
    arr = np.array(latencies_ms, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if n > 1:
        se = std / np.sqrt(n)
        t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
        ci_half = t_crit * se
    else:
        ci_half = 0.0
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "std": std,
        "cv": std / mean if mean > 0 else 0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_lower": round(mean - ci_half, 3),
        "ci95_upper": round(mean + ci_half, 3),
        "ci95_half": round(ci_half, 3),
    }

# --- Get slide dimensions helper ---
def get_slide_dimensions(metadata, slide_id):
    m = metadata["metas"][slide_id]
    return float(m["image_width"]), float(m["image_height"])

# --- Random viewport helper (same as framework.random_viewport) ---
def random_viewport(width, height, frac, rng):
    vw = width * np.sqrt(frac)
    vh = height * np.sqrt(frac)
    x0 = float(rng.uniform(0, max(1, width - vw)))
    y0 = float(rng.uniform(0, max(1, height - vh)))
    return x0, y0, float(x0 + vw), float(y0 + vh)

# --- Configuration ---
SO_TABLE = config.TABLE_SLIDE_ONLY   # "objects_slide_only"
FRACS = [0.01, 0.02, 0.05, 0.10, 0.20]
N_TRIALS = 200
SEED = 42

print(f"  SO table: {SO_TABLE}")
print(f"  Viewport fractions: {FRACS}")
print(f"  Trials per fraction: {N_TRIALS}")

# --- Connect ---
conn = psycopg2.connect(config.dsn())

# --- Warmup ---
print("  Warming up cache...")
with conn.cursor() as cur:
    for _ in range(3):
        cur.execute(f"SELECT COUNT(*) FROM {SO_TABLE}")
        cur.fetchone()
        cur.execute(f"SELECT * FROM {SO_TABLE} ORDER BY random() LIMIT 1000")
        cur.fetchall()

# --- Run benchmark ---
results = {}
for frac in FRACS:
    print(f"\n  Viewport fraction: {frac}")
    rng = np.random.RandomState(SEED)
    latencies = []

    for trial in range(N_TRIALS):
        sid = rng.choice(slide_ids)
        w, h = get_slide_dimensions(metadata, sid)
        x0, y0, x1, y1 = random_viewport(w, h, frac, rng)

        sql = f"""
            SELECT * FROM {SO_TABLE}
            WHERE slide_id = %s
              AND centroid_x BETWEEN %s AND %s
              AND centroid_y BETWEEN %s AND %s
        """
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql, (sid, x0, x1, y0, y1))
            rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    stats = compute_stats(latencies)
    results[str(frac)] = {
        "SO": stats,
    }
    print(f"    SO p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  "
          f"mean={stats['mean']:.1f}ms  n_trials={stats['n']}")

conn.close()

# --- Save ---
out_path = os.path.join(config.RAW_DIR, "viewport_sensitivity_so.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to {out_path}")
print("  SO viewport sensitivity complete.")
PYEOF
    STEP_RC=$?
    set -e
    if [[ ${STEP_RC} -eq 0 ]]; then
        add_summary "SO viewport sensitivity: DONE"
    else
        add_summary "SO viewport sensitivity: FAILED (exit code ${STEP_RC}, see log)"
    fi
fi

# ===========================================================================
# STEP 3: TCIA Multi-Cancer Download
# ===========================================================================
banner "STEP 3/4: TCIA Multi-Cancer Data (BRCA, LUAD, COAD)"

TCIA_DIR="${PROJ_DIR}/data_cache/tcia"
TCIA_RESULT="${RAW_DIR}/tcia_multi_cancer.json"
TCIA_CANCER_TYPES=("BRCA" "LUAD" "COAD")
TCIA_ALL_PRESENT=true

for CT in "${TCIA_CANCER_TYPES[@]}"; do
    CT_DIR="${TCIA_DIR}/${CT}_polygon"
    if [[ ! -d "${CT_DIR}" ]] || [[ -z "$(ls -A "${CT_DIR}" 2>/dev/null)" ]]; then
        TCIA_ALL_PRESENT=false
        break
    fi
done

if [[ "${TCIA_ALL_PRESENT}" == "true" ]] && [[ -f "${TCIA_RESULT}" ]] && [[ -s "${TCIA_RESULT}" ]]; then
    echo "  SKIP: TCIA multi-cancer data already downloaded and processed"
    add_summary "TCIA multi-cancer: SKIPPED (already exists)"
else
    echo "  Attempting TCIA Pan-Cancer-Nuclei-Seg download for: ${TCIA_CANCER_TYPES[*]}"
    echo ""
    mkdir -p "${TCIA_DIR}"

    set +e
    python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, '/home/ubuntu/SpatialPathDB_CLEAN')
os.chdir('/home/ubuntu/SpatialPathDB_CLEAN')

from spdb.multi_dataset import TCIADatasetAdapter, download_tcia_cancer_type

TCIA_DIR = "/home/ubuntu/SpatialPathDB_CLEAN/data_cache/tcia"
CANCER_TYPES = ["BRCA", "LUAD", "COAD"]
results = {}

for ct in CANCER_TYPES:
    print(f"\n--- {ct} ---")
    ct_dir = os.path.join(TCIA_DIR, f"{ct}_polygon")

    # Check if already downloaded
    if os.path.isdir(ct_dir) and len(os.listdir(ct_dir)) > 0:
        n_files = len([f for f in os.listdir(ct_dir) if f.endswith(".csv")])
        print(f"  Already downloaded: {n_files} CSV files")
        results[ct] = {"status": "already_present", "n_files": n_files}
        continue

    # Try automatic download
    try:
        polygon_dir = download_tcia_cancer_type(ct, output_dir=TCIA_DIR)
        if os.path.isdir(polygon_dir) and len(os.listdir(polygon_dir)) > 0:
            n_files = len([f for f in os.listdir(polygon_dir) if f.endswith(".csv")])
            print(f"  Downloaded: {n_files} CSV files")
            results[ct] = {"status": "downloaded", "n_files": n_files}
        else:
            results[ct] = {"status": "manual_download_needed"}
    except Exception as e:
        print(f"  Automatic download failed: {e}")
        results[ct] = {"status": "manual_download_needed", "error": str(e)}

# If any cancer types have data, try parsing them
adapter = TCIADatasetAdapter(cache_dir=TCIA_DIR)
parse_results = {}

for ct in CANCER_TYPES:
    csv_files = adapter.list_csv_files(ct)
    if csv_files:
        print(f"\n  Parsing {ct}: {len(csv_files)} CSV files found")
        slides = adapter.load_cancer_type(ct, n_slides=10, seed=42)
        total_nuclei = sum(len(df) for df, _ in slides)
        parse_results[ct] = {
            "n_slides": len(slides),
            "total_nuclei": total_nuclei,
            "csv_files_available": len(csv_files),
        }
        print(f"    Loaded {len(slides)} slides, {total_nuclei:,} nuclei")
    else:
        parse_results[ct] = {"n_slides": 0, "total_nuclei": 0}

# Save status
output = {
    "download_status": results,
    "parse_results": parse_results,
}

out_path = os.path.join("/home/ubuntu/SpatialPathDB_CLEAN/results/raw",
                        "tcia_multi_cancer.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  TCIA status saved to {out_path}")

# Print manual download instructions for any missing data
missing = [ct for ct, v in results.items()
           if v.get("status") == "manual_download_needed"]
if missing:
    print("\n" + "=" * 60)
    print("  MANUAL DOWNLOAD REQUIRED for: " + ", ".join(missing))
    print("=" * 60)
    print("  1. Visit: https://www.cancerimagingarchive.net/analysis-result/pan-cancer-nuclei-seg/")
    print("  2. Download the polygon CSV archives for:", ", ".join(missing))
    print(f"  3. Extract to: {TCIA_DIR}/")
    print(f"     Expected structure:")
    for ct in missing:
        print(f"       {TCIA_DIR}/{ct}_polygon/*.csv")
    print("  4. Re-run this script to process the downloaded data.")
    print("=" * 60)
PYEOF
    STEP_RC=$?
    set -e
    if [[ ${STEP_RC} -eq 0 ]]; then
        add_summary "TCIA multi-cancer: DONE (check log for manual download needs)"
    else
        add_summary "TCIA multi-cancer: FAILED (exit code ${STEP_RC}, see log)"
    fi
fi

# ===========================================================================
# STEP 4: OSM Buildings Experiment
# ===========================================================================
banner "STEP 4/4: OSM Buildings Experiment"

OSM_RESULT="${RAW_DIR}/osm_buildings.json"

if check_result "${OSM_RESULT}" "OSM buildings"; then
    add_summary "OSM buildings: SKIPPED (already exists)"
else
    echo "  Running OSM Buildings experiment (Manhattan)..."
    set +e
    python3 << 'PYEOF'
import sys, os, json, time
sys.path.insert(0, '/home/ubuntu/SpatialPathDB_CLEAN')
os.chdir('/home/ubuntu/SpatialPathDB_CLEAN')

import numpy as np
import psycopg2

from spdb import config
from spdb.multi_dataset import OSMDatasetAdapter
from benchmarks.framework import compute_stats

# --- Download OSM buildings ---
print("  Downloading Manhattan buildings from Overpass API...")
adapter = OSMDatasetAdapter(region_name="manhattan")
try:
    buildings = adapter.download_osm_buildings(
        bbox=(40.7000, -74.0200, 40.8200, -73.9300),  # Manhattan
        timeout=300,
    )
except Exception as e:
    print(f"  ERROR: Overpass API download failed: {e}")
    print("  Retrying with smaller bbox (Midtown only)...")
    try:
        buildings = adapter.download_osm_buildings(
            bbox=(40.7480, -73.9930, 40.7680, -73.9680),  # Midtown
            timeout=300,
        )
        adapter.region_name = "midtown"
    except Exception as e2:
        print(f"  ERROR: Retry also failed: {e2}")
        print("  Saving error status and exiting OSM step.")
        out_path = os.path.join(config.RAW_DIR, "osm_buildings.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"status": "download_failed", "error": str(e2)}, f, indent=2)
        sys.exit(0)

n_buildings = len(buildings)
print(f"  Downloaded {n_buildings} buildings")

if n_buildings < 100:
    print("  WARNING: Too few buildings for meaningful benchmark. Saving status.")
    out_path = os.path.join(config.RAW_DIR, "osm_buildings.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"status": "insufficient_data", "n_buildings": n_buildings}, f, indent=2)
    sys.exit(0)

# --- Transform to SPDB schema ---
print("  Transforming to SPDB schema...")
df, meta = adapter.transform_to_spdb_schema()
print(f"  Region: {meta['slide_id']}, {meta['num_objects']:,} objects")
print(f"  Extent: {meta['image_width']:.0f}m x {meta['image_height']:.0f}m")

# --- Create temporary table and ingest ---
conn = psycopg2.connect(config.dsn())
conn.autocommit = True

TABLE_OSM = "objects_osm_buildings"

print(f"  Creating table {TABLE_OSM}...")
with conn.cursor() as cur:
    cur.execute(f"DROP TABLE IF EXISTS {TABLE_OSM} CASCADE")
    cur.execute(f"""
        CREATE TABLE {TABLE_OSM} (
            object_id       BIGSERIAL PRIMARY KEY,
            slide_id        TEXT        NOT NULL,
            geom            GEOMETRY(Point, 0),
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
        )
    """)

# Batch insert using COPY for speed
print("  Ingesting data via INSERT batches...")
conn.autocommit = False
from psycopg2.extras import execute_values
batch_size = 5000
rows_inserted = 0

for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    chunk = df.iloc[start:end]
    values = []
    for _, row in chunk.iterrows():
        values.append((
            str(row["slide_id"]),
            float(row["centroid_x"]), float(row["centroid_y"]),
            str(row["class_label"]), str(row["tile_id"]),
            int(row["hilbert_key"]), int(row["zorder_key"]),
            float(row["area"]), float(row["perimeter"]),
            float(row["confidence"]), str(row["pipeline_id"]),
        ))
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""INSERT INTO {TABLE_OSM}
                (slide_id, centroid_x, centroid_y, class_label, tile_id,
                 hilbert_key, zorder_key, area, perimeter, confidence, pipeline_id)
                VALUES %s""",
            values,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        )
    conn.commit()
    rows_inserted += len(chunk)
    if rows_inserted % 20000 == 0 or end == len(df):
        print(f"    Inserted {rows_inserted:,} / {len(df):,} rows")

# Update geometry column from centroid coords
print("  Setting geometry from centroids...")
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(f"""
        UPDATE {TABLE_OSM}
        SET geom = ST_SetSRID(ST_MakePoint(centroid_x, centroid_y), 0)
    """)

# Create indexes
print("  Creating indexes...")
with conn.cursor() as cur:
    cur.execute(f"CREATE INDEX idx_osm_geom ON {TABLE_OSM} USING gist (geom)")
    cur.execute(f"CREATE INDEX idx_osm_slide ON {TABLE_OSM} USING btree (slide_id)")
    cur.execute(f"CREATE INDEX idx_osm_hilbert ON {TABLE_OSM} USING btree (slide_id, hilbert_key)")
    cur.execute(f"ANALYZE {TABLE_OSM}")

print(f"  Table {TABLE_OSM} ready with {rows_inserted:,} rows")

# --- Run viewport benchmark ---
print("\n  Running viewport benchmark on OSM data...")
FRACS = [0.01, 0.05, 0.10]
N_TRIALS = 200
SEED = 42
rng = np.random.RandomState(SEED)

w = meta["image_width"]
h = meta["image_height"]
region = meta["slide_id"]

def random_viewport(width, height, frac, rng):
    vw = width * np.sqrt(frac)
    vh = height * np.sqrt(frac)
    x0 = float(rng.uniform(0, max(1, width - vw)))
    y0 = float(rng.uniform(0, max(1, height - vh)))
    return x0, y0, float(x0 + vw), float(y0 + vh)

# Warmup
conn.autocommit = False
with conn.cursor() as cur:
    for _ in range(3):
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_OSM}")
        cur.fetchone()

benchmark_results = {}
for frac in FRACS:
    print(f"\n  Viewport frac={frac}")
    rng_frac = np.random.RandomState(SEED)
    latencies = []

    for trial in range(N_TRIALS):
        x0, y0, x1, y1 = random_viewport(w, h, frac, rng_frac)
        sql = f"""
            SELECT object_id, centroid_x, centroid_y, class_label
            FROM {TABLE_OSM}
            WHERE slide_id = %s
              AND centroid_x BETWEEN %s AND %s
              AND centroid_y BETWEEN %s AND %s
        """
        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(sql, (region, x0, x1, y0, y1))
            rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    stats = compute_stats(latencies)
    benchmark_results[str(frac)] = stats
    print(f"    p50={stats['p50']:.1f}ms  p95={stats['p95']:.1f}ms  mean={stats['mean']:.1f}ms")

# --- Get storage stats ---
print("\n  Measuring storage...")
with conn.cursor() as cur:
    cur.execute(f"SELECT pg_total_relation_size('{TABLE_OSM}')")
    total_bytes = cur.fetchone()[0]
    cur.execute(f"SELECT COUNT(*) FROM {TABLE_OSM}")
    row_count = cur.fetchone()[0]

conn.close()

# --- Save results ---
output = {
    "dataset": "osm_buildings",
    "region": meta["slide_id"],
    "n_buildings": int(n_buildings),
    "n_ingested": int(row_count),
    "extent_m": {"width": meta["image_width"], "height": meta["image_height"]},
    "storage_bytes": int(total_bytes),
    "storage_mb": round(total_bytes / 1048576, 1),
    "bytes_per_row": round(total_bytes / max(1, row_count), 1),
    "viewport_benchmarks": benchmark_results,
    "meta": {k: v for k, v in meta.items() if k != "num_buckets"},
}

out_path = os.path.join(config.RAW_DIR, "osm_buildings.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  OSM results saved to {out_path}")
print(f"  Storage: {output['storage_mb']} MB for {row_count:,} buildings")
print("  OSM buildings experiment complete.")
PYEOF
    STEP_RC=$?
    set -e
    if [[ ${STEP_RC} -eq 0 ]]; then
        add_summary "OSM buildings: DONE"
    else
        add_summary "OSM buildings: FAILED (exit code ${STEP_RC}, see log)"
    fi
fi

# ===========================================================================
# SUMMARY
# ===========================================================================
banner "ALL REMAINING BENCHMARKS COMPLETE"

echo ""
echo "Summary:"
echo -e "${SUMMARY}"
echo ""
echo "Result files:"
for f in \
    "${RAW_DIR}/duckdb_baseline.json" \
    "${RAW_DIR}/viewport_sensitivity_so.json" \
    "${RAW_DIR}/tcia_multi_cancer.json" \
    "${RAW_DIR}/osm_buildings.json"; do
    if [[ -f "$f" ]] && [[ -s "$f" ]]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  [OK]   ${f}  (${SIZE})"
    else
        echo "  [MISS] ${f}"
    fi
done

echo ""
echo "Log: ${LOG}"
echo "Finished at: $(ts)"
