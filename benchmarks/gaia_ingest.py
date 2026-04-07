"""Gaia DR3 astronomy catalog ingestion: download, load, build HCCI + GiST indexes.

Downloads stellar sources from the ESA Gaia DR3 catalog via TAP (Table Access
Protocol). Uses right ascension (ra) and declination (dec) as spatial coordinates,
and color-based stellar classification (bp_rp color index) as categorical label.

Target: 50M+ sources from full sky declination strips.
Class label = stellar color class (Blue, White, Yellow, Orange, Red, Very_Red)

Usage:
    python -m benchmarks.gaia_ingest
    python -m benchmarks.gaia_ingest --target-rows 50000000
    python -m benchmarks.gaia_ingest --index-only
    python -m benchmarks.gaia_ingest --stats-only
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

import numpy as np
import psycopg2

from spdb import config, hilbert, hcci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TABLE = "gaia_sources"
INDEX_HCCI = "idx_gaia_hcci_covering"
INDEX_GIST = "idx_gaia_gist"
DATASET_ID = "gaia_dr3"

CACHE_DIR = "/tmp/gaia_cache"

# ESA Gaia TAP endpoint
TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

# Stellar color classification based on BP-RP color index
# Maps physical stellar types to color bins
COLOR_CLASSES = {
    "Blue":      (-1.0, 0.5),    # O/B stars, hot
    "White":     (0.5, 1.0),     # A/F stars
    "Yellow":    (1.0, 1.5),     # G stars (Sun-like)
    "Orange":    (1.5, 2.0),     # K stars
    "Red":       (2.0, 3.0),     # M stars, cool
    "Very_Red":  (3.0, 10.0),    # Very cool / reddened
}

# Sky patches: (dec_min, dec_max, ra_min, ra_max, magnitude_limit)
# Split sky into dec strips × RA slices for reliable TAP queries.
# Each patch covers a smaller sky area → more likely to succeed.
def _generate_sky_patches():
    """Generate sky patches covering the full sky."""
    patches = []
    dec_bands = [
        (-90, -60, 19.0),
        (-60, -30, 19.0),
        (-30,   0, 19.0),
        (  0,  30, 19.0),
        ( 30,  60, 19.0),
        ( 60,  90, 18.0),
    ]
    ra_slices = [(0, 90), (90, 180), (180, 270), (270, 360)]

    for dec_min, dec_max, mag in dec_bands:
        for ra_min, ra_max in ra_slices:
            patches.append((dec_min, dec_max, ra_min, ra_max, mag))
    return patches

SKY_PATCHES = _generate_sky_patches()  # 24 patches


# ---------------------------------------------------------------------------
# TAP query helpers
# ---------------------------------------------------------------------------

def _submit_tap_async(adql: str) -> str:
    """Submit an async TAP query and return the job URL."""
    data = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }).encode()

    req = urllib.request.Request(f"{TAP_URL}/async", data=data, method="POST")
    req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")

    with urllib.request.urlopen(req, timeout=120) as resp:
        # The response redirects to the job URL
        job_url = resp.url
    return job_url


def _run_tap_phase(job_url: str):
    """Set job phase to RUN."""
    data = urllib.parse.urlencode({"PHASE": "RUN"}).encode()
    req = urllib.request.Request(f"{job_url}/phase", data=data, method="POST")
    req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")
    with urllib.request.urlopen(req, timeout=30) as resp:
        pass


def _wait_tap_job(job_url: str, timeout_sec: int = 600) -> str:
    """Poll job status until COMPLETED or ERROR."""
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        req = urllib.request.Request(f"{job_url}/phase")
        req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            phase = resp.read().decode().strip()
        if phase == "COMPLETED":
            return "COMPLETED"
        elif phase in ("ERROR", "ABORTED"):
            return phase
        time.sleep(5)
    return "TIMEOUT"


def _download_tap_result(job_url: str) -> str:
    """Download the result CSV from a completed TAP job."""
    req = urllib.request.Request(f"{job_url}/results/result")
    req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")
    with urllib.request.urlopen(req, timeout=600) as resp:
        return resp.read().decode()


def query_gaia_patch(dec_min: float, dec_max: float,
                     ra_min: float, ra_max: float,
                     mag_limit: float,
                     max_rows: int = 5_000_000) -> list[dict]:
    """Query Gaia DR3 for sources in a sky patch (dec + RA bounded)."""
    cache_file = os.path.join(
        CACHE_DIR, f"gaia_dec{dec_min}_{dec_max}_ra{ra_min}_{ra_max}.csv"
    )
    if os.path.exists(cache_file):
        print(f"    Cached: dec [{dec_min},{dec_max}] ra [{ra_min},{ra_max}]")
        return _parse_csv_cache(cache_file)

    adql = f"""
    SELECT TOP {max_rows}
        source_id, ra, dec, phot_g_mean_mag, bp_rp,
        phot_bp_mean_mag, phot_rp_mean_mag, parallax
    FROM gaiadr3.gaia_source
    WHERE dec >= {dec_min} AND dec < {dec_max}
      AND ra >= {ra_min} AND ra < {ra_max}
      AND phot_g_mean_mag < {mag_limit}
      AND bp_rp IS NOT NULL
      AND ra IS NOT NULL AND dec IS NOT NULL
    """

    label = f"dec [{dec_min:+4.0f},{dec_max:+4.0f}] ra [{ra_min:.0f},{ra_max:.0f}]"
    print(f"    Querying {label}, mag < {mag_limit}...")

    try:
        job_url = _submit_tap_async(adql)
        _run_tap_phase(job_url)
        status = _wait_tap_job(job_url, timeout_sec=900)

        if status != "COMPLETED":
            print(f"      TAP job {status}, trying sync fallback...")
            return _query_gaia_sync_patch(
                dec_min, dec_max, ra_min, ra_max, mag_limit, max_rows
            )

        csv_data = _download_tap_result(job_url)

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "w") as f:
            f.write(csv_data)

        return _parse_csv_string(csv_data)

    except Exception as e:
        print(f"      Async failed ({e}), trying sync...")
        return _query_gaia_sync_patch(
            dec_min, dec_max, ra_min, ra_max, mag_limit, max_rows
        )


def _query_gaia_sync_patch(dec_min: float, dec_max: float,
                           ra_min: float, ra_max: float,
                           mag_limit: float, max_rows: int) -> list[dict]:
    """Fallback: synchronous TAP query for a sky patch."""
    adql = f"""
    SELECT TOP {min(max_rows, 3000000)}
        source_id, ra, dec, phot_g_mean_mag, bp_rp,
        phot_bp_mean_mag, phot_rp_mean_mag, parallax
    FROM gaiadr3.gaia_source
    WHERE dec >= {dec_min} AND dec < {dec_max}
      AND ra >= {ra_min} AND ra < {ra_max}
      AND phot_g_mean_mag < {mag_limit}
      AND bp_rp IS NOT NULL
    """
    data = urllib.parse.urlencode({
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": adql,
    }).encode()

    req = urllib.request.Request(f"{TAP_URL}/sync", data=data)
    req.add_header("User-Agent", "SpatialPathDB-HCCI-Research/1.0")

    with urllib.request.urlopen(req, timeout=900) as resp:
        csv_data = resp.read().decode()

    cache_file = os.path.join(
        CACHE_DIR, f"gaia_dec{dec_min}_{dec_max}_ra{ra_min}_{ra_max}.csv"
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write(csv_data)

    return _parse_csv_string(csv_data)


def _parse_csv_string(csv_data: str) -> list[dict]:
    """Parse Gaia CSV response into records."""
    records = []
    reader = csv.DictReader(io.StringIO(csv_data))
    for row in reader:
        try:
            ra = float(row["ra"])
            dec = float(row["dec"])
            bp_rp = float(row["bp_rp"])
            mag = float(row["phot_g_mean_mag"])

            # Classify by color
            class_label = "Unknown"
            for name, (lo, hi) in COLOR_CLASSES.items():
                if lo <= bp_rp < hi:
                    class_label = name
                    break

            records.append({
                "source_id": int(row["source_id"]),
                "ra": ra,
                "dec": dec,
                "mag": mag,
                "bp_rp": bp_rp,
                "class_label": class_label,
            })
        except (ValueError, KeyError):
            continue

    return records


def _parse_csv_cache(path: str) -> list[dict]:
    """Parse cached CSV file."""
    with open(path) as f:
        return _parse_csv_string(f.read())


def download_all_patches(patches: list, target_rows: int) -> list[dict]:
    """Download Gaia sources from sky patches (dec × RA grid)."""
    print(f"\n[Download] Fetching Gaia DR3 sources from {len(patches)} sky patches...")
    print(f"  Target: {target_rows:,} total rows")
    t0 = time.time()

    all_records = []
    # Request 3M per patch to overshoot target, then trim at the end
    rows_per_patch = max(3_000_000, target_rows // len(patches) + 1)

    for dec_min, dec_max, ra_min, ra_max, mag_limit in patches:
        records = query_gaia_patch(dec_min, dec_max, ra_min, ra_max,
                                   mag_limit, max_rows=rows_per_patch)
        all_records.extend(records)
        print(f"    dec [{dec_min:+4.0f},{dec_max:+4.0f}] ra [{ra_min:.0f},{ra_max:.0f}]: "
              f"{len(records):,} sources  (total: {len(all_records):,})")

        if len(all_records) >= target_rows:
            print(f"  Reached target ({target_rows:,}), stopping.")
            all_records = all_records[:target_rows]
            break

        time.sleep(3)  # Rate limit courtesy

    elapsed = time.time() - t0
    print(f"\n  Downloaded {len(all_records):,} sources in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    return all_records


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def create_table(conn):
    """Create the gaia_sources table."""
    print(f"\n[Create Table] Setting up {TABLE}...")
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        cur.execute(f"""
            CREATE TABLE {TABLE} (
                id SERIAL PRIMARY KEY,
                source_id BIGINT,
                dataset_id TEXT DEFAULT '{DATASET_ID}',
                centroid_x DOUBLE PRECISION,
                centroid_y DOUBLE PRECISION,
                class_label TEXT,
                mag DOUBLE PRECISION,
                bp_rp DOUBLE PRECISION,
                geom GEOMETRY(Point, 0),
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
            f"{r['source_id']}\t{DATASET_ID}\t{r['ra']}\t{r['dec']}\t"
            f"{r['class_label']}\t{r['mag']}\t{r['bp_rp']}\n"
        )

    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_from(
            buf, TABLE,
            columns=("source_id", "dataset_id", "centroid_x", "centroid_y",
                     "class_label", "mag", "bp_rp"),
            sep="\t",
        )
    conn.commit()
    elapsed = time.time() - t0
    rate = len(records) / elapsed if elapsed > 0 else 0
    print(f"  Loaded in {elapsed:.1f}s ({rate:,.0f} rows/s)")


def build_geometry(conn):
    """Populate geom column from ra/dec (SRID=0 for sky coordinates)."""
    print(f"\n[Geometry] Building point geometries (ra, dec)...")
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE {TABLE}
            SET geom = ST_MakePoint(centroid_x, centroid_y)
        """)
    conn.commit()
    print(f"  Done in {time.time() - t0:.1f}s")


def compute_hilbert_keys(conn, hilbert_order: int = config.HILBERT_ORDER):
    """Compute Hilbert keys from normalized ra/dec coordinates."""
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

    print(f"  Bounds: ra=[{x_min:.2f}, {x_max:.2f}], dec=[{y_min:.2f}, {y_max:.2f}]")
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
    """Build color class enum."""
    print(f"\n[Enum] Building stellar color class enum...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        rows = cur.fetchall()

    class_enum = {}
    print(f"  {len(rows)} color classes:")
    for i, (label, cnt) in enumerate(rows):
        class_enum[label] = i
        print(f"    {i:>3}: {label:<12} {cnt:>12,}")

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
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total = cur.fetchone()[0]
        print(f"  Total rows: {total:,}")

        cur.execute(f"""
            SELECT class_label, COUNT(*) as cnt,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) as pct
            FROM {TABLE}
            GROUP BY class_label
            ORDER BY cnt DESC
        """)
        print("\n  Stellar color class distribution:")
        for cat, cnt, pct in cur.fetchall():
            print(f"    {cat:<12} {cnt:>12,}  ({pct}%)")

        cur.execute(f"""
            SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
            FROM pg_indexes WHERE tablename = '{TABLE}'
        """)
        print("\n  Indexes:")
        for name, size in cur.fetchall():
            print(f"    {name}: {size}")

        cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{TABLE}'))")
        total_size = cur.fetchone()[0]
        print(f"  Total table size (with indexes): {total_size}")

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


def save_metadata(bounds: dict, class_enum: dict[str, int], path: str):
    """Save dataset metadata for benchmark use."""
    meta = {
        "table": TABLE,
        "dataset_id": DATASET_ID,
        "bounds": bounds,
        "class_enum": class_enum,
        "hilbert_order": config.HILBERT_ORDER,
        "source": "Gaia DR3",
        "coordinate_system": "ICRS (ra, dec)",
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
        description="Gaia DR3 astronomy catalog ingestion for HCCI cross-domain validation"
    )
    parser.add_argument("--target-rows", type=int, default=50_000_000,
                        help="Target number of sources to download (default: 50M)")
    parser.add_argument("--index-only", action="store_true",
                        help="Skip download and load, just build indexes")
    parser.add_argument("--stats-only", action="store_true",
                        help="Just print stats")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gaia DR3 Astronomy Catalog Ingestion")
    print(f"  Target: {args.target_rows:,} stellar sources")
    print(f"  Sky patches: {len(SKY_PATCHES)} (dec × RA grid)")
    print(f"  Class label: stellar color (BP-RP index)")
    print("=" * 60)

    conn = psycopg2.connect(config.dsn())
    t_start = time.time()

    if args.stats_only:
        class_enum = build_class_enum(conn)
        verify(conn, class_enum)
        conn.close()
        return

    if not args.index_only:
        all_records = download_all_patches(SKY_PATCHES, args.target_rows)
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
    meta_path = os.path.join(config.RAW_DIR, "gaia_metadata.json")
    save_metadata(bounds, class_enum, meta_path)

    total = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Gaia ingestion complete in {total:.0f}s ({total / 60:.1f}m)")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
