"""SpatialPathDB configuration: DB connections, Hilbert params, benchmark settings.
  SPDB       -- LIST(slide_id) + RANGE(hilbert_key), hybrid indexes
  SPDB-Z     -- same as SPDB but Z-order keys
"""

import os

DB_HOST = os.getenv("SPDB_DB_HOST", "localhost")
DB_PORT = int(os.getenv("SPDB_DB_PORT", "5432"))
DB_NAME = os.getenv("SPDB_DB_NAME", "spdb")
DB_USER = os.getenv("SPDB_DB_USER", os.getenv("USER", "postgres"))
DB_PASSWORD = os.getenv("SPDB_DB_PASSWORD", "")

HILBERT_ORDER = 8
BUCKET_TARGET = 50_000
VIEWPORT_FRACTION = 0.05

HF_DATASET = "longevity-db/pan-cancer-nuclei-seg"

RANDOM_SEED = 42

BENCHMARK_TRIALS_Q1 = 500
BENCHMARK_TRIALS_Q2 = 500
BENCHMARK_TRIALS_Q3 = 500
BENCHMARK_TRIALS_Q4 = 100
CONCURRENCY_LEVELS = [1, 4, 16, 32, 64]
CONCURRENCY_WINDOW_SEC = 30

HILBERT_ORDERS_SWEEP = [6, 8, 10, 12]
VIEWPORT_FRACTIONS_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.20]
KNN_K_SWEEP = [10, 25, 50, 100, 200]
WORKLOAD_MIX = {"Q1": 0.70, "Q2": 0.15, "Q3": 0.10, "Q4": 0.05}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

CONFIG_NAMES = [
    "mono", "mono_clustered",
    "slide_only", "slide_only_clustered",
    "spdb", "spdb_zorder",
]

# Configs used in the primary evaluation (Mono-T dropped: see paper Section 5)
PRIMARY_CONFIGS = ["mono", "slide_only", "spdb", "spdb_zorder"]

TABLE_MONO = "objects_mono"
TABLE_MONO_TUNED = "objects_mono_tuned"
TABLE_MONO_CLUSTERED = "objects_mono_clustered"
TABLE_SLIDE_ONLY = "objects_slide_only"
TABLE_SLIDE_ONLY_CLUSTERED = "objects_so_clustered"
TABLE_SPDB = "objects_spdb"
TABLE_SPDB_ZORDER = "objects_spdb_zorder"

ALL_TABLES = [
    TABLE_MONO, TABLE_MONO_TUNED, TABLE_MONO_CLUSTERED,
    TABLE_SLIDE_ONLY, TABLE_SLIDE_ONLY_CLUSTERED,
    TABLE_SPDB, TABLE_SPDB_ZORDER,
]

BENCH_CONFIGS = {
    "Mono":   TABLE_MONO,
    "Mono-T": TABLE_MONO_TUNED,
    "Mono-C": TABLE_MONO_CLUSTERED,
    "SO":     TABLE_SLIDE_ONLY,
    "SO-C":   TABLE_SLIDE_ONLY_CLUSTERED,
    "SPDB":   TABLE_SPDB,
}

BENCH_CONFIGS_WITH_ZORDER = {**BENCH_CONFIGS, "SPDB-Z": TABLE_SPDB_ZORDER}

CLASS_LABELS = ["Epithelial", "Stromal", "Tumor", "Lymphocyte"]
CLASS_DISTRIBUTION = {
    "Epithelial": 0.425,
    "Stromal": 0.297,
    "Tumor": 0.164,
    "Lymphocyte": 0.114,
}

BRIN_PAGES_PER_RANGE = 32

PG_RANDOM_PAGE_COST = 1.1
PG_SEQ_PAGE_COST = 1.0
PG_TUPLE_COST = 0.01
PG_INDEX_TUPLE_COST = 0.005


def dsn():
    parts = [f"host={DB_HOST}", f"port={DB_PORT}", f"dbname={DB_NAME}", f"user={DB_USER}"]
    if DB_PASSWORD:
        parts.append(f"password={DB_PASSWORD}")
    return " ".join(parts)


def asyncpg_dsn():
    pwd = f":{DB_PASSWORD}" if DB_PASSWORD else ""
    return f"postgresql://{DB_USER}{pwd}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
