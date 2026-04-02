# SpatialPathDB

**Locality-Aware Partitioned Storage for Scalable Spatial Querying in Digital Pathology**

## Key Results (42.1M nuclei, 29 TCGA BLCA slides)

| Metric | Value |
|--------|-------|
| Q1 Viewport (SPDB p50) | **63 ms** |
| Q1 Viewport (Mono p50) | 159 ms |
| Improvement over Mono | 2.5x |
| Cold-cache improvement | **9.1x** (53 ms vs 486 ms) |
| Partition pruning rate | **89%** (5.7 of ~59 sub-partitions scanned) |
| Hilbert vs Z-order | Hilbert **28% faster** |
| kNN (k=50, SPDB) | 15 ms p50 |
| Mixed workload throughput | 16.4 QPS |
| Max concurrent throughput (SO) | 65 QPS @ 16 clients |

## Architecture

```
┌─────────────────────────────────────────┐
│             Application Layer           │
│   (Hilbert key range computation)       │
├─────────────────────────────────────────┤
│           PostgreSQL 17 / PostGIS 3.6   │
├─────────────────────────────────────────┤
│  Level 1: LIST(slide_id)  [29 slides]   │
│  Level 2: RANGE(hilbert_key) [~30/slide]│
│  = 857 leaf partitions                  │
├─────────────────────────────────────────┤
│  Per-partition hybrid indexes:          │
│  - GiST(geom)                          │
│  - B-tree(slide_id, hilbert_key)        │
│  - B-tree(slide_id, class_label)        │
│  - B-tree(tile_id)                      │
└─────────────────────────────────────────┘
```

## Project Structure

```
spdb/
  config.py          - Configuration (DB, Hilbert order, benchmark params)
  hilbert.py         - Vectorized Hilbert curve encoder/decoder
  zorder.py          - Z-order (Morton) curve encoder
  schema.py          - Schema creation for all 5 DB configurations
  ingest.py          - Data pipeline: HuggingFace → transform → COPY
  pruning_model.py   - Formal pruning model
  adaptive.py        - Adaptive repartitioning prototype

benchmarks/
  framework.py       - Timing, statistics, EXPLAIN parsing, I/O decomposition
  q1_viewport.py     - Q1: Viewport range queries (with Hilbert pruning)
  q2_knn.py          - Q2: k-Nearest Neighbor queries
  q3_aggregation.py  - Q3: Tile-level aggregation
  q4_spatial_join.py - Q4: Spatial join with phenotype filter
  concurrent.py      - Concurrent throughput benchmark (asyncpg)
  extended.py        - Extended experiments (12 experiments total)

visualization/
  figures.py         - Publication-ready Matplotlib figures
  tables.py          - LaTeX table generation

paper/
  spdb.tex           - Full LaTeX paper

results/
  raw/               - JSON results + CSV raw latencies
  figures/           - PDF/PNG figures
  tables/            - LaTeX table fragments
```

## Quick Start

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Create database
createdb spdb
psql -d spdb -c "CREATE EXTENSION postgis;"

# 3. Ingest data
python run_ingest.py

# 4. Run benchmarks
python run_benchmarks.py all

# 5. Generate figures
python run_benchmarks.py figures
```

## Experiments

- **Q1-Q4**: Core query benchmarks across all 5 configurations
- **Viewport sensitivity**: 1%, 2%, 5%, 10%, 20% viewport fractions
- **Hilbert order sweep**: p ∈ {6, 8, 10, 12}
- **Hilbert vs Z-order**: Controlled comparison, identical partition structure
- **kNN k-sweep**: k ∈ {10, 25, 50, 100, 200}
- **Concurrency**: 1, 4, 16, 32, 64 concurrent clients
- **Workload mix**: 70% Q1, 15% Q2, 10% Q3, 5% Q4
- **Storage overhead**: Recursive partition size measurement
- **Density analysis**: Per-slide object density distribution
- **Cold cache**: Fresh connection per trial
- **Pruning analysis**: EXPLAIN-based partition scan counting

## Citation

```bibtex
@article{yamani2026spatialpathdb,
  title={SpatialPathDB: Locality-Aware Partitioned Storage for
         Scalable Spatial Querying in Digital Pathology},
  author={Yamani, Sai Asish},
  year={2026}
}
```
