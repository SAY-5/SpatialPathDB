# SpatialPathDB

Spatial data infrastructure for digital pathology. Built to handle the scale problem that comes with whole-slide imaging — a single digitized tissue sample produces millions of cell geometries that need to be stored, indexed, and queried in real time.

![Stack](https://img.shields.io/badge/PostGIS%20%7C%20Spring%20Boot%20%7C%20Redis%20%7C%20Celery%20%7C%20Spark%20%7C%20React-00e5cc?style=flat-square&labelColor=0a0a0f)
![Platform](https://img.shields.io/badge/Docker-ARM64%20%2F%20AMD64-00e5cc?style=flat-square&labelColor=0a0a0f)

---

## Background

Computational pathology workflows generate enormous amounts of spatial data. A single H&E slide scanned at 40x magnification can contain upward of 5 million cells, each with a location, morphology, and classification. Standard relational databases collapse under this load — full table scans on geometry columns, no partition isolation, no spatial indexing strategy.

This project is my attempt at building the data layer properly: spatially-indexed, partitioned, cached, and queryable at sub-100ms latency.

---

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     React + Leaflet UI                       │
│            Spatial viewer · Query panel · Job monitor        │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST
┌───────────────────────────▼─────────────────────────────────┐
│                   Spring Boot API (Java 17)                   │
│              Spatial endpoints · Redis cache layer           │
└──────────┬────────────────────────────┬─────────────────────┘
           │                            │
┌──────────▼──────────┐     ┌──────────▼──────────┐
│  PostgreSQL + PostGIS│     │        Redis         │
│  Hash-partitioned    │     │   5–30 min TTL cache │
│  spatial_objects × 8 │     └─────────────────────┘
│  GIST indexes        │
│  7 stored functions  │
└──────────┬───────────┘
           │
┌──────────▼─────────────────────────────────────────────────┐
│                   Celery Workers (Python)                    │
│    Cell detection · Tissue segmentation · Spatial stats     │
└──────────┬─────────────────────────────────────────────────┘
           │
┌──────────▼─────────────────────────────────────────────────┐
│                     Apache Spark Jobs                        │
│         Cross-slide cohort analytics · Feature extraction   │
└────────────────────────────────────────────────────────────┘
```

---

## Technical Decisions

**Hash partitioning on `slide_id`**
Almost every query filters by slide. Partitioning the `spatial_objects` table into 8 hash buckets means Postgres only ever touches one partition per query — no cross-partition scans, and the planner can parallelize across buckets when needed.

**GIST spatial indexes per partition**
Each partition gets its own GIST index on both `geometry` (polygon) and `centroid` (point). Spatial intersection queries go from O(n) full scans to O(log n) index lookups. The difference between unusable and sub-100ms at this data volume.

**COPY-based ingestion**
Bulk loading uses PostgreSQL's native COPY protocol with in-memory CSV buffers rather than parameterized INSERT statements. Throughput: ~14,000 objects/sec. Parameterized inserts top out around 150/sec at this row size.

**Separate ObjectMappers for Redis**
Spring's default Redis cache serializer doesn't handle `java.time` types. Configured a dedicated `ObjectMapper` with `JavaTimeModule` for both the `RedisTemplate` and `RedisCacheManager` — a common footgun when mixing Spring Cache annotations with custom serialization.

**Leaflet with `CRS.Simple`**
Microscopy data lives in pixel space, not geographic coordinates. Using `CRS.Simple` lets Leaflet treat pixel coordinates natively without the distortion that comes from projecting onto a lat/lng system.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Leaflet, TanStack Query |
| API | Spring Boot 3, Java 17, Hibernate Spatial |
| Database | PostgreSQL 15, PostGIS 3.6 |
| Cache | Redis 7 |
| Workers | Python 3.11, Celery, Shapely, scikit-learn, NumPy |
| Analytics | Apache Spark 3.5 (PySpark) |
| Infra | Docker, Docker Compose (ARM64 + AMD64) |

---

## Running Locally

**Requirements:** Docker, Docker Compose, Python 3.11+
```bash
# Start infrastructure
docker-compose up -d --build

# Apply database migrations
for f in database/migrations/V*.sql; do
  cat "$f" | docker-compose exec -T postgres psql -U pathdb_user -d spatialpathdb
done

# Load synthetic data
pip3 install -r spatial-engine/requirements.txt
python3 scripts/generate_synthetic_data.py --slides 5 --cells-per-slide 100000
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8080 |
| Swagger | http://localhost:8080/swagger-ui.html |
| Health | http://localhost:8080/api/v1/health |

---

## API
```bash
# Bounding box query
POST /api/v1/spatial/bbox
{
  "slideId": "<uuid>",
  "minX": 10000, "minY": 10000,
  "maxX": 50000, "maxY": 50000,
  "limit": 1000
}

# K-nearest neighbors
POST /api/v1/spatial/knn
{
  "slideId": "<uuid>",
  "x": 25000, "y": 25000,
  "k": 50,
  "objectType": "epithelial"
}

# Density grid
POST /api/v1/spatial/density?slideId=<uuid>&gridSize=256
```

---

## Schema
```sql
CREATE TABLE spatial_objects (
    id          UUID PRIMARY KEY,
    slide_id    UUID NOT NULL,
    geometry    GEOMETRY(Polygon, 0),
    centroid    GEOMETRY(Point, 0),
    object_type VARCHAR(50),
    area        DOUBLE PRECISION,
    confidence  DOUBLE PRECISION,
    properties  JSONB
) PARTITION BY HASH (slide_id);
```

---

## Performance

Tested locally on Apple M-series, 300K objects across 3 slides:

| Query | Latency |
|-------|---------|
| Bounding box | ~45ms |
| KNN (k=50) | ~38ms |
| Density grid | ~120ms |
| Bulk ingest | 14K obj/s |

---

## Project Structure
```
SpatialPath/
├── backend/                  # Spring Boot API
├── database/migrations/      # SQL schema, indexes, stored functions
├── frontend/                 # React + TypeScript
├── spatial-engine/           # Python Celery workers
├── spark-jobs/               # PySpark cohort analytics
├── benchmarks/               # Query performance tests
├── scripts/                  # Data generation
└── docker-compose.yml
```
