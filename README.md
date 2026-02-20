# SpatialPathDB

I built this because I wanted to understand what it actually takes to store and query spatial data at scale — not toy scale, real scale.

A digitized pathology slide is not a normal image. When a hospital scans a tissue sample at 40x magnification, the result is something like a 100,000 × 90,000 pixel file. An AI model then runs over that image and detects every single cell — where it is, what shape it has, whether it looks like a cancer cell or an immune cell or normal tissue. One slide produces somewhere between 500,000 and 5,000,000 data points, each one a polygon with coordinates, a label, a confidence score, and properties.

Now multiply that by a research cohort of 200 slides.

The question I wanted to answer: how do you actually build the infrastructure layer for that?

---

## What this is

A full-stack spatial data platform. You can load slides, store millions of cell geometries, run spatial queries against them (bounding box, K-nearest neighbors, density estimation), submit async analysis jobs, and visualize everything in an interactive map viewer in the browser.

The interesting parts are not the frontend. The interesting parts are what's happening underneath:

- The database table that stores cell geometries is hash-partitioned across 8 buckets by slide ID. Every query filters by slide, so this means Postgres never touches data it doesn't need. At 5 million rows, this is the difference between a 2-second scan and a 40ms index lookup.

- Each partition has its own GIST spatial index on both the polygon geometry and the centroid point. GIST is the same index type that PostGIS uses for geographic queries — it builds an R-tree structure that makes spatial intersection queries logarithmic instead of linear.

- Bulk data loading uses PostgreSQL's native COPY protocol with in-memory buffers rather than INSERT statements. The throughput difference is not small: 14,000 objects per second vs roughly 150 with parameterized inserts.

- Query results are cached in Redis. Researchers exploring a slide tend to query the same regions repeatedly. The first query hits the database; subsequent identical queries return from cache in under a millisecond.

- Heavy analysis jobs (spatial statistics, tissue segmentation) run asynchronously through Celery workers so they don't block the API. Job status is tracked in the database.

- There's a separate Spark layer for cohort-level analytics — things that don't make sense to run per-slide, like ranking slides by immune cell density across a study population.

---

## The stack and why

**PostgreSQL + PostGIS** for the core store. PostGIS gives you real spatial types — not latitude/longitude tricks, actual geometric primitives with proper operators. `ST_Within`, `ST_DWithin`, `ST_Centroid` — these are first-class database operations with index support.

**Spring Boot** for the API. Java's verbosity is annoying but Hibernate Spatial handles the geometry type mapping cleanly, and the ecosystem around caching annotations (`@Cacheable`) made wiring Redis straightforward.

**Redis** for query caching. Simple TTL-based caching with separate expiry windows per cache type (5 minutes for spatial queries, 30 minutes for slide metadata).

**Celery + Python** for workers. The scientific Python ecosystem (NumPy, Shapely, scikit-learn) is where spatial statistics tooling actually lives. Running those in Java would mean wrapping C libraries with JNI or reimplementing algorithms. Not worth it.

**Apache Spark** for batch analytics. When you need to do population-scale joins across hundreds of slides, you want a distributed execution engine, not a for loop.

**React + Leaflet** for the viewer. Leaflet's `CRS.Simple` mode lets you use pixel coordinates directly without any geographic projection. The cells render as colored markers on a zoomable canvas that treats the slide as a flat coordinate plane.

---

## Running it

You need Docker and Docker Compose. That's it.

```bash
git clone https://github.com/SAY-5/SpatialPathDB.git
cd SpatialPathDB

# Start everything
docker-compose up -d --build

# Apply the database schema (partitioned tables, indexes, stored functions)
for f in database/migrations/V*.sql; do
  cat "$f" | docker-compose exec -T postgres psql -U pathdb_user -d spatialpathdb
done

# Generate synthetic slides with cell data
pip3 install -r spatial-engine/requirements.txt
python3 scripts/generate_synthetic_data.py --slides 3 --cells-per-slide 100000
```

Open http://localhost:3000. You'll see three slides, each with 100,000 spatial objects loaded into the partitioned store. Click into any slide to open the spatial viewer.

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| API | http://localhost:8080 |
| Swagger | http://localhost:8080/swagger-ui.html |
| Health check | http://localhost:8080/api/v1/health |

Works on both Apple Silicon (ARM64) and standard AMD64. The PostGIS image doesn't publish ARM64 builds, so there's a custom Dockerfile in `db/` that installs PostGIS from apt on the official Postgres base image.

---

## Querying the data

```bash
# Cells within a bounding box
curl -X POST http://localhost:8080/api/v1/spatial/bbox \
  -H "Content-Type: application/json" \
  -d '{
    "slideId": "<uuid>",
    "minX": 10000, "minY": 10000,
    "maxX": 50000, "maxY": 50000,
    "limit": 1000
  }'

# 50 nearest cells to a point, filtered by type
curl -X POST http://localhost:8080/api/v1/spatial/knn \
  -H "Content-Type: application/json" \
  -d '{
    "slideId": "<uuid>",
    "x": 25000, "y": 25000,
    "k": 50,
    "objectType": "lymphocyte"
  }'

# Density grid (for heatmap rendering)
curl -X POST "http://localhost:8080/api/v1/spatial/density?slideId=<uuid>&gridSize=256"
```

---

## Schema

The core table. Simple on the surface, the partitioning and indexing strategy is where the work is:

```sql
CREATE TABLE spatial_objects (
    id          UUID DEFAULT gen_random_uuid(),
    slide_id    UUID NOT NULL,
    geometry    GEOMETRY(Polygon, 0),   -- full cell outline
    centroid    GEOMETRY(Point, 0),     -- precomputed center point
    object_type VARCHAR(50),            -- epithelial, stromal, lymphocyte, etc
    area        DOUBLE PRECISION,
    confidence  DOUBLE PRECISION,
    properties  JSONB                   -- arbitrary model outputs
) PARTITION BY HASH (slide_id);

-- 8 hash partitions
CREATE TABLE spatial_objects_p0 PARTITION OF spatial_objects
    FOR VALUES WITH (modulus 8, remainder 0);
-- ... p1 through p7

-- GIST indexes on every partition
CREATE INDEX ON spatial_objects_p0 USING GIST (geometry);
CREATE INDEX ON spatial_objects_p0 USING GIST (centroid);
-- ... repeated for all 8 partitions
```

---

## Numbers

On a MacBook Pro M-series, 3 slides × 100K cells = 300K objects:

| Operation | Time |
|-----------|------|
| Bounding box query | ~45ms |
| KNN query (k=50) | ~38ms |
| Density grid (256px cells) | ~120ms |
| Bulk ingest via COPY | 14,000 objects/sec |
| Same query after cache hit | <1ms |

---

## What I'd do differently at production scale

The partitioning strategy works well for slide-isolated queries but cross-slide analytics that need to touch many partitions simultaneously would benefit from a different sharding key. At true production scale (thousands of slides), you'd want to think about whether `slide_id` hash partitioning still makes sense or whether time-based or study-based partitioning gives better query isolation.

The Redis caching layer is also very basic right now — fixed TTLs, no cache invalidation strategy beyond expiry. A real system would need cache invalidation when slide data is updated.

---

## Project layout

```
SpatialPathDB/
├── backend/                   # Spring Boot REST API (Java 17)
│   └── src/main/java/com/spatialpathdb/
│       ├── controller/        # REST endpoints
│       ├── service/           # Business logic, caching
│       ├── repository/        # JPA repositories, stored function calls
│       └── model/             # JPA entities, request/response DTOs
├── database/
│   └── migrations/
│       ├── V1__create_schema.sql     # Tables + hash partitions
│       ├── V2__create_indexes.sql    # GIST spatial indexes
│       └── V3__create_functions.sql  # 7 stored spatial functions
├── frontend/                  # React + TypeScript
│   └── src/
│       ├── components/        # SpatialViewer, QueryPanel, SlideList
│       ├── hooks/             # TanStack Query data fetching
│       └── api/               # Axios client
├── spatial-engine/            # Python workers
│   └── src/
│       ├── workers/           # Celery task definitions
│       ├── spatial_analysis/  # Ripley's K, density estimation, KNN
│       └── data_ingestion/    # COPY loader, synthetic data generator
├── spark-jobs/                # PySpark cohort analytics
├── benchmarks/                # Query performance benchmarking
├── scripts/                   # CLI tools
└── docker-compose.yml
```
