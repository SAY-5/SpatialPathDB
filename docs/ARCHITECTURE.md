# SpatialPathDB Architecture

## System Overview

SpatialPathDB is a distributed platform for spatial pathology data management. The architecture is designed for:

- **High-throughput ingestion**: Bulk loading millions of spatial objects
- **Low-latency queries**: Sub-100ms spatial queries with indexes
- **Horizontal scalability**: Partitioned storage and distributed processing
- **Async processing**: Background jobs for compute-intensive analysis

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Load Balancer                                   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐               ┌───────────────┐
            │   Frontend    │               │   REST API    │
            │   (React)     │               │ (Spring Boot) │
            │   Port 3000   │               │   Port 8080   │
            └───────┬───────┘               └───────┬───────┘
                    │                               │
                    │                               ▼
                    │                       ┌───────────────┐
                    │                       │    Redis      │
                    │                       │    Cache      │
                    │                       │   Port 6379   │
                    │                       └───────┬───────┘
                    │                               │
                    ▼                               ▼
            ┌───────────────────────────────────────────────┐
            │              PostgreSQL + PostGIS              │
            │                   Port 5432                    │
            │  ┌─────────────────────────────────────────┐  │
            │  │         spatial_objects (partitioned)    │  │
            │  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │  │
            │  │  │ p0  │ │ p1  │ │ ... │ │ p7  │       │  │
            │  │  └─────┘ └─────┘ └─────┘ └─────┘       │  │
            │  └─────────────────────────────────────────┘  │
            └───────────────────────────────────────────────┘
                                    ▲
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
    ┌───────┴───────┐       ┌───────┴───────┐       ┌───────┴───────┐
    │ Celery Worker │       │ Celery Worker │       │    PySpark    │
    │      #1       │       │      #2       │       │     Jobs      │
    └───────────────┘       └───────────────┘       └───────────────┘
```

## Data Flow

### Ingestion Pipeline

```
GeoJSON/Synthetic  ──▶  Python Parser  ──▶  PostgreSQL COPY  ──▶  Partitioned Table
     Data                                        (bulk)
```

1. **Data Source**: GeoJSON annotations or synthetic generator
2. **Parser**: Validates geometries, computes features (area, centroid)
3. **Bulk Load**: PostgreSQL COPY command for 100x faster inserts
4. **Storage**: Hash-partitioned by slide_id across 8 partitions

### Query Pipeline

```
Client Request  ──▶  REST API  ──▶  Cache Check  ──▶  PostGIS Query  ──▶  Response
                                       │                   │
                                       ▼                   ▼
                                   Cache Hit           Cache Miss
                                   (5min TTL)         (index scan)
```

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐         ┌─────────────────────────┐
│     slides      │         │    spatial_objects      │
├─────────────────┤         ├─────────────────────────┤
│ id (PK, UUID)   │◀────────│ slide_id (PK, FK)       │
│ slide_name      │         │ id (PK, BIGSERIAL)      │
│ file_path       │         │ object_type             │
│ width_pixels    │         │ label                   │
│ height_pixels   │         │ confidence              │
│ microns_per_px  │         │ geometry (GEOMETRY)     │
│ stain_type      │         │ centroid (POINT)        │
│ organ           │         │ area_pixels             │
│ metadata (JSONB)│         │ properties (JSONB)      │
└─────────────────┘         └─────────────────────────┘
        │                              │
        │                              │
        ▼                              │
┌─────────────────┐                    │
│  analysis_jobs  │                    │
├─────────────────┤                    │
│ id (PK, UUID)   │                    │
│ slide_id (FK)   │────────────────────┘
│ job_type        │
│ status          │
│ parameters      │
│ result_summary  │
└─────────────────┘
```

### Partitioning Strategy

The `spatial_objects` table is hash-partitioned by `slide_id`:

```sql
CREATE TABLE spatial_objects (
    id BIGSERIAL,
    slide_id UUID NOT NULL,
    ...
    PRIMARY KEY (id, slide_id)
) PARTITION BY HASH (slide_id);

-- 8 partitions for parallel execution
CREATE TABLE spatial_objects_p0 PARTITION OF spatial_objects
    FOR VALUES WITH (MODULUS 8, REMAINDER 0);
-- ... p1 through p7
```

Benefits:
- Queries filtered by slide_id only scan relevant partitions
- Parallel query execution across partitions
- Easier maintenance (per-partition vacuum, reindex)

### Indexing Strategy

```sql
-- GIST spatial index (critical for bbox queries)
CREATE INDEX idx_geometry ON spatial_objects USING GIST (geometry);

-- GIST index on centroids (KNN queries)
CREATE INDEX idx_centroid ON spatial_objects USING GIST (centroid);

-- B-tree for filtering
CREATE INDEX idx_slide_type ON spatial_objects (slide_id, object_type);

-- Partial index for high-confidence objects
CREATE INDEX idx_high_conf ON spatial_objects (slide_id, label)
    WHERE confidence > 0.8;
```

## Caching Strategy

Redis caches are used at multiple levels:

| Cache | TTL | Key Pattern | Content |
|-------|-----|-------------|---------|
| Spatial queries | 5 min | `bbox:{slide}:{coords}` | Query results |
| Slide metadata | 30 min | `slide:{id}` | Slide details |
| Statistics | 15 min | `stats:{slide}` | Aggregated stats |

Cache invalidation occurs on:
- Slide deletion
- New spatial objects inserted
- Manual cache clear

## Job Processing Architecture

### Celery Task Flow

```
API Request  ──▶  Create Job Record  ──▶  Push to Redis Queue
                       │
                       ▼
                  Job Status: QUEUED
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
 Worker 1          Worker 2          Worker N
    │                  │                  │
    ▼                  ▼                  ▼
 Process           Process           Process
    │                  │                  │
    └──────────────────┼──────────────────┘
                       ▼
              Update Job Record
              (COMPLETED/FAILED)
```

### Job Types

1. **cell_detection**: Generate synthetic cell annotations
2. **tissue_segmentation**: Create tissue region boundaries
3. **spatial_statistics**: Compute distribution metrics

## API Design

### REST Principles

- Resource-oriented URLs (`/slides`, `/spatial/bbox`)
- Standard HTTP methods (GET, POST, DELETE)
- JSON request/response bodies
- Pagination via query parameters (`?page=0&size=20`)

### Error Handling

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "status": 404,
  "error": "Not Found",
  "message": "Slide not found: abc123"
}
```

## Scalability Considerations

### Vertical Scaling
- Increase PostgreSQL `shared_buffers` and `work_mem`
- Add more Celery worker processes
- Increase Redis memory

### Horizontal Scaling
- Add read replicas for query load distribution
- Shard by slide_id ranges across multiple databases
- Deploy multiple API instances behind load balancer
- Scale Celery workers independently

### Performance Tuning

| Parameter | Recommended | Impact |
|-----------|-------------|--------|
| `shared_buffers` | 25% of RAM | Query caching |
| `work_mem` | 256MB | Sort/hash operations |
| `effective_cache_size` | 75% of RAM | Planner estimates |
| `max_parallel_workers_per_gather` | 4 | Parallel queries |
