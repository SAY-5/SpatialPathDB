-- SpatialPathDB Schema
-- Spatial data management for digital pathology

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Slide metadata
CREATE TABLE slides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slide_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    width_pixels BIGINT NOT NULL,
    height_pixels BIGINT NOT NULL,
    microns_per_pixel DOUBLE PRECISION,
    stain_type VARCHAR(50),
    organ VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT chk_slide_dimensions CHECK (width_pixels > 0 AND height_pixels > 0)
);

-- Spatial annotations (cells, regions, tissue boundaries)
-- Partitioned by hash on slide_id for scalable query performance
CREATE TABLE spatial_objects (
    id BIGSERIAL,
    slide_id UUID NOT NULL REFERENCES slides(id) ON DELETE CASCADE,
    object_type VARCHAR(50) NOT NULL,
    label VARCHAR(100),
    confidence DOUBLE PRECISION,
    geometry GEOMETRY(Geometry, 0) NOT NULL,
    centroid GEOMETRY(Point, 0),
    area_pixels DOUBLE PRECISION,
    perimeter_pixels DOUBLE PRECISION,
    properties JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, slide_id),
    CONSTRAINT chk_confidence_range CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
) PARTITION BY HASH (slide_id);

-- Create 8 hash partitions for parallel query execution
CREATE TABLE spatial_objects_p0 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE spatial_objects_p1 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE spatial_objects_p2 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE spatial_objects_p3 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE spatial_objects_p4 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE spatial_objects_p5 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE spatial_objects_p6 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE spatial_objects_p7 PARTITION OF spatial_objects FOR VALUES WITH (MODULUS 8, REMAINDER 7);

-- Analysis jobs for async processing
CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slide_id UUID NOT NULL REFERENCES slides(id) ON DELETE CASCADE,
    job_type VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'QUEUED',
    parameters JSONB,
    result_summary JSONB,
    submitted_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    CONSTRAINT chk_job_status CHECK (status IN ('QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'))
);

-- Query execution logs for performance monitoring
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    query_type VARCHAR(100),
    slide_id UUID,
    bbox_wkt TEXT,
    num_results INT,
    execution_time_ms DOUBLE PRECISION,
    used_index BOOLEAN,
    executed_at TIMESTAMP DEFAULT NOW()
);

-- Benchmark results storage
CREATE TABLE benchmark_results (
    id BIGSERIAL PRIMARY KEY,
    benchmark_name VARCHAR(100) NOT NULL,
    parameters JSONB,
    results JSONB,
    mean_latency_ms DOUBLE PRECISION,
    p95_latency_ms DOUBLE PRECISION,
    p99_latency_ms DOUBLE PRECISION,
    throughput_qps DOUBLE PRECISION,
    executed_at TIMESTAMP DEFAULT NOW()
);
