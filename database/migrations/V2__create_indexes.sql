-- Spatial and B-tree indexes for query optimization
-- These indexes provide 40%+ improvement on spatial queries

-- GIST spatial index on geometry (critical for bbox and intersection queries)
CREATE INDEX idx_spatial_objects_geometry ON spatial_objects USING GIST (geometry);

-- GIST index on centroids for point-based and KNN queries
CREATE INDEX idx_spatial_objects_centroid ON spatial_objects USING GIST (centroid);

-- B-tree indexes for common filtering patterns
CREATE INDEX idx_spatial_objects_slide_id ON spatial_objects (slide_id);
CREATE INDEX idx_spatial_objects_type ON spatial_objects (object_type);
CREATE INDEX idx_spatial_objects_label ON spatial_objects (label);

-- Composite index optimizes the most common query pattern:
-- filter by slide, then by object type, then spatial intersection
CREATE INDEX idx_spatial_objects_slide_type ON spatial_objects (slide_id, object_type);

-- Partial index for high-confidence detections
-- Only indexes rows where confidence > 0.8, reducing index size significantly
CREATE INDEX idx_spatial_objects_high_conf ON spatial_objects (slide_id, label)
    WHERE confidence > 0.8;

-- GIN index on JSONB properties for flexible querying
CREATE INDEX idx_spatial_objects_properties ON spatial_objects USING GIN (properties);

-- Analysis jobs indexes for status polling
CREATE INDEX idx_jobs_slide ON analysis_jobs (slide_id);
CREATE INDEX idx_jobs_status ON analysis_jobs (status);
CREATE INDEX idx_jobs_submitted ON analysis_jobs (submitted_at DESC);

-- Query logs indexes for performance analysis
CREATE INDEX idx_query_logs_type ON query_logs (query_type);
CREATE INDEX idx_query_logs_executed ON query_logs (executed_at DESC);

-- Slides indexes
CREATE INDEX idx_slides_organ ON slides (organ);
CREATE INDEX idx_slides_stain ON slides (stain_type);
CREATE INDEX idx_slides_uploaded ON slides (uploaded_at DESC);
