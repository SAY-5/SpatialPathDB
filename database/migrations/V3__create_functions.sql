-- Optimized spatial query functions
-- These stored procedures leverage indexes and provide consistent interfaces

-- Bounding box range query with pagination
-- Uses && operator for index-accelerated intersection test
CREATE OR REPLACE FUNCTION query_spatial_objects_bbox(
    p_slide_id UUID,
    p_min_x DOUBLE PRECISION,
    p_min_y DOUBLE PRECISION,
    p_max_x DOUBLE PRECISION,
    p_max_y DOUBLE PRECISION,
    p_object_type VARCHAR DEFAULT NULL,
    p_min_confidence DOUBLE PRECISION DEFAULT 0.0,
    p_limit INT DEFAULT 1000,
    p_offset INT DEFAULT 0
)
RETURNS TABLE (
    id BIGINT,
    object_type VARCHAR,
    label VARCHAR,
    confidence DOUBLE PRECISION,
    geometry_wkt TEXT,
    centroid_x DOUBLE PRECISION,
    centroid_y DOUBLE PRECISION,
    area_pixels DOUBLE PRECISION,
    properties JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        so.id,
        so.object_type,
        so.label,
        so.confidence,
        ST_AsText(so.geometry) AS geometry_wkt,
        ST_X(so.centroid) AS centroid_x,
        ST_Y(so.centroid) AS centroid_y,
        so.area_pixels,
        so.properties
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
      AND so.geometry && ST_MakeEnvelope(p_min_x, p_min_y, p_max_x, p_max_y, 0)
      AND (p_object_type IS NULL OR so.object_type = p_object_type)
      AND (so.confidence IS NULL OR so.confidence >= p_min_confidence)
    ORDER BY so.id
    LIMIT p_limit OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

-- K-nearest neighbors query using <-> distance operator
-- PostGIS optimizes KNN queries with GIST index
CREATE OR REPLACE FUNCTION query_knn(
    p_slide_id UUID,
    p_x DOUBLE PRECISION,
    p_y DOUBLE PRECISION,
    p_k INT DEFAULT 10,
    p_object_type VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    label VARCHAR,
    confidence DOUBLE PRECISION,
    centroid_x DOUBLE PRECISION,
    centroid_y DOUBLE PRECISION,
    distance_pixels DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        so.id,
        so.label,
        so.confidence,
        ST_X(so.centroid) AS centroid_x,
        ST_Y(so.centroid) AS centroid_y,
        ST_Distance(so.centroid, ST_SetSRID(ST_MakePoint(p_x, p_y), 0)) AS distance_pixels
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
      AND so.centroid IS NOT NULL
      AND (p_object_type IS NULL OR so.object_type = p_object_type)
    ORDER BY so.centroid <-> ST_SetSRID(ST_MakePoint(p_x, p_y), 0)
    LIMIT p_k;
END;
$$ LANGUAGE plpgsql STABLE;

-- Spatial density heatmap using grid-based aggregation
-- Divides the slide into grid cells and counts objects per cell
CREATE OR REPLACE FUNCTION compute_density_grid(
    p_slide_id UUID,
    p_grid_size DOUBLE PRECISION DEFAULT 256.0,
    p_object_type VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    grid_x INT,
    grid_y INT,
    cell_count BIGINT,
    avg_confidence DOUBLE PRECISION,
    dominant_label VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        FLOOR(ST_X(so.centroid) / p_grid_size)::INT AS grid_x,
        FLOOR(ST_Y(so.centroid) / p_grid_size)::INT AS grid_y,
        COUNT(*)::BIGINT AS cell_count,
        AVG(so.confidence) AS avg_confidence,
        MODE() WITHIN GROUP (ORDER BY so.label) AS dominant_label
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
      AND so.centroid IS NOT NULL
      AND (p_object_type IS NULL OR so.object_type = p_object_type)
    GROUP BY FLOOR(ST_X(so.centroid) / p_grid_size)::INT,
             FLOOR(ST_Y(so.centroid) / p_grid_size)::INT
    ORDER BY cell_count DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Spatial statistics summary per slide
CREATE OR REPLACE FUNCTION compute_slide_statistics(p_slide_id UUID)
RETURNS TABLE (
    object_type VARCHAR,
    label VARCHAR,
    total_count BIGINT,
    avg_area DOUBLE PRECISION,
    std_area DOUBLE PRECISION,
    avg_confidence DOUBLE PRECISION,
    spatial_extent TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        so.object_type,
        so.label,
        COUNT(*)::BIGINT AS total_count,
        AVG(so.area_pixels) AS avg_area,
        STDDEV(so.area_pixels) AS std_area,
        AVG(so.confidence) AS avg_confidence,
        ST_AsText(ST_Extent(so.geometry)) AS spatial_extent
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
    GROUP BY so.object_type, so.label
    ORDER BY total_count DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- Point-in-polygon containment query
CREATE OR REPLACE FUNCTION query_objects_within_polygon(
    p_slide_id UUID,
    p_polygon_wkt TEXT,
    p_object_type VARCHAR DEFAULT NULL,
    p_limit INT DEFAULT 10000
)
RETURNS TABLE (
    id BIGINT,
    object_type VARCHAR,
    label VARCHAR,
    confidence DOUBLE PRECISION,
    centroid_x DOUBLE PRECISION,
    centroid_y DOUBLE PRECISION,
    area_pixels DOUBLE PRECISION
) AS $$
DECLARE
    query_geom GEOMETRY;
BEGIN
    query_geom := ST_GeomFromText(p_polygon_wkt, 0);

    RETURN QUERY
    SELECT
        so.id,
        so.object_type,
        so.label,
        so.confidence,
        ST_X(so.centroid) AS centroid_x,
        ST_Y(so.centroid) AS centroid_y,
        so.area_pixels
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
      AND ST_Within(so.centroid, query_geom)
      AND (p_object_type IS NULL OR so.object_type = p_object_type)
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql STABLE;

-- Count objects in bounding box (lightweight version for UI feedback)
CREATE OR REPLACE FUNCTION count_objects_in_bbox(
    p_slide_id UUID,
    p_min_x DOUBLE PRECISION,
    p_min_y DOUBLE PRECISION,
    p_max_x DOUBLE PRECISION,
    p_max_y DOUBLE PRECISION,
    p_object_type VARCHAR DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    obj_count BIGINT;
BEGIN
    SELECT COUNT(*) INTO obj_count
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id
      AND so.geometry && ST_MakeEnvelope(p_min_x, p_min_y, p_max_x, p_max_y, 0)
      AND (p_object_type IS NULL OR so.object_type = p_object_type);

    RETURN obj_count;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get slide bounding box (useful for initializing viewer)
CREATE OR REPLACE FUNCTION get_slide_bounds(p_slide_id UUID)
RETURNS TABLE (
    min_x DOUBLE PRECISION,
    min_y DOUBLE PRECISION,
    max_x DOUBLE PRECISION,
    max_y DOUBLE PRECISION,
    total_objects BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ST_XMin(ST_Extent(so.geometry)) AS min_x,
        ST_YMin(ST_Extent(so.geometry)) AS min_y,
        ST_XMax(ST_Extent(so.geometry)) AS max_x,
        ST_YMax(ST_Extent(so.geometry)) AS max_y,
        COUNT(*)::BIGINT AS total_objects
    FROM spatial_objects so
    WHERE so.slide_id = p_slide_id;
END;
$$ LANGUAGE plpgsql STABLE;
