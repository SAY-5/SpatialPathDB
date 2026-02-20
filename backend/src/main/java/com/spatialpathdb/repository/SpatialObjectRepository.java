package com.spatialpathdb.repository;

import com.spatialpathdb.model.entity.SpatialObject;
import com.spatialpathdb.model.entity.SpatialObjectId;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface SpatialObjectRepository extends JpaRepository<SpatialObject, SpatialObjectId> {

    Page<SpatialObject> findBySlideId(UUID slideId, Pageable pageable);

    Page<SpatialObject> findBySlideIdAndObjectType(UUID slideId, String objectType, Pageable pageable);

    @Query(value = "SELECT COUNT(*) FROM spatial_objects WHERE slide_id = :slideId", nativeQuery = true)
    long countBySlideId(@Param("slideId") UUID slideId);

    // Bounding box query using stored function
    @Query(value = """
        SELECT * FROM query_spatial_objects_bbox(
            :slideId, :minX, :minY, :maxX, :maxY,
            :objectType, :minConfidence, :limit, :offset
        )
        """, nativeQuery = true)
    List<Object[]> findByBoundingBox(
        @Param("slideId") UUID slideId,
        @Param("minX") double minX,
        @Param("minY") double minY,
        @Param("maxX") double maxX,
        @Param("maxY") double maxY,
        @Param("objectType") String objectType,
        @Param("minConfidence") double minConfidence,
        @Param("limit") int limit,
        @Param("offset") int offset
    );

    // KNN query using stored function
    @Query(value = """
        SELECT * FROM query_knn(:slideId, :x, :y, :k, :objectType)
        """, nativeQuery = true)
    List<Object[]> findKNearestNeighbors(
        @Param("slideId") UUID slideId,
        @Param("x") double x,
        @Param("y") double y,
        @Param("k") int k,
        @Param("objectType") String objectType
    );

    // Density grid using stored function
    @Query(value = """
        SELECT * FROM compute_density_grid(:slideId, :gridSize, :objectType)
        """, nativeQuery = true)
    List<Object[]> computeDensityGrid(
        @Param("slideId") UUID slideId,
        @Param("gridSize") double gridSize,
        @Param("objectType") String objectType
    );

    // Slide statistics using stored function
    @Query(value = """
        SELECT * FROM compute_slide_statistics(:slideId)
        """, nativeQuery = true)
    List<Object[]> computeSlideStatistics(@Param("slideId") UUID slideId);

    // Point in polygon query
    @Query(value = """
        SELECT * FROM query_objects_within_polygon(:slideId, :polygonWkt, :objectType, :limit)
        """, nativeQuery = true)
    List<Object[]> findObjectsWithinPolygon(
        @Param("slideId") UUID slideId,
        @Param("polygonWkt") String polygonWkt,
        @Param("objectType") String objectType,
        @Param("limit") int limit
    );

    // Count objects in bounding box (lightweight)
    @Query(value = """
        SELECT count_objects_in_bbox(:slideId, :minX, :minY, :maxX, :maxY, :objectType)
        """, nativeQuery = true)
    long countInBoundingBox(
        @Param("slideId") UUID slideId,
        @Param("minX") double minX,
        @Param("minY") double minY,
        @Param("maxX") double maxX,
        @Param("maxY") double maxY,
        @Param("objectType") String objectType
    );

    // Get slide bounds
    @Query(value = """
        SELECT * FROM get_slide_bounds(:slideId)
        """, nativeQuery = true)
    List<Object[]> getSlideBounds(@Param("slideId") UUID slideId);

    // Delete all objects for a slide
    void deleteBySlideId(UUID slideId);

    // Get distinct labels for a slide
    @Query(value = """
        SELECT DISTINCT label FROM spatial_objects
        WHERE slide_id = :slideId AND label IS NOT NULL
        ORDER BY label
        """, nativeQuery = true)
    List<String> findDistinctLabelsBySlideId(@Param("slideId") UUID slideId);

    // Get distinct object types for a slide
    @Query(value = """
        SELECT DISTINCT object_type FROM spatial_objects
        WHERE slide_id = :slideId
        ORDER BY object_type
        """, nativeQuery = true)
    List<String> findDistinctObjectTypesBySlideId(@Param("slideId") UUID slideId);
}
