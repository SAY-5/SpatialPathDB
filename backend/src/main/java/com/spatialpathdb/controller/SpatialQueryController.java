package com.spatialpathdb.controller;

import com.spatialpathdb.model.dto.*;
import com.spatialpathdb.service.SpatialQueryService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/spatial")
@RequiredArgsConstructor
@Tag(name = "Spatial Queries", description = "High-performance spatial query endpoints")
public class SpatialQueryController {

    private final SpatialQueryService spatialQueryService;

    @PostMapping("/bbox")
    @Operation(
        summary = "Bounding box query",
        description = "Find all spatial objects within a bounding box. Uses spatial index for optimal performance."
    )
    public ResponseEntity<List<SpatialObjectResponse>> bboxQuery(
            @Valid @RequestBody BBoxQueryRequest request) {
        List<SpatialObjectResponse> results = spatialQueryService.executeBBoxQuery(request);
        return ResponseEntity.ok(results);
    }

    @PostMapping("/knn")
    @Operation(
        summary = "K-nearest neighbors query",
        description = "Find the K nearest spatial objects to a given point."
    )
    public ResponseEntity<List<SpatialObjectResponse>> knnQuery(
            @Valid @RequestBody KNNQueryRequest request) {
        List<SpatialObjectResponse> results = spatialQueryService.executeKnnQuery(request);
        return ResponseEntity.ok(results);
    }

    @PostMapping("/density")
    @Operation(
        summary = "Density grid computation",
        description = "Compute density heatmap by aggregating objects into grid cells."
    )
    public ResponseEntity<List<DensityGridResponse>> densityGrid(
            @Parameter(description = "Slide ID", required = true)
            @RequestParam UUID slideId,
            @Parameter(description = "Grid cell size in pixels")
            @RequestParam(defaultValue = "256") double gridSize,
            @Parameter(description = "Filter by object type")
            @RequestParam(required = false) String objectType) {

        List<DensityGridResponse> results = spatialQueryService.computeDensityGrid(
            slideId, gridSize, objectType
        );
        return ResponseEntity.ok(results);
    }

    @PostMapping("/within")
    @Operation(
        summary = "Point-in-polygon query",
        description = "Find all objects whose centroids are within a polygon."
    )
    public ResponseEntity<List<SpatialObjectResponse>> withinQuery(
            @Parameter(description = "Slide ID", required = true)
            @RequestParam UUID slideId,
            @Parameter(description = "Polygon in WKT format", required = true)
            @RequestParam String polygonWkt,
            @Parameter(description = "Filter by object type")
            @RequestParam(required = false) String objectType,
            @Parameter(description = "Maximum results to return")
            @RequestParam(defaultValue = "10000") int limit) {

        List<SpatialObjectResponse> results = spatialQueryService.executeWithinQuery(
            slideId, polygonWkt, objectType, limit
        );
        return ResponseEntity.ok(results);
    }

    @PostMapping("/count")
    @Operation(
        summary = "Count objects in bounding box",
        description = "Lightweight query to count objects without returning data."
    )
    public ResponseEntity<Map<String, Long>> countInBbox(
            @Valid @RequestBody BBoxQueryRequest request) {
        long count = spatialQueryService.countInBoundingBox(request);
        return ResponseEntity.ok(Map.of("count", count));
    }
}
