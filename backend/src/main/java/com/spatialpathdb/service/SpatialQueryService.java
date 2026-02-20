package com.spatialpathdb.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.spatialpathdb.model.dto.*;
import com.spatialpathdb.model.entity.QueryLog;
import com.spatialpathdb.repository.QueryLogRepository;
import com.spatialpathdb.repository.SpatialObjectRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class SpatialQueryService {

    private final SpatialObjectRepository spatialObjectRepository;
    private final QueryLogRepository queryLogRepository;
    private final ObjectMapper objectMapper;

    @Cacheable(value = "spatialQueries", key = "#request.toCacheKey()")
    public List<SpatialObjectResponse> executeBBoxQuery(BBoxQueryRequest request) {
        long startTime = System.nanoTime();

        List<Object[]> results = spatialObjectRepository.findByBoundingBox(
            request.getSlideId(),
            request.getMinX(),
            request.getMinY(),
            request.getMaxX(),
            request.getMaxY(),
            request.getObjectType(),
            request.getMinConfidence(),
            request.getLimit(),
            request.getOffset()
        );

        List<SpatialObjectResponse> response = results.stream()
            .map(this::mapToSpatialObjectResponse)
            .collect(Collectors.toList());

        long elapsed = (System.nanoTime() - startTime) / 1_000_000;

        logQueryAsync("bbox", request.getSlideId(), elapsed, response.size(), buildBboxWkt(request));

        return response;
    }

    @Cacheable(value = "spatialQueries", key = "#request.toCacheKey()")
    public List<SpatialObjectResponse> executeKnnQuery(KNNQueryRequest request) {
        long startTime = System.nanoTime();

        List<Object[]> results = spatialObjectRepository.findKNearestNeighbors(
            request.getSlideId(),
            request.getX(),
            request.getY(),
            request.getK(),
            request.getObjectType()
        );

        List<SpatialObjectResponse> response = results.stream()
            .map(this::mapToKnnResponse)
            .collect(Collectors.toList());

        long elapsed = (System.nanoTime() - startTime) / 1_000_000;

        logQueryAsync("knn", request.getSlideId(), elapsed, response.size(), null);

        return response;
    }

    public List<DensityGridResponse> computeDensityGrid(UUID slideId, double gridSize, String objectType) {
        long startTime = System.nanoTime();

        List<Object[]> results = spatialObjectRepository.computeDensityGrid(
            slideId, gridSize, objectType
        );

        List<DensityGridResponse> response = results.stream()
            .map(row -> DensityGridResponse.builder()
                .gridX(((Number) row[0]).intValue())
                .gridY(((Number) row[1]).intValue())
                .cellCount(((Number) row[2]).longValue())
                .avgConfidence(row[3] != null ? ((Number) row[3]).doubleValue() : null)
                .dominantLabel((String) row[4])
                .centerX((((Number) row[0]).intValue() + 0.5) * gridSize)
                .centerY((((Number) row[1]).intValue() + 0.5) * gridSize)
                .build())
            .collect(Collectors.toList());

        long elapsed = (System.nanoTime() - startTime) / 1_000_000;
        logQueryAsync("density", slideId, elapsed, response.size(), null);

        return response;
    }

    public SlideStatisticsResponse computeSlideStatistics(UUID slideId) {
        long startTime = System.nanoTime();

        List<Object[]> results = spatialObjectRepository.computeSlideStatistics(slideId);

        List<SlideStatisticsResponse.TypeStatistics> stats = results.stream()
            .map(row -> SlideStatisticsResponse.TypeStatistics.builder()
                .objectType((String) row[0])
                .label((String) row[1])
                .totalCount(((Number) row[2]).longValue())
                .avgArea(row[3] != null ? ((Number) row[3]).doubleValue() : null)
                .stdArea(row[4] != null ? ((Number) row[4]).doubleValue() : null)
                .avgConfidence(row[5] != null ? ((Number) row[5]).doubleValue() : null)
                .spatialExtent((String) row[6])
                .build())
            .collect(Collectors.toList());

        long totalObjects = stats.stream()
            .mapToLong(SlideStatisticsResponse.TypeStatistics::getTotalCount)
            .sum();

        long elapsed = (System.nanoTime() - startTime) / 1_000_000;
        logQueryAsync("statistics", slideId, elapsed, stats.size(), null);

        return SlideStatisticsResponse.builder()
            .slideId(slideId.toString())
            .totalObjects(totalObjects)
            .byType(stats)
            .build();
    }

    public List<SpatialObjectResponse> executeWithinQuery(
            UUID slideId, String polygonWkt, String objectType, int limit) {
        long startTime = System.nanoTime();

        List<Object[]> results = spatialObjectRepository.findObjectsWithinPolygon(
            slideId, polygonWkt, objectType, limit
        );

        List<SpatialObjectResponse> response = results.stream()
            .map(row -> SpatialObjectResponse.builder()
                .id(((Number) row[0]).longValue())
                .objectType((String) row[1])
                .label((String) row[2])
                .confidence(row[3] != null ? ((Number) row[3]).doubleValue() : null)
                .centroidX(row[4] != null ? ((Number) row[4]).doubleValue() : null)
                .centroidY(row[5] != null ? ((Number) row[5]).doubleValue() : null)
                .areaPixels(row[6] != null ? ((Number) row[6]).doubleValue() : null)
                .build())
            .collect(Collectors.toList());

        long elapsed = (System.nanoTime() - startTime) / 1_000_000;
        logQueryAsync("within", slideId, elapsed, response.size(), polygonWkt);

        return response;
    }

    public long countInBoundingBox(BBoxQueryRequest request) {
        return spatialObjectRepository.countInBoundingBox(
            request.getSlideId(),
            request.getMinX(),
            request.getMinY(),
            request.getMaxX(),
            request.getMaxY(),
            request.getObjectType()
        );
    }

    public Map<String, Object> getSlideBounds(UUID slideId) {
        List<Object[]> results = spatialObjectRepository.getSlideBounds(slideId);

        if (results.isEmpty() || results.get(0)[0] == null) {
            return Map.of(
                "minX", 0,
                "minY", 0,
                "maxX", 0,
                "maxY", 0,
                "totalObjects", 0
            );
        }

        Object[] row = results.get(0);
        return Map.of(
            "minX", ((Number) row[0]).doubleValue(),
            "minY", ((Number) row[1]).doubleValue(),
            "maxX", ((Number) row[2]).doubleValue(),
            "maxY", ((Number) row[3]).doubleValue(),
            "totalObjects", ((Number) row[4]).longValue()
        );
    }

    private SpatialObjectResponse mapToSpatialObjectResponse(Object[] row) {
        Map<String, Object> properties = null;
        if (row[8] != null) {
            try {
                properties = objectMapper.readValue(
                    row[8].toString(),
                    new TypeReference<Map<String, Object>>() {}
                );
            } catch (Exception e) {
                log.warn("Failed to parse properties JSON", e);
            }
        }

        return SpatialObjectResponse.builder()
            .id(((Number) row[0]).longValue())
            .objectType((String) row[1])
            .label((String) row[2])
            .confidence(row[3] != null ? ((Number) row[3]).doubleValue() : null)
            .geometryWkt((String) row[4])
            .centroidX(row[5] != null ? ((Number) row[5]).doubleValue() : null)
            .centroidY(row[6] != null ? ((Number) row[6]).doubleValue() : null)
            .areaPixels(row[7] != null ? ((Number) row[7]).doubleValue() : null)
            .properties(properties)
            .build();
    }

    private SpatialObjectResponse mapToKnnResponse(Object[] row) {
        return SpatialObjectResponse.builder()
            .id(((Number) row[0]).longValue())
            .label((String) row[1])
            .confidence(row[2] != null ? ((Number) row[2]).doubleValue() : null)
            .centroidX(row[3] != null ? ((Number) row[3]).doubleValue() : null)
            .centroidY(row[4] != null ? ((Number) row[4]).doubleValue() : null)
            .distance(row[5] != null ? ((Number) row[5]).doubleValue() : null)
            .build();
    }

    private String buildBboxWkt(BBoxQueryRequest request) {
        return String.format("POLYGON((%f %f, %f %f, %f %f, %f %f, %f %f))",
            request.getMinX(), request.getMinY(),
            request.getMaxX(), request.getMinY(),
            request.getMaxX(), request.getMaxY(),
            request.getMinX(), request.getMaxY(),
            request.getMinX(), request.getMinY()
        );
    }

    @Async
    protected void logQueryAsync(String queryType, UUID slideId, long elapsedMs, int numResults, String bboxWkt) {
        try {
            QueryLog log = QueryLog.builder()
                .queryType(queryType)
                .slideId(slideId)
                .executionTimeMs((double) elapsedMs)
                .numResults(numResults)
                .bboxWkt(bboxWkt)
                .usedIndex(true)
                .build();

            queryLogRepository.save(log);
        } catch (Exception e) {
            log.warn("Failed to log query", e);
        }
    }
}
