package com.spatialpathdb.controller;

import com.spatialpathdb.model.dto.SlideCreateRequest;
import com.spatialpathdb.model.dto.SlideResponse;
import com.spatialpathdb.model.dto.SlideStatisticsResponse;
import com.spatialpathdb.service.SlideService;
import com.spatialpathdb.service.SpatialQueryService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/v1/slides")
@RequiredArgsConstructor
@Tag(name = "Slides", description = "Slide metadata management")
public class SlideController {

    private final SlideService slideService;
    private final SpatialQueryService spatialQueryService;

    @PostMapping
    @Operation(summary = "Upload slide metadata", description = "Register a new slide with its metadata")
    public ResponseEntity<SlideResponse> createSlide(
            @Valid @RequestBody SlideCreateRequest request) {
        SlideResponse response = slideService.createSlide(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping
    @Operation(summary = "List all slides", description = "Get paginated list of all slides")
    public ResponseEntity<Page<SlideResponse>> getAllSlides(
            @Parameter(description = "Filter by organ")
            @RequestParam(required = false) String organ,
            @Parameter(description = "Filter by stain type")
            @RequestParam(required = false) String stainType,
            @PageableDefault(size = 20) Pageable pageable) {

        Page<SlideResponse> slides;

        if (organ != null) {
            slides = slideService.getSlidesByOrgan(organ, pageable);
        } else if (stainType != null) {
            slides = slideService.getSlidesByStainType(stainType, pageable);
        } else {
            slides = slideService.getAllSlides(pageable);
        }

        return ResponseEntity.ok(slides);
    }

    @GetMapping("/{slideId}")
    @Operation(summary = "Get slide details", description = "Get detailed information about a specific slide")
    public ResponseEntity<SlideResponse> getSlide(
            @PathVariable UUID slideId) {
        return ResponseEntity.ok(slideService.getSlide(slideId));
    }

    @PutMapping("/{slideId}")
    @Operation(summary = "Update slide", description = "Update slide metadata")
    public ResponseEntity<SlideResponse> updateSlide(
            @PathVariable UUID slideId,
            @Valid @RequestBody SlideCreateRequest request) {
        return ResponseEntity.ok(slideService.updateSlide(slideId, request));
    }

    @DeleteMapping("/{slideId}")
    @Operation(summary = "Delete slide", description = "Delete a slide and all associated data")
    public ResponseEntity<Void> deleteSlide(@PathVariable UUID slideId) {
        slideService.deleteSlide(slideId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/{slideId}/statistics")
    @Operation(summary = "Get slide statistics", description = "Get spatial statistics summary for a slide")
    public ResponseEntity<SlideStatisticsResponse> getSlideStatistics(
            @PathVariable UUID slideId) {
        return ResponseEntity.ok(spatialQueryService.computeSlideStatistics(slideId));
    }

    @GetMapping("/{slideId}/bounds")
    @Operation(summary = "Get slide bounds", description = "Get the bounding box of all spatial objects in a slide")
    public ResponseEntity<Map<String, Object>> getSlideBounds(
            @PathVariable UUID slideId) {
        return ResponseEntity.ok(spatialQueryService.getSlideBounds(slideId));
    }
}
