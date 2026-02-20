package com.spatialpathdb.model.dto;

import jakarta.validation.constraints.*;
import lombok.*;

import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BBoxQueryRequest {

    @NotNull(message = "Slide ID is required")
    private UUID slideId;

    @NotNull(message = "minX is required")
    private Double minX;

    @NotNull(message = "minY is required")
    private Double minY;

    @NotNull(message = "maxX is required")
    private Double maxX;

    @NotNull(message = "maxY is required")
    private Double maxY;

    private String objectType;

    @DecimalMin(value = "0.0", message = "minConfidence must be >= 0")
    @DecimalMax(value = "1.0", message = "minConfidence must be <= 1")
    @Builder.Default
    private Double minConfidence = 0.0;

    @Min(value = 1, message = "limit must be at least 1")
    @Max(value = 10000, message = "limit cannot exceed 10000")
    @Builder.Default
    private Integer limit = 1000;

    @Min(value = 0, message = "offset cannot be negative")
    @Builder.Default
    private Integer offset = 0;

    public String toCacheKey() {
        return String.format("bbox:%s:%.0f:%.0f:%.0f:%.0f:%s:%.2f:%d:%d",
            slideId, minX, minY, maxX, maxY,
            objectType != null ? objectType : "all",
            minConfidence, limit, offset);
    }
}
