package com.spatialpathdb.model.dto;

import jakarta.validation.constraints.*;
import lombok.*;

import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class KNNQueryRequest {

    @NotNull(message = "Slide ID is required")
    private UUID slideId;

    @NotNull(message = "x coordinate is required")
    private Double x;

    @NotNull(message = "y coordinate is required")
    private Double y;

    @Min(value = 1, message = "k must be at least 1")
    @Max(value = 1000, message = "k cannot exceed 1000")
    @Builder.Default
    private Integer k = 10;

    private String objectType;

    public String toCacheKey() {
        return String.format("knn:%s:%.0f:%.0f:%d:%s",
            slideId, x, y, k,
            objectType != null ? objectType : "all");
    }
}
