package com.spatialpathdb.model.dto;

import lombok.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class DensityGridResponse {

    private Integer gridX;
    private Integer gridY;
    private Long cellCount;
    private Double avgConfidence;
    private String dominantLabel;

    // Computed coordinates for visualization
    private Double centerX;
    private Double centerY;
}
