package com.spatialpathdb.model.dto;

import lombok.*;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SpatialObjectResponse {

    private Long id;
    private String objectType;
    private String label;
    private Double confidence;
    private String geometryWkt;
    private Double centroidX;
    private Double centroidY;
    private Double areaPixels;
    private Double perimeterPixels;
    private Map<String, Object> properties;
    private Double distance;  // Used for KNN results
}
