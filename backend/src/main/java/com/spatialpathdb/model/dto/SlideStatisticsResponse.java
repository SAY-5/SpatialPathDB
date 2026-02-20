package com.spatialpathdb.model.dto;

import lombok.*;

import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SlideStatisticsResponse {

    private String slideId;
    private Long totalObjects;
    private List<TypeStatistics> byType;

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    public static class TypeStatistics {
        private String objectType;
        private String label;
        private Long totalCount;
        private Double avgArea;
        private Double stdArea;
        private Double avgConfidence;
        private String spatialExtent;
    }
}
