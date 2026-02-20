package com.spatialpathdb.model.dto;

import com.spatialpathdb.model.entity.Slide;
import lombok.*;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SlideResponse {

    private UUID id;
    private String slideName;
    private String filePath;
    private Long widthPixels;
    private Long heightPixels;
    private Double micronsPerPixel;
    private String stainType;
    private String organ;
    private LocalDateTime uploadedAt;
    private Map<String, Object> metadata;
    private Long objectCount;  // Optional, populated when requested

    public static SlideResponse fromEntity(Slide slide) {
        return SlideResponse.builder()
            .id(slide.getId())
            .slideName(slide.getSlideName())
            .filePath(slide.getFilePath())
            .widthPixels(slide.getWidthPixels())
            .heightPixels(slide.getHeightPixels())
            .micronsPerPixel(slide.getMicronsPerPixel())
            .stainType(slide.getStainType())
            .organ(slide.getOrgan())
            .uploadedAt(slide.getUploadedAt())
            .metadata(slide.getMetadata())
            .build();
    }
}
