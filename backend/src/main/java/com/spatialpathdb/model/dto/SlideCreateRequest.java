package com.spatialpathdb.model.dto;

import jakarta.validation.constraints.*;
import lombok.*;

import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SlideCreateRequest {

    @NotBlank(message = "Slide name is required")
    @Size(max = 255, message = "Slide name cannot exceed 255 characters")
    private String slideName;

    @NotBlank(message = "File path is required")
    private String filePath;

    @NotNull(message = "Width is required")
    @Positive(message = "Width must be positive")
    private Long widthPixels;

    @NotNull(message = "Height is required")
    @Positive(message = "Height must be positive")
    private Long heightPixels;

    @Positive(message = "Microns per pixel must be positive")
    private Double micronsPerPixel;

    @Size(max = 50, message = "Stain type cannot exceed 50 characters")
    private String stainType;

    @Size(max = 100, message = "Organ cannot exceed 100 characters")
    private String organ;

    private Map<String, Object> metadata;
}
