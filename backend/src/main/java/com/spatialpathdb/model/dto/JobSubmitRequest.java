package com.spatialpathdb.model.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.*;

import java.util.Map;
import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class JobSubmitRequest {

    @NotNull(message = "Slide ID is required")
    private UUID slideId;

    @NotBlank(message = "Job type is required")
    private String jobType;

    private Map<String, Object> parameters;
}
