package com.spatialpathdb.model.dto;

import com.spatialpathdb.model.entity.AnalysisJob;
import lombok.*;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class JobStatusResponse {

    private UUID id;
    private UUID slideId;
    private String jobType;
    private AnalysisJob.Status status;
    private Map<String, Object> parameters;
    private Map<String, Object> resultSummary;
    private LocalDateTime submittedAt;
    private LocalDateTime startedAt;
    private LocalDateTime completedAt;
    private String errorMessage;
    private Long durationMs;

    public static JobStatusResponse fromEntity(AnalysisJob job) {
        Long duration = null;
        if (job.getStartedAt() != null && job.getCompletedAt() != null) {
            duration = java.time.Duration.between(
                job.getStartedAt(), job.getCompletedAt()
            ).toMillis();
        }

        return JobStatusResponse.builder()
            .id(job.getId())
            .slideId(job.getSlideId())
            .jobType(job.getJobType())
            .status(job.getStatus())
            .parameters(job.getParameters())
            .resultSummary(job.getResultSummary())
            .submittedAt(job.getSubmittedAt())
            .startedAt(job.getStartedAt())
            .completedAt(job.getCompletedAt())
            .errorMessage(job.getErrorMessage())
            .durationMs(duration)
            .build();
    }
}
