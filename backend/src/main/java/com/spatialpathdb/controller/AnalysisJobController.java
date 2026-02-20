package com.spatialpathdb.controller;

import com.spatialpathdb.model.dto.JobStatusResponse;
import com.spatialpathdb.model.dto.JobSubmitRequest;
import com.spatialpathdb.model.entity.AnalysisJob;
import com.spatialpathdb.service.AnalysisJobService;
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
@RequestMapping("/api/v1/jobs")
@RequiredArgsConstructor
@Tag(name = "Analysis Jobs", description = "Asynchronous analysis job management")
public class AnalysisJobController {

    private final AnalysisJobService jobService;

    @PostMapping
    @Operation(
        summary = "Submit analysis job",
        description = "Submit a new analysis job for asynchronous processing. " +
            "Supported job types: cell_detection, tissue_segmentation, spatial_statistics"
    )
    public ResponseEntity<JobStatusResponse> submitJob(
            @Valid @RequestBody JobSubmitRequest request) {
        JobStatusResponse response = jobService.submitJob(request);
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(response);
    }

    @GetMapping("/{jobId}")
    @Operation(summary = "Get job status", description = "Get the current status of an analysis job")
    public ResponseEntity<JobStatusResponse> getJobStatus(
            @PathVariable UUID jobId) {
        return ResponseEntity.ok(jobService.getJobStatus(jobId));
    }

    @GetMapping
    @Operation(summary = "List all jobs", description = "Get paginated list of all analysis jobs")
    public ResponseEntity<Page<JobStatusResponse>> getAllJobs(
            @Parameter(description = "Filter by status")
            @RequestParam(required = false) AnalysisJob.Status status,
            @Parameter(description = "Filter by slide ID")
            @RequestParam(required = false) UUID slideId,
            @PageableDefault(size = 20, sort = "submittedAt") Pageable pageable) {

        Page<JobStatusResponse> jobs;

        if (slideId != null) {
            jobs = jobService.getJobsBySlide(slideId, pageable);
        } else if (status != null) {
            jobs = jobService.getJobsByStatus(status, pageable);
        } else {
            jobs = jobService.getAllJobs(pageable);
        }

        return ResponseEntity.ok(jobs);
    }

    @DeleteMapping("/{jobId}")
    @Operation(summary = "Cancel job", description = "Cancel a queued or running job")
    public ResponseEntity<Void> cancelJob(@PathVariable UUID jobId) {
        jobService.cancelJob(jobId);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/counts")
    @Operation(summary = "Get job counts", description = "Get counts of jobs by status")
    public ResponseEntity<Map<String, Long>> getJobCounts() {
        return ResponseEntity.ok(jobService.getJobCounts());
    }
}
