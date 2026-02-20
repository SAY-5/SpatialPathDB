package com.spatialpathdb.service;

import com.spatialpathdb.exception.JobExecutionException;
import com.spatialpathdb.exception.SlideNotFoundException;
import com.spatialpathdb.model.dto.JobStatusResponse;
import com.spatialpathdb.model.dto.JobSubmitRequest;
import com.spatialpathdb.model.entity.AnalysisJob;
import com.spatialpathdb.repository.AnalysisJobRepository;
import com.spatialpathdb.repository.SlideRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.UUID;

@Service
@RequiredArgsConstructor
@Slf4j
public class AnalysisJobService {

    private final AnalysisJobRepository jobRepository;
    private final SlideRepository slideRepository;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String JOB_QUEUE_KEY = "analysis:jobs:queue";

    @Transactional
    public JobStatusResponse submitJob(JobSubmitRequest request) {
        // Validate slide exists
        if (!slideRepository.existsById(request.getSlideId())) {
            throw new SlideNotFoundException(request.getSlideId());
        }

        // Validate job type
        if (!isValidJobType(request.getJobType())) {
            throw new JobExecutionException("Invalid job type: " + request.getJobType());
        }

        // Create job record
        AnalysisJob job = AnalysisJob.builder()
            .slideId(request.getSlideId())
            .jobType(request.getJobType())
            .parameters(request.getParameters())
            .status(AnalysisJob.Status.QUEUED)
            .build();

        job = jobRepository.save(job);
        log.info("Job submitted: {} for slide {}", job.getId(), request.getSlideId());

        // Queue job for processing (in real implementation, this would go to Celery/Redis)
        queueJob(job);

        return JobStatusResponse.fromEntity(job);
    }

    public JobStatusResponse getJobStatus(UUID jobId) {
        AnalysisJob job = jobRepository.findById(jobId)
            .orElseThrow(() -> new JobExecutionException("Job not found: " + jobId));

        return JobStatusResponse.fromEntity(job);
    }

    public Page<JobStatusResponse> getAllJobs(Pageable pageable) {
        return jobRepository.findAll(pageable)
            .map(JobStatusResponse::fromEntity);
    }

    public Page<JobStatusResponse> getJobsByStatus(AnalysisJob.Status status, Pageable pageable) {
        return jobRepository.findByStatus(status, pageable)
            .map(JobStatusResponse::fromEntity);
    }

    public Page<JobStatusResponse> getJobsBySlide(UUID slideId, Pageable pageable) {
        return jobRepository.findBySlideId(slideId, pageable)
            .map(JobStatusResponse::fromEntity);
    }

    @Transactional
    public void cancelJob(UUID jobId) {
        AnalysisJob job = jobRepository.findById(jobId)
            .orElseThrow(() -> new JobExecutionException("Job not found: " + jobId));

        if (job.getStatus() == AnalysisJob.Status.COMPLETED ||
            job.getStatus() == AnalysisJob.Status.FAILED) {
            throw new JobExecutionException("Cannot cancel job in status: " + job.getStatus());
        }

        job.setStatus(AnalysisJob.Status.CANCELLED);
        jobRepository.save(job);

        log.info("Job cancelled: {}", jobId);
    }

    public Map<String, Long> getJobCounts() {
        return Map.of(
            "queued", jobRepository.countByStatus(AnalysisJob.Status.QUEUED),
            "running", jobRepository.countByStatus(AnalysisJob.Status.RUNNING),
            "completed", jobRepository.countByStatus(AnalysisJob.Status.COMPLETED),
            "failed", jobRepository.countByStatus(AnalysisJob.Status.FAILED),
            "cancelled", jobRepository.countByStatus(AnalysisJob.Status.CANCELLED)
        );
    }

    private boolean isValidJobType(String jobType) {
        return jobType != null && (
            jobType.equals("cell_detection") ||
            jobType.equals("tissue_segmentation") ||
            jobType.equals("spatial_statistics")
        );
    }

    private void queueJob(AnalysisJob job) {
        try {
            // Push job ID to Redis queue for Celery workers
            Map<String, Object> jobMessage = Map.of(
                "job_id", job.getId().toString(),
                "slide_id", job.getSlideId().toString(),
                "job_type", job.getJobType(),
                "parameters", job.getParameters() != null ? job.getParameters() : Map.of()
            );

            redisTemplate.opsForList().rightPush(JOB_QUEUE_KEY, jobMessage);
            log.debug("Job queued to Redis: {}", job.getId());
        } catch (Exception e) {
            log.error("Failed to queue job to Redis", e);
            // Job is still created in DB, workers can poll from there as fallback
        }
    }
}
