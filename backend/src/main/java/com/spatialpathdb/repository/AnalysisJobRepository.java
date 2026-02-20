package com.spatialpathdb.repository;

import com.spatialpathdb.model.entity.AnalysisJob;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Repository
public interface AnalysisJobRepository extends JpaRepository<AnalysisJob, UUID> {

    Page<AnalysisJob> findBySlideId(UUID slideId, Pageable pageable);

    Page<AnalysisJob> findByStatus(AnalysisJob.Status status, Pageable pageable);

    Page<AnalysisJob> findBySlideIdAndStatus(UUID slideId, AnalysisJob.Status status, Pageable pageable);

    List<AnalysisJob> findByStatusIn(List<AnalysisJob.Status> statuses);

    @Query("SELECT j FROM AnalysisJob j WHERE j.slideId = :slideId ORDER BY j.submittedAt DESC")
    List<AnalysisJob> findRecentBySlideId(@Param("slideId") UUID slideId, Pageable pageable);

    @Modifying
    @Query("UPDATE AnalysisJob j SET j.status = :status, j.startedAt = :startedAt WHERE j.id = :jobId")
    void updateStatusToRunning(
        @Param("jobId") UUID jobId,
        @Param("status") AnalysisJob.Status status,
        @Param("startedAt") LocalDateTime startedAt
    );

    @Modifying
    @Query("""
        UPDATE AnalysisJob j
        SET j.status = :status, j.completedAt = :completedAt, j.resultSummary = :resultSummary
        WHERE j.id = :jobId
        """)
    void updateStatusToCompleted(
        @Param("jobId") UUID jobId,
        @Param("status") AnalysisJob.Status status,
        @Param("completedAt") LocalDateTime completedAt,
        @Param("resultSummary") java.util.Map<String, Object> resultSummary
    );

    @Modifying
    @Query("""
        UPDATE AnalysisJob j
        SET j.status = :status, j.completedAt = :completedAt, j.errorMessage = :errorMessage
        WHERE j.id = :jobId
        """)
    void updateStatusToFailed(
        @Param("jobId") UUID jobId,
        @Param("status") AnalysisJob.Status status,
        @Param("completedAt") LocalDateTime completedAt,
        @Param("errorMessage") String errorMessage
    );

    long countByStatus(AnalysisJob.Status status);

    void deleteBySlideId(UUID slideId);
}
