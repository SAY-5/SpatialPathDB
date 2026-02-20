package com.spatialpathdb.repository;

import com.spatialpathdb.model.entity.QueryLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Repository
public interface QueryLogRepository extends JpaRepository<QueryLog, Long> {

    Page<QueryLog> findByQueryType(String queryType, Pageable pageable);

    Page<QueryLog> findBySlideId(UUID slideId, Pageable pageable);

    @Query("""
        SELECT q FROM QueryLog q
        WHERE q.executedAt >= :since
        ORDER BY q.executedAt DESC
        """)
    List<QueryLog> findRecentQueries(@Param("since") LocalDateTime since);

    // Performance statistics by query type
    @Query(value = """
        SELECT query_type,
               COUNT(*) as total_queries,
               AVG(execution_time_ms) as avg_time,
               MIN(execution_time_ms) as min_time,
               MAX(execution_time_ms) as max_time,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) as p95_time,
               PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) as p99_time,
               AVG(num_results) as avg_results
        FROM query_logs
        WHERE executed_at >= :since
        GROUP BY query_type
        """, nativeQuery = true)
    List<Object[]> getPerformanceStatsByType(@Param("since") LocalDateTime since);

    // Hourly query counts for the past 24 hours
    @Query(value = """
        SELECT date_trunc('hour', executed_at) as hour,
               COUNT(*) as query_count,
               AVG(execution_time_ms) as avg_time
        FROM query_logs
        WHERE executed_at >= NOW() - INTERVAL '24 hours'
        GROUP BY date_trunc('hour', executed_at)
        ORDER BY hour
        """, nativeQuery = true)
    List<Object[]> getHourlyQueryStats();
}
