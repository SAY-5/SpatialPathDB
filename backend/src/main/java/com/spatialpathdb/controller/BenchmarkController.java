package com.spatialpathdb.controller;

import com.spatialpathdb.model.entity.QueryLog;
import com.spatialpathdb.repository.QueryLogRepository;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/benchmarks")
@RequiredArgsConstructor
@Tag(name = "Benchmarks", description = "Performance benchmarking and monitoring")
public class BenchmarkController {

    private final QueryLogRepository queryLogRepository;

    @GetMapping("/results")
    @Operation(summary = "Get benchmark results", description = "Get query performance statistics")
    public ResponseEntity<Map<String, Object>> getBenchmarkResults(
            @RequestParam(defaultValue = "24") int hoursBack) {

        LocalDateTime since = LocalDateTime.now().minusHours(hoursBack);

        // Get performance stats by query type
        List<Object[]> statsResults = queryLogRepository.getPerformanceStatsByType(since);

        List<Map<String, Object>> statsByType = statsResults.stream()
            .map(row -> {
                Map<String, Object> stat = new LinkedHashMap<>();
                stat.put("queryType", row[0]);
                stat.put("totalQueries", ((Number) row[1]).longValue());
                stat.put("avgTimeMs", row[2] != null ? ((Number) row[2]).doubleValue() : 0);
                stat.put("minTimeMs", row[3] != null ? ((Number) row[3]).doubleValue() : 0);
                stat.put("maxTimeMs", row[4] != null ? ((Number) row[4]).doubleValue() : 0);
                stat.put("p95TimeMs", row[5] != null ? ((Number) row[5]).doubleValue() : 0);
                stat.put("p99TimeMs", row[6] != null ? ((Number) row[6]).doubleValue() : 0);
                stat.put("avgResults", row[7] != null ? ((Number) row[7]).doubleValue() : 0);
                return stat;
            })
            .collect(Collectors.toList());

        // Get hourly query stats
        List<Object[]> hourlyResults = queryLogRepository.getHourlyQueryStats();

        List<Map<String, Object>> hourlyStats = hourlyResults.stream()
            .map(row -> {
                Map<String, Object> stat = new LinkedHashMap<>();
                stat.put("hour", row[0]);
                stat.put("queryCount", ((Number) row[1]).longValue());
                stat.put("avgTimeMs", row[2] != null ? ((Number) row[2]).doubleValue() : 0);
                return stat;
            })
            .collect(Collectors.toList());

        // Get recent queries
        List<QueryLog> recentQueries = queryLogRepository.findRecentQueries(
            LocalDateTime.now().minusMinutes(30)
        );

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("period", Map.of(
            "since", since,
            "until", LocalDateTime.now()
        ));
        response.put("statsByQueryType", statsByType);
        response.put("hourlyStats", hourlyStats);
        response.put("recentQueryCount", recentQueries.size());

        // Compute overall summary
        if (!statsByType.isEmpty()) {
            double totalQueries = statsByType.stream()
                .mapToLong(s -> ((Number) s.get("totalQueries")).longValue())
                .sum();

            double weightedAvg = statsByType.stream()
                .mapToDouble(s -> ((Number) s.get("avgTimeMs")).doubleValue() *
                    ((Number) s.get("totalQueries")).longValue())
                .sum() / totalQueries;

            response.put("summary", Map.of(
                "totalQueries", (long) totalQueries,
                "overallAvgTimeMs", weightedAvg
            ));
        }

        return ResponseEntity.ok(response);
    }

    @PostMapping("/run")
    @Operation(
        summary = "Run benchmark suite",
        description = "Trigger benchmark suite execution (placeholder for external benchmark runner)"
    )
    public ResponseEntity<Map<String, Object>> runBenchmarks(
            @RequestParam(required = false) UUID slideId) {

        // In a real implementation, this would trigger the Python benchmark suite
        // For now, return instructions

        Map<String, Object> response = new LinkedHashMap<>();
        response.put("status", "acknowledged");
        response.put("message", "Run the benchmark suite from command line:");
        response.put("command", "python benchmarks/src/benchmark_spatial_queries.py" +
            (slideId != null ? " --slide-id " + slideId : ""));

        return ResponseEntity.accepted().body(response);
    }
}
