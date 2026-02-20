package com.spatialpathdb.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.sql.DataSource;
import java.sql.Connection;
import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/health")
@RequiredArgsConstructor
@Tag(name = "Health", description = "Health check endpoints")
public class HealthController {

    private final DataSource dataSource;
    private final RedisTemplate<String, Object> redisTemplate;

    @GetMapping
    @Operation(summary = "Health check", description = "Check the health of all system components")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        Map<String, Object> health = new LinkedHashMap<>();
        health.put("status", "UP");
        health.put("timestamp", LocalDateTime.now());

        // Check database
        Map<String, Object> dbHealth = new LinkedHashMap<>();
        try (Connection conn = dataSource.getConnection()) {
            dbHealth.put("status", "UP");
            dbHealth.put("database", conn.getMetaData().getDatabaseProductName());
            dbHealth.put("version", conn.getMetaData().getDatabaseProductVersion());
        } catch (Exception e) {
            dbHealth.put("status", "DOWN");
            dbHealth.put("error", e.getMessage());
            health.put("status", "DEGRADED");
        }
        health.put("database", dbHealth);

        // Check Redis
        Map<String, Object> redisHealth = new LinkedHashMap<>();
        try {
            String pong = redisTemplate.getConnectionFactory()
                .getConnection()
                .ping();
            redisHealth.put("status", "UP");
            redisHealth.put("ping", pong);
        } catch (Exception e) {
            redisHealth.put("status", "DOWN");
            redisHealth.put("error", e.getMessage());
            health.put("status", "DEGRADED");
        }
        health.put("redis", redisHealth);

        return ResponseEntity.ok(health);
    }

    @GetMapping("/ready")
    @Operation(summary = "Readiness check", description = "Check if the service is ready to accept requests")
    public ResponseEntity<Map<String, Object>> readinessCheck() {
        Map<String, Object> ready = new LinkedHashMap<>();

        boolean isReady = true;

        // Check database connection
        try (Connection conn = dataSource.getConnection()) {
            conn.isValid(5);
        } catch (Exception e) {
            isReady = false;
        }

        ready.put("ready", isReady);
        ready.put("timestamp", LocalDateTime.now());

        if (isReady) {
            return ResponseEntity.ok(ready);
        } else {
            return ResponseEntity.status(503).body(ready);
        }
    }

    @GetMapping("/live")
    @Operation(summary = "Liveness check", description = "Check if the service is alive")
    public ResponseEntity<Map<String, Object>> livenessCheck() {
        return ResponseEntity.ok(Map.of(
            "alive", true,
            "timestamp", LocalDateTime.now()
        ));
    }
}
