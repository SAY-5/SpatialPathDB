package com.spatialpathdb.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "query_logs")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class QueryLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "query_type", length = 100)
    private String queryType;

    @Column(name = "slide_id")
    private UUID slideId;

    @Column(name = "bbox_wkt", columnDefinition = "text")
    private String bboxWkt;

    @Column(name = "num_results")
    private Integer numResults;

    @Column(name = "execution_time_ms")
    private Double executionTimeMs;

    @Column(name = "used_index")
    private Boolean usedIndex;

    @Column(name = "executed_at")
    private LocalDateTime executedAt;

    @PrePersist
    protected void onCreate() {
        if (executedAt == null) {
            executedAt = LocalDateTime.now();
        }
    }
}
