package com.spatialpathdb.model.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.Point;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "spatial_objects")
@IdClass(SpatialObjectId.class)
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SpatialObject {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Id
    @Column(name = "slide_id", nullable = false)
    private UUID slideId;

    @Column(name = "object_type", nullable = false, length = 50)
    private String objectType;

    @Column(name = "label", length = 100)
    private String label;

    @Column(name = "confidence")
    private Double confidence;

    @Column(name = "geometry", columnDefinition = "geometry(Geometry, 0)")
    private Geometry geometry;

    @Column(name = "centroid", columnDefinition = "geometry(Point, 0)")
    private Point centroid;

    @Column(name = "area_pixels")
    private Double areaPixels;

    @Column(name = "perimeter_pixels")
    private Double perimeterPixels;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "properties", columnDefinition = "jsonb")
    private Map<String, Object> properties;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }
}
