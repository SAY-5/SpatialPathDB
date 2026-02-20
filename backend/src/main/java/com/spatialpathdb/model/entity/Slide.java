package com.spatialpathdb.model.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import java.time.LocalDateTime;
import java.util.Map;
import java.util.UUID;

@Entity
@Table(name = "slides")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Slide {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "slide_name", nullable = false)
    private String slideName;

    @Column(name = "file_path", nullable = false)
    private String filePath;

    @Column(name = "width_pixels", nullable = false)
    private Long widthPixels;

    @Column(name = "height_pixels", nullable = false)
    private Long heightPixels;

    @Column(name = "microns_per_pixel")
    private Double micronsPerPixel;

    @Column(name = "stain_type", length = 50)
    private String stainType;

    @Column(name = "organ", length = 100)
    private String organ;

    @Column(name = "uploaded_at")
    private LocalDateTime uploadedAt;

    @JdbcTypeCode(SqlTypes.JSON)
    @Column(name = "metadata", columnDefinition = "jsonb")
    private Map<String, Object> metadata;

    @PrePersist
    protected void onCreate() {
        if (uploadedAt == null) {
            uploadedAt = LocalDateTime.now();
        }
    }
}
