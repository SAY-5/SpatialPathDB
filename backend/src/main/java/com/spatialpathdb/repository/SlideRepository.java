package com.spatialpathdb.repository;

import com.spatialpathdb.model.entity.Slide;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface SlideRepository extends JpaRepository<Slide, UUID> {

    Page<Slide> findByOrgan(String organ, Pageable pageable);

    Page<Slide> findByStainType(String stainType, Pageable pageable);

    List<Slide> findByOrganAndStainType(String organ, String stainType);

    @Query("SELECT s FROM Slide s WHERE s.organ = :organ OR s.stainType = :stainType")
    Page<Slide> findByOrganOrStainType(
        @Param("organ") String organ,
        @Param("stainType") String stainType,
        Pageable pageable
    );

    @Query(value = """
        SELECT s.*, COUNT(so.id) as object_count
        FROM slides s
        LEFT JOIN spatial_objects so ON s.id = so.slide_id
        GROUP BY s.id
        ORDER BY s.uploaded_at DESC
        """, nativeQuery = true)
    List<Object[]> findAllWithObjectCounts();

    boolean existsBySlideName(String slideName);
}
