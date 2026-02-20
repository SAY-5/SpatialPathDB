package com.spatialpathdb.service;

import com.spatialpathdb.exception.SlideNotFoundException;
import com.spatialpathdb.model.dto.SlideCreateRequest;
import com.spatialpathdb.model.dto.SlideResponse;
import com.spatialpathdb.model.entity.Slide;
import com.spatialpathdb.repository.AnalysisJobRepository;
import com.spatialpathdb.repository.SlideRepository;
import com.spatialpathdb.repository.SpatialObjectRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class SlideService {

    private final SlideRepository slideRepository;
    private final SpatialObjectRepository spatialObjectRepository;
    private final AnalysisJobRepository analysisJobRepository;

    @Transactional
    public SlideResponse createSlide(SlideCreateRequest request) {
        Slide slide = Slide.builder()
            .slideName(request.getSlideName())
            .filePath(request.getFilePath())
            .widthPixels(request.getWidthPixels())
            .heightPixels(request.getHeightPixels())
            .micronsPerPixel(request.getMicronsPerPixel())
            .stainType(request.getStainType())
            .organ(request.getOrgan())
            .metadata(request.getMetadata())
            .build();

        slide = slideRepository.save(slide);
        log.info("Created slide: {}", slide.getId());

        return SlideResponse.fromEntity(slide);
    }

    @Cacheable(value = "slides", key = "#slideId")
    public SlideResponse getSlide(UUID slideId) {
        Slide slide = slideRepository.findById(slideId)
            .orElseThrow(() -> new SlideNotFoundException(slideId));

        SlideResponse response = SlideResponse.fromEntity(slide);

        // Add object count
        long objectCount = spatialObjectRepository.countBySlideId(slideId);
        response.setObjectCount(objectCount);

        return response;
    }

    public Page<SlideResponse> getAllSlides(Pageable pageable) {
        return slideRepository.findAll(pageable)
            .map(SlideResponse::fromEntity);
    }

    public Page<SlideResponse> getSlidesByOrgan(String organ, Pageable pageable) {
        return slideRepository.findByOrgan(organ, pageable)
            .map(SlideResponse::fromEntity);
    }

    public Page<SlideResponse> getSlidesByStainType(String stainType, Pageable pageable) {
        return slideRepository.findByStainType(stainType, pageable)
            .map(SlideResponse::fromEntity);
    }

    public List<SlideResponse> getSlidesWithObjectCounts() {
        List<Object[]> results = slideRepository.findAllWithObjectCounts();

        return results.stream()
            .map(row -> {
                Slide slide = new Slide();
                slide.setId((UUID) row[0]);
                slide.setSlideName((String) row[1]);
                slide.setFilePath((String) row[2]);
                slide.setWidthPixels(((Number) row[3]).longValue());
                slide.setHeightPixels(((Number) row[4]).longValue());

                SlideResponse response = SlideResponse.fromEntity(slide);
                response.setObjectCount(((Number) row[row.length - 1]).longValue());

                return response;
            })
            .collect(Collectors.toList());
    }

    @Transactional
    @CacheEvict(value = "slides", key = "#slideId")
    public void deleteSlide(UUID slideId) {
        if (!slideRepository.existsById(slideId)) {
            throw new SlideNotFoundException(slideId);
        }

        // Delete associated data
        spatialObjectRepository.deleteBySlideId(slideId);
        analysisJobRepository.deleteBySlideId(slideId);
        slideRepository.deleteById(slideId);

        log.info("Deleted slide and associated data: {}", slideId);
    }

    @Transactional
    @CacheEvict(value = "slides", key = "#slideId")
    public SlideResponse updateSlide(UUID slideId, SlideCreateRequest request) {
        Slide slide = slideRepository.findById(slideId)
            .orElseThrow(() -> new SlideNotFoundException(slideId));

        if (request.getSlideName() != null) {
            slide.setSlideName(request.getSlideName());
        }
        if (request.getFilePath() != null) {
            slide.setFilePath(request.getFilePath());
        }
        if (request.getMicronsPerPixel() != null) {
            slide.setMicronsPerPixel(request.getMicronsPerPixel());
        }
        if (request.getStainType() != null) {
            slide.setStainType(request.getStainType());
        }
        if (request.getOrgan() != null) {
            slide.setOrgan(request.getOrgan());
        }
        if (request.getMetadata() != null) {
            slide.setMetadata(request.getMetadata());
        }

        slide = slideRepository.save(slide);
        return SlideResponse.fromEntity(slide);
    }
}
