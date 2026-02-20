"""
Celery workers for asynchronous analysis job processing.
These workers handle long-running analysis tasks in the background.
"""

import json
from datetime import datetime
from typing import Dict
import numpy as np

from celery import shared_task

from ..utils.db_connection import get_db_connection
from ..spatial_analysis.cell_detection import CellDetectionSimulator
from ..spatial_analysis.tissue_segmentation import TissueSegmentationSimulator
from ..spatial_analysis.spatial_statistics import SpatialStatisticsEngine
from ..spatial_analysis.density_estimation import DensityEstimator
from ..data_ingestion.db_loader import BulkDataLoader


def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status in database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if status == 'RUNNING':
                cur.execute("""
                    UPDATE analysis_jobs
                    SET status = %s, started_at = NOW()
                    WHERE id = %s
                """, (status, job_id))
            elif status == 'COMPLETED':
                cur.execute("""
                    UPDATE analysis_jobs
                    SET status = %s, completed_at = NOW(), result_summary = %s
                    WHERE id = %s
                """, (status, json.dumps(kwargs.get('result_summary', {})), job_id))
            elif status == 'FAILED':
                cur.execute("""
                    UPDATE analysis_jobs
                    SET status = %s, completed_at = NOW(), error_message = %s
                    WHERE id = %s
                """, (status, kwargs.get('error_message', 'Unknown error'), job_id))


def get_slide_info(slide_id: str) -> Dict:
    """Fetch slide metadata from database."""
    with get_db_connection(dict_cursor=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, slide_name, width_pixels, height_pixels,
                       microns_per_pixel, stain_type, organ
                FROM slides WHERE id = %s
            """, (slide_id,))
            result = cur.fetchone()
            if result:
                return dict(result)
            return {}


@shared_task(bind=True, max_retries=3)
def run_cell_detection(self, job_id: str, slide_id: str, parameters: Dict):
    """
    Celery task: Run cell detection on a slide.

    1. Update job status to RUNNING
    2. Generate synthetic cell detections
    3. Bulk insert results into database
    4. Compute summary statistics
    5. Update job status to COMPLETED
    """
    try:
        update_job_status(job_id, 'RUNNING')

        # Get slide dimensions
        slide_info = get_slide_info(slide_id)
        if not slide_info:
            raise ValueError(f"Slide not found: {slide_id}")

        # Configure detection parameters
        detection_params = {
            'target_cells': parameters.get('target_cells', 100000),
            'n_clusters': parameters.get('n_clusters', 15),
            'confidence_threshold': parameters.get('confidence_threshold', 0.5),
            'slide_width': slide_info['width_pixels'],
            'slide_height': slide_info['height_pixels']
        }

        # Run detection
        detector = CellDetectionSimulator(seed=hash(slide_id) % 2**31)
        loader = BulkDataLoader()

        total_inserted = 0
        batch_count = 0

        for batch in detector.generate_detections(slide_id, detection_params):
            total_inserted += loader.bulk_insert_objects(batch, slide_id)
            batch_count += 1

            # Update progress periodically
            if batch_count % 10 == 0:
                self.update_state(
                    state='PROGRESS',
                    meta={'current': total_inserted, 'status': 'Inserting cells...'}
                )

        # Compute summary
        summary = loader.get_slide_summary(slide_id)
        result_summary = {
            'total_cells_detected': total_inserted,
            'cell_type_breakdown': summary.get('by_type_and_label', []),
            'parameters': detection_params
        }

        update_job_status(job_id, 'COMPLETED', result_summary=result_summary)
        return result_summary

    except Exception as e:
        update_job_status(job_id, 'FAILED', error_message=str(e))
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def run_tissue_segmentation(self, job_id: str, slide_id: str, parameters: Dict):
    """
    Celery task: Run tissue segmentation on a slide.

    Generates tissue region boundaries and inserts them into the database.
    """
    try:
        update_job_status(job_id, 'RUNNING')

        slide_info = get_slide_info(slide_id)
        if not slide_info:
            raise ValueError(f"Slide not found: {slide_id}")

        # Configure segmentation
        n_regions = parameters.get('n_regions', 8)
        seg_params = {
            'min_region_size': parameters.get('min_region_size', 3000),
            'max_region_size': parameters.get('max_region_size', 20000)
        }

        # Run segmentation
        segmenter = TissueSegmentationSimulator(seed=hash(slide_id) % 2**31)
        regions = segmenter.generate_regions(
            slide_id,
            slide_info['width_pixels'],
            slide_info['height_pixels'],
            n_regions,
            seg_params
        )

        # Insert regions
        loader = BulkDataLoader()
        total_inserted = loader.bulk_insert_objects(regions, slide_id)

        # Compute summary
        result_summary = segmenter.run_segmentation(
            slide_id,
            slide_info['width_pixels'],
            slide_info['height_pixels'],
            n_regions,
            seg_params
        )
        result_summary['regions_created'] = total_inserted

        update_job_status(job_id, 'COMPLETED', result_summary=result_summary)
        return result_summary

    except Exception as e:
        update_job_status(job_id, 'FAILED', error_message=str(e))
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def run_spatial_statistics(self, job_id: str, slide_id: str, parameters: Dict):
    """
    Celery task: Compute spatial statistics for a slide.

    Analyzes the spatial distribution of cells already in the database.
    """
    try:
        update_job_status(job_id, 'RUNNING')

        slide_info = get_slide_info(slide_id)
        if not slide_info:
            raise ValueError(f"Slide not found: {slide_id}")

        # Fetch cell centroids from database
        with get_db_connection(dict_cursor=True) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT ST_X(centroid) as x, ST_Y(centroid) as y, label
                    FROM spatial_objects
                    WHERE slide_id = %s AND centroid IS NOT NULL
                    LIMIT 1000000
                """, (slide_id,))
                rows = cur.fetchall()

        if not rows:
            update_job_status(job_id, 'COMPLETED', result_summary={
                'error': 'No spatial objects found for this slide'
            })
            return

        # Prepare data
        centroids = np.array([[r['x'], r['y']] for r in rows])
        labels = np.array([r['label'] for r in rows])

        # Compute statistics
        slide_area = slide_info['width_pixels'] * slide_info['height_pixels']
        stats_engine = SpatialStatisticsEngine(centroids, labels)

        self.update_state(state='PROGRESS', meta={'status': 'Computing nearest neighbors...'})
        nn_stats = stats_engine.compute_nearest_neighbor_distribution()

        self.update_state(state='PROGRESS', meta={'status': 'Computing hotspots...'})
        hotspots = stats_engine.compute_hotspot_detection(
            cell_size=parameters.get('hotspot_grid_size', 500),
            min_density=parameters.get('min_hotspot_density', 5)
        )

        self.update_state(state='PROGRESS', meta={'status': 'Computing colocalization...'})
        coloc = stats_engine.compute_label_colocalization(
            radius=parameters.get('colocalization_radius', 100)
        )

        # Compute Ripley's K if requested
        ripleys_k = None
        if parameters.get('compute_ripleys_k', False):
            self.update_state(state='PROGRESS', meta={'status': "Computing Ripley's K..."})
            radii = np.linspace(10, 500, 20)
            ripleys_k = stats_engine.compute_ripleys_k(radii, slide_area)

        # Compute density grid
        self.update_state(state='PROGRESS', meta={'status': 'Computing density grid...'})
        density_estimator = DensityEstimator(
            centroids,
            bounds=(0, 0, slide_info['width_pixels'], slide_info['height_pixels'])
        )
        density = density_estimator.compute_grid_density(
            grid_size=parameters.get('density_grid_size', 512),
            labels=labels
        )

        result_summary = {
            'total_objects_analyzed': len(centroids),
            'nearest_neighbor_stats': nn_stats,
            'hotspot_analysis': {
                'n_hotspots': hotspots['n_hotspots'],
                'top_hotspots': hotspots['hotspots'][:10]
            },
            'colocalization': coloc,
            'density_summary': {
                'max_count': density['max_count'],
                'grid_size': density['grid_size']
            }
        }

        if ripleys_k:
            result_summary['ripleys_k'] = ripleys_k

        update_job_status(job_id, 'COMPLETED', result_summary=result_summary)
        return result_summary

    except Exception as e:
        update_job_status(job_id, 'FAILED', error_message=str(e))
        raise self.retry(exc=e, countdown=60)
