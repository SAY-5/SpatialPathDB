"""
Celery application configuration for SpatialPathDB.
Handles asynchronous job processing for analysis tasks.
"""

import os
from celery import Celery

# Get Redis URL from environment
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
app = Celery(
    'spatialpathdb',
    broker=redis_url,
    backend=redis_url,
    include=['src.workers.analysis_worker']
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Result settings
    result_expires=3600 * 24,  # Results expire after 24 hours

    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Task routing
    task_routes={
        'src.workers.analysis_worker.run_cell_detection': {'queue': 'analysis'},
        'src.workers.analysis_worker.run_tissue_segmentation': {'queue': 'analysis'},
        'src.workers.analysis_worker.run_spatial_statistics': {'queue': 'analysis'},
    },

    # Rate limiting
    task_annotations={
        'src.workers.analysis_worker.run_cell_detection': {'rate_limit': '10/m'},
    },

    # Scheduled tasks (optional)
    beat_schedule={
        # Example: cleanup old results every hour
        # 'cleanup-old-results': {
        #     'task': 'src.workers.maintenance.cleanup_old_results',
        #     'schedule': 3600.0,
        # },
    }
)

if __name__ == '__main__':
    app.start()
