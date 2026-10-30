"""Celery application. Start a worker with::

    celery -A docai.api.celery_app worker --loglevel=INFO --concurrency=1
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.signals import worker_process_init

from docai.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

celery_app = Celery("docai", broker=_settings.redis_url, backend=_settings.redis_url, include=["docai.api.tasks"])
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,  # a worker crash re-queues the document instead of losing it
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # inference is long-running; don't hoard tasks
    task_time_limit=_settings.task_time_limit_s,
    task_soft_time_limit=max(1, _settings.task_time_limit_s - 15),
    result_expires=_settings.result_ttl_s,
    broker_connection_retry_on_startup=True,
)


@worker_process_init.connect
def _warm_up(**_: Any) -> None:
    """Load the model when each worker process starts, not on its first task."""
    from docai.pipeline import get_pipeline

    _ = get_pipeline().classifier
    logger.info("Model loaded in worker process")
