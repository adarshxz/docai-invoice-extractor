"""Celery tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from docai.api.celery_app import celery_app
from docai.pipeline import get_pipeline


@celery_app.task(name="docai.extract")
def extract_document(upload_path: str, file_name: str) -> dict[str, Any]:
    """Run the pipeline on an uploaded file and return the JSON-ready result.

    The upload is deleted once processed. Document-level failures are reported
    inside the payload (``status = FAILED``), not as task failures.
    """
    path = Path(upload_path)
    result = get_pipeline().process_path(path, file_name)
    path.unlink(missing_ok=True)
    return result.model_dump(mode="json")
