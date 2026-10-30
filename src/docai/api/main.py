"""FastAPI service.

Endpoints:

* ``POST /v1/documents`` - upload a document, get ``202`` and a task id.
* ``GET  /v1/documents/{task_id}`` - poll for the extraction result.
* ``POST /v1/extract`` - synchronous extraction (for development / low volume).
* ``GET  /health`` - liveness plus broker reachability.

Run with ``uvicorn docai.api.main:app``.
"""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Protocol

import anyio
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from docai import __version__
from docai.config import Settings, get_settings
from docai.ingestion import SUPPORTED_EXTENSIONS, is_pdf
from docai.pipeline import InvoiceExtractionPipeline, get_pipeline
from docai.schemas import ExtractionResult

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_CHUNK = 1 << 20


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class TaskState(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


class TaskAccepted(BaseModel):
    task_id: str
    status: TaskState = TaskState.PENDING
    status_url: str


class TaskStatus(BaseModel):
    task_id: str
    status: TaskState
    result: ExtractionResult | None = None
    error: str | None = None


class Health(BaseModel):
    status: str
    version: str
    broker: str


# --------------------------------------------------------------------------
# Task queue abstraction (swapped for a fake in tests)
# --------------------------------------------------------------------------


class TaskQueue(Protocol):
    def submit(self, task_id: str, upload_path: Path, file_name: str) -> None: ...

    def status(self, task_id: str) -> tuple[TaskState, dict[str, Any] | None, str | None]: ...

    def ping(self) -> bool: ...


class CeleryTaskQueue:
    """Submits to and reads from the Celery/Redis backend."""

    def submit(self, task_id: str, upload_path: Path, file_name: str) -> None:
        from docai.api.tasks import extract_document

        extract_document.apply_async(args=[str(upload_path), file_name], task_id=task_id)

    def status(self, task_id: str) -> tuple[TaskState, dict[str, Any] | None, str | None]:
        from celery.result import AsyncResult

        from docai.api.celery_app import celery_app

        res = AsyncResult(task_id, app=celery_app)
        state = TaskState(res.state) if res.state in TaskState.__members__ else TaskState.PENDING
        if state is TaskState.SUCCESS:
            return state, res.result, None
        if state is TaskState.FAILURE:
            return state, None, f"{type(res.result).__name__}: {res.result}"
        return state, None, None

    def ping(self) -> bool:
        from docai.api.celery_app import celery_app

        try:
            with celery_app.connection_for_write() as conn:
                conn.ensure_connection(max_retries=1, timeout=2)
            return True
        except Exception:
            return False


def get_task_queue() -> TaskQueue:
    return CeleryTaskQueue()


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------


def _sniff_ok(head: bytes) -> bool:
    return is_pdf(head) or head.startswith(_PNG_MAGIC) or head.startswith(_JPEG_MAGIC)


async def read_upload(file: UploadFile, settings: Settings) -> tuple[bytes, str]:
    """Read and validate an upload by extension, size and magic bytes.

    Raises:
        HTTPException: 415 for unsupported types, 413 for oversize files, 400 for empty ones.
    """
    file_name = Path(file.filename or "document").name
    suffix = Path(file_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type {suffix or '(none)'}; expected one of {sorted(SUPPORTED_EXTENSIONS)}",
        )

    limit = settings.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    size = 0
    while chunk := await file.read(_CHUNK):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
        chunks.append(chunk)
    data = b"".join(chunks)

    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty file")
    if not _sniff_ok(data[:1024]):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "File content is not a PDF, PNG or JPEG")
    return data, file_name


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="Document & Invoice AI",
    version=__version__,
    description="Extracts structured invoice fields from scanned documents with LayoutLMv3.",
)

# The model is not thread-safe; serialise synchronous inference inside this process.
_inference_limiter = anyio.CapacityLimiter(1)

SettingsDep = Annotated[Settings, Depends(get_settings)]
QueueDep = Annotated[TaskQueue, Depends(get_task_queue)]


@app.post("/v1/documents", status_code=status.HTTP_202_ACCEPTED, response_model=TaskAccepted)
async def submit_document(
    request: Request, settings: SettingsDep, queue: QueueDep, file: UploadFile = File(...)
) -> TaskAccepted:
    """Queue a document for extraction. Poll ``status_url`` for the result."""
    data, file_name = await read_upload(file, settings)
    task_id = str(uuid.uuid4())
    upload_path = settings.upload_dir / f"{task_id}{Path(file_name).suffix.lower()}"

    def _write() -> None:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(data)

    await run_in_threadpool(_write)
    try:
        await run_in_threadpool(queue.submit, task_id, upload_path, file_name)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Task queue unavailable") from exc

    return TaskAccepted(task_id=task_id, status_url=str(request.url_for("get_document_status", task_id=task_id)))


@app.get("/v1/documents/{task_id}", response_model=TaskStatus, response_model_exclude_none=True)
async def get_document_status(task_id: uuid.UUID, queue: QueueDep) -> TaskStatus:
    """Return the task state and, once finished, the extraction result.

    Unknown ids report ``PENDING`` (Celery cannot tell them apart from queued tasks).
    """
    state, payload, error = await run_in_threadpool(queue.status, str(task_id))
    result = ExtractionResult.model_validate(payload) if payload is not None else None
    return TaskStatus(task_id=str(task_id), status=state, result=result, error=error)


@app.post("/v1/extract", response_model=ExtractionResult)
async def extract_sync(
    settings: SettingsDep,
    pipeline: Annotated[InvoiceExtractionPipeline, Depends(get_pipeline)],
    file: UploadFile = File(...),
) -> ExtractionResult:
    """Extract synchronously in this process; prefer ``/v1/documents`` under load."""
    data, file_name = await read_upload(file, settings)
    return await anyio.to_thread.run_sync(pipeline.process_bytes, data, file_name, limiter=_inference_limiter)


@app.get("/health", response_model=Health)
async def health(queue: QueueDep) -> Health:
    """Liveness, plus whether the Redis broker is reachable."""
    broker_ok = await run_in_threadpool(queue.ping)
    return Health(status="ok", version=__version__, broker="up" if broker_ok else "down")
