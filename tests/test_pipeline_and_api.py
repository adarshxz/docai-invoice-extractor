from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docai.api.main import TaskState, app, get_task_queue
from docai.config import Settings, get_settings
from docai.ingestion import UnsupportedDocumentError, load_pages
from docai.model import aggregate_word_probabilities, first_subtoken_positions
from docai.pipeline import InvoiceExtractionPipeline, get_pipeline

# --- ingestion -------------------------------------------------------------


def test_load_png_and_jpeg(png_bytes):
    pages, total = load_pages(png_bytes)
    assert total == 1 and pages[0].mode == "RGB"

    buf = io.BytesIO()
    Image.new("L", (40, 30)).save(buf, format="JPEG")
    pages, _ = load_pages(buf.getvalue())
    assert pages[0].size == (40, 30) and pages[0].mode == "RGB"


@pytest.mark.parametrize("data", [b"", b"not an image", b"GIF89a" + b"\0" * 20])
def test_load_rejects_bad_input(data):
    with pytest.raises(UnsupportedDocumentError):
        load_pages(data)


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="poppler not installed")
def test_load_pdf_respects_max_pages():
    buf = io.BytesIO()
    imgs = [Image.new("RGB", (200, 300), "white") for _ in range(3)]
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    pages, total = load_pages(buf.getvalue(), dpi=72, max_pages=2)
    assert total == 3 and len(pages) == 2


# --- model helpers (no weights needed) ------------------------------------


def test_first_subtoken_positions():
    assert first_subtoken_positions([None, 0, 0, 1, 2, 2, None]) == {0: 1, 1: 3, 2: 4}


def test_aggregate_word_probabilities_averages_overlapping_windows():
    n_labels = 3
    w1 = np.array([[1, 0, 0], [0.2, 0.8, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=float)
    w2 = np.array([[1, 0, 0], [0.4, 0.6, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    # window 1 covers words 0, 1; window 2 covers words 1, 2; word 3 is never seen.
    probs = aggregate_word_probabilities([w1, w2], [[None, 0, 1, None], [None, 1, 2, None]], num_words=4)
    np.testing.assert_allclose(probs[0], [0.2, 0.8, 0])
    np.testing.assert_allclose(probs[1], [0.65, 0.35, 0])  # mean of w1[2] and w2[1]
    np.testing.assert_allclose(probs[2], [0, 0, 1])
    np.testing.assert_allclose(probs[3], [1, 0, 0])  # unseen word -> "O"
    assert probs.shape == (4, n_labels)


# --- pipeline --------------------------------------------------------------


def make_pipeline(fake_classifier, invoice_words) -> InvoiceExtractionPipeline:
    return InvoiceExtractionPipeline(
        Settings(upload_dir=Path("/tmp/unused")), classifier=fake_classifier, ocr=lambda _img: invoice_words
    )


def test_pipeline_success(fake_classifier, invoice_words, png_bytes):
    result = make_pipeline(fake_classifier, invoice_words).process_bytes(png_bytes, "inv.png")
    payload = json.loads(result.model_dump_json())

    assert payload["document_metadata"]["status"] == "SUCCESS"
    assert payload["document_metadata"]["total_pages"] == 1
    assert payload["document_metadata"]["file_name"] == "inv.png"
    assert payload["extracted_entities"]["invoice_number"] == {"value": "INV-2024-0042", "confidence": 0.93}
    assert payload["extracted_entities"]["invoice_date"]["value"] == "2024-03-05"
    assert payload["extracted_entities"]["total_amount"] == {"value": 165.55, "currency": "USD", "confidence": 0.93}
    assert payload["line_items"] == [{"description": "Widget A", "amount": 120.0}, {"description": "Gadget", "amount": 30.5}]
    assert set(payload) == {"document_metadata", "extracted_entities", "line_items"}


def test_pipeline_failure_returns_failed_payload(fake_classifier, invoice_words):
    result = make_pipeline(fake_classifier, invoice_words).process_bytes(b"garbage", "x.png")
    assert result.document_metadata.status.value == "FAILED"
    assert result.extracted_entities.total_amount.value is None
    assert result.line_items == []
    assert fake_classifier.calls == 0


# --- API -------------------------------------------------------------------


class FakeQueue:
    def __init__(self) -> None:
        self.submitted: dict[str, tuple[Path, str]] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def submit(self, task_id: str, upload_path: Path, file_name: str) -> None:
        assert upload_path.exists()
        self.submitted[task_id] = (upload_path, file_name)

    def status(self, task_id: str):
        if task_id in self.results:
            return TaskState.SUCCESS, self.results[task_id], None
        return TaskState.PENDING, None, None

    def ping(self) -> bool:
        return True


@pytest.fixture
def client(tmp_path, fake_classifier, invoice_words):
    queue = FakeQueue()
    settings = Settings(upload_dir=tmp_path, max_upload_mb=1)
    pipeline = make_pipeline(fake_classifier, invoice_words)
    app.dependency_overrides = {
        get_task_queue: lambda: queue,
        get_settings: lambda: settings,
        get_pipeline: lambda: pipeline,
    }
    with TestClient(app) as c:
        c.queue = queue  # type: ignore[attr-defined]
        c.pipeline = pipeline  # type: ignore[attr-defined]
        yield c
    app.dependency_overrides = {}


def test_async_submit_and_poll(client, png_bytes):
    resp = client.post("/v1/documents", files={"file": ("scan.PNG", png_bytes, "image/png")})
    assert resp.status_code == 202
    body = resp.json()
    task_id = body["task_id"]
    assert body["status"] == "PENDING"
    assert body["status_url"].endswith(f"/v1/documents/{task_id}")

    path, name = client.queue.submitted[task_id]
    assert name == "scan.PNG" and path.read_bytes() == png_bytes

    assert client.get(f"/v1/documents/{task_id}").json() == {"task_id": task_id, "status": "PENDING"}

    client.queue.results[task_id] = client.pipeline.process_bytes(png_bytes, name).model_dump(mode="json")
    done = client.get(f"/v1/documents/{task_id}").json()
    assert done["status"] == "SUCCESS"
    assert done["result"]["extracted_entities"]["vendor_name"]["value"] == "Acme Supplies Ltd"


def test_sync_extract(client):
    resp = client.post("/v1/extract", files={"file": ("inv.jpg", _jpeg(), "image/jpeg")})
    assert resp.status_code == 200
    assert resp.json()["extracted_entities"]["tax_amount"]["value"] == 15.05


@pytest.mark.parametrize(
    ("name", "data", "code"),
    [
        ("notes.txt", b"hello", 415),
        ("fake.pdf", b"plain text", 415),
        ("empty.png", b"", 400),
        ("big.png", b"\x89PNG\r\n\x1a\n" + b"\0" * (1024 * 1024 + 1), 413),
    ],
)
def test_upload_validation(client, name, data, code):
    assert client.post("/v1/documents", files={"file": (name, data)}).status_code == code


def test_status_rejects_non_uuid(client):
    assert client.get("/v1/documents/not-a-uuid").status_code == 422
    assert client.get(f"/v1/documents/{uuid.uuid4()}").status_code == 200


def test_health(client):
    assert client.get("/health").json()["broker"] == "up"


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 140), "white").save(buf, format="JPEG")
    return buf.getvalue()
