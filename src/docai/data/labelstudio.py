"""Labelling your own invoices with Label Studio.

Workflow:

1. ``docai-data ls-prepare raw_invoices/ data/invoices`` runs the *inference*
   preprocessing (deskew + CLAHE) and Tesseract on every page, saving
   ``images/<page>.png`` and ``ocr/<page>.json``. Training on the same cleaned
   image and the same OCR words the pipeline will see in production avoids a
   train/serve mismatch. It also writes ``tasks.json`` and ``label_config.xml``.
2. In Label Studio, annotators draw one rectangle around each entity
   instance (the whole vendor name, one line item's description, ...) and
   pick its label. Nothing needs transcribing: the words come from the OCR.
3. Export the project as JSON and run
   ``docai-data ls-convert export.json data/invoices``. Each OCR word whose
   centre lies inside a rectangle takes that rectangle's label; the first such
   word (in OCR order) is ``B-``, the rest ``I-``. Everything else is ``O``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from docai.config import Settings
from docai.data.common import LabelledWord, bio_tags, write_records
from docai.ingestion import SUPPORTED_EXTENSIONS, load_pages_from_path
from docai.labels import ENTITY_TYPES
from docai.ocr import run_ocr
from docai.preprocessing import preprocess_page
from docai.schemas import OCRWord

logger = logging.getLogger(__name__)

# Distinct, colour-blind-friendly-ish label colours for the annotation UI.
_COLOURS = ("#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080")

# Label Studio prefixes uploaded files with 8 hex chars and a dash.
_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}-")


def label_config_xml() -> str:
    """Label Studio labelling interface: one rectangle tool with the six entity types."""
    labels = "\n".join(
        f'    <Label value="{e}" background="{c}"/>' for e, c in zip(ENTITY_TYPES, _COLOURS, strict=True)
    )
    return (
        "<View>\n"
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>\n'
        '  <RectangleLabels name="label" toName="image" strokeWidth="2" opacity="0.4">\n'
        f"{labels}\n"
        "  </RectangleLabels>\n"
        "</View>\n"
    )


# --------------------------------------------------------------------------
# Prepare
# --------------------------------------------------------------------------


@dataclass
class PrepareStats:
    documents: int = 0
    pages: int = 0
    failed: list[str] = field(default_factory=list)


def prepare(input_dir: Path, workdir: Path, settings: Settings | None = None) -> PrepareStats:
    """Preprocess + OCR every PDF/PNG/JPEG under ``input_dir`` into ``workdir``.

    Writes ``images/``, ``ocr/``, ``tasks.json`` (image URLs served by Label
    Studio's local-files storage with ``workdir`` as document root) and
    ``label_config.xml``. Existing pages are re-processed.
    """
    s = settings or Settings()
    (workdir / "images").mkdir(parents=True, exist_ok=True)
    (workdir / "ocr").mkdir(parents=True, exist_ok=True)
    stats = PrepareStats()
    tasks: list[dict[str, Any]] = []

    files = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in files:
        stem = re.sub(r"[^\w.-]+", "_", str(path.relative_to(input_dir).with_suffix("")))
        try:
            pages, _ = load_pages_from_path(path, dpi=s.pdf_dpi, max_pages=s.max_pages, poppler_path=s.poppler_path)
        except Exception as exc:  # keep going; report at the end
            logger.warning("Skipping %s: %s", path, exc)
            stats.failed.append(str(path))
            continue
        stats.documents += 1
        for n, page in enumerate(pages, start=1):
            page_id = f"{stem}_p{n}"
            cleaned = preprocess_page(
                page,
                min_angle=s.deskew_min_angle,
                max_angle=s.deskew_max_angle,
                clip_limit=s.clahe_clip_limit,
                tile_grid=s.clahe_tile_grid,
            ).image
            words = run_ocr(cleaned, lang=s.tesseract_lang, config=s.tesseract_config,
                            min_confidence=s.ocr_min_confidence, tesseract_cmd=s.tesseract_cmd)
            cleaned.save(workdir / "images" / f"{page_id}.png")
            sidecar = {
                "page_id": page_id,
                "source": str(path),
                "width": cleaned.width,
                "height": cleaned.height,
                "words": [w.model_dump(mode="json") for w in words],
            }
            (workdir / "ocr" / f"{page_id}.json").write_text(json.dumps(sidecar))
            tasks.append({"data": {"image": f"/data/local-files/?d=images/{page_id}.png", "page_id": page_id}})
            stats.pages += 1

    (workdir / "tasks.json").write_text(json.dumps(tasks, indent=1))
    (workdir / "label_config.xml").write_text(label_config_xml())
    return stats


# --------------------------------------------------------------------------
# Convert
# --------------------------------------------------------------------------


@dataclass
class ConvertStats:
    pages: int = 0
    entities: int = 0
    unlabelled_tasks: int = 0
    empty_boxes: int = 0
    missing_ocr: list[str] = field(default_factory=list)


def page_id_from_task(task: dict[str, Any]) -> str:
    """Recover the page id from a task's data (``page_id`` field, else the image URL)."""
    data = task.get("data", {})
    if data.get("page_id"):
        return str(data["page_id"])
    url = urlparse(str(data.get("image", "")))
    path = parse_qs(url.query).get("d", [url.path])[0]
    return _UPLOAD_PREFIX.sub("", Path(unquote(path)).stem)


def latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    """The most recently updated annotation that was not skipped/cancelled."""
    anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "", default=None)


def assign_labels(
    words: Sequence[OCRWord], rectangles: Sequence[tuple[str, tuple[float, float, float, float]]]
) -> tuple[list[LabelledWord], int]:
    """Label words by the rectangle their centre falls in (the smallest, if several overlap).

    Args:
        words: OCR words in reading order.
        rectangles: ``(entity_type, (x0, y0, x1, y1))`` in the same pixel frame as the words.

    Returns:
        ``(labelled_words, empty_rectangles)``; a rectangle is empty when no word centre is inside it.
    """
    labelled = [LabelledWord(text=w.text, box=w.pixel_box) for w in words]
    hits = [0] * len(rectangles)
    for lw in labelled:
        cx, cy = (lw.box[0] + lw.box[2]) / 2, (lw.box[1] + lw.box[3]) / 2
        inside = [
            (abs((r[2] - r[0]) * (r[3] - r[1])), k)
            for k, (_, r) in enumerate(rectangles)
            if r[0] <= cx <= r[2] and r[1] <= cy <= r[3]
        ]
        if inside:
            _, k = min(inside)
            lw.entity_type, lw.entity_id = rectangles[k][0], k
            hits[k] += 1
    return labelled, sum(1 for h in hits if h == 0)


def rectangles_from_annotation(annotation: dict[str, Any], width: int, height: int
                               ) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Rectangle results (percent coordinates) converted to pixels of a ``width`` x ``height`` image."""
    rects: list[tuple[str, tuple[float, float, float, float]]] = []
    for res in annotation.get("result", []):
        value = res.get("value", {})
        labels = value.get("rectanglelabels") or []
        if res.get("type") != "rectanglelabels" or not labels:
            continue
        if labels[0] not in ENTITY_TYPES:
            logger.warning("Ignoring unknown label %r", labels[0])
            continue
        if abs(float(value.get("rotation", 0) or 0)) > 0.5:
            logger.warning("Rotated rectangle treated as axis-aligned (rotation %s)", value.get("rotation"))
        x0 = float(value["x"]) * width / 100
        y0 = float(value["y"]) * height / 100
        rects.append((labels[0], (x0, y0, x0 + float(value["width"]) * width / 100, y0 + float(value["height"]) * height / 100)))
    return rects


def convert(export_path: Path, workdir: Path, output: Path | None = None) -> ConvertStats:
    """Turn a Label Studio JSON export into training JSONL (default ``workdir/labelled.jsonl``)."""
    output = output or workdir / "labelled.jsonl"
    tasks = json.loads(export_path.read_text())
    stats = ConvertStats()
    records: list[dict[str, Any]] = []

    for task in tasks:
        annotation = latest_annotation(task)
        if annotation is None:
            stats.unlabelled_tasks += 1
            continue
        page_id = page_id_from_task(task)
        sidecar_path = workdir / "ocr" / f"{page_id}.json"
        if not sidecar_path.exists():
            stats.missing_ocr.append(page_id)
            continue
        sidecar = json.loads(sidecar_path.read_text())
        words = [OCRWord.model_validate(w) for w in sidecar["words"]]
        rects = rectangles_from_annotation(annotation, sidecar["width"], sidecar["height"])
        labelled, empty = assign_labels(words, rects)
        stats.empty_boxes += empty
        stats.entities += len(rects) - empty
        stats.pages += 1
        records.append(
            {
                "id": page_id,
                "image": os.path.relpath(workdir / "images" / f"{page_id}.png", output.parent),
                "words": [w.text for w in labelled],
                "boxes": [list(w.box) for w in labelled],
                "labels": bio_tags(labelled),
                "box_format": "pixel",
            }
        )

    write_records(output, records)
    return stats
