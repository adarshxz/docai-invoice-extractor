"""Helpers shared by the dataset converters."""

from __future__ import annotations

import io
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from PIL import Image

from docai.labels import OUTSIDE
from docai.ocr import normalize_box
from docai.preprocessing import enhance_contrast, to_grayscale
from docai.schemas import BBox


@dataclass
class LabelledWord:
    """A word, its pixel box, and the entity instance it belongs to (if any)."""

    text: str
    box: BBox
    entity_type: str | None = None
    entity_id: int | None = None


def reading_order(words: Sequence[LabelledWord]) -> list[LabelledWord]:
    """Sort words top-to-bottom into rows, then left-to-right within a row.

    A word joins the current row when its vertical centre is within half a
    line height of the row's mean centre. Using the row mean rather than one
    word keeps "Supply" (descenders make its box taller) on the same row as
    "Tidewater".
    """
    rows: list[list[LabelledWord]] = []
    row_center = row_height = 0.0
    for w in sorted(words, key=lambda w: (w.box[1] + w.box[3]) / 2):
        center, height = (w.box[1] + w.box[3]) / 2, max(w.box[3] - w.box[1], 1)
        if rows and abs(center - row_center) <= 0.5 * min(height, row_height):
            rows[-1].append(w)
            n = len(rows[-1])
            row_center += (center - row_center) / n
            row_height += (height - row_height) / n
        else:
            rows.append([w])
            row_center, row_height = center, height
    return [w for row in rows for w in sorted(row, key=lambda w: w.box[0])]


def bio_tags(words: Sequence[LabelledWord]) -> list[str]:
    """BIO tags from entity membership; a word opens an entity unless it continues the previous word's."""
    tags: list[str] = []
    prev_id: int | None = None
    for w in words:
        if w.entity_type is None or w.entity_id is None:
            tags.append(OUTSIDE)
            prev_id = None
            continue
        tags.append(f"{'I' if w.entity_id == prev_id else 'B'}-{w.entity_type}")
        prev_id = w.entity_id
    return tags


def split_line(text: str, box: Sequence[int]) -> list[tuple[str, BBox]]:
    """Split a text-line box into word boxes, apportioning width by character count.

    Datasets such as SROIE annotate whole lines; LayoutLMv3 works best with
    word-level boxes, and character-proportional splitting is a close
    approximation for printed text.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    if not text.strip():
        return []
    width = max(x1 - x0, 1)
    n_chars = max(len(text), 1)
    out: list[tuple[str, BBox]] = []
    pos = 0
    for token in text.split():
        start = text.index(token, pos)
        end = start + len(token)
        pos = end
        out.append((token, (x0 + width * start // n_chars, y0, x0 + width * end // n_chars, y1)))
    return out


def normalize_training_image(image: Image.Image, max_side: int = 1600) -> tuple[Image.Image, float]:
    """Grayscale + CLAHE (as at inference, minus deskew) and downscale.

    Deskew is skipped because it would move the annotated boxes. Returns the
    image and the scale factor applied to it.
    """
    gray = enhance_contrast(to_grayscale(image.convert("RGB")))
    out = Image.fromarray(gray)
    scale = min(1.0, max_side / max(out.size))
    if scale < 1.0:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.Resampling.LANCZOS)
    return out, scale


def decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return img.copy()


class JsonlWriter:
    """Writes one page per line plus its image, with the image path relative to the JSONL file."""

    def __init__(self, jsonl_path: Path, image_dir: Path, *, normalize_images: bool = True, max_side: int = 1600) -> None:
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        self.normalize_images = normalize_images
        self.max_side = max_side
        self.count = 0
        self._fh: IO[str] | None = None

    def __enter__(self) -> "JsonlWriter":
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self._fh = self.jsonl_path.open("w")
        return self

    def __exit__(self, *_: object) -> None:
        if self._fh:
            self._fh.close()

    def write(self, page_id: str, image: Image.Image, words: Sequence[LabelledWord]) -> None:
        """Save the image and append a record with boxes normalised to [0, 1000]."""
        assert self._fh is not None, "use JsonlWriter as a context manager"
        width, height = image.size  # boxes are in original-image pixels
        if self.normalize_images:
            image, _ = normalize_training_image(image, self.max_side)
        else:
            image = image.convert("RGB")
        image_path = self.image_dir / f"{page_id}.jpg"
        image.save(image_path, quality=90)

        record = {
            "id": page_id,
            "image": os.path.relpath(image_path, self.jsonl_path.parent),
            "words": [w.text for w in words],
            "boxes": [list(normalize_box(w.box, width, height)) for w in words],
            "labels": bio_tags(words),
            "box_format": "normalized",
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1


def write_records(path: Path, records: Iterable[dict[str, object]]) -> int:
    """Write already-built JSONL records; returns how many were written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
