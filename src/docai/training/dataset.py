"""Annotated-invoice dataset for token classification.

Each line of a JSONL file is one page::

    {
      "image": "pages/inv_001.png",          # relative to the JSONL file
      "words": ["Invoice", "#", "INV-001", ...],
      "boxes": [[x0, y0, x1, y1], ...],      # one per word
      "labels": ["O", "O", "B-INVOICE_NUMBER", ...],
      "box_format": "pixel"                  # or "normalized" if already in [0, 1000]
    }

Word labels are aligned to sub-tokens by the LayoutLMv3 processor: the first
sub-token of a word carries the word's label, the rest get ``-100`` and are
ignored by the loss. Every sub-token carries its word's normalised box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

from docai.labels import IGNORE_INDEX, LABEL2ID
from docai.ocr import normalize_box


@dataclass(frozen=True)
class AnnotatedPage:
    image_path: Path
    words: list[str]
    boxes: list[tuple[int, int, int, int]]
    labels: list[str]
    box_format: Literal["pixel", "normalized"] = "pixel"


def read_jsonl(path: str | Path) -> list[AnnotatedPage]:
    """Load and validate annotations.

    Raises:
        ValueError: On length mismatches or unknown labels, naming the line.
    """
    path = Path(path)
    pages: list[AnnotatedPage] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        words, boxes, labels = rec["words"], rec["boxes"], rec["labels"]
        if not len(words) == len(boxes) == len(labels):
            raise ValueError(f"{path}:{lineno}: words/boxes/labels lengths differ")
        if unknown := set(labels) - LABEL2ID.keys():
            raise ValueError(f"{path}:{lineno}: unknown labels {sorted(unknown)}")
        pages.append(
            AnnotatedPage(
                image_path=(path.parent / rec["image"]).resolve(),
                words=[str(w) for w in words],
                boxes=[tuple(int(v) for v in b) for b in boxes],  # type: ignore[misc]
                labels=labels,
                box_format=rec.get("box_format", "pixel"),
            )
        )
    return pages


class InvoiceTokenDataset(Dataset[dict[str, Any]]):
    """Encodes :class:`AnnotatedPage` s on the fly for ``LayoutLMv3ForTokenClassification``."""

    def __init__(self, pages: list[AnnotatedPage], processor: Any, max_length: int = 512) -> None:
        self.pages = pages
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        page = self.pages[idx]
        image = Image.open(page.image_path).convert("RGB")
        if page.box_format == "pixel":
            boxes = [list(normalize_box(b, *image.size)) for b in page.boxes]
        else:
            boxes = [list(b) for b in page.boxes]

        encoding = self.processor(
            image,
            page.words,
            boxes=boxes,
            word_labels=[LABEL2ID[label] for label in page.labels],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def collate_batch(features: list[dict[str, Any]], pad_token_id: int = 1) -> dict[str, torch.Tensor]:
    """Pad a batch to its longest page instead of always to 512 tokens.

    Padding positions get ``pad_token_id``, a zero box, attention 0 and the
    ignored label, matching what the processor itself would produce.
    """
    longest = max(f["input_ids"].shape[0] for f in features)
    pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "labels": IGNORE_INDEX, "bbox": 0}
    batch: dict[str, torch.Tensor] = {}
    for key, value in pad_values.items():
        rows = []
        for f in features:
            t = f[key]
            missing = longest - t.shape[0]
            if missing:
                pad_shape = (missing, *t.shape[1:])
                t = torch.cat([t, torch.full(pad_shape, value, dtype=t.dtype)])
            rows.append(t)
        batch[key] = torch.stack(rows)
    batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    return batch
