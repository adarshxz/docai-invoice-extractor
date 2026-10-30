from __future__ import annotations

import io
from collections.abc import Sequence

import pytest
from PIL import Image

from docai.ocr import normalize_box
from docai.schemas import OCRWord, WordPrediction

PAGE_W, PAGE_H = 1000, 1400


def word(text: str, x0: int, y0: int, x1: int, y1: int, line: int = 1, block: int = 1) -> OCRWord:
    box = (x0, y0, x1, y1)
    return OCRWord(
        text=text,
        pixel_box=box,
        norm_box=normalize_box(box, PAGE_W, PAGE_H),
        ocr_confidence=95.0,
        block_num=block,
        line_num=line,
    )


def pred(w: OCRWord, label: str, conf: float = 0.95) -> WordPrediction:
    return WordPrediction(word=w, label=label, confidence=conf)


# A small invoice laid out the way Tesseract would read it, with gold labels.
INVOICE: list[tuple[OCRWord, str]] = [
    (word("Acme", 50, 40, 140, 70, line=1), "B-VENDOR_NAME"),
    (word("Supplies", 150, 40, 300, 70, line=1), "I-VENDOR_NAME"),
    (word("Ltd.", 310, 40, 370, 70, line=1), "I-VENDOR_NAME"),
    (word("Invoice", 600, 40, 720, 70, line=2), "O"),
    (word("#:", 725, 40, 750, 70, line=2), "O"),
    (word("INV-2024-0042", 760, 40, 950, 70, line=2), "B-INVOICE_NUMBER"),
    (word("Date:", 600, 90, 680, 120, line=3), "O"),
    (word("March", 690, 90, 780, 120, line=3), "B-DATE"),
    (word("5,", 785, 90, 815, 120, line=3), "I-DATE"),
    (word("2024", 820, 90, 900, 120, line=3), "I-DATE"),
    (word("Widget", 50, 400, 160, 430, line=4), "B-ITEM_DESCRIPTION"),
    (word("A", 165, 400, 185, 430, line=4), "I-ITEM_DESCRIPTION"),
    (word("2", 500, 400, 520, 430, line=4), "O"),
    (word("$60.00", 600, 400, 700, 430, line=4), "O"),
    (word("$120.00", 800, 400, 920, 430, line=4), "O"),
    (word("Gadget", 50, 450, 160, 480, line=5), "B-ITEM_DESCRIPTION"),
    (word("$30.50", 800, 450, 920, 480, line=5), "O"),
    (word("Tax", 600, 600, 680, 630, line=6), "O"),
    (word("$15.05", 800, 600, 920, 630, line=6), "B-TAX_AMOUNT"),
    (word("Total", 600, 650, 690, 680, line=7), "O"),
    (word("$165.55", 800, 650, 920, 680, line=7), "B-TOTAL_AMOUNT"),
]


class FakeClassifier:
    """Returns pre-set labels keyed by word text."""

    def __init__(self, labels: dict[str, str], confidence: float = 0.93) -> None:
        self.labels = labels
        self.confidence = confidence
        self.calls = 0

    def predict(self, image: Image.Image, words: Sequence[OCRWord]) -> list[WordPrediction]:
        self.calls += 1
        return [pred(w, self.labels.get(w.text, "O"), self.confidence) for w in words]


@pytest.fixture
def invoice_words() -> list[OCRWord]:
    return [w for w, _ in INVOICE]


@pytest.fixture
def fake_classifier() -> FakeClassifier:
    return FakeClassifier({w.text: label for w, label in INVOICE})


@pytest.fixture
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (PAGE_W // 4, PAGE_H // 4), "white").save(buf, format="PNG")
    return buf.getvalue()
