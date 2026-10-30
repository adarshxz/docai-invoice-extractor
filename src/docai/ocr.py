"""Stage 2: Tesseract OCR and bounding-box normalisation to LayoutLM's [0, 1000] grid."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image

from docai.schemas import BBox, OCRWord

_TESSERACT_WORD_LEVEL = 5


def _scale(value: float, extent: int) -> int:
    """``min(1000, max(0, floor(1000 * value / extent)))``."""
    return min(1000, max(0, math.floor(1000 * value / extent)))


def normalize_box(box: Sequence[float], width: int, height: int) -> BBox:
    """Map a pixel box ``(x_min, y_min, x_max, y_max)`` to integer [0, 1000] coordinates.

    Implements, for an image of width ``W`` and height ``H``::

        x_norm = min(1000, max(0, floor(1000 * x / W)))
        y_norm = min(1000, max(0, floor(1000 * y / H)))

    Raises:
        ValueError: If the image has a non-positive dimension.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")
    x0, y0, x1, y1 = box
    return (_scale(x0, width), _scale(y0, height), _scale(x1, width), _scale(y1, height))


def _int_field(data: Mapping[str, Sequence[Any]], key: str, i: int) -> int:
    column = data.get(key)
    return int(column[i]) if column is not None else 0


def words_from_tesseract_data(
    data: Mapping[str, Sequence[Any]],
    width: int,
    height: int,
    *,
    min_confidence: float = 0.0,
) -> list[OCRWord]:
    """Convert ``pytesseract.image_to_data(..., output_type=Output.DICT)`` into :class:`OCRWord` s.

    Non-word rows (page/block/line levels, which carry ``conf == -1``), blank
    strings and words below ``min_confidence`` are dropped. Reading order is
    Tesseract's.
    """
    words: list[OCRWord] = []
    for i, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        if not text:
            continue
        if "level" in data and int(data["level"][i]) != _TESSERACT_WORD_LEVEL:
            continue
        conf = float(data["conf"][i])
        if conf < 0 or conf < min_confidence:
            continue

        left, top = int(data["left"][i]), int(data["top"][i])
        right, bottom = left + int(data["width"][i]), top + int(data["height"][i])
        pixel_box: BBox = (left, top, right, bottom)
        words.append(
            OCRWord(
                text=text,
                pixel_box=pixel_box,
                norm_box=normalize_box(pixel_box, width, height),
                ocr_confidence=conf,
                block_num=_int_field(data, "block_num", i),
                par_num=_int_field(data, "par_num", i),
                line_num=_int_field(data, "line_num", i),
            )
        )
    return words


def run_ocr(
    image: Image.Image,
    *,
    lang: str = "eng",
    config: str = "--oem 3 --psm 3",
    min_confidence: float = 0.0,
    tesseract_cmd: str | None = None,
) -> list[OCRWord]:
    """Run Tesseract on a page and return normalised words.

    Args:
        tesseract_cmd: Path to the binary when it is not on ``PATH``.

    Raises:
        RuntimeError: If the ``tesseract`` binary is not installed.
    """
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError("Tesseract is not installed or not on PATH.") from exc
    width, height = image.size
    return words_from_tesseract_data(data, width, height, min_confidence=min_confidence)
