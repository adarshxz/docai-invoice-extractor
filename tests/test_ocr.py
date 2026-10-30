from __future__ import annotations

import pytest

from docai.ocr import normalize_box, words_from_tesseract_data


@pytest.mark.parametrize(
    ("box", "size", "expected"),
    [
        ((0, 0, 100, 50), (1000, 1000), (0, 0, 100, 50)),
        ((10, 20, 30, 40), (2000, 4000), (5, 5, 15, 10)),
        ((1, 1, 2, 2), (3, 3), (333, 333, 666, 666)),  # floor, not round
        ((-5, -1, 1200, 900), (1000, 800), (0, 0, 1000, 1000)),  # clamped
        ((0, 0, 1700, 2200), (1700, 2200), (0, 0, 1000, 1000)),
    ],
)
def test_normalize_box(box, size, expected):
    assert normalize_box(box, *size) == expected


def test_normalize_box_rejects_empty_image():
    with pytest.raises(ValueError):
        normalize_box((0, 0, 1, 1), 0, 10)


def test_words_from_tesseract_data_filters_and_normalises():
    data = {
        "level": [1, 4, 5, 5, 5, 5],
        "text": ["", "", "Invoice", "  ", "#123", "smudge"],
        "conf": ["-1", "-1", "96.5", "90", 88, 12.0],
        "left": [0, 10, 100, 300, 400, 500],
        "top": [0, 10, 50, 50, 50, 50],
        "width": [2000, 900, 150, 10, 80, 30],
        "height": [1000, 40, 30, 30, 30, 30],
        "block_num": [0, 1, 1, 1, 1, 1],
        "par_num": [0, 1, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 1, 1, 1],
    }
    words = words_from_tesseract_data(data, 2000, 1000, min_confidence=50)

    assert [w.text for w in words] == ["Invoice", "#123"]
    assert words[0].pixel_box == (100, 50, 250, 80)
    assert words[0].norm_box == (50, 50, 125, 80)
    assert words[1].line_key == (1, 1, 1)
