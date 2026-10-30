from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from docai.preprocessing import deskew, enhance_contrast, estimate_skew_angle, preprocess_page, rotate


def synthetic_page(width: int = 1200, height: int = 1600) -> np.ndarray:
    """White page with left-aligned lines of black 'text' of varying length."""
    page = np.full((height, width), 255, np.uint8)
    rng = np.random.default_rng(0)
    for i, y in enumerate(range(150, height - 150, 60)):
        line = "INVOICE TOTAL AMOUNT DUE 1,234.56 " * 2
        n = int(rng.integers(15, len(line)))
        cv2.putText(page, line[:n], (100, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    return page


def test_straight_page_has_no_skew():
    assert abs(estimate_skew_angle(synthetic_page())) < 0.3


@pytest.mark.parametrize("angle", [-7.0, -3.0, 2.0, 5.0, 10.0])
def test_estimate_and_correct_skew(angle):
    # rotate() turns the page counter-clockwise by `angle`, i.e. a skew of -angle.
    skewed = rotate(synthetic_page(), -angle)
    estimated = estimate_skew_angle(skewed)
    assert estimated == pytest.approx(angle, abs=0.5)

    straightened, applied = deskew(skewed, max_angle=15)
    assert applied == pytest.approx(estimated)
    assert abs(estimate_skew_angle(straightened)) < 0.5


def test_deskew_ignores_tiny_and_huge_angles():
    page = synthetic_page()
    out, applied = deskew(page, min_angle=0.5)
    assert applied == 0.0 and out is page

    skewed = rotate(page, -8)
    out, applied = deskew(skewed, max_angle=5)
    assert applied == 0.0 and out is skewed


def test_blank_page_is_left_alone():
    blank = np.full((500, 400), 255, np.uint8)
    assert estimate_skew_angle(blank) == 0.0


def test_rotate_grows_canvas_with_white_border():
    page = np.zeros((100, 200), np.uint8)
    out = rotate(page, 30)
    assert out.shape[0] > 100 and out.shape[1] > 200
    assert out[0, 0] == 255


def test_enhance_contrast_stretches_low_contrast_image():
    flat = np.tile(np.linspace(110, 140, 256, dtype=np.uint8), (256, 1))
    out = enhance_contrast(flat)
    assert out.shape == flat.shape
    assert int(out.max()) - int(out.min()) > int(flat.max()) - int(flat.min())


def test_preprocess_page_returns_rgb_and_angle():
    skewed = rotate(synthetic_page(), -4)
    result = preprocess_page(Image.fromarray(skewed).convert("RGB"))
    assert result.image.mode == "RGB"
    assert result.skew_angle == pytest.approx(4, abs=0.5)
