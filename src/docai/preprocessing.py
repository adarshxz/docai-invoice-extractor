"""Stage 1b: deskew and contrast-enhance scanned pages with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessedPage:
    """A cleaned page plus the correction that was applied to it."""

    image: Image.Image
    skew_angle: float


def to_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return a single-channel ``uint8`` array from a PIL image or RGB/gray array."""
    arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _long_side_angle(rect: tuple[tuple[float, float], tuple[float, float], float]) -> float:
    """Angle in degrees, in ``(-90, 90]``, of a ``cv2.minAreaRect`` box's long side.

    OpenCV changed its ``minAreaRect`` angle convention in 4.5.1; working from
    the long side makes the result independent of that.
    """
    (_, _), (w, h), angle = rect
    if w < h:
        angle += 90.0
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate page skew in degrees using ``cv2.minAreaRect``.

    Text is binarised and smeared horizontally so each text line becomes one
    elongated blob; ``minAreaRect`` gives each blob's orientation and the
    length-weighted median is the page skew. A positive result means the text
    runs downhill to the right (clockwise on screen).

    Returns 0.0 when there is not enough text to measure.
    """
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 40), 3))
    smeared = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(smeared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles: list[float] = []
    weights: list[float] = []
    min_len = max(20.0, w * 0.05)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        long_side, short_side = max(rect[1]), min(rect[1])
        if long_side < min_len or long_side < 3 * max(short_side, 1.0):
            continue  # not line-shaped: specks, logos, table borders' corners
        angle = _long_side_angle(rect)
        if abs(angle) > 45:
            continue
        angles.append(angle)
        weights.append(long_side)

    if angles:
        order = np.argsort(angles)
        cum = np.cumsum(np.asarray(weights)[order])
        return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])

    # Fallback: one rectangle around every foreground pixel.
    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < 50:
        return 0.0
    angle = _long_side_angle(cv2.minAreaRect(coords))
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return float(angle)


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees, growing the canvas so nothing is cropped.

    New border pixels are filled with white.
    """
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    matrix[0, 2] += new_w / 2.0 - center[0]
    matrix[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        gray, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


def deskew(gray: np.ndarray, *, min_angle: float = 0.1, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    """Straighten a grayscale page.

    Args:
        gray: Grayscale page.
        min_angle: Skews smaller than this are left alone (avoids needless resampling).
        max_angle: Skews larger than this are assumed to be misestimates and left alone.

    Returns:
        ``(image, applied_angle)``; ``applied_angle`` is 0.0 if nothing was done.
    """
    angle = estimate_skew_angle(gray)
    if abs(angle) < min_angle or abs(angle) > max_angle:
        return gray, 0.0
    # A clockwise skew of +angle is undone by a counter-clockwise rotation of +angle.
    return rotate(gray, angle), angle


def enhance_contrast(gray: np.ndarray, *, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Apply CLAHE (contrast-limited adaptive histogram equalisation)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def preprocess_page(
    page: Image.Image,
    *,
    min_angle: float = 0.1,
    max_angle: float = 15.0,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> PreprocessedPage:
    """Grayscale -> deskew -> CLAHE, returned as an RGB PIL image.

    The same cleaned image is fed to both OCR and the model, so OCR boxes and
    image patches stay in one coordinate frame.
    """
    gray = to_grayscale(page)
    gray, angle = deskew(gray, min_angle=min_angle, max_angle=max_angle)
    gray = enhance_contrast(gray, clip_limit=clip_limit, tile_grid=tile_grid)
    return PreprocessedPage(image=Image.fromarray(gray).convert("RGB"), skew_angle=angle)
