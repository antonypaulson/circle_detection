"""Shared rendering helpers for overlays (used by infer and the viewer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2 as cv
except ImportError as exc:  # pragma: no cover - optional at import for unit tests
    cv = None
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None


def require_cv2():
    if cv is None:
        raise ImportError(
            "OpenCV is required for image overlays. Install opencv-python-headless "
            "or opencv-python."
        ) from _CV_IMPORT_ERROR
    return cv


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)


def draw_prediction_overlay(
    image: np.ndarray,
    row: float,
    col: float,
    rad: float,
    color=(200, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the predicted circle on a BGR overlay. (row, col) are matrix coords."""
    cv2 = require_cv2()
    gray = image_to_uint8(np.squeeze(image))
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    radius = max(int(round(rad)), 1)
    center = (int(round(col)), int(round(row)))
    cv2.circle(overlay, center, radius, color=color, thickness=thickness)
    return overlay


def upscale(image: np.ndarray, factor: int = 6) -> np.ndarray:
    cv2 = require_cv2()
    return cv2.resize(
        image,
        dsize=(0, 0),
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_NEAREST,
    )


def save_overlay(image: np.ndarray, prediction_px, overlay_path, raw_path=None, factor: int = 6) -> None:
    cv2 = require_cv2()
    row, col, rad = prediction_px
    overlay = upscale(draw_prediction_overlay(image, row, col, rad), factor=factor)
    raw = upscale(image_to_uint8(np.squeeze(image)), factor=factor)
    Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    if raw_path is not None:
        cv2.imwrite(str(raw_path), raw)
