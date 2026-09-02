"""Synthetic noisy-circle data and IOU helpers."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from skimage.draw import circle_perimeter_aa


def draw_circle(image: np.ndarray, row: int, col: int, rad: int) -> np.ndarray:
    """Draw an anti-aliased circle perimeter onto `image` in-place."""
    rr, cc, val = circle_perimeter_aa(int(row), int(col), int(rad))
    valid = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    image[rr[valid], cc[valid]] = val[valid]
    return image


def noisy_circle(size: int, radius: int, noise: float) -> tuple[tuple[int, int, int], np.ndarray]:
    """Return ((row, col, rad), image) with a circle on a noisy background.

    Pixel values are floats in ``[0, 1]``. The circle may be partially off-image.
    """
    image = np.zeros((size, size), dtype=np.float32)
    image += noise * np.random.random(image.shape).astype(np.float32)

    row = int(np.random.randint(0, size))
    col = int(np.random.randint(0, size))
    # np.random.randint high is exclusive; keep a non-empty range.
    rad = int(np.random.randint(4, max(5, int(radius))))
    draw_circle(image, row, col, rad)
    return (row, col, rad), np.clip(image, 0.0, 1.0)


def iou(params0, params1) -> float:
    """Intersection-over-union of two (row, col, radius) circles."""
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    rad0 = max(float(rad0), 1.0)
    rad1 = max(float(rad1), 1.0)
    shape0 = Point(float(row0), float(col0)).buffer(rad0)
    shape1 = Point(float(row1), float(col1)).buffer(rad1)

    union = shape0.union(shape1).area
    if union <= 0:
        return 0.0
    return float(shape0.intersection(shape1).area / union)


def create_training_data(
    samples: int,
    image_size: int,
    max_radius: int,
    noise_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and labels in relative image coordinates.

    Labels are ``(row, col, radius)`` divided by ``image_size``, so the image
    center is about ``(0.5, 0.5)``.
    """
    training_images = np.zeros((samples, image_size, image_size), dtype=np.float32)
    training_labels = np.zeros((samples, 3), dtype=np.float64)

    for i in range(samples):
        params, image = noisy_circle(image_size, max_radius, noise_level)
        training_images[i] = image
        training_labels[i] = params

    training_labels /= image_size
    return training_images, training_labels
