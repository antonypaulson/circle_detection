"""Backward-compatible re-exports. Prefer: circle_detection.shapes"""

from circle_detection.shapes import (  # noqa: F401
    create_training_data,
    draw_circle,
    iou,
    noisy_circle,
)
