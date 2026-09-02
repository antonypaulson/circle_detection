"""Backward-compatible re-exports. Prefer: circle_detection.model"""

from circle_detection.model import (  # noqa: F401
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    build_model,
    compile_model,
)
