"""Backward-compatible re-exports. Prefer: circle_detection.model"""

from circle_detection.constants import DEFAULT_LEARNING_RATE, DEFAULT_MODEL_PATH, IMAGE_SIZE  # noqa: F401
from circle_detection.model import build_model, compile_model  # noqa: F401
