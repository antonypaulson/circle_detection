"""Shared defaults. Kept out of model.py so the CLI can parse args without TensorFlow."""

IMAGE_SIZE = 64
DEFAULT_MODEL_PATH = "circle_detection_model.keras"
DEFAULT_LEARNING_RATE = 0.08
DEFAULT_DROPOUT = 0.4
DEFAULT_STEPS = 8000
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_SAMPLES = 500
DEFAULT_NOISE = 0.5
DEFAULT_INFER_SAMPLES = 100
