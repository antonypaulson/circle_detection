import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from circle_detection.infer import evaluate_predictions, predict_locations
from circle_detection.model import IMAGE_SIZE, build_model, compile_model
from circle_detection.shapes import create_training_data
from circle_detection.smoke import run_smoke


def test_model_output_shape():
    model = build_model()
    x = np.random.rand(2, IMAGE_SIZE, IMAGE_SIZE, 1).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 3)


def test_predict_locations_accepts_hw_images():
    model = build_model()
    images = np.random.rand(3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    preds = predict_locations(model, images)
    assert preds.shape == (3, 3)


def test_evaluate_predictions_returns_valid_iou():
    labels = np.array([[0.5, 0.5, 0.2], [0.2, 0.2, 0.1]], dtype=np.float64)
    preds = labels.copy()
    scores = evaluate_predictions(labels, preds, image_size=IMAGE_SIZE)
    assert scores.shape == (2,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert scores[0] > 0.99


def test_compile_and_one_train_step():
    model = compile_model(build_model())
    images, labels = create_training_data(8, IMAGE_SIZE, IMAGE_SIZE // 2, 0.5)
    x = images[..., np.newaxis]
    history = model.fit(x, labels.astype(np.float32), batch_size=4, epochs=1, verbose=0)
    assert "loss" in history.history


def test_smoke_train_and_infer(tmp_path):
    root = run_smoke(workdir=str(tmp_path))
    assert (root / "smoke.keras").exists()
    overlays = list((root / "overlays").glob("*_overlay.png"))
    assert overlays
