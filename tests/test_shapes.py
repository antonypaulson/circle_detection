import numpy as np
import pytest

from circle_detection.shapes import create_training_data, iou, noisy_circle


def test_noisy_circle_shape_and_range():
    params, image = noisy_circle(64, 32, 0.5)
    row, col, rad = params
    assert image.shape == (64, 64)
    assert image.dtype == np.float32
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert 0 <= row < 64
    assert 0 <= col < 64
    assert rad >= 4


def test_identical_circles_have_iou_one():
    assert iou((32, 32, 10), (32, 32, 10)) == pytest.approx(1.0)


def test_disjoint_circles_have_low_iou():
    assert iou((8, 8, 4), (50, 50, 4)) < 0.01


def test_iou_clamps_invalid_radius():
    score = iou((10, 10, 0), (10, 10, 0))
    assert 0.0 <= score <= 1.0


def test_create_training_data_normalizes_labels():
    images, labels = create_training_data(4, 64, 32, 0.5)
    assert images.shape == (4, 64, 64)
    assert labels.shape == (4, 3)
    assert np.all(labels[:, 0] < 1.0)
    assert np.all(labels[:, 1] < 1.0)
    assert np.all(labels[:, 2] > 0)
