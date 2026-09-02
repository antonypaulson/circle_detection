"""Run inference on synthetic circles and report IOU statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from circle_detection.constants import DEFAULT_INFER_SAMPLES, DEFAULT_MODEL_PATH, DEFAULT_NOISE, IMAGE_SIZE
from circle_detection.shapes import create_training_data, iou
from circle_detection.train import prepare_xy
from circle_detection.visualize import save_overlay

if TYPE_CHECKING:
    import tensorflow as tf


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Trained Keras model to load (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_INFER_SAMPLES)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, write raw/overlay PNGs for each sample here.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-save",
        type=int,
        default=16,
        help="Maximum overlays to write when --save-dir is set.",
    )
    return parser


def load_model(model_path: str) -> tf.keras.Model:
    from circle_detection import tf_config  # noqa: F401
    import tensorflow as tf

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first:\n"
            f"  python -m circle_detection train --model-path {path}"
        )
    return tf.keras.models.load_model(path)


def predict_locations(model: Any, images: np.ndarray) -> np.ndarray:
    """Predict relative (row, col, radius) for images of shape (N, H, W) or (N, H, W, 1)."""
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return model.predict(x, verbose=0)


def evaluate_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """IOU per sample after converting relative coords back to pixels."""
    pred_px = image_size * np.asarray(predictions, dtype=np.float64)
    true_px = image_size * np.asarray(labels, dtype=np.float64)
    scores = np.zeros(len(pred_px), dtype=np.float64)
    for i, (truth, pred) in enumerate(zip(true_px, pred_px)):
        scores[i] = iou((truth[0], truth[1], max(truth[2], 1.0)), (pred[0], pred[1], pred[2]))
    return scores


def print_iou_stats(scores: np.ndarray) -> None:
    scores = np.asarray(scores)
    print()
    print(
        f"{len(scores)} samples. Average IOU: {np.average(scores):2.2f} "
        f"Min IOU: {np.min(scores):2.2f}"
    )
    print(f"Samples with IOU > .5: {100.0 * np.mean(scores > 0.5):.1f}%")
    print("------------------------")


def run_inference(
    model_path: str = DEFAULT_MODEL_PATH,
    samples: int = DEFAULT_INFER_SAMPLES,
    noise: float = DEFAULT_NOISE,
    image_size: int = IMAGE_SIZE,
    save_dir: str | None = None,
    seed: int | None = None,
    max_save: int = 16,
    model: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a model on fresh synthetic circles.

    Returns ``(images, labels, predictions, ious)``.
    """
    if seed is not None:
        np.random.seed(seed)

    if model is None:
        model = load_model(model_path)

    max_radius = image_size // 2
    images, labels = create_training_data(samples, image_size, max_radius, noise)
    x, _ = prepare_xy(images, labels)
    predictions = model.predict(x, verbose=0)
    scores = evaluate_predictions(labels, predictions, image_size=image_size)
    print_iou_stats(scores)

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        n_save = min(samples, max_save)
        for i in range(n_save):
            pred_px = image_size * predictions[i]
            save_overlay(
                images[i],
                pred_px,
                out / f"sample_{i:03d}_overlay.png",
                raw_path=out / f"sample_{i:03d}_raw.png",
            )
        print(f"Wrote {n_save} overlay images to {out}")

    return images, labels, predictions, scores


def run(args: argparse.Namespace):
    return run_inference(
        model_path=args.model_path,
        samples=args.samples,
        noise=args.noise,
        image_size=args.image_size,
        save_dir=args.save_dir,
        seed=args.seed,
        max_save=args.max_save,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run circle-detection inference on synthetic images.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
