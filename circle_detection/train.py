"""Train the circle detector on synthetic noisy circles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from circle_detection.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MODEL_PATH,
    DEFAULT_NOISE,
    DEFAULT_STEPS,
    IMAGE_SIZE,
)
from circle_detection.shapes import create_training_data

if TYPE_CHECKING:
    import tensorflow as tf


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Where to write the trained Keras model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Optimizer steps. Original default is {DEFAULT_STEPS}.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Synthetic training images to generate (default: same as --steps).",
    )
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--noise", type=float, default=DEFAULT_NOISE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def prepare_xy(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(images, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(labels, dtype=np.float32)
    return x, y


def train(
    model_path: str = DEFAULT_MODEL_PATH,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    samples: int | None = None,
    eval_samples: int = DEFAULT_EVAL_SAMPLES,
    noise: float = DEFAULT_NOISE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    image_size: int = IMAGE_SIZE,
    seed: int | None = None,
) -> tf.keras.Model:
    """Generate synthetic data, train, and save ``model_path``.

    Matches the original Estimator loop: generate a finite set of images, then
    shuffle/repeat them for ``steps`` batches of ``batch_size``.
    """
    from circle_detection import tf_config  # noqa: F401
    import tensorflow as tf

    from circle_detection.model import build_model, compile_model

    if seed is not None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    n_train = samples if samples is not None else steps
    max_radius = image_size // 2

    print(f"Generating {n_train} training and {eval_samples} eval images ({image_size}x{image_size})...")
    train_images, train_labels = create_training_data(n_train, image_size, max_radius, noise)
    eval_images, eval_labels = create_training_data(eval_samples, image_size, max_radius, noise)

    x_train, y_train = prepare_xy(train_images, train_labels)
    x_eval, y_eval = prepare_xy(eval_images, eval_labels)

    model = compile_model(build_model(image_size=image_size), learning_rate=learning_rate)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train))
        .repeat()
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    eval_ds = tf.data.Dataset.from_tensor_slices((x_eval, y_eval)).batch(batch_size)

    print(f"Training for {steps} steps (batch_size={batch_size}, lr={learning_rate})...")
    model.fit(
        train_ds,
        steps_per_epoch=steps,
        epochs=1,
        validation_data=eval_ds,
        verbose=1,
        shuffle=False,
    )

    dest = Path(model_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(dest)
    print(f"Saved model to {dest}")
    return model


def run(args: argparse.Namespace):
    return train(
        model_path=args.model_path,
        steps=args.steps,
        batch_size=args.batch_size,
        samples=args.samples,
        eval_samples=args.eval_samples,
        noise=args.noise,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        seed=args.seed,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the circle-detection CNN on synthetic data.")
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
