"""Keras port of the original TF1 Estimator circle-detector CNN.

Architecture is unchanged from the TensorFlow MNIST / LeNet demo the project
was adapted from: two conv+pool stages, a 1024-unit dense layer, dropout, and
a 3-unit regression head for (row, col, radius).
"""

from __future__ import annotations

from circle_detection import tf_config  # noqa: F401
import tensorflow as tf

from circle_detection.constants import DEFAULT_DROPOUT, DEFAULT_LEARNING_RATE, DEFAULT_MODEL_PATH, IMAGE_SIZE

# Re-export so `from circle_detection.model import IMAGE_SIZE` still works.
__all__ = [
    "IMAGE_SIZE",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_DROPOUT",
    "build_model",
    "compile_model",
]


def build_model(image_size: int = IMAGE_SIZE, dropout_rate: float = DEFAULT_DROPOUT) -> tf.keras.Model:
    """Build the original circle-detection CNN as a Keras model."""
    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="x")

    conv1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1",
    )(inputs)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool1")(conv1)

    conv2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv2",
    )(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, name="pool2")(conv2)

    flat = tf.keras.layers.Flatten(name="flatten")(pool2)
    dense = tf.keras.layers.Dense(1024, activation="relu", name="dense")(flat)
    dropout = tf.keras.layers.Dropout(dropout_rate, name="dropout")(dense)
    outputs = tf.keras.layers.Dense(3, name="location")(dropout)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="circle_detector")


def compile_model(model: tf.keras.Model, learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.Model:
    """Compile with the original SGD + MSE regression objective."""
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss="mse",
        metrics=["mse"],
    )
    return model
