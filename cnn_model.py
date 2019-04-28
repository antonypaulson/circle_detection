import tensorflow as tf
import numpy as np

IMAGE_SIZE = 52

# Adapted from Tensorflow mnist digit recognition example:
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/estimators/cnn.ipynb
def model_function(features, labels, mode):
    """Model function for CNN."""
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    # Convolutional, pooling layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional, pooling layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense layer, 1024 units is unchanged from example
    pool2_flat = tf.reshape(pool2, [-1, 13 * 13 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output layer, very simple
    output_layer = tf.layers.dense(inputs=dropout, units=3)
    predictions = {"location": output_layer}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # This has to be after the PREDICT part, since when in PREDICT mode the labels will be None.
    loss = tf.losses.mean_squared_error(labels, output_layer)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"MSE": tf.metrics.mean_squared_error(labels=labels, predictions=output_layer)}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
