import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = ((rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1]))
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(2, max(2, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (shape0.intersection(shape1).area / shape0.union(shape1).area)

def create_training_data(samples, image_size):
    training = np.zeros((samples, image_size, image_size))
    labels = np.zeros((samples, 3), dtype=np.float64)

    image = np.zeros((image_size, image_size), dtype=np.float)
    for i in range(samples):
        params, image = noisy_circle(image_size, 20, 2)
        training[i, :, :] = image
        labels[i] = params
        #labels[:, 1] = params[1]
        #labels[:, 2] = params[2]

    # Normalize to relative image coordinates: every image is size 1.0 x 1.0, pixels are stored as floats
    #labels /= image_size

    return training, labels

# Adapted from Tensorflow mnist digit recognition example:
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/estimators/cnn.ipynb
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 52, 52, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 13 * 13 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    #logits = tf.layers.dense(inputs=dropout, units=10)

    output_layer = tf.layers.dense(inputs=dropout, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        #"classes": tf.argmax(input=output_layer, axis=1),
        "location": output_layer,
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        #"probabilities": tf.nn.softmax(output_layer, name="softmax_tensor")
    }


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=output_layer)
    # https://stackoverflow.com/questions/49703387/tensorflow-valueerror-shapes-1-and-are-incompatible/50180397
    # this was what worked for 1d
    #loss = tf.losses.mean_squared_error(tf.expand_dims(labels, axis=1), output_layer)
    loss = tf.losses.mean_squared_error(labels, output_layer)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    #eval_metric_ops = {"MSE": tf.metrics.mean_squared_error(labels=labels, predictions=predictions['radius'])}\
    eval_metric_ops = {"MSE": tf.metrics.mean_squared_error(labels=labels, predictions=output_layer)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    training_steps = 3000
    image_size = 52

    # Load training and eval data
    train_data, train_labels = create_training_data(training_steps, image_size)
    eval_data, eval_labels = create_training_data(500, image_size)

    print(eval_labels)
    #((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    print("training data shape: ", train_data.shape)
    print("training labels shape: ", train_labels.shape)

    #train_data = train_data / np.float32(255)
    #train_labels = train_labels.astype(np.int32)  # not required

    #eval_data = eval_data / np.float32(255)
    #eval_labels = eval_labels.astype(np.int32)  # not required

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./circle_finder_v24")

    # Set up logging for predictions
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=32,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn, steps=training_steps, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(" 8888888", eval_results)

    inference_data, inference_labels = create_training_data(1, image_size)
    print(inference_labels.shape)
    print("-- Truth x, y, r: {0} {1} {2}".format(inference_labels[-1, 0], inference_labels[-1, 1], inference_labels[-1, 2]))

    predict_input_function = tf.estimator.inputs.numpy_input_fn(
        x={"x": inference_data},
        num_epochs=1,
        shuffle=False)

    inference_result = mnist_classifier.predict(input_fn=predict_input_function)
    # Results will be returned as a generator, wrapped in the Tensorflow verbiage defined in PREDICT mode
    # We're only interested in doing this on a single circle at a time, rounded back to integers.
    inference_result = np.round(list(inference_result)[0]['location'])
    print("-- Predict x, y, r: {0} {1} {2}".format(inference_result[0], inference_result[1], inference_result[2]))


