import numpy as np
import tensorflow as tf

from cnn_model import model_function
from shapes import noisy_circle

def create_training_data(samples, image_size, max_radius, noise_level):
    training_images = np.zeros((samples, image_size, image_size))
    training_labels = np.zeros((samples, 3), dtype=np.float64)

    image = np.zeros((image_size, image_size), dtype=np.float)
    for i in range(samples):
        params, image = noisy_circle(image_size, max_radius, noise_level)
        training_images[i, :, :] = image
        training_labels[i] = params

    # Normalize to relative image coordinates: every image is size 1.0 x 1.0, pixels are stored as floats
    # I like the idea of this, since it'd make generating a good loss function with different units easier.
    # But I had trouble getting convergence on earlier versions. Worth retrying now.
    #labels /= image_size

    return training_images, training_labels

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    model_pathed_name = "./circle_finder_v1"
    training_steps = 10
    batch_size = 32
    eval_set_size = 500

    image_size = 52
    max_circle_radius = 20
    noise_level = .2

    train_data, train_labels = create_training_data(training_steps, image_size, max_circle_radius, noise_level)
    eval_data, eval_labels = create_training_data(eval_set_size, image_size, max_circle_radius, noise_level)

    tf_circle_detector = tf.estimator.Estimator(model_fn=model_function, model_dir=model_pathed_name)

    tensors_to_log = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    tf_circle_detector.train(input_fn=train_input_fn, steps=training_steps, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = tf_circle_detector.evaluate(input_fn=eval_input_fn)

    inference_data, inference_labels = create_training_data(1, image_size, max_circle_radius, noise_level)
    print("-- Truth x, y, r: {0} {1} {2}".format(inference_labels[-1, 0], inference_labels[-1, 1], inference_labels[-1, 2]))

    predict_input_function = tf.estimator.inputs.numpy_input_fn(
        x={"x": inference_data},
        num_epochs=1,
        shuffle=False)

    inference_result = tf_circle_detector.predict(input_fn=predict_input_function)

    # Results will be returned as a generator, wrapped in the Tensorflow verbiage defined in PREDICT mode
    # We're only interested in doing this on a single circle at a time, rounded back to integers.
    inference_result = np.round(list(inference_result)[0]['location'])
    print("-- Predict x, y, r: {0} {1} {2}".format(inference_result[0], inference_result[1], inference_result[2]))
