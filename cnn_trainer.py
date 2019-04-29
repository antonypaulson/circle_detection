import numpy as np
import tensorflow as tf

import cnn_model
from shapes import noisy_circle, create_training_data

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    model_pathed_name = "./circle_detection_model"
    training_steps = 8000
    batch_size = 32
    eval_set_size = 500

    image_size = cnn_model.IMAGE_SIZE
    max_circle_radius = image_size // 2
    noise_level = .5

    train_data, train_labels = create_training_data(training_steps, image_size, max_circle_radius, noise_level)
    eval_data, eval_labels = create_training_data(eval_set_size, image_size, max_circle_radius, noise_level)

    tf_circle_detector = tf.estimator.Estimator(model_fn=cnn_model.model_function, model_dir=model_pathed_name)

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
