import sys

import cv2 as cv
import numpy as np
import tensorflow as tf

import shapes
import cnn_model

if __name__ == '__main__':
    number = 100
    model_path_name = "./circle_detection_model"
    image_size = cnn_model.IMAGE_SIZE
    max_radius = 28
    noise_level = .5

    tf.logging.set_verbosity(tf.logging.ERROR)
    inference_data, inference_labels = shapes.create_training_data(number, image_size, max_radius, noise_level)

    predict_input_function = tf.estimator.inputs.numpy_input_fn(
        x={"x": inference_data},
        num_epochs=1,
        shuffle=False)
    tf_circle_detector = tf.estimator.Estimator(model_fn=cnn_model.model_function, model_dir=model_path_name)
    inference_result = tf_circle_detector.predict(input_fn=predict_input_function)

    # Results will be returned from Tensorflow as a generator,
    # wrapped in the Tensorflow verbiage defined in PREDICT mode
    inference_results = list(inference_result)

    # Batch statistics: average IOU. Doing this separately from the display loop below because I don't expect
    # that to actually accumulate all results
    ious = np.zeros(number)
    for i, prediction in enumerate(inference_results):
        prediction = cnn_model.IMAGE_SIZE * prediction['location']
        true_x, true_y, true_r = cnn_model.IMAGE_SIZE * inference_labels[i]
        ious[i] = shapes.iou((true_x, true_y, true_r), (prediction[0], prediction[1], prediction[2]))
    print()
    print("{0} samples. Average IOU: {1:2.2f} Min IOU: {2:2.2f} \n".format(number, np.average(ious), np.min(ious)))

    for i, prediction in enumerate(inference_results):
        # Unwrap the Tensorflow output
        prediction = cnn_model.IMAGE_SIZE * prediction['location']
        prediction_int = [int(round(prediction[0])), int(round(prediction[1])), int(round(prediction[2]))]

        image = np.array(np.reshape(inference_data[i], (image_size, image_size)) * 256, dtype=np.uint8)

        overlay_image = cv.cvtColor(np.copy(image), cv.COLOR_GRAY2BGR)
        # Reverse the order of the coordinates to go from matrix indexing to image indexing
        cv.circle(overlay_image, (prediction_int[1], prediction_int[0]), prediction_int[2], color=(200, 0, 255), thickness=2)
        # Declare these outside the imshow() loop so they can be resized. See https://stackoverflow.com/questions/24842382/fitting-an-image-to-screen-using-imshow-opencv
        cv.namedWindow("Raw Image", cv.WINDOW_NORMAL)
        cv.namedWindow("Overlay Image", cv.WINDOW_NORMAL)

        true_x, true_y, true_r = cnn_model.IMAGE_SIZE * inference_labels[i]
        print("True x, y, r: {0} {1} {2}".format(int(true_x), int(true_y), int(true_r)))
        print("Pred x, y, r: {0} {1} {2}".format(prediction_int[0], prediction_int[1], prediction_int[2]))
        print("IOU: {0:2.2f} \n".format(shapes.iou((true_x, true_y, true_r), (prediction[0], prediction[1], prediction[2]))))

        while True:
            k = cv.waitKey(1)
            # Press j or f for next image, q or escape to quit
            if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
                sys.exit("Quitting")
            elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
                break

            cv.imshow("Raw Image", image)
            cv.resizeWindow("Raw Image", 400, 400)
            cv.imshow("Overlay Image", overlay_image)
            cv.resizeWindow("Overlay Image", 400, 400)
