import sys

import cv2 as cv
import numpy as np

import shapes

image_size = 200
circle_size = 100
params, image = shapes.noisy_circle(image_size, circle_size, 1)
image_uint8 = np.array(np.clip(image, 0, 1) * 256, dtype=np.uint8)

if __name__ == '__main__':
    while True:
        k = cv.waitKey(1)
        # Press j or f for next image, q or escape to quit
        if k == 'q' or k == 27 or k == 'Q' or k == 1048603 or k == 1048689:
            sys.exit("Quitting")
        elif k == 'j' or k == 102 or k == 'f' or k == 106 or k == 65363:
            params, image = shapes.noisy_circle(image_size, circle_size, .8)
            image_uint8 = np.array(image * 256, dtype=np.uint8)

        cv.imshow("corrected", image_uint8)