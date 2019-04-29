import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa

def draw_circle(image, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = ((rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1]))
    image[rr[valid], cc[valid]] = val[valid]

    return image

def noisy_circle(size, radius, noise):
    image = np.zeros((size, size), dtype=np.float)
    image += noise * np.random.rand(*image.shape)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(4, max(4, radius))
    draw_circle(image, row, col, rad)

    return (row, col, rad), np.clip(image, 0, 1)

def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (shape0.intersection(shape1).area / shape0.union(shape1).area)

def create_training_data(samples, image_size, max_radius, noise_level):
    training_images = np.zeros((samples, image_size, image_size))
    training_labels = np.zeros((samples, 3), dtype=np.float64)

    image = np.zeros((image_size, image_size), dtype=np.float)
    for i in range(samples):
        params, image = noisy_circle(image_size, max_radius, noise_level)
        training_images[i, :, :] = image
        training_labels[i] = params

    # Normalize to relative image coordinates: every image has dimensions 1.0 x 1.0, pixels are stored as floats
    training_labels /= image_size

    return training_images, training_labels
