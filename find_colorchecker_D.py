"""
Python-Macduff: "the Macbeth ColorChecker finder," ported to Python.

This Python module is a port of the original "Macduff" software, designed for detecting the Macbeth ColorChecker in images. The original C++ code for Macduff is available at: https://github.com/ryanfb/macduff/

Please note that this module is a Python adaptation of the Macduff ColorChecker detector, and the original C++ code is not authored by the developer of this Python version.

Usage:
- Use this module to detect the Macbeth ColorChecker in digital images.
- It provides functionality for color checker identification and analysis in photographs.

For more information on how to use this module and its features, please refer to the documentation and examples provided in the repository at: https://github.com/mathandy/python-macduff-colorchecker-detector/blob/master/macduff.py

Disclaimer: This module is not authored by the developer and is copied from the above-mentioned repository.

Original C++ code repository: https://github.com/ryanfb/macduff/
"""
import cv2 as cv
from numpy.linalg import norm
from math import sqrt
import numpy as np


def crop_patch(center, size, image):
    """Returns mean color in intersection of `image` and `rectangle`."""
    x, y = center - np.array(size) / 2
    w, h = size
    x0, y0, x1, y1 = map(round, [x, y, x + w, y + h])
    return image[int(max(y0, 0)): int(min(y1, image.shape[0])), int(max(x0, 0)): int(min(x1, image.shape[1]))]


def check_colorchecker(values, expected):
    """Find deviation of colorchecker `values` from expected values."""
    diff = (values - expected).ravel(order='K')
    return sqrt(np.dot(diff, diff))


class ColorChecker:
    def __init__(self, error, values, points, size):
        self.error = error
        self.values = values
        self.points = points
        self.size = size


def find_colorchecker(boxes, image, expected, use_patch_std=True, macbeth_width=6, macbeth_height=5):  # Change width and height if you want to use a different colorchecker
    points = np.array([[box.center[0], box.center[1]] for box in boxes])
    passport_box = cv.minAreaRect(points.astype('float32'))
    _, _, a = passport_box
    box_corners = cv.boxPoints(passport_box)

    # sort `box_corners` to be in order TL, TR, BR, BL
    top_corners = sorted(enumerate(box_corners), key=lambda contours: contours[1][1])[:2]
    top_left_idx = min(top_corners, key=lambda contours: contours[1][0])[0]
    box_corners = np.roll(box_corners, -top_left_idx, 0)
    tl, tr, br, bl = box_corners

    landscape_orientation = True  # `passport_box` is wider than tall
    if norm(tr - tl) < norm(bl - tl):
        landscape_orientation = False

    average_size = int(sum(min(box.size) for box in boxes) / len(boxes))
    if landscape_orientation:
        dx = (tr - tl) / (macbeth_width - 1)
        dy = (bl - tl) / (macbeth_height - 1)
    else:
        dx = (bl - tl) / (macbeth_width - 1)
        dy = (tr - tl) / (macbeth_height - 1)

    # calculate the averages for our oriented colorchecker
    checker_dims = (macbeth_height, macbeth_width)
    patch_values = np.empty(checker_dims + (3,), dtype='float32')
    patch_points = np.empty(checker_dims + (2,), dtype='float32')
    sum_of_patch_stds = np.array((0.0, 0.0, 0.0))
    for x in range(macbeth_width):
        for y in range(macbeth_height):
            center = tl + x * dx + y * dy

            img_patch = crop_patch(center, [average_size] * 2, image)

            if not landscape_orientation:
                y = macbeth_height - 1 - y

            patch_points[y, x] = center
            patch_values[y, x] = img_patch.mean(axis=(0, 1))
            sum_of_patch_stds += img_patch.std(axis=(0, 1))

    # determine which orientation has lower error
    orient_1_error = check_colorchecker(patch_values, expected)
    orient_2_error = check_colorchecker(patch_values[::-1, ::-1], expected)

    if orient_1_error > orient_2_error:  # rotate by 180 degrees
        patch_values = patch_values[::-1, ::-1]
        patch_points = patch_points[::-1, ::-1]

    if use_patch_std:
        error = sum_of_patch_stds.mean() / (macbeth_width * macbeth_height)
    else:
        error = min(orient_1_error, orient_2_error)

    return ColorChecker(error=error,
                        values=patch_values,
                        points=patch_points,
                        size=average_size)
