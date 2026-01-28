import numpy as np
from math import ceil
import cv2

def make_blur(image):
    """
    Makes a Gaussian blur using 21x21 kernels
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur


def make_bloodstain_mask(image, blurred_image, threshold=60):
    """
    Makes the bloodstain mask.
    """
    _, mask = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(np.uint8)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    return mask, masked_img


def create_background_mask(img, blur, threshold, iterations=1):
    """
    Old version of creating the background mask, can be deleted.
    """
    _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((21, 21), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=iterations)
    opened = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations * 3)
    erosion2 = cv2.erode(closed, kernel, iterations=iterations)
    masked_img = cv2.bitwise_and(img, img, mask=erosion2)
    return np.invert(erosion2), masked_img


def divide_mask(img, blur, threshold, ratio=0.7, inner=False, background=False):
    """
    Created all other masks. Select the right ratio and threshold values for your mask.
    """
    complete_mask, _ = make_bloodstain_mask(img, blur, threshold=threshold)
    mask, _ = make_bloodstain_mask(img, blur, threshold=threshold)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Created contours found in the image
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        if r > 400:  # Only if the circle is big enough will it get used
            if background:
                r *= 1.1  # So that the circle is always bigger than the bloodstain
                cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)  # Makes a white circle around the entire BS
            else:
                cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)  # Same as above
                cv2.circle(mask, (int(x), int(y)), int(r * ratio), (0, 0, 0), -1)  # Makes a black circle in the center of the BS
    if inner or background:
        mask = np.invert(mask)
    if not background:
        mask = cv2.bitwise_and(complete_mask, mask)
    bloodstain = cv2.bitwise_and(img, img, mask=mask)
    return mask, bloodstain


# ~~~~~~~~~~~~~~~~~~~~~~~~OLD FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_splice(filename, angle, threshold, iteration=1):
    img = cv2.imread(filename)
    rot_img = rotate_image(img, angle)
    blur = make_blur(rot_img)
    mask, _ = create_background_mask(img, blur, threshold, iterations=iteration)
    rot_mask = rotate_image(mask, angle)
    contours, _ = cv2.findContours(rot_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = map(ceil, cv2.boundingRect(contours[0]))
    splice = rot_img[y:y + h, x:x + w]
    return splice, h, w


def rotate_image(img, angle):
    shape = img.shape
    m = cv2.getRotationMatrix2D((shape[1] // 2, shape[0] // 2), angle, -1)
    rotated_img = cv2.warpAffine(img, m, (shape[1], shape[0]))
    return rotated_img


def resize_images(s1, s2, h1, h2, w1, w2):
    h, w = min(h1, h2), min(w1, w2)
    s1, s2 = cv2.resize(s1, (h, w)), cv2.resize(s2, (h, w))
    return s1, s2


def show_image(name, image):
    pic = cv2.resize(image, (800, 600))
    cv2.imshow(name, pic)
