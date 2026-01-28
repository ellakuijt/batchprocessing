import numpy as np
import cv2 as cv
from interactive_image_D import InteractiveImageEditor

red_text = "\033[91m"  # For red warning texts
reset_color = "\033[0m"  # To reset the color


def make_combined_image(cropped_image, size):
    """
    Create a combined image from a list of cropped images.

    This function arranges a list of cropped images into a grid to create a combined image.

    Args:
        cropped_image (list of numpy.ndarray): A list of cropped images to combine.
        size (int): The size of each cell in the grid for arranging images.

    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: The combined image with images arranged in a grid.
            - list of tuple: A list of center positions of each cell in the grid.
    """
    shape = (6, 5)  # Change this if you want to use a different colorchecker
    combined_image = np.zeros((size * shape[1], size * shape[0], 3), dtype=np.uint8)
    positions = []
    for row in range(shape[1]):
        for col in range(shape[0]):
            x_pos = col * size
            y_pos = row * size
            positions.append((x_pos, y_pos))
    center_positions = []
    for i in range(len(cropped_image)):
        x_pos, y_pos = positions[i]
        combined_image[y_pos:y_pos + size, x_pos:x_pos + size] = cropped_image[i]
    for pos in positions:
        center_positions.append((pos[0] + size // 2, pos[1] + size // 2))
    return combined_image, center_positions


def create_rects(number, scale):
    """
    Create a list of cropped images and return the images, width, and size.

    This function loads a series of images, allows the user to interactively select regions,
    and then crops and returns those regions.

    Args:
        number (int): The number of images to process.
        scale (int): Scaling factor for resizing the input images.

    Returns:
        tuple: A tuple containing three elements:
            - list of numpy.ndarray: A list of cropped images.
            - int: The width of the cropped images.
            - int: The size parameter used for creating rotated rectangles.

    Raises:
        SystemExit: If the user fails to click on 30 squares as required.
    """
    cropped_images = []
    positions = np.arange(1, number + 1)
    interactive_image = InteractiveImageEditor()
    interactive_image.create_slider()

    for i in positions:
        image_path = f'uv_pics/after/{i}.jpg'
        try:
            image = cv.imread(image_path)
            if image is None:
                raise FileNotFoundError(f'{red_text}Failed to load image: {image_path}.{reset_color}')
        except FileNotFoundError as e:
            print(f'{red_text}Error: {e}.{reset_color}')
            continue

        img = resize_image(image, scale)
        cv.imshow('Image', img)
        coordinates, size = interactive_image.create_interactive_image(img)
        cv.destroyAllWindows()
        boxes = make_boxes(coordinates, size, scale)

        for b in boxes:
            x, y = map(int, b.center)
            w, _ = b.size
            w = int(w // 2)
            cropped_images.append(image[y:y + w, x:x + w])  # Add the slice of the current image based on x, y and w to the list
        coordinates.clear()

    if not len(cropped_images) == 30:  # Change this if you want to use a different colorchecker with for example 24 squares
        print(f'{red_text}You need to click the 30 squares.\n'
              f'You clicked {len(cropped_images)} times.')
        raise SystemExit
    return cropped_images, w, size


def make_boxes(cors, size, scale=1):
    """
    Create a list of rotated rectangles based on the given coordinates, size, and scale.

    Args:
        cors (list of tuples): List of (x, y) coordinates.
        size (int): The size of the rectangles.
        scale (float, optional): Scaling factor for coordinates. Defaults to 1.

    Returns:
        list of cv2.RotatedRect: List of rotated rectangles.
    """
    rects = []
    for i in range(len(cors)):
        center = (cors[i][0] * scale, cors[i][1] * scale)  # Due to resizing sometimes the coordinates need to be multiplied by that resizing scale
        rect = cv.RotatedRect(center, (size * scale, size * scale), 0)  # Can be changed to a normal rect. No longer use rotations
        rects.append(rect)
    return rects


def resize_image(img, scale):
    """
    Resize an image based on the specified scaling factor.

    Args:
        img (numpy.ndarray): The input image as a NumPy array.
        scale (int): Scaling factor for resizing the image.

    Returns:
        numpy.ndarray: The resized image as a NumPy array.
    """
    height, width = img.shape[:2]
    img = cv.resize(img, (width // scale, height // scale))
    return img
