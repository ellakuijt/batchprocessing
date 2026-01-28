print("Loading file... Please wait")
import colour
import cv2 as cv
import numpy as np
from image_processing_D import create_rects, make_combined_image, make_boxes, resize_image
from utility import choose_method
from find_colorchecker_D import find_colorchecker
from tests import test_method, test_number_of_pictures, test_resize_scale
import pickle


def main():
    # User inputs
    number_of_pictures = test_number_of_pictures()
    method = test_method()
    order = choose_method(method)
    resize_scale = test_resize_scale()
    print('Loading picture')

    # Make combined image
    cropped_images, size, square_size = create_rects(number_of_pictures, resize_scale)
    ordered_images = [cropped_images[i] for i in order]
    combined_image, centre_positions = make_combined_image(ordered_images, size)
    # Find colorchecker
    reference_data = np.flip(np.loadtxt('reference.csv', skiprows=0, delimiter=','), 1)
    reshaped_reference = reference_data.reshape(5, 6,
                                                3)  # Change this if you use a different colorchecker (height, width)
    boxes = make_boxes(centre_positions, square_size)
    found_colorchecker = find_colorchecker(boxes, combined_image, reshaped_reference)
    camera_data = found_colorchecker.values.reshape(-1, 3)

    # Correct image and calculate correction matrix
    correction_matrix = colour.characterisation.matrix_colour_correction_Finlayson2015(camera_data / 255,
                                                                                       reference_data / 255, 2)
    corrected_image = colour.characterisation.apply_matrix_colour_correction_Finlayson2015(combined_image / 255,
                                                                                           correction_matrix, 2)

    # Show images and save the correction matrix
    cv.imwrite('Combined-image2-27-1.png', combined_image)
    cv.imwrite('Corrected-image2-27-1.png', 255 * corrected_image)
    # np.savetxt('matrix.csv', correction_matrix,
    #            delimiter=',')  # Note that this correction matrix works for BGR images, not RGB
    with open('low.pickle', 'wb') as f:
        pickle.dump(found_colorchecker, f)


if __name__ == "__main__":
    main()
