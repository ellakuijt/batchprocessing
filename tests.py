import numpy as np

red_text = "\033[91m"  # For red warning texts
reset_color = "\033[0m"  # To reset the color


def test_method():
    """
    Prompt the user to input a method number and validates it.

    Returns:
        int: The selected method number (1, 2, 3, or 4).
    """
    while True:
        method = input('Enter the method number: ')
        if method.isdigit() and int(method) in np.arange(1, 5):
            return int(method)
        else:
            print(f'{red_text}Please enter a valid method number (1, 2, 3, or 4).{reset_color}')


def test_number_of_pictures():
    """
    Prompt the user to input the number of pictures and validates it.

    Returns:
        int: The number of pictures (1 to 30) selected by the user.
    """
    while True:
        number_of_pictures = input('Enter the number of pictures: ')
        if not number_of_pictures.isdigit():
            print(f'{red_text}Please enter a valid number.{reset_color}')
        else:
            number = int(number_of_pictures)
            if number not in range(1, 31):
                print(f'{red_text}Please select a number of pictures between 1 and 30.{reset_color}')
            elif number in [1, 4, 9, 30]:
                return number
            else:
                print(f'{red_text}You have entered a non-recommended number of pictures.\n'
                      f'Take care to select the squares in the right order.{reset_color}')
                return number


def test_resize_scale():
   """
    Prompt the user to input the resize scale and validates it.

    Returns:
        int: The resize scale (a positive integer).
    """
   while True:
        resize_scale = input('Enter the resize scale: ')
        if not resize_scale.isdigit():
            print(f'{red_text}Please enter a valid integer for resize scale.{reset_color}')
        else:
            resize_scale = int(resize_scale)
            if resize_scale > 5:
                print(
                    f'{red_text}Warning: A resize scale greater than 5 may result in significant quality loss.{reset_color}')
                return resize_scale
            elif resize_scale <= 0:
                print(f'{red_text}Resize scale must be a positive integer.{reset_color}')
            else:
                return resize_scale
