import numpy as np

red_text = "\033[91m"  # For red warning texts


def choose_method(method):
    """
    Selects the order of colors based on the chosen method.

    This function takes a method parameter to determine the order of colors.
    It returns a list specifying the order in which colors should be processed. The
    available methods are:

    - Method 1: Orders colors from 0 to 30 for a 1x1 clicking method.
    - Method 2: A predefined order for a 2x2 clicking method.
    - Method 3: A predifined oredr for a 3x3 clicking method.
    - Method 4: A custom order that the user can modify. If it is None it raises a SystemExit.

    Parameters:
    method (int): The method chosen for ordering colors.

    Returns:
    list: A list specifying the order of colors

    Raises:
    SystemExit: If the method is 4 and no custom order has been filled.
    """
    if method == 1:
        order = np.arange(31)
    elif method == 2:
        order = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 12, 13, 16, 17, 20, 21, 14, 15, 18, 19, 22, 23, 24, 25, 26, 27,
                 28, 29]
    elif method == 3:
        order = [0, 1, 2, 9, 10, 11, 3, 4, 5, 12, 13, 14, 6, 7, 8, 15, 16, 17, 18, 19, 20, 24, 25, 26, 21, 22, 23, 27,
                 28, 29]
    elif method == 4:
        order = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23, 24, 25, 26,
                 27, 28, 29]  # Change this to the custom order you want, the length should be the same as the amount of squares on the colorcheker
        if order is None:
            print(f'{red_text}You have not filled in your custom order.\n'
                  f'{red_text}Please create a custom order before selecting method 4.')
            raise SystemExit
    return order
