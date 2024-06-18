import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

CV_COLORS = {
    "rgb": cv2.COLOR_BGR2RGB,
    "gray": cv2.COLOR_BGR2GRAY,
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
}


def show_image(image: np.ndarray, title: str = "", color: str = "rgb"):
    """
    Show an image with a specified title and color space.

    Args:
        image (np.ndarray): The image to display.
        title (str): The title of the displayed image.
            Default is an empty string.
        color (str): The color space of the image.
            Default is "rgb".

    Raises:
        ValueError: If the color space is not valid.

    Returns:
        None
    """
    valid_colors = set(CV_COLORS.keys())

    if color not in valid_colors:
        raise ValueError(f"Invalid color space {color}")

    if color == "gray":
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, CV_COLORS[color]))

    plt.title(title)
    plt.show()


def save_image(image: np.ndarray, output_path: str = None):
    """
    Save the image to a specified output path as a .jpg file.

    Args:
        image (np.ndarray): The image to save.
        output_path (str): The path where the image will be
            saved as a .jpg file.

    Raises:
        ValueError: If the output path does not end with ".jpg".

    Returns:
        None
    """
    if not output_path.endswith(".jpg"):
        raise ValueError("Output path must be a .jpg file")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def count_images(directory: str) -> int:
    """
    Count the number of images in each category within the
    specified directory.

    Args:
        directory (str): The directory containing the images.

    Returns:
        int: The total number of images in the directory.
    """
    return sum(
        len([file for file in files if file.casefold().endswith("jpg")])
        for subdir, _, files in os.walk(directory)
    )
