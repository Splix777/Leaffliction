import copy
import os
import shutil
import sys

import cv2
from tqdm import tqdm

from data_augmentation.utils.augmentation_utils import (
    flip_image,
    rotate_image,
    random_skew,
    shear_image,
    crop_image,
    apply_contrast,
    perspective_transform_manual,
    elastic_transform,
    color_jitter,
    affine_transform,
    add_noise,
    barrel_distortion,
    pincushion_distortion,
    mustache_distortion,
)

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

AUGMENTATION_FUNCTIONS = {
    "Flip": flip_image,
    "Rotate": rotate_image,
    "Skew": random_skew,
    "Shear": shear_image,
    "Crop": crop_image,
    "Distortion": elastic_transform,
    "Contrast": apply_contrast,
    "Jitter": color_jitter,
    "Projective": perspective_transform_manual,
    "Affine": affine_transform,
    "Noise": add_noise,
    "Barrel": barrel_distortion,
    "Pincushion": pincushion_distortion,
    "Mustache": mustache_distortion,
}


def load_image(image_path: str) -> cv2.imread:
    """
    Load an image from a file path.

    Args:
        image_path: The path to the image file.

    Returns:
        The image as a numpy array.
    """
    return cv2.imread(image_path)


def save_image(image: cv2.imread, path: str) -> None:
    """
    Save an image to a file path.

    Args:
        image: The image to save.
        path: The path to save the image to.

    Returns:
        None
    """
    cv2.imwrite(path, image)


def augment_image(image: cv2.imread, augmentation_type: str) -> cv2.imread:
    """
    Apply an augmentation to an image.

    Args:
        image: The image to augment.
        augmentation_type: The type of augmentation to apply.

    Returns:
        The augmented image.

    Raises:
        ValueError: If the augmentation type is not supported.
    """
    if augmentation_type in AUGMENTATION_FUNCTIONS:
        return AUGMENTATION_FUNCTIONS[augmentation_type](image)
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type}")


def perform_augmentation(image_path: str, input_directory: str,
                         output_directory: str) -> list:
    """
    Perform augmentation on an image and save the
    augmented images to a directory.

    Args:
        image_path: The path to the image to augment.
        input_directory: The directory containing
            the input images.
        output_directory: The directory to save the
            augmented images to.

    Returns:
        A list of paths to the augmented images.
    """
    image = load_image(image_path)
    relative_path = os.path.relpath(image_path, input_directory)
    base_name, ext = os.path.splitext(relative_path)

    output_subdirectory = os.path.join(
        output_directory, os.path.dirname(base_name)
    )
    os.makedirs(output_subdirectory, exist_ok=True)

    augmented_images = []
    for aug_type in AUGMENTATION_FUNCTIONS.keys():
        augmented_image = augment_image(copy.deepcopy(image), aug_type)
        new_image_name = f"{os.path.basename(base_name)}_{aug_type}{ext}"
        new_image_path = os.path.join(output_subdirectory, new_image_name)
        save_image(augmented_image, new_image_path)
        augmented_images.append(new_image_path)

    return augmented_images


def count_and_copy_images(input_directory: str, output_directory: str) -> dict:
    """
    Count the number of images in each class and copy
    the images to the output directory.

    Args:
        input_directory: The directory containing
            the input images.
        output_directory: The directory to save the
            images to.

    Returns:
        A dictionary containing the number of images in each class.
    """
    class_counts = {}
    for subdir, _, files in os.walk(input_directory):
        class_name = os.path.basename(subdir).lower()
        if class_name not in class_counts:
            class_counts[class_name] = 0
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                class_counts[class_name] += 1
                image_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(image_path, input_directory)
                output_path = os.path.join(output_directory, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(str(image_path), str(output_path))

    return class_counts


def find_data_directory(start_directory: str) -> str:
    """
    Find the directory containing multiple subdirectories
    under the starting directory.

    Args:
        start_directory: The directory to start looking from.

    Returns:
        The directory containing multiple subdirectories.
    """
    current_dir = start_directory
    while True:
        subdirectories = next(os.walk(str(current_dir)))[1]
        if len(subdirectories) > 1:
            break
        if len(subdirectories) == 1:
            current_dir = os.path.join(current_dir, subdirectories[0])
        else:
            raise ValueError(
                f"No multiple subdirectories found under {start_directory}")

    return current_dir


def balance_dataset(input_directory: str, output_directory: str) -> None:
    """
    Balance the dataset by augmenting the images in each class
    to have the same number of images.

    Args:
        input_directory: The directory containing
            the input images.
        output_directory: The directory to save the
            augmented images to.

    Returns:
        None
    """
    class_counts = count_and_copy_images(input_directory, output_directory)
    max_images = max(class_counts.values())

    for subdir, _, files in os.walk(input_directory):
        class_name = os.path.basename(subdir).lower()
        num_images = class_counts.get(class_name, 0)
        if num_images == 0:
            continue
        progress_bar = tqdm(
            total=(max_images - num_images), desc=str(class_name)
        )
        while num_images < max_images:
            for file in files:
                if num_images >= max_images:
                    break
                if file.casefold().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(subdir, file)
                    augmented_images = perform_augmentation(
                        image_path, input_directory, output_directory
                    )
                    num_images += len(augmented_images)
                    progress_bar.update(len(augmented_images))

        progress_bar.close()


def augmentation(input_dir: str = None, output_dir: str = None) -> None:
    """
    Perform data augmentation on a dataset.

    Args:
        input_dir: The directory containing the input images.
        output_dir: The directory to save the augmented images to.

    Returns:
        None
    """
    if len(sys.argv) != 2 and input_dir is None:
        print("Usage: python Augmentation.py <directory>")
        sys.exit(1)

    directory = sys.argv[1] if input_dir is None else input_dir
    if not os.path.isdir(directory):
        print("The provided path is not a directory.")
        sys.exit(1)

    data_dir = find_data_directory(directory)
    augmented_dir = output_dir or f"data/{os.path.basename(data_dir)}_augmented"
    os.makedirs(augmented_dir, exist_ok=True)

    balance_dataset(data_dir, augmented_dir)


if __name__ == "__main__":
    apples = "../leaves/images/Apple_rust"
    augmentation(input_dir=apples)
