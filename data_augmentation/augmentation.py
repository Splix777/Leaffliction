import copy
import os
import shutil
import sys

import cv2
from tqdm import tqdm

from utils.augmentation_utils import (
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


def load_image(image_path):
    return cv2.imread(image_path)


def save_image(image, path):
    cv2.imwrite(path, image)


def augment_image(image, augmentation_type):
    if augmentation_type in AUGMENTATION_FUNCTIONS:
        return AUGMENTATION_FUNCTIONS[augmentation_type](image)
    else:
        raise ValueError(f"Unsupported augmentation type: {augmentation_type}")


def perform_augmentation(image_path, input_directory, output_directory):
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


def count_and_copy_images(input_directory, output_directory):
    class_counts = {}
    for subdir, _, files in os.walk(input_directory):
        class_name = os.path.basename(subdir).lower()
        if class_name not in class_counts:
            class_counts[class_name] = 0
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                class_counts[class_name] += 1
                image_path = os.path.join(subdir, file)
                relative_path = os.path.relpath(
                    str(image_path), str(input_directory)
                )
                output_path = os.path.join(
                    str(output_directory), str(relative_path)
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(str(image_path), str(output_path))
    return class_counts


def balance_dataset(input_directory, output_directory):
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
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(subdir, file)
                    augmented_images = perform_augmentation(
                        image_path, input_directory, output_directory
                    )
                    num_images += len(augmented_images)
                    progress_bar.update(len(augmented_images))
        progress_bar.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python Augmentation.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("The provided path is not a directory.")
        sys.exit(1)

    augmented_dir = "data/augmented_directory"
    os.makedirs(augmented_dir, exist_ok=True)

    balance_dataset(directory, augmented_dir)


if __name__ == "__main__":
    main()
