import json
import os
import subprocess
import zipfile

from keras.src.saving import load_model


def contains_multiple_dirs_jpgs(directory: str) -> bool:
    """
    Check if the directory contains jpg files and at least two subdirectories
    with jpg files.

    Args:
        directory (str): The path to the directory to check.

    Returns:
        bool: True if the directory contains jpg files and at least two
              subdirectories with jpg files, False otherwise.
    """
    subdirectories_with_jpgs = set()

    for subdir, _, files in os.walk(directory):
        if subdir == directory:
            continue
        if any(file.casefold().endswith('.jpg') for file in files):
            subdirectories_with_jpgs.add(subdir)
        if len(subdirectories_with_jpgs) >= 2:
            return True
    return False


def contains_jpgs(directory: str) -> bool:
    """
    Check if the directory contains jpg files

    Args:
        directory (str): The path to the directory to check.

    Returns:
        bool: True if the directory contains jpg files
    """
    return any(
        any(file.casefold().endswith('.jpg') for file in files)
        for subdir, _, files in os.walk(directory)
    )


def count_image(directory: str) -> int:
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


def zip_directory(directory, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                zipf.write(str(os.path.join(root, file)))


def compute_sha1(file_path):
    result = subprocess.run(
        ['sha1sum', file_path],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.split()[0]
    else:
        raise RuntimeError(f"Failed to compute SHA1 hash for {file_path}")


def extract_model_from_zip(zip_filename, model_name):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        model_filename = f'{model_name}.keras'
        if model_filename not in zipf.namelist():
            raise FileNotFoundError(f"Model file '{model_filename}' "
                                    f"not found in '{zip_filename}'")

        with zipf.open(model_filename, 'r') as model_file:
            model_bytes = model_file.read()

        with open(model_filename, 'wb') as f:
            f.write(model_bytes)

        return load_model(model_filename)


def extract_labels_from_zip(zip_filename, labels_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        if labels_filename not in zipf.namelist():
            raise FileNotFoundError(f"Labels file '{labels_filename}' "
                                    f"not found in '{zip_filename}'")

        with zipf.open(labels_filename, 'r') as labels_file:
            return json.load(labels_file)
