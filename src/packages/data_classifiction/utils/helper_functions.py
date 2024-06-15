import os
import zipfile

import click


def validate_source(
        ctx: click.Context, param: click.Parameter, value: str) -> str:
    if os.path.isdir(value):
        return value
    else:
        raise click.BadParameter('Source must be a valid directory.')


def contains_jpgs(directory: str) -> bool:
    for _, _, files in os.walk(directory):
        for file in files:
            if file.casefold().endswith('.jpg'):
                return True
    return False


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
