import os
import sys
import zipfile

import click
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")


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


def core_train(src: str) -> None:
    if not os.path.isdir(src) or not contains_jpgs(src):
        print("The provided path is not a valid directory.")
        sys.exit(1)

    augmented_dir = f"{src}_augmented"
    output_zip = '../data/'
    # augmentation(src)
    # transformation(
    #     src=augmented_dir,
    #     dst=augmented_dir,
    #     keep_dir_structure=True)

    # zip_directory(augmented_dir, output_zip)


@click.command()
@click.option('--src', required=True,
              help='Source directory.', callback=validate_source)
def cli_train(src: str) -> None:
    core_train(src)


if __name__ == '__main__':
    src_dir = "../leaves"
    core_train(src_dir)
