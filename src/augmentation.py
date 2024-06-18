import argparse
import os
import subprocess
import sys

from packages.data_augmentation.augmentation_utils import (
    balance_dataset,
    find_data_directory,
    perform_augmentation
)
from packages.utils.config import Config
from packages.utils.decorators import error_handling_decorator
from packages.utils.logger import Logger

config = Config()
logger = Logger("Augmentation").get_logger()


@error_handling_decorator(handle_exceptions=(FileNotFoundError,))
def augment_images(input_src: str = None, output_dir: str = None):
    """
    Perform data augmentation on a dataset.

    Args:
        input_src: The directory containing the input images.
        output_dir: The directory to save the augmented images to.

    Returns:
        None
    """
    if input_src is None and output_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--src", required=True,
            help="Path to the folder containing the input images"
        )
        parser.add_argument(
            "--dst", required=False,
            help="Path to the folder to save the augmented images"
        )
        args = parser.parse_args()
        input_src = args.src
        output_dir = args.dst

    if os.path.isdir(input_src):
        data_dir = find_data_directory(input_src)
        augmented_dir = (
                output_dir
                or config.output_dir / f"{os.path.basename(data_dir)}_aug"
        )
        os.makedirs(augmented_dir, exist_ok=True)
        balance_dataset(data_dir, augmented_dir)
        return augmented_dir

    elif os.path.isfile(input_src):
        file_dir = os.path.dirname(input_src)
        perform_augmentation(
            image_path=input_src,
            input_directory=file_dir,
            output_directory=config.temp_dir)
        subprocess.Popen(["open", config.temp_dir])

    else:
        logger.error(f"Invalid input file or directory: {input_src}")
        print(f"Invalid input file or directory: {input_src}")
        sys.exit(1)


if __name__ == "__main__":
    augment_images()
