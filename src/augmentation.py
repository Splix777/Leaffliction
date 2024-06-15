import argparse
import os
import subprocess
import sys

from packages.data_augmentation.augmentation_workflow import (
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
def main(input_dir: str = None, output_dir: str = None):
    """
    Perform data augmentation on a dataset.

    Args:
        input_dir: The directory containing the input images.
        output_dir: The directory to save the augmented images to.

    Returns:
        None
    """
    if input_dir is None and output_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "input_dir",
            help="Path to the folder containing the input images"
        )
        args = parser.parse_args()
        input_dir = args.input_dir

    if os.path.isdir(input_dir):
        data_dir = find_data_directory(input_dir)
        augmented_dir = config.output_dir / f"{os.path.basename(data_dir)}_augmented"
        os.makedirs(augmented_dir, exist_ok=True)

        balance_dataset(data_dir, augmented_dir)
    elif os.path.isfile(input_dir):
        file_dir = os.path.dirname(input_dir)
        augmented_images = perform_augmentation(
            image_path=input_dir,
            input_directory=file_dir,
            output_directory=config.temp_dir)
        subprocess.Popen(["open", config.temp_dir])
    else:
        logger.error(f"Invalid input file or directory: {input_dir}")
        print(f"Invalid input file or directory: {input_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()