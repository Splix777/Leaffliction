import argparse
import os
import sys

from packages.data_transformation.transformation_class import Transformation
from packages.utils.config import Config
from packages.utils.decorators import error_handling_decorator
from packages.utils.logger import Logger

config = Config()
logger = Logger("Transformation").get_logger()


@error_handling_decorator(handle_exceptions=(FileNotFoundError,))
def transformation(src: str = None, dst: None = None) -> None:
    """
     Perform the transformation based on the
     source and destination.

     Args:
         src (str): The source image or directory.
         dst (Union[str, None]): The destination directory.
         keep_dir_structure (bool): Whether to keep the
            directory structure.
     """
    if src is None and dst is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--src", required=True,
            help="Path to the folder containing the input images"
        )
        parser.add_argument(
            "--dst",
            help="Path to the folder to save the output images"
        )
        args = parser.parse_args()
        src = args.src
        dst = args.dst

    if os.path.isdir(src):
        if dst is None:
            print("Please provide a destination directory.")
            sys.exit(1)
        Transformation(
            input_dir=src,
            output_dir=dst,
            keep_dir_structure=True
        )
    elif os.path.isfile(src):
        Transformation(image_path=src, keep_dir_structure=False)
    else:
        logger.error(f"Invalid input file or directory: {src}")
        print(f"Invalid input file or directory: {src}")
        sys.exit(1)


if __name__ == '__main__':

    transformation()
