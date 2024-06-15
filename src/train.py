import argparse
import os
import sys

from packages.data_classifiction.classification_utils import contains_jpgs
from packages.data_classifiction.create_model import ModelMaker
from packages.utils.config import Config
from packages.utils.decorators import error_handling_decorator
from packages.utils.logger import Logger

config = Config()
logger = Logger("Train").get_logger()


@error_handling_decorator(handle_exceptions=(FileNotFoundError,))
def main(src: str = None) -> None:
    if src is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--src", required=True,
            help="Path to the folder containing the input images"
        )
        args = parser.parse_args()
        src = args.src

    if not os.path.isdir(src) or not contains_jpgs(src):
        print("The provided path is not a valid directory.")
        sys.exit(1)

    ModelMaker(src)


if __name__ == "__main__":
    main()
