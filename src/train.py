import argparse
import os
import sys

from augmentation import augment_images
from packages.data_classifiction.classification_utils import (
    compute_sha1,
)
from packages.data_classifiction.classification_utils import (
    contains_multiple_dirs_jpgs
)
from packages.data_classifiction.create_model import ModelMaker
from packages.utils.config import Config
from packages.utils.decorators import error_handling_decorator
from packages.utils.logger import Logger

logger = Logger("Train").get_logger()
config = Config()


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

    if not os.path.isdir(src) or not contains_multiple_dirs_jpgs(src):
        print("The provided path is not a valid directory.")
        sys.exit(1)

    aug_dir = augment_images(input_dir=src)
    model = ModelMaker(src_data=aug_dir, model_name="model_v1")
    model.run()

    zip_filename = config.output_dir / 'data.zip'
    sha1_hash = compute_sha1(zip_filename)

    signature_filename = config.root_dir / 'signature.txt'
    with open(signature_filename, 'w') as f:
        f.write(sha1_hash)


if __name__ == "__main__":
    main()
