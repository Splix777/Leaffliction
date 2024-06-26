import argparse
import os
import sys

from packages.data_analysis.distribution_utils import (
    fetch_and_analyze_images,
    generate_charts
)
from packages.utils.config import Config
from packages.utils.logger import Logger

config = Config()
logger = Logger("Distribution").get_logger()


def create_data_distribution_plots(directory: str = None):
    """
    Main function to generate distribution charts
    for the plant types and their respective categories
    and image counts.

    Args:
        directory (str): Directory path containing images of plants.

    Returns:
        None

    Raises:
        SystemExit: If the provided path is not a directory.
    """
    if directory is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--src", required=True,
            help="Path to the folder containing the image data"
        )
        args = parser.parse_args()
        directory = args.src

    if not os.path.isdir(directory):
        print("The provided path is not a directory.")
        sys.exit(1)

    plant_types = fetch_and_analyze_images(directory)
    generate_charts(plant_types)


if __name__ == "__main__":
    create_data_distribution_plots()
