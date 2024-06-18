import os

import matplotlib.pyplot as plt

from ..utils.config import Config
from ..utils.decorators import error_handling_decorator
from ..utils.logger import Logger

config = Config()
logger = Logger("Distribution").get_logger()


@error_handling_decorator(handle_exceptions=(FileNotFoundError,))
def fetch_and_analyze_images(directory: str) -> dict[str, dict[str, int]]:
    """
    Fetches all the images in the provided directory and analyzes
    them to determine the distribution of images per category.

    Args:
        directory (str): Directory path containing images of plants.

    Returns:
        dict: A dictionary containing the plant types and their
            respective categories and image counts.

    Raises:
        FileNotFoundError: If the provided directory path
            does not exist.
    """
    plant_types = {}
    by_fruit_type = set()

    for subdir, _, files in os.walk(directory):
        if subdir == directory:
            continue
        category = os.path.basename(subdir).lower()
        image_count = sum(bool(file.casefold().endswith("jpg"))
                          for file in files)

        if image_count > 0:
            fruit_type = category.split("_")[0]
            by_fruit_type.add(fruit_type)
            if fruit_type not in plant_types:
                plant_types[fruit_type] = {}
            plant_types[fruit_type][category] = image_count

    logger.info(plant_types)
    return plant_types


@error_handling_decorator()
def generate_charts(plants_by_fruit: dict[str, dict[str, int]]):
    """
    Generates charts for each plant type and saves
    them in the output_charts directory.

    Args:
        plants_by_fruit (dict): A dictionary containing the
            plant types and their respective categories and
            image counts.

    Returns:
        None
    """
    output_dir = config.output_directory / "output_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    for plant_name, plant_types in plants_by_fruit.items():
        categories = list(plant_types.keys())
        counts = list(plant_types.values())

        # Pie Chart
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.pie(counts, labels=categories, autopct="%1.1f%%", startangle=90)
        plt.title(
            f"{plant_name.capitalize()} - Distribution of Image Categories"
        )

        # Bar Chart
        plt.subplot(1, 2, 2)
        plt.bar(categories, counts, color="skyblue")
        plt.xlabel("Categories")
        plt.ylabel("Number of Images")
        plt.title(f"{plant_name.capitalize()} - Number of Images per Category")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        file_path = output_dir / f"{plant_name}_chart.png"
        if os.path.exists(file_path):
            response = input(
                f'File {plant_name}_chart.png already exists. '
                f'Do you want to overwrite it? (y/n): ')
            if response.lower() != 'y':
                plt.close()
                return
        plt.savefig(file_path)
        plt.close()
