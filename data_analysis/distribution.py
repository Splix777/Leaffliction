import os
import sys

import matplotlib.pyplot as plt


def fetch_and_analyze_images(directory: str) -> dict[str, dict[str, int]]:
    """
    Fetches all the images in the provided directory and analyzes them to
    determine the distribution of images per category.
    """
    plant_types = {}
    by_fruit_type = set()

    for subdir, _, files in os.walk(directory):
        if subdir == directory:
            continue
        category = os.path.basename(subdir).lower()
        image_count = len(
            [
                file
                for file in files
                if file.lower().endswith(("jpg", "jpeg", "png"))
            ]
        )
        if image_count > 0:
            fruit_type = category.split("_")[0]
            by_fruit_type.add(fruit_type)
            if fruit_type not in plant_types:
                plant_types[fruit_type] = {}
            plant_types[fruit_type][category] = image_count

    return plant_types


def generate_charts(plants_by_fruit: dict[str, dict[str, int]]):
    """
    Generates charts for each plant type and saves them in the output_charts
    directory.
    """
    if not os.path.exists("data/output_charts"):
        os.makedirs("data/output_charts", exist_ok=True)

    for plant_name, plant_types in plants_by_fruit.items():
        categories = list(plant_types.keys())
        counts = list(plant_types.values())

        # Pie Chart
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.pie(counts, labels=categories, autopct="%1.1f%%", startangle=140)
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
        plt.savefig(f"data/output_charts/{plant_name}_distribution.png")
        plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("The provided path is not a directory.")
        sys.exit(1)

    plant_types = fetch_and_analyze_images(directory)
    generate_charts(plant_types)


if __name__ == "__main__":
    main()
