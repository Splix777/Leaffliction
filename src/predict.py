import argparse
import os
import sys

import cv2
import numpy as np
from keras.src.models import Sequential
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from plantcv import plantcv as pcv
from tensorflow.data.experimental import AUTOTUNE
from tqdm import tqdm

from packages.data_classifiction.classification_utils import (
    contains_jpgs,
    extract_labels_from_zip,
    extract_model_from_zip,
    contains_multiple_dirs_jpgs
)
from packages.data_transformation.transformation_class import Transformation
from packages.utils.config import Config
from packages.utils.decorators import error_handling_decorator
from packages.utils.logger import Logger

logger = Logger("Predict").get_logger()
config = Config()


@error_handling_decorator(handle_exceptions=(ValueError, cv2.error))
def preprocess_image(image_path: str) -> np.ndarray:
    image, _, _ = pcv.readimage(image_path, mode='rgb')
    if image is None:
        raise ValueError("The image could not be loaded.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.expand_dims(image, axis=0)


def plot_predictions(prediction: str, original: np.ndarray,
                     masked: np.ndarray, confidence: float) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    fig.patch.set_facecolor('black')
    _load_image(ax1, original, "Original Image")
    _load_image(ax2, masked, "Masked Image")
    fig.subplots_adjust(top=0.85)
    fig.add_subplot(111, frameon=False)
    plt.grid(False)

    fig.suptitle(f"Predicted class: {prediction}", fontsize=16, color='green')

    plt.text(
        x=0.5,
        y=0.05,
        s=f"Confidence: {confidence:.2f}",
        ha='center',
        va='center',
        fontsize=14,
        color='white'
    )

    plt.show()


def _load_image(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
    ax.set_facecolor('black')


def predict_image(src: str, model, class_names: dict,
                  plot: bool = False) -> tuple[str, float]:
    image = preprocess_image(src)

    predictions = model.predict(image, verbose=0)

    confidence = np.max(predictions)
    predicted_class_name = class_names[str(np.argmax(predictions))]
    pred_class = predicted_class_name.replace("_", " ").capitalize()

    if plot:
        transformation = Transformation(image_path=src)
        images = transformation.get_images()

        plot_predictions(
            pred_class,
            images['original'],
            images['masked'],
            confidence
        )

    return pred_class, confidence


def predict_batch(src: str, model: Sequential, class_names: dict,
                  val_count: int) -> None:
    all_predictions = {}
    for subdir, _, files in os.walk(src):
        if not files:
            continue

        subdir_name = (
            os.path.basename(subdir.rstrip('/'))
            .replace("_", " ")
            .capitalize()
        )

        if subdir_name not in all_predictions:
            all_predictions[subdir_name] = []

        if val_count > 0:
            files = files[:val_count]
        pbar = tqdm(files, desc=f"Predicting {subdir_name}", unit='files')
        for file in files:
            if file.casefold().endswith('.jpg'):
                file_path = os.path.join(subdir, file)
                pred, conf = predict_image(file_path, model, class_names)
                all_predictions[subdir_name].append((file_path, pred, conf))
                pbar.update(1)
        pbar.close()

    calculate_prediction_accuracy(all_predictions)


def calculate_prediction_accuracy(all_predictions: dict) -> None:
    overall_correct_percentage = 0
    for subdir_name, predictions in all_predictions.items():
        total_correct = 0
        total_mistakes = 0
        prediction_confidence = 0

        for src, prediction, confidence in predictions:
            if subdir_name == prediction:
                total_correct += 1
            else:
                total_mistakes += 1
            prediction_confidence += confidence

        total_predictions = total_correct + total_mistakes
        average_confidence = prediction_confidence / total_predictions
        overall_correct_percentage += total_correct

        if total_predictions == 0:
            percentage_correct = 0.0
        else:
            percentage_correct = (total_correct / total_predictions) * 100

        print(
            f"Category: {subdir_name}",
            f"Correct: {total_correct}",
            f"Wrong: {total_mistakes}",
            f"Accuracy: {percentage_correct:.2f}%",
            f"Average Confidence: {average_confidence:.2f}",
            sep='\n'
        )
        print('-' * 25)

    total_predictions = sum(
        len(predictions) for predictions in all_predictions.values()
    )
    overall_accuracy = (overall_correct_percentage / total_predictions) * 100
    print(f"Overall accuracy: {overall_accuracy:.2f}%")


@error_handling_decorator()
def evaluate_model(src: str, model: Sequential, batch_size: int) -> None:
    # Load data from directory
    print(src)
    full_path = os.path.join(config.root_dir, src)
    dataset = image_dataset_from_directory(
        directory=full_path,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=False,
        seed=69,
        validation_split=0.2,
        subset='validation',
        verbose=0
    )

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    # Evaluate the model
    loss, accuracy = model.evaluate(dataset)
    print(
        f"Evaluation results - Loss: {loss}, Accuracy: {accuracy * 100:.2f}%")


@error_handling_decorator(handle_exceptions=(FileNotFoundError,))
def main(src: str = None, val_count: int = 0) -> None:
    if src is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--src", required=True,
                            help="Path to image or directory containing jpgs")
        parser.add_argument("--val_count", type=int, default=0,
                            help="Amount of validation data to use")
        args = parser.parse_args()
        src = args.src
        val_count = args.val_count

    if not os.path.exists(src):
        print("The provided path does not exist.")
        sys.exit(1)
    if os.path.isdir(src) and not contains_jpgs(src):
        print("The provided path is not a valid directory.")
        sys.exit(1)
    if os.path.isfile(src) and not src.casefold().endswith('.jpg'):
        print("The provided path is not a valid image file.")
        sys.exit(1)

    zip_filename = config.output_dir / 'data.zip'

    if os.path.isfile(zip_filename):
        model = extract_model_from_zip(zip_filename, 'model_v1')
        labels = extract_labels_from_zip(zip_filename, 'labels.json')

        if os.path.isfile(src):
            predict_image(
                src=src,
                model=model,
                class_names=labels,
                plot=True
            )
        else:
            predict_batch(
                src=src,
                model=model,
                class_names=labels,
                val_count=val_count
            )
            if contains_multiple_dirs_jpgs(src):
                evaluate_model(src=src, model=model, batch_size=8)

    else:
        logger.error("The model zip file is missing.")
        print("The model zip file is missing.")


if __name__ == "__main__":
    main()
