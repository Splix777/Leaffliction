import json
import os
import sys
from typing import Tuple

import click
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import (
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.regularizers import L2
from keras.src.saving.saving_api import save_model
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from matplotlib import pyplot as plt

from src.packages.data_classifiction.utils.helper_functions import (
    validate_source,
    contains_jpgs,
)


def plot_training_metrics(history: tf.keras.callbacks.History, model_path: str) -> None:
    """
    Plot the training metrics

    Args:
        history (tf.keras.callbacks.History): The history object
        model_path (str): The path to save the plot

    Returns:
        None
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = len(loss)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    fig.suptitle(
        "Metrics evolution during the training",
        fontsize=13,
        fontweight="bold"
    )

    # Axes[0] is the loss
    axes[0].plot(range(epochs), loss, label="Train loss")
    axes[0].plot(range(epochs), val_loss, label="Validation loss")
    axes[0].set_title("Evolution of the loss during the training")
    axes[0].set_xlabel("Epochs")
    axes[0].set_xticks(range(epochs))
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(
        linestyle="--",
        linewidth=0.5
    )

    # Axes[1] is the accuracy
    axes[1].plot(range(epochs), acc, label="Train accuracy")
    axes[1].plot(range(epochs), val_acc, label="Validation accuracy")
    axes[1].set_title("Evolution of the accuracy during the training")
    axes[1].set_xlabel("Epochs")
    axes[1].set_xticks(range(epochs))
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_ylim(bottom=0, top=1)
    axes[1].grid(
        linestyle="--",
        linewidth=0.5
    )

    plt.savefig(f"../{model_path}/plot_pre.png")

    plt.show()


def load_data_to_keras(src: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load the data from the source directory to Keras
    dataset objects.

    Args:
        src (str): The source directory containing the images

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]:
            The training and validation datasets
    """
    train_df, val_df = image_dataset_from_directory(
        directory=src,          # The directory containing the images
        labels='inferred',      # The labels are inferred from dir structure
        label_mode='int',       # The labels are integers
        batch_size=32,          # The batch size, 32 images per batch
        image_size=(256, 256),  # The images are resized to 256x256
        shuffle=True,           # Shuffles the dataset
        seed=69,                # The seed for shuffling
        validation_split=0.2,   # 20% of the data is used for validation
        subset='both',          # Training and Validation sets are returned
        verbose=True,           # Print the dataset information
    )

    return train_df, val_df


def create_model(src_data) -> Sequential:
    train_df, val_df = load_data_to_keras(src_data)
    num_classes = len(train_df.class_names)

    # model = Sequential([
    #     Rescaling(1. / 255),
    #     Conv2D(filters=16, kernel_size=4, activation=ReLU()),
    #     MaxPooling2D(),
    #     Conv2D(filters=32, kernel_size=4, activation=ReLU()),
    #     MaxPooling2D(),
    #     Dropout(0.1),
    #     Conv2D(filters=64, kernel_size=4, activation=ReLU()),
    #     MaxPooling2D(),
    #     Dropout(0.1),
    #     Conv2D(filters=128, kernel_size=4, activation=ReLU()),
    #     MaxPooling2D(),
    #     Flatten(),
    #     Dense(units=128, activation=ReLU()),
    #     Dense(units=num_classes, activation=Softmax())
    # ])

    model = Sequential([
        Rescaling(1. / 255),  # Rescale pixel values to [0, 1]

        # Convolutional layers with 3x3 kernel size, ReLU activation, and L2 regularization
        Conv2D(filters=16, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(filters=32, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(filters=64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=L2(0.001)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(units=256, activation='relu'),
        Dropout(0.5),
        Dense(units=128, activation='relu'),
        Dropout(0.5),
        Dense(units=num_classes, activation='softmax')
    ])

    model.build(input_shape=(None, 256, 256, 3))
    print(model.summary())

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    history = model.fit(
        train_df,
        validation_data=val_df,
        epochs=10,
        callbacks=[EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            restore_best_weights=True,
        )],
        batch_size=16,
        verbose=1,
    )

    history_dict = history.history
    # Create a JSON string with indentation and sorted keys
    history_json_str = json.dumps(history_dict, indent=4, sort_keys=True)

    # Write the JSON string to a file
    history_json_file = f'{src_data}_history_pre.json'
    with open(history_json_file, 'w') as f:
        f.write(history_json_str)

    plot_training_metrics(history, f"{src_data}")

    save_model(model=model, filepath=f"{src_data}_pre.keras")

    return model


def core_train(src: str) -> None:
    if not os.path.isdir(src) or not contains_jpgs(src):
        print("The provided path is not a valid directory.")
        sys.exit(1)

    src_name = os.path.basename(src)
    output_zip = f'../data/{src_name}_augmented.zip'
    augmented_dir = f'../data/{src_name}_augmented'
    # augmentation(input_dir=packages, output_dir=augmented_dir)

    model = create_model(augmented_dir)

    # zip_directory(directory=augmented_dir, output_zip=output_zip)


@click.command()
@click.option('--packages', required=True, help='Source directory.',
              callback=validate_source)
def cli_train(src: str) -> None:
    core_train(src)


if __name__ == '__main__':
    src_dir = "../leaves"
    core_train(src_dir)
