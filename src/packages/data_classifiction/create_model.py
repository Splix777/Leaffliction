import json

import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping, TensorBoard
from keras.src.layers import (
    ReLU,
    Softmax,
    Flatten,
    Input,
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

from ..utils.config import Config
from ..utils.logger import Logger

config = Config()
logger = Logger("ModelMaker").get_logger()

def plot_training_metrics(history: tf.keras.callbacks.History,
                          model_path: str) -> None:
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


def create_model(src_data) -> Sequential:
    train_df, val_df = load_data_to_keras(src_data)
    num_classes = len(train_df.class_names)

    model = Sequential([
        Rescaling(1. / 255),
        Conv2D(filters=16, kernel_size=4, activation=ReLU()),
        MaxPooling2D(),
        Conv2D(filters=32, kernel_size=4, activation=ReLU()),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(filters=64, kernel_size=4, activation=ReLU()),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(filters=128, kernel_size=4, activation=ReLU()),
        MaxPooling2D(),
        Flatten(),
        Dense(units=128, activation=ReLU()),
        Dense(units=num_classes, activation=Softmax())
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


class ModelMaker:
    def __init__(self, src_data: str):
        self.src_data = src_data
        self.train_df = None
        self.val_df = None
        self.num_classes = None
        self.model = None

        self._load_data_to_keras()
        self._create_model()
        self._fit_model()

    def _load_data_to_keras(self) -> None:
        train_df, val_df = image_dataset_from_directory(
            directory=self.src_data,  # The directory containing the images
            labels='inferred',  # The labels are inferred from dir structure
            label_mode='int',  # The labels are integers
            batch_size=32,  # The batch size, 32 images per batch
            image_size=(256, 256),  # The images are resized to 256x256
            shuffle=True,  # Shuffles the dataset
            seed=69,  # The seed for shuffling
            validation_split=0.2,  # 20% of the data is used for validation
            subset='both',  # Training and Validation sets are returned
            verbose=True,  # Print the dataset information
        )
        self.train_df = train_df
        self.val_df = val_df
        self.num_classes = len(train_df.class_names)

    @staticmethod
    def _add_conv2d(model, filters):
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=3,
                activation='relu',
                padding='same',
                kernel_regularizer=L2(0.001),
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D())

    def _create_model(self) -> None:
        model = Sequential()
        model.add(Input(shape=(256, 256, 3)))
        model.add(Rescaling(1. / 255))
        self._add_conv2d(model, 16)

        self._add_conv2d(model, 32)
        model.add(Dropout(0.2))

        self._add_conv2d(model, 64)
        model.add(Dropout(0.2))

        self._add_conv2d(model, 128)
        model.add(Dropout(0.2))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(units=self.num_classes, activation='softmax'))

        print(model.summary())

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        self.model = model

    def _fit_model(self) -> None:
        log_dir = config.logs_dir

        history = self.model.fit(
            self.train_df,
            validation_data=self.val_df,
            epochs=10,
            callbacks=[EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=1,
                restore_best_weights=True,
            ),
                TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1)],
            batch_size=16,
            verbose=1,
        )

        self.history_dict = history.history
        # Create a JSON string with indentation and sorted keys
        history_json = json.dumps(self.history_dict, indent=4, sort_keys=True)

        # Write the JSON string to a file
        history_json_file = f'{self.src_data}_history.json'
        with open(history_json_file, 'w') as f:
            f.write(history_json)

        plot_training_metrics(history, f"{self.src_data}")

        save_model(model=self.model, filepath=f"{self.src_data}.keras")
