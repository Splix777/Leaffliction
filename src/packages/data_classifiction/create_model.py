import json
import tempfile
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from keras import Sequential
from keras.src.callbacks import EarlyStopping, TensorBoard
from keras.src.layers import (
    Input,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
)
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.saving.saving_api import save_model
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from tensorflow.data import AUTOTUNE, Dataset

from ..utils.config import Config
from ..utils.decorators import error_handling_decorator
from ..utils.logger import Logger

config = Config()
logger = Logger("ModelMaker").get_logger()


class ModelMaker:
    """
    A class to create a model for image classification

    Attributes:
        src_data: The path to the directory containing the images
        train_df: The training dataset
        val_df: The validation dataset
        num_classes: The number of classes in the dataset
        model: The model created for image classification
        history: The training history of the model
        model_name: The name of the model
        epochs: The number of epochs to train the model
        batch_size: The number of images per batch
        verbose: The verbosity level of the training
        val_split: The fraction of the dataset used for validation
        learning_rate: The learning rate of the model

    Methods:
        __call__: Run the model creation workflow and return the model
        _save_dataset_to_zip: Saves image datasets and their labels to a zip
        _save_model_to_zip: Saves a machine learning model to a zip file
        _save_history_to_zip: Saves the training history dictionary as a JSON
        _save_image_to_zip: Saves a given figure as an image to a zip file
        _add_conv2d: Add a Conv2D layer to the model
        _load_data_to_keras: Loads data and saves split datasets to a zip
        _create_model: Creates and compiles the model
        _fit_model: Fit the model
        _plot_training_metrics: Plot the training metrics
        run: Run the model creation workflow
    """

    def __init__(self, src_data: str, model_name: str, epochs: int = 10,
                 batch_size: int = 32, verbose: int = 1,
                 val_split: float = 0.2, learning_rate: float = 0.001) -> None:
        self.src_data = src_data
        self.train_df = None
        self.val_df = None
        self.num_classes = None
        self.model = None
        self.history = None
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.val_split = val_split
        self.learning_rate = learning_rate

    def __call__(self) -> Sequential:
        """
        Run the model creation workflow and return the model
        """
        self.run()
        return self.model

    @staticmethod
    def _save_dataset_to_zip(train_df: Dataset, val_df: Dataset) -> None:
        """
        Saves image datasets and their labels to a zip file.

        Args:
            train_df: The training dataset.
            val_df: The validation dataset.

        Returns:
            None
        """

        def save_images_to_zip(dataset, zip_handle, subdir_name):
            """
            Saves images from a dataset to a zip file under
            the specified subdirectory name.

            Args:
                dataset: The dataset containing images and labels.
                zip_handle: The handle to the zip file where
                    images will be saved.
                subdir_name: The name of the subdirectory
                    within the zip file.

            Returns:
                None
            """
            for i, (image, label) in enumerate(dataset.unbatch()):
                img = image.numpy().astype('uint8')
                label = label.numpy()
                img_filename = (f'{subdir_name}/{dataset.class_names[label]}/'
                                f'image_{i + 1}.jpg')

                # Save image to a BytesIO buffer
                img_buffer = BytesIO()
                plt.imsave(img_buffer, img, format='jpg')
                img_buffer.seek(0)

                # Write the image to the zip file
                with zip_handle.open(img_filename, 'w') as img_file:
                    img_file.write(img_buffer.read())

        def save_labels_to_zip(class_names, zip_handle):
            """
            Saves class names as JSON data to a zip file.

            Args:
                class_names: The list of class names to be saved.
                zip_handle: The handle to the zip file where
                    class names will be saved.

            Returns:
                None
            """
            labels_filename = 'labels.json'
            labels_data = {i: class_names[i] for i in range(len(class_names))}
            labels_json = json.dumps(labels_data).encode('utf-8')
            with zip_handle.open(labels_filename, 'w') as labels_file:
                labels_file.write(labels_json)

        zip_output = config.output_dir / 'data.zip'
        with ZipFile(zip_output, 'a', ZIP_DEFLATED) as zipf:
            if 'train' not in zipf.namelist() and 'val' not in zipf.namelist():
                print(f"Saving dataset to '{zip_output}'")
                save_images_to_zip(train_df, zipf, 'train')
                save_images_to_zip(val_df, zipf, 'val')
            if 'labels.json' not in zipf.namelist():
                print(f"Saving labels to '{zip_output}'")
                save_labels_to_zip(train_df.class_names, zipf)

    @staticmethod
    def _save_model_to_zip(model: Sequential, zip_filename: str,
                           model_name: str) -> None:
        """
        Saves a machine learning model to a zip file with the
        specified model name.

        Args:
            model: The model to be saved.
            zip_filename: The path to the zip file where
                the model will be saved.
            model_name: The name of the model associated
                with the model.

        Returns:
            None
        """
        with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}.keras' in zipf.namelist():
                print(f"Model '{model_name}.keras' already exists in "
                      f"'{zip_filename}'. Skipping save.")
                return

        print(f"Saving model '{model_name}.keras' to '{zip_filename}'")
        with tempfile.NamedTemporaryFile(suffix='.keras') as temp_file:
            temp_filename = temp_file.name

            save_model(model, temp_filename)

            with open(temp_filename, 'rb') as f:
                model_bytes = f.read()
                model_buffer = BytesIO(model_bytes)

            with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
                zipf.writestr(f'{model_name}.keras', model_buffer.getvalue())

    @staticmethod
    def _save_history_to_zip(history_dict: dict, zip_filename: str,
                             model_name: str) -> None:
        """
        Saves the training history dictionary as a JSON file
        to a zip archive with the specified model name.

        Args:
            history_dict: The dictionary containing the
                training history.
            zip_filename: The path to the zip file where
                the history will be saved.
            model_name: The name of the model associated
                with the history.

        Returns:
            None
        """
        with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}_history.json' in zipf.namelist():
                print(f"History '{model_name}_history.json' already "
                      f"exists in '{zip_filename}'. Skipping save.")
                return

        print(f"Saving history '{model_name}_history.json' "
              f"to '{zip_filename}'")
        history_json = json.dumps(history_dict, indent=4, sort_keys=True)
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            temp_filename = temp_file.name
            with open(temp_filename, 'w') as f:
                f.write(history_json)

            with open(temp_filename, 'rb') as f:
                history_bytes = f.read()
                history_buffer = BytesIO(history_bytes)

            zip_output = config.output_dir / 'data.zip'
            with ZipFile(zip_output, 'a', ZIP_DEFLATED) as zipf:
                zipf.writestr(f'{model_name}_history.json',
                              history_buffer.getvalue())

    @staticmethod
    def _save_image_to_zip(fig: plt.Figure, image_output: str,
                           model_name: str) -> None:
        """
        Saves a given figure as an image to a zip file with
        the specified model name.

        Args:
            fig: The figure to be saved as an image.
            image_output: The path to the zip file
                where the image will be saved.
            model_name: The name of the model associated
                with the image.

        Returns:
            None
        """
        with ZipFile(image_output, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}_metrics.png' in zipf.namelist():
                print(f"Image '{model_name}_metrics.png' already exists in "
                      f"'{image_output}'. Skipping save.")
                return

        print(f"Saving metrics image '{model_name}_metrics.png' to "
              f"'{image_output}'")
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            temp_filename = temp_file.name
            fig.savefig(temp_filename)

            with open(temp_filename, 'rb') as f:
                image_bytes = f.read()
                image_buffer = BytesIO(image_bytes)

            with ZipFile(image_output, 'a', ZIP_DEFLATED) as zipf:
                zipf.writestr(f'{model_name}_metrics.png',
                              image_buffer.getvalue())

    @staticmethod
    def _add_conv2d(model: Sequential, filters: int, kernel_size: int) -> None:
        """
        Add a Conv2D layer to the model

        Args:
            model: The model to add the layer to
            filters: The number of filters to use
            kernel_size: The size of the kernel

        Returns:
            None
        """
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same'
            )
        )
        # model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.1))

    @error_handling_decorator()
    def _load_data_to_keras(self) -> None:
        """
        Load the data to Keras and saves split datasets to a zip file
        along with dataset labels.
        """
        train_df, val_df = image_dataset_from_directory(
            directory=self.src_data,  # The directory containing the images
            labels='inferred',  # The labels are inferred from dir structure
            label_mode='int',  # The labels are integers
            batch_size=self.batch_size,  # The amount of images per batch
            image_size=(256, 256),  # The images are resized to 256x256
            shuffle=True,  # Shuffles the dataset
            seed=69,  # The seed for shuffling (The funny number)
            validation_split=self.val_split,  # 20% used for validation
            subset='both',  # Training and Validation sets are returned
            verbose=bool(self.verbose),  # Print the dataset information
        )
        self._save_dataset_to_zip(train_df, val_df)
        self.train_df = train_df.prefetch(buffer_size=AUTOTUNE)
        self.val_df = val_df.prefetch(buffer_size=AUTOTUNE)
        self.num_classes = len(train_df.class_names)

    @error_handling_decorator()
    def _create_model(self) -> None:
        """
        Creates and compiles the model
        """
        model = Sequential()
        model.add(Input(shape=(256, 256, 3)))
        model.add(Rescaling(1.0 / 255))

        self._add_conv2d(model=model, filters=16, kernel_size=3)
        self._add_conv2d(model=model, filters=32, kernel_size=3)
        self._add_conv2d(model=model, filters=64, kernel_size=3)
        self._add_conv2d(model=model, filters=128, kernel_size=3)

        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=self.num_classes, activation='softmax'))

        print(model.summary())

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        self.model = model

    @error_handling_decorator()
    def _fit_model(self) -> None:
        """
        Fit the model
        """
        history = self.model.fit(
            self.train_df,
            validation_data=self.val_df,
            epochs=self.epochs,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    verbose=self.verbose,
                    restore_best_weights=True,
                ),
                TensorBoard(
                    log_dir=config.logs_dir,
                    histogram_freq=1)
            ],
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        zip_output = config.output_dir / 'data.zip'
        self._save_model_to_zip(self.model, zip_output, self.model_name)

        self.history = history.history
        self._save_history_to_zip(self.history, zip_output, self.model_name)

    @error_handling_decorator()
    def _plot_training_metrics(self) -> None:
        """
        Plot the training metrics
        """
        if self.history is None:
            return
        acc = self.history['accuracy']
        val_acc = self.history['val_accuracy']

        loss = self.history['loss']
        val_loss = self.history['val_loss']

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
        image_output = config.output_dir / 'data.zip'
        self._save_image_to_zip(fig, image_output, self.model_name)

    @error_handling_decorator()
    def run(self) -> None:
        """
        Run the model creation workflow
        """
        self._load_data_to_keras()
        self._create_model()
        self._fit_model()
        self._plot_training_metrics()
