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
    GlobalAveragePooling2D,
    BatchNormalization,
)
from keras.src.losses import SparseCategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.saving.saving_api import save_model
from keras.src.utils.image_dataset_utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from tensorflow.data import AUTOTUNE

from ..utils.config import Config
from ..utils.decorators import error_handling_decorator
from ..utils.logger import Logger

config = Config()
logger = Logger("ModelMaker").get_logger()


class ModelMaker:
    def __init__(self, src_data: str, model_name: str) -> None:
        self.src_data = src_data
        self.train_df = None
        self.val_df = None
        self.num_classes = None
        self.model = None
        self.history = None
        self.model_name = model_name
        self._workflow()

    @staticmethod
    def _save_dataset_to_zip(train_df, val_df) -> None:
        def save_images_to_zip(dataset, zip_handle, subdir_name):
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

        zip_output = config.output_dir / 'data.zip'
        if zip_output.exists():
            return

        with ZipFile(zip_output, 'w', ZIP_DEFLATED) as zipf:
            save_images_to_zip(train_df, zipf, 'train')
            save_images_to_zip(val_df, zipf, 'val')

    @staticmethod
    def _save_model_to_zip(model, zip_filename, model_name):
        with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}.keras' in zipf.namelist():
                print(f"Model '{model_name}.keras' already exists in "
                      f"'{zip_filename}'. Skipping save.")
                return

        with tempfile.NamedTemporaryFile(suffix='.keras') as temp_file:
            temp_filename = temp_file.name

            save_model(model, temp_filename)

            with open(temp_filename, 'rb') as f:
                model_bytes = f.read()
                model_buffer = BytesIO(model_bytes)

            with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
                zipf.writestr(f'{model_name}.keras', model_buffer.getvalue())

    @staticmethod
    def _save_history_to_zip(history_dict, zip_filename, model_name) -> None:
        with ZipFile(zip_filename, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}_history.json' in zipf.namelist():
                print(f"History '{model_name}_history.json' already "
                      f"exists in '{zip_filename}'. Skipping save.")
                return

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
    def _save_image_to_zip(fig, image_output, model_name):
        with ZipFile(image_output, 'a', ZIP_DEFLATED) as zipf:
            if f'{model_name}_metrics.png' in zipf.namelist():
                print(f"Image '{model_name}_metrics.png' already exists in "
                      f"'{image_output}'. Skipping save.")
                return

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
        model.add(
            Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

    @error_handling_decorator()
    def _load_data_to_keras(self) -> None:
        train_df, val_df = image_dataset_from_directory(
            directory=self.src_data,  # The directory containing the images
            labels='inferred',  # The labels are inferred from dir structure
            label_mode='int',  # The labels are integers
            batch_size=8,  # The batch size, 16 images per batch
            image_size=(256, 256),  # The images are resized to 256x256
            shuffle=True,  # Shuffles the dataset
            seed=69,  # The seed for shuffling
            validation_split=0.2,  # 20% of the data is used for validation
            subset='both',  # Training and Validation sets are returned
            verbose=True,  # Print the dataset information
        )
        self._save_dataset_to_zip(train_df, val_df)
        self.train_df = train_df.prefetch(buffer_size=AUTOTUNE)
        self.val_df = val_df.prefetch(buffer_size=AUTOTUNE)
        self.num_classes = len(train_df.class_names)

    @error_handling_decorator()
    def _create_model(self) -> None:
        model = Sequential()
        model.add(Input(shape=(256, 256, 3)))
        model.add(Rescaling(1.0 / 255))

        self._add_conv2d(model=model, filters=16, kernel_size=3)
        self._add_conv2d(model=model, filters=32, kernel_size=3)
        self._add_conv2d(model=model, filters=64, kernel_size=3)
        self._add_conv2d(model=model, filters=128, kernel_size=3)

        model.add(GlobalAveragePooling2D())

        # model.add(Dense(units=256, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=self.num_classes, activation='softmax'))

        print(model.summary())

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )

        self.model = model

    @error_handling_decorator()
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

        zip_output = config.output_dir / 'data.zip'
        self._save_model_to_zip(self.model, zip_output, self.model_name)

        self.history = history.history
        self._save_history_to_zip(self.history, zip_output, self.model_name)

    @error_handling_decorator()
    def _plot_training_metrics(self) -> None:
        """
        Plot the training metrics

        Returns:
            None
        """
        if self.history is None:
            self.history = {
                'accuracy': [0.1, 0.2, 0.3, 0.4, 0.5],
                'val_accuracy': [0.15, 0.25, 0.35, 0.45, 0.55],
                'loss': [0.9, 0.8, 0.7, 0.6, 0.5],
                'val_loss': [0.95, 0.85, 0.75, 0.65, 0.55],
            }

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
    def _workflow(self) -> None:
        self._load_data_to_keras()
        self._create_model()
        self._fit_model()
        self._plot_training_metrics()
