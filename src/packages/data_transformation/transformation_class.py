import os
from typing import Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
from tqdm import tqdm

from .helper_functions import (
    save_image,
    show_image,
    count_images
)
from ..utils.decorators import (
    error_handling_decorator,
    timeit,
    ensure_image_loaded
)


class Transformation:
    """
    Initialize the transformation utility with optional
    image path, input directory, and output directory.

    Args:
        image_path (str): The path to the image file.
            Default is None.
        input_dir (str): The input directory path.
            Default is None.
        output_dir (str): The output directory path.
            Default is None.

    Returns:
        None
    """
    def __init__(self, image_path: str = None, input_dir: str = None,
                 output_dir: str = None, keep_dir_structure: bool = True
                 ) -> None:
        # Intermediate images
        self.gray_image = None
        self.refined_mask = None
        self.final_masked_image = None
        self.combined_mask = None
        self.disease_mask = None
        # Final images
        self.image = None
        self.gaussian_blur = None
        self.diseased_image = None
        self.boundary_image_h = None
        self.shape_image = None
        self.image_with_landmarks = None
        # Image paths
        self.image_path = image_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.keep_dir_structure = keep_dir_structure
        # Apply transformations
        if input_dir is not None and output_dir is not None:
            self._apply_transformation_from_file()
        elif image_path is not None:
            self._load_image(image_path)
            self._run_workflow()

    @property
    def image_name(self) -> str:
        if self.image_path:
            return os.path.basename(self.image_path).split(".")[0]
        return "unknown_image"

    @staticmethod
    def calculate_roi(image: np.ndarray) -> tuple[int, int, int, int]:
        """
        Calculate the Region of Interest (ROI) dimensions
        within an image.

        Args:
            image: Input image for which the ROI dimensions
                are calculated.

        Raises:
            ValueError: If the image is not loaded or has
                invalid dimensions.

        Returns:
            Tuple[int, int, int, int]: The starting coordinates (x, y)
            and dimensions (width, height) of the ROI rectangle.
        """
        if image is None:
            raise ValueError("Image not loaded")

        # Calculate the dimensions of the image
        height, width = image.shape[:2]

        # Ensure that the image dimensions are not zero
        if width == 0 or height == 0:
            raise ValueError("Image dimensions cannot be zero")

        # Define the size of the ROI rectangle relative to the image size
        roi_width = int(width * 0.9)
        roi_height = int(height * 0.9)

        # Adjust the ROI dimensions to be at least 1 pixel
        roi_width = max(roi_width, 1)
        roi_height = max(roi_height, 1)

        # Calculate the starting coordinates for the ROI rectangle
        x = int((width - roi_width) / 2)
        y = int((height - roi_height) / 2)

        return x, y, roi_width, roi_height

    @staticmethod
    def draw_landmarks(image: np.ndarray, landmarks: np.ndarray,
                       color: tuple[int, int, int]) -> None:
        """
        Draw landmarks on the given image using the provided
        coordinates and color.

        Args:
            image (np.ndarray): The image on which
                landmarks will be drawn.
            landmarks (np.ndarray): The array of
                landmark coordinates.
            color (tuple): The color of the landmarks.

        Raises:
            RuntimeError: If an error occurs while drawing landmarks.

        Returns:
            None
        """
        for point in landmarks:
            try:
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(
                    image, center=(x, y), radius=3, color=color,
                    thickness=-1
                )
            except Exception as ie:
                raise RuntimeError(
                    f"Failed to draw landmarks: {ie}") from ie

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _load_image(self, image_path: str) -> None:
        """
        Load an image from the specified image path and process it.

        Args:
            image_path (str): The path to the image file to load.

        Raises:
            ValueError: If the image path is invalid.

        Returns:
            None
        """
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError(f"Invalid image path: {image_path}")

        self.image_path = image_path
        self.image, _, _ = pcv.readimage(image_path)
        if self.image is None:
            raise ValueError("Failed to load the image")
        self._save_or_show_img(self.image, "original")

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _white_balance(self) -> None:
        """
        Apply white balance correction to the loaded image
        using a specified region of interest (ROI).

        Raises:
            ValueError: If the image has not been loaded.
            ValueError: If the image dimensions are invalid.

        Returns:
            None
        """
        # Get the dimensions of the image
        h, w, _ = self.image.shape
        if h <= 0 or w <= 0:
            raise ValueError("Image dimensions cannot be zero")

        # Define the region of interest (ROI) as a rectangle
        roi = [w // 4, h // 4, w // 2, h // 2]
        if any(dim <= 0 for dim in roi):
            raise ValueError("Invalid ROI dimensions")

        self.image = pcv.white_balance(img=self.image, mode="hist", roi=roi)

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _grayscale_conversion(self) -> None:
        """
        Convert the loaded image to grayscale using the
        HSV color space.

        Raises:
            ValueError: If the image has not been loaded.
            RuntimeError: If the grayscale conversion fails.

        Returns:
            None
        """
        self.gray_image = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _apply_gaussian_blur(self) -> None:
        """
        Apply Gaussian blur to the generated grayscale image.

        Raises:
            ValueError: If the grayscale image has not been generated.
            RuntimeError: If the Gaussian blur fails.

        Returns:
            None
        """
        img_thresh = pcv.threshold.binary(
            gray_img=self.gray_image,
            threshold=60,
            object_type="light",
        )
        self.gaussian_blur = pcv.gaussian_blur(
            img=img_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )
        self._save_or_show_img(
            image=self.gaussian_blur,
            image_suffix="gaussian_blur",
            color="gray"
        )

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _create_masks(self) -> None:
        """
        Perform various image processing computations
        to create masks for disease detection, including
        thresholding, masking, and logical operations.

        Raises:
            ValueError: If the Gaussian blur has not been applied.

        Returns:
            None
        """
        # Convert image to LAB and apply binary thresholding on the 'b' channel
        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b_channel,
            threshold=135,
            object_type="light",
        )

        # Combine Gaussian blur and thresholded 'b' channel masks
        combined_mask = pcv.logical_or(
            bin_img1=self.gaussian_blur, bin_img2=b_thresh
        )

        # Apply the mask to the original image to remove the background
        masked_image = pcv.apply_mask(
            img=self.image, mask=combined_mask, mask_color="white"
        )

        # Get 'a' and 'b' channels from the masked image for further analysis
        a_channel_masked = pcv.rgb2gray_lab(rgb_img=masked_image,
                                            channel="a")
        b_channel_masked = pcv.rgb2gray_lab(rgb_img=masked_image,
                                            channel="b")

        # Threshold the 'a' channel to detect dark regions
        # of the leaf (indicative of disease)
        a_thresh_dark = pcv.threshold.binary(
            gray_img=a_channel_masked,
            threshold=115,
            object_type="dark",
        )
        # Threshold the 'a' channel to detect light regions
        # of the leaf (indicative of healthy regions)
        a_thresh_light = pcv.threshold.binary(
            gray_img=a_channel_masked,
            threshold=125,
            object_type="light",
        )
        # Threshold the 'b' channel to detect light regions
        # of the leaf (indicative of healthy regions)
        b_thresh = pcv.threshold.binary(
            gray_img=b_channel_masked,
            threshold=135,
            object_type="light",
        )

        # Combine all the thresholded images to create a final mask
        combined_ab = pcv.logical_or(bin_img1=a_thresh_dark,
                                     bin_img2=b_thresh)
        final_combined_mask = pcv.logical_or(
            bin_img1=a_thresh_light, bin_img2=combined_ab
        )

        # XOR operation between 'a' channel dark threshold and 'b'
        # channel threshold to isolate disease regions
        self.disease_mask = pcv.logical_xor(
            bin_img1=a_thresh_dark, bin_img2=b_thresh
        )

        # Apply the disease mask to the original image
        self.diseased_image = pcv.apply_mask(
            img=self.image, mask=self.disease_mask, mask_color="white"
        )

        # Apply the filled mask to the masked image
        self.refined_mask = pcv.fill(bin_img=final_combined_mask, size=200)
        self.final_masked_image = pcv.apply_mask(
            img=masked_image, mask=self.refined_mask, mask_color="white"
        )

        self._save_or_show_img(self.diseased_image, "diseased")

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _create_roi_and_objects(self) -> None:
        """
        Create regions of interest (ROI) and analyze objects
        within the image using image processing techniques.

        Raises:
            ValueError: If the image or disease mask is not loaded.
            RuntimeError: If an error occurs during the creation
                of ROI and objects.

        Returns:
            None
        """
        x, y, w, h = self.calculate_roi(self.image)

        roi = pcv.roi.rectangle(
            img=self.image,
            x=x,
            y=y,
            h=h,
            w=w,
        )
        disease_mask = pcv.threshold.binary(
            gray_img=self.disease_mask, threshold=127, object_type="light"
        )

        mask = pcv.roi.filter(
            mask=disease_mask, roi=roi, roi_type="partial"
        )

        self.shape_image = pcv.analyze.size(img=self.image,
                                            labeled_mask=mask)

        self.boundary_image_h = pcv.analyze.bound_horizontal(
            img=self.image,
            labeled_mask=mask,
            line_position=250,
        )

        self._save_or_show_img(self.shape_image, "analyze_object")
        self._save_or_show_img(self.boundary_image_h, "roi_objects")

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _pseudolandmarks(self) -> None:
        """
        Generate pseudolandmarks on the image based on the
        provided image and disease mask, and draw these
        landmarks on a copy of the original image.

        Raises:
            ValueError: If the image or disease mask is not loaded.
            RuntimeError: If an error occurs during the creation of
                pseudolandmarks or drawing landmarks.

        Returns:
            None
        """
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.image,
            mask=self.disease_mask,
        )

        # Create a copy of the original image to draw the landmarks on
        self.image_with_landmarks = np.copy(self.image)

        # Draw top landmarks in blue
        self.draw_landmarks(
            image=self.image_with_landmarks,
            landmarks=top,
            color=(255, 0, 0))
        # Draw bottom landmarks in magenta
        self.draw_landmarks(
            image=self.image_with_landmarks,
            landmarks=bottom,
            color=(255, 0, 255)
        )
        # Draw center landmarks in orange
        self.draw_landmarks(
            image=self.image_with_landmarks,
            landmarks=center_v,
            color=(0, 165, 255)
        )

        self._save_or_show_img(
            image=self.image_with_landmarks,
            image_suffix="pseudolandmarks")

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _color_histogram(self) -> None:
        """
        Calculate and plot color histograms for the loaded
        image based on the provided disease mask.

        Returns:
            None
        """
        # Calculate the histogram of the image
        pcv.analyze.color(
            rgb_img=self.image,
            labeled_mask=self.disease_mask,
            colorspaces='all'
        )

        # Define a list of the histogram keys to be retrieved
        histogram_keys = [
            ('Blue', 'blue_frequencies'),
            ('Blue-Yellow', 'blue-yellow_frequencies'),
            ('Green', 'green_frequencies'),
            ('Green-Magenta', 'green-magenta_frequencies'),
            ('Hue', 'hue_frequencies'),
            ('Lightness', 'lightness_frequencies'),
            ('Red', 'red_frequencies'),
            ('Saturation', 'saturation_frequencies'),
            ('Value', 'value_frequencies')
        ]

        # Retrieve the histogram data from pcv.outputs
        observations = pcv.outputs.observations['default_1']
        histograms = {
            name: observations[key]['value']
            for name, key in histogram_keys
        }

        # Plot all histograms on the same plot using matplotlib
        plt.figure(figsize=(10, 7))

        # Define a list of colors for each histogram
        colors = [
            'blue', 'yellow', 'green', 'pink', 'purple',
            'brown', 'red', 'cyan', 'orange'
        ]

        for (title, data), color in zip(histograms.items(), colors):
            plt.plot(data, color=color, label=title)

        plt.title("Color Histograms")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Proportion of Pixels (%)")
        plt.legend()
        plt.tight_layout()
        if self.output_dir is not None and not self.keep_dir_structure:
            plt.savefig(os.path.join(
                self.output_dir,
                f"{self.image_name}_color_histogram.jpg"))
        elif self.output_dir is None and not self.keep_dir_structure:
            plt.show()
        plt.close()

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    @ensure_image_loaded
    def _run_workflow(self) -> None:
        """
        Run a workflow of image transformations including
        white balance correction, grayscale conversion, Gaussian blur,
        mask creation, ROI and object analysis, pseudolandmark
        generation, and color histogram plotting.

        Raises:
            Any exceptions that occur during the image transformation
                workflow.

        Returns:
            None
        """
        # Step 1: Apply white balance correction
        self._white_balance()
        # Step 2: Convert the image to grayscale
        self._grayscale_conversion()
        # Step 3: Apply Gaussian blur to the grayscale image
        self._apply_gaussian_blur()
        # Step 4: Create masks for disease detection
        self._create_masks()
        # Step 5: Create regions of interest and analyze objects
        self._create_roi_and_objects()
        # Step 6: Generate pseudolandmarks on the image
        self._pseudolandmarks()
        if self.output_dir is None:
            # Step 7: Plot color histograms for the image
            self._color_histogram()

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    @timeit
    def _apply_transformation_from_file(self) -> None:
        """
        Apply a series of image transformations to images
        loaded from files in the input directory
        and save the processed images to the output directory.

        Raises:
            ValueError: If the output directory is not specified.

        Returns:
            None
        """
        total_images = count_images(self.input_dir)
        progress_bar = tqdm(
            total=total_images, desc="Processing images"
        )
        # Traverse through all files in input_dir
        for subdir, _, files in os.walk(self.input_dir):
            for file in files:
                if file.casefold().endswith(".jpg"):
                    self._process_single_image(subdir, file)
                    progress_bar.update(1)

        progress_bar.close()

    def _process_single_image(self, subdir: str, file: str) -> None:
        """
        Process a single image from the input directory
        and save the processed images to the output directory.

        Args:
            subdir (str): The subdirectory within the input directory.
            file (str): The image file to process.

        Returns:
            None
        """
        image_path = os.path.join(subdir, file)
        self._load_image(image_path)
        self._run_workflow()

        if self.keep_dir_structure:
            relative_path = os.path.relpath(subdir, self.input_dir)
            output_subdir = os.path.join(self.output_dir, relative_path)
            self._save_batch_images(output_subdir)

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _save_batch_images(self, output_subdir: str) -> None:
        """
        Save all the intermediate and final images
        to the specified output subdirectory.

        Args:
            output_subdir (str): The sub-subdirectory
                within the output directory to save the images.

        Returns:
            None
        """
        if self.output_dir:
            save_image(
                self.image,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_original.jpg"))
            save_image(
                self.gaussian_blur,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_gaussian_blur.jpg"))
            save_image(
                self.diseased_image,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_diseased.jpg"))
            save_image(
                self.shape_image,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_analyze_object.jpg"))
            save_image(
                self.boundary_image_h,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_roi_objects.jpg"))
            save_image(
                self.image_with_landmarks,
                os.path.join(
                    output_subdir,
                    f"{self.image_name}_pseudolandmarks.jpg"))

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    def _save_or_show_img(self, image: np.ndarray, image_suffix: str,
                          color: str = 'rgb') -> None:
        """
        Save the image if the output directory is set
        and the directory structure should not be kept.
        Otherwise, show the image.

        Args:
            image (np.ndarray): The image to be saved or shown.
            image_suffix (str): Suffix to be added to
                the image name for saving.
            color (str): The color space of the image.
        """
        if self.output_dir is not None and not self.keep_dir_structure:
            save_image(
                image,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_{image_suffix}.jpg"
                )
            )
        elif self.output_dir is None and not self.keep_dir_structure:
            show_image(
                image,
                title=f"Image with {image_suffix.capitalize()}",
                color=color
            )

    @error_handling_decorator(handle_exceptions=(ValueError, RuntimeError))
    @ensure_image_loaded
    def get_images(self) -> dict[str, Any]:
        """
        Return the final images generated
        during the image transformation workflow.

        Returns:
            dict: A dictionary containing the final images.
        """
        return {
            "original": self.image,
            "gaussian_blur": self.gaussian_blur,
            "diseased": self.diseased_image,
            "analyze_object": self.shape_image,
            "roi_objects": self.boundary_image_h,
            "pseudolandmarks": self.image_with_landmarks,
            "masked": self.final_masked_image,
        }


if __name__ == "__main__":
    img = "../../leaves/images/Apple_healthy/image (65).JPG"
    Transformation(image_path=img)
