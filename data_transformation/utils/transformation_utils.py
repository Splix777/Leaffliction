import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
from tqdm import tqdm

CV_COLORS = {
    "rgb": cv2.COLOR_BGR2RGB,
    "gray": cv2.COLOR_BGR2GRAY,
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
}


def show_image(image: np.ndarray, title: str = "", color: str = "rgb"):
    """
    Show an image with a specified title and color space.

    Args:
        image (np.ndarray): The image to display.
        title (str): The title of the displayed image.
            Default is an empty string.
        color (str): The color space of the image.
            Default is "rgb".

    Raises:
        ValueError: If the color space is not valid.

    Returns:
        None
    """
    valid_colors = set(CV_COLORS.keys())

    if color not in valid_colors:
        raise ValueError(f"Invalid color space {color}")

    if color == "gray":
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, CV_COLORS[color]))

    plt.title(title)
    plt.show()


def save_image(image: np.ndarray, output_path: str = None):
    """
    Save the image to a specified output path as a .jpg file.

    Args:
        image (np.ndarray): The image to save.
        output_path (str): The path where the image will be
            saved as a .jpg file.

    Raises:
        ValueError: If the output path does not end with ".jpg".

    Returns:
        None
    """
    if not output_path.endswith(".jpg"):
        raise ValueError("Output path must be a .jpg file")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def count_image(directory: str) -> int:
    """
    Count the number of images in each category within the
    specified directory.

    Args:
        directory (str): The directory containing the images.

    Returns:
        int: The total number of images in the directory.
    """
    return sum(
        len([file for file in files if file.casefold().endswith("jpg")])
        for subdir, _, files in os.walk(directory)
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

    def __init__(self, image_path: str = None, input_dir: str = None,
                 output_dir: str = None, keep_dir_structure: bool = False
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
        self.image_name = None
        self.image_path = image_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.keep_dir_structure = keep_dir_structure
        # Apply transformations
        if input_dir is not None and output_dir is not None:
            self.apply_transformation_from_file()
        elif image_path is not None:
            self.load_image(image_path)
            self.run_workflow()

    def load_image(self, image_path: str) -> None:
        """
        Load an image from the specified image path and process it.

        Args:
            image_path (str): The path to the image file to load.

        Raises:
            ValueError: If the image path is invalid.
            RuntimeError: If the image processing fails.

        Returns:
            None
        """
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError(f"Invalid image path: {image_path}")

        try:
            self.image, _, _ = pcv.readimage(image_path)
            self.image_name = os.path.basename(image_path).split(".")[0]
            self.image_path = image_path

            if self.output_dir is not None and not self.keep_dir_structure:
                save_image(self.image, os.path.join(
                    self.output_dir,
                    f"{self.image_name}_original.jpg"))
            elif self.output_dir is None:
                show_image(self.image, title="Original Image", color="rgb")

        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}") from e

    def _white_balance(self) -> None:
        """
        Apply white balance correction to the loaded image
        using a specified region of interest (ROI).

        Raises:
            ValueError: If the image has not been loaded.
            RuntimeError: If the white balance correction fails.

        Returns:
            None
        """
        if self.image is None:
            raise ValueError("Image not loaded")

        # Get the dimensions of the image
        h, w, _ = self.image.shape
        if h <= 0 or w <= 0:
            raise ValueError("Image dimensions cannot be zero")

        # Define the region of interest (ROI) as a rectangle
        roi = [w // 4, h // 4, w // 2, h // 2]
        if any(dim <= 0 for dim in roi):
            raise ValueError("Invalid ROI dimensions")

        try:
            self.image = pcv.white_balance(
                img=self.image, mode="hist", roi=roi)

        except Exception as e:
            raise RuntimeError(f"Failed to apply white balance: {e}") from e

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
        if self.image is None:
            raise ValueError("Image not loaded")

        try:
            self.gray_image = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")

        except Exception as e:
            raise RuntimeError(
                "Failed to convert image to grayscale: {e}") from e

    def _apply_gaussian_blur(self) -> None:
        """
        Apply Gaussian blur to the generated grayscale image.

        Raises:
            ValueError: If the grayscale image has not been generated.
            RuntimeError: If the Gaussian blur fails.

        Returns:
            None
        """
        if self.gray_image is None:
            raise ValueError("Grayscale image not generated")

        try:
            img_thresh = pcv.threshold.binary(
                gray_img=self.gray_image,
                threshold=60,
                object_type="light",
            )
            self.gaussian_blur = pcv.gaussian_blur(
                img=img_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
            )

        except Exception as e:
            raise RuntimeError(
                "Failed to apply Gaussian blur: {e}") from e

        if self.output_dir is not None and not self.keep_dir_structure:
            save_image(
                self.gaussian_blur,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_gaussian_blur.jpg"))
        elif self.output_dir is None:
            show_image(self.gaussian_blur, title="Gaussian Blur", color="gray")

    def _create_masks(self) -> None:
        """
        Attempt to create masks by calling the _mask_computations
        method and handle specific exceptions that may occur
        during the process.

        Raises:
            ValueError: If the Gaussian blur has not been applied.
            RuntimeError: If the mask computations fail.
            Exception: If an unexpected error occurs during the process.

        Returns:
            None
        """
        try:
            self._mask_computations()
        except ValueError as ve:
            print(f"ValueError occurred creating masks: {ve}")
        except RuntimeError as re:
            print(f"RuntimeError occurred creating masks: {re}")
        except Exception as e:
            print(f"Unexpected error occurred creating masks: {e}")

    def _mask_computations(self) -> None:
        """
        Perform various image processing computations
        to create masks for disease detection, including
        thresholding, masking, and logical operations.

        Raises:
            ValueError: If the Gaussian blur has not been applied.

        Returns:
            None
        """
        if any(img is None for img in [self.image, self.gaussian_blur]):
            raise ValueError("Gaussian blur not applied")

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

        if self.output_dir is not None and not self.keep_dir_structure:
            save_image(
                self.diseased_image,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_diseased.jpg"))
        elif self.output_dir is None:
            show_image(
                self.diseased_image,
                title="Cropped Diseased Image",
                color="rgb")

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
        if any(img is None for img in [self.image, self.disease_mask]):
            raise ValueError("Image or disease mask not loaded")

        x, y, w, h = self.calculate_roi(self.image)
        try:
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

        except Exception as e:
            raise RuntimeError(f"Failed to create ROI and objects: {e}") from e

        if self.output_dir is not None and not self.keep_dir_structure:
            save_image(self.shape_image, os.path.join(
                self.output_dir,
                f"{self.image_name}_analyze_object.jpg"))
            save_image(self.boundary_image_h, os.path.join(
                self.output_dir,
                f"{self.image_name}_roi_objects.jpg"))
        elif self.output_dir is None:
            show_image(self.shape_image, title="Analyze Object", color="rgb")
            show_image(self.boundary_image_h, title="ROI Objects", color="rgb")

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
        if any(img is None for img in [self.image, self.disease_mask]):
            raise ValueError("Image or disease mask not loaded")

        try:
            top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
                img=self.image,
                mask=self.disease_mask,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to create pseudolandmarks: {e}") from e

        # Create a copy of the original image to draw the landmarks on
        self.image_with_landmarks = np.copy(self.image)

        # Function to draw landmarks on the image
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

        # Draw top landmarks in blue
        draw_landmarks(self.image_with_landmarks, top, (255, 0, 0))
        # Draw bottom landmarks in magenta
        draw_landmarks(self.image_with_landmarks, bottom, (255, 0, 255))
        # Draw center landmarks in orange
        draw_landmarks(self.image_with_landmarks, center_v, (0, 165, 255))

        if self.output_dir is not None and not self.keep_dir_structure:
            save_image(
                self.image_with_landmarks,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_pseudolandmarks.jpg"))
        elif self.output_dir is None:
            show_image(
                self.image_with_landmarks,
                title="Image with Pseudolandmarks",
                color="rgb")

    def _color_histogram(self) -> None:
        """
        Calculate and plot color histograms for the
        loaded image based on the provided disease mask.

        Raises:
            ValueError: If the image or disease mask is not loaded.
            RuntimeError: If an error occurs during the
                plotting of color histograms.

        Returns:
            None
        """
        if any(img is None for img in [self.image, self.disease_mask]):
            raise ValueError("Image or disease mask not loaded")

        try:
            self._draw_histogram()

        except Exception as e:
            raise RuntimeError(f"Failed to plot color histograms: {e}") from e

    def _draw_histogram(self) -> None:
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
        elif self.output_dir is None:
            plt.show()
        plt.close()

    def run_workflow(self) -> None:
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
        try:
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

        except Exception as e:
            print(f"An error occurred during image transformation: {e}")

    def apply_transformation_from_file(self) -> None:
        """
        Apply a series of image transformations to images
        loaded from files in the input directory
        and save the processed images to the output directory.

        Raises:
            ValueError: If the output directory is not specified.

        Returns:
            None
        """
        if any(
                dir_path is None
                for dir_path in [self.input_dir, self.output_dir]
        ):
            raise ValueError("Output directory not specified")

        total_images = count_image(self.input_dir)
        progress_bar = tqdm(
            total=total_images, desc="Processing images"
        )
        # Traverse through all files in input_dir
        for subdir, _, files in os.walk(self.input_dir):
            for file in files:
                if file.casefold().endswith(".jpg"):
                    image_path = os.path.join(subdir, file)
                    try:
                        self.load_image(image_path)
                    except (ValueError, RuntimeError) as e:
                        print(f"Failed to load image {image_path}: {e}")
                        continue

                    try:
                        self.run_workflow()
                    except Exception as e:
                        print(f"Failed to process image {image_path}: {e}")
                        continue

                    if self.keep_dir_structure:
                        relative_path = os.path.relpath(subdir, self.input_dir)
                        output_subdir = os.path.join(self.output_dir,
                                                     relative_path)
                        self.save_transformed_images(output_subdir)

                    progress_bar.update(1)

        progress_bar.close()

    def save_transformed_images(self, output_subdir: str) -> None:
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


if __name__ == "__main__":
    img = "../../leaves/images/Apple_healthy/image (65).JPG"
    Transformation(image_path=img)
