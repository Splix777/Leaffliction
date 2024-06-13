import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

CV_COLORS = {
    "rgb": cv2.COLOR_BGR2RGB,
    "gray": cv2.COLOR_BGR2GRAY,
    "hsv": cv2.COLOR_BGR2HSV,
    "lab": cv2.COLOR_BGR2LAB,
}


def show_image(image: np.ndarray, title: str = "", color: str = "rgb"):
    if color not in CV_COLORS:
        raise ValueError(f"Invalid color space {color}")

    if color == "gray":
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image, CV_COLORS[color]))

    plt.title(title)
    plt.show()


def save_image(image: np.ndarray, output_path: str):
    if not output_path.endswith(".jpg"):
        raise ValueError("Output path must be a .jpg file")
    cv2.imwrite(output_path, image)


class Transformation:
    def __init__(self, image_path: str = None, input_dir: str = None, output_dir: str = None):
        self.image_name = image_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        if input_dir is not None and output_dir is not None:
            self.apply_transformation_from_file()
        if image_path is not None:
            self.load_image(image_path)
            self.apply_transformations()
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

    @staticmethod
    def calculate_roi(image):
        # Calculate the dimensions of the image
        height, width = image.shape[:2]

        # Define the size of the ROI rectangle relative to the image size
        roi_width = int(width * 0.9)
        roi_height = int(height * 0.9)

        # Calculate the starting coordinates for the ROI rectangle
        x = int((width - roi_width) / 2)
        y = int((height - roi_height) / 2)

        return x, y, roi_width, roi_height

    def load_image(self, image_path: str):
        self.image, _, _ = pcv.readimage(image_path)
        self.image_name = os.path.basename(image_path).split(".")[0]

        if self.output_dir is not None:
            save_image(self.image, os.path.join(
                self.output_dir,
                f"{self.image_name}_original.jpg"))
        else:
            show_image(self.image, title="Original Image", color="rgb")

    def _white_balance(self):
        if self.image is None:
            raise ValueError("Image not loaded")
        h, w, _ = self.image.shape
        roi = [w // 4, h // 4, w // 2, h // 2]
        self.image = pcv.white_balance(img=self.image, mode="hist", roi=roi)

    def _grayscale_conversion(self):
        if self.image is None:
            raise ValueError("Image not loaded")
        self.gray_image = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")

    def _apply_gaussian_blur(self):
        if self.gray_image is None:
            raise ValueError("Grayscale image not generated")
        img_thresh = pcv.threshold.binary(
            gray_img=self.gray_image,
            threshold=60,
            object_type="light",
        )
        self.gaussian_blur = pcv.gaussian_blur(
            img=img_thresh, ksize=(5, 5), sigma_x=0, sigma_y=None
        )

        if self.output_dir is not None:
            save_image(
                self.gaussian_blur,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_gaussian_blur.jpg"))
        else:
            show_image(self.gaussian_blur, title="Gaussian Blur", color="gray")

    def _create_masks(self):
        if self.gaussian_blur is None:
            raise ValueError("Gaussian blur not applied")

        # Convert image to LAB and apply binary thresholding on the 'b' channel
        # l: Lightness, a: Green-Red, b: Blue-Yellow
        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b_channel,
            threshold=135,
            object_type="light",
        )

        # Combine Gaussian blur and thresholded 'b' channel masks
        # We use the logical OR operation to combine the masks which
        # better differentiate the background from the foreground
        combined_mask = pcv.logical_or(
            bin_img1=self.gaussian_blur, bin_img2=b_thresh
        )

        # Apply the mask to the original image to effectively
        # remove the background
        masked_image = pcv.apply_mask(
            img=self.image, mask=combined_mask, mask_color="white"
        )

        # Extract 'a' and 'b' channels from the masked image
        # Helps us better analyze the masked image for disease detection
        a_channel_masked = pcv.rgb2gray_lab(rgb_img=masked_image, channel="a")
        b_channel_masked = pcv.rgb2gray_lab(rgb_img=masked_image, channel="b")

        # Threshold the 'a' channel to detect dark regions of the leaf
        # Dark regions are typically indicative of disease
        a_thresh_dark = pcv.threshold.binary(
            gray_img=a_channel_masked,
            threshold=115,
            object_type="dark",
        )
        # Threshold the 'a' channel to detect light regions of the leaf
        # Light regions are typically indicative of healthy regions
        a_thresh_light = pcv.threshold.binary(
            gray_img=a_channel_masked,
            threshold=125,
            object_type="light",
        )
        # Threshold the 'b' channel to detect disease regions
        b_thresh = pcv.threshold.binary(
            gray_img=b_channel_masked,
            threshold=135,
            object_type="light",
        )

        # Combine all the thresholded images to create a final mask
        combined_ab = pcv.logical_or(bin_img1=a_thresh_dark, bin_img2=b_thresh)
        final_combined_mask = pcv.logical_or(
            bin_img1=a_thresh_light, bin_img2=combined_ab
        )

        # XOR operation between 'a' channel dark threshold and
        # 'b' channel threshold. This helps us single out the
        # disease regions of the leaf
        self.disease_mask = pcv.logical_xor(
            bin_img1=a_thresh_dark, bin_img2=b_thresh
        )

        self.diseased_image = pcv.apply_mask(
            img=self.image, mask=self.disease_mask, mask_color="white"
        )

        # Applies the filled mask to the masked image which
        # helps better remove the background
        self.refined_mask = pcv.fill(bin_img=final_combined_mask, size=200)

        # Apply the filled mask to the masked image
        self.final_masked_image = pcv.apply_mask(
            img=masked_image, mask=self.refined_mask, mask_color="white"
        )

        if self.output_dir is not None:
            save_image(
                self.diseased_image,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_diseased.jpg"))
        else:
            show_image(
                self.diseased_image,
                title="Cropped Diseased Image",
                color="rgb")

    def _create_roi_and_objects(self):
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

        self.shape_image = pcv.analyze.size(img=self.image, labeled_mask=mask)

        self.boundary_image_h = pcv.analyze.bound_horizontal(
            img=self.image,
            labeled_mask=mask,
            line_position=250,
        )

        if self.output_dir:
            save_image(self.shape_image, os.path.join(
                self.output_dir,
                f"{self.image_name}_analyze_object.jpg"))
            save_image(self.boundary_image_h, os.path.join(
                self.output_dir,
                f"{self.image_name}_roi_objects.jpg"))
        else:
            show_image(self.shape_image, title="Analyze Object", color="rgb")
            show_image(self.boundary_image_h, title="ROI Objects", color="rgb")

    def _pseudolandmarks(self):
        # Generate pseudolandmarks
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.image,
            mask=self.disease_mask,
        )

        # Create a copy of the original image to draw the landmarks on
        self.image_with_landmarks = copy.deepcopy(self.image)

        # Function to draw landmarks on the image
        def draw_landmarks(image, landmarks, color):
            for point in landmarks:
                x, y = int(point[0][0]), int(point[0][1])
                cv2.circle(
                    image, (x, y), radius=3, color=color, thickness=-1
                )

        # Draw top landmarks in blue
        draw_landmarks(self.image_with_landmarks, top, (255, 0, 0))
        # Draw bottom landmarks in magenta
        draw_landmarks(self.image_with_landmarks, bottom, (255, 0, 255))
        # Draw center landmarks in orange
        draw_landmarks(self.image_with_landmarks, center_v, (0, 165, 255))

        if self.output_dir:
            save_image(
                self.image_with_landmarks,
                os.path.join(
                    self.output_dir,
                    f"{self.image_name}_pseudolandmarks.jpg"))
        else:
            show_image(
                self.image_with_landmarks,
                title="Image with Pseudolandmarks",
                color="rgb")

    def _color_histogram(self):
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
        histograms = {
            name: pcv.outputs.observations['default_1'][key]['value']
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
        if self.output_dir:
            plt.savefig(os.path.join(
                self.output_dir,
                f"{self.image_name}_color_histogram.jpg"))
        else:
            plt.show()
        # Close the figure after saving or showing it
        plt.close()

    def apply_transformations(self):
        self._white_balance()
        self._grayscale_conversion()
        self._apply_gaussian_blur()
        self._create_masks()
        self._create_roi_and_objects()
        self._pseudolandmarks()
        self._color_histogram()

    def apply_transformation_from_file(self):
        if self.output_dir is None:
            raise ValueError("Output directory not specified")
        for subdir, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    image_path = os.path.join(subdir, file)
                    self.load_image(str(image_path))
                    self.apply_transformations()


if __name__ == "__main__":
    img = "../data/augmented_directory/images/Apple_healthy/image (65).JPG"
    transform = Transformation(output_dir="output")
    transform.load_image(img)

