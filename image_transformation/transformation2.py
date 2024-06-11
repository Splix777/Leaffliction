import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


class Transformation:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.image = None
        self.gray_image = None
        self.gaussian_blur = None

    @staticmethod
    def find_contour(image, cnts):
        contains = []
        y_ri, x_ri = image.shape[:2]
        center = (x_ri // 2, y_ri // 2)

        for cc in cnts:
            yn = cv2.pointPolygonTest(cc, center, False)
            contains.append(yn)

        val = [contains.index(temp) for temp in contains if temp > 0]
        if val:
            return val[0]
        else:
            # Return the index of the contour with the maximum area
            areas = [cv2.contourArea(cnt) for cnt in cnts]
            max_area_idx = np.argmax(areas)
            # Check if the largest contour is near the center
            if cv2.pointPolygonTest(cnts[max_area_idx], center, False) > 0:
                return max_area_idx
            # Otherwise, return the contour closest to the center
            distances = [
                cv2.pointPolygonTest(cnt, center, True) for cnt in cnts
            ]
            return np.argmin(distances)

    def load_image(self, image_path: str):
        self.image, _, _ = pcv.readimage(image_path)
        if self.debug:
            plt.imshow(X=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.show()

    def white_balance(self):
        if self.image is None:
            raise ValueError("Image not loaded")
        h, w, _ = self.image.shape
        roi = [w // 4, h // 4, w // 2, h // 2]
        self.image = pcv.white_balance(img=self.image, mode="hist", roi=roi)
        if self.debug:
            plt.imshow(X=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            plt.title("White Balanced Image")
            plt.show()

    def grayscale_conversion(self):
        if self.image is None:
            raise ValueError("Image not loaded")
        self.gray_image = pcv.rgb2gray_hsv(rgb_img=self.image, channel="s")
        if self.debug:
            plt.imshow(X=self.gray_image, cmap="gray")
            plt.title("Grayscale Image")
            plt.show()

    def apply_gaussian_blur(self):
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
        if self.debug:
            plt.imshow(X=self.gaussian_blur, cmap="gray")
            plt.title("Gaussian Blurred Image")
            plt.show()

    def create_mask(self):
        if self.gaussian_blur is None:
            raise ValueError("Gaussian blur not applied")

        b_channel = pcv.rgb2gray_lab(rgb_img=self.image, channel="b")
        b_thresh = pcv.threshold.binary(
            gray_img=b_channel,
            threshold=200,
            object_type="light",
        )

        bg = pcv.logical_or(bin_img1=self.gaussian_blur, bin_img2=b_thresh)

        masked = pcv.apply_mask(img=self.image, mask=bg, mask_color="white")

        a_masked = pcv.rgb2gray_lab(rgb_img=masked, channel="a")
        b_masked = pcv.rgb2gray_lab(rgb_img=masked, channel="b")

        if self.debug:
            plt.imshow(X=a_masked, cmap="gray")
            plt.title("A Channel Masked Image")
            plt.show()

            plt.imshow(X=b_masked, cmap="gray")
            plt.title("B Channel Masked Image")
            plt.show()

            plt.imshow(X=cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
            plt.title("Masked Image")
            plt.show()

        a_thresh_dark = pcv.threshold.binary(
            gray_img=a_masked,
            threshold=115,
            object_type="dark",
        )
        a_thresh_light = pcv.threshold.binary(
            gray_img=a_masked,
            threshold=135,
            object_type="light",
        )
        b_thresh = pcv.threshold.binary(
            gray_img=b_masked,
            threshold=128,
            object_type="light",
        )

        if self.debug:
            plt.imshow(X=a_thresh_dark, cmap="gray")
            plt.title("A Channel Dark Threshold")
            plt.show()

            plt.imshow(X=a_thresh_light, cmap="gray")
            plt.title("A Channel Light Threshold")
            plt.show()

            plt.imshow(X=b_thresh, cmap="gray")
            plt.title("B Channel Threshold")
            plt.show()

        ab1 = pcv.logical_or(bin_img1=a_thresh_dark, bin_img2=b_thresh)
        ab = pcv.logical_or(bin_img1=a_thresh_light, bin_img2=ab1)

        if self.debug:
            plt.imshow(X=ab, cmap="gray")
            plt.title("A and B Channel Threshold Combined")
            plt.show()

        xor_img = pcv.logical_xor(bin_img1=a_thresh_dark, bin_img2=b_thresh)

        xor_img_color = pcv.apply_mask(
            img=self.image, mask=xor_img, mask_color="white"
        )

        filled_ab = pcv.fill(bin_img=ab, size=200)

        filled_mask = pcv.apply_mask(
            img=masked, mask=filled_ab, mask_color="white"
        )

        if self.debug:
            plt.imshow(X=cv2.cvtColor(xor_img_color, cv2.COLOR_BGR2RGB))
            plt.title("XOR A and B Channel")
            plt.show()

            plt.imshow(X=cv2.cvtColor(filled_mask, cv2.COLOR_BGR2RGB))
            plt.title("Filled A and B Channel")
            plt.show()


if __name__ == "__main__":
    img_path = "../leaves/images/Apple_Black_rot/image (1).JPG"
    transform = Transformation(debug=True)
    transform.load_image(img_path)
    transform.white_balance()
    transform.grayscale_conversion()
    transform.apply_gaussian_blur()
    transform.create_mask()
