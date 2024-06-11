import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


def find_contour(image, cnts):
    contains = []
    y_ri, x_ri = image.shape
    for cc in cnts:
        yn = cv2.pointPolygonTest(cc, (x_ri // 2, y_ri // 2), False)
        contains.append(yn)

    val = [contains.index(temp) for temp in contains if temp > 0]
    print(contains)
    if val:
        return val[0]
    else:
        # Return the index of the contour with the maximum area
        areas = [cv2.contourArea(cnt) for cnt in cnts]
        return np.argmax(areas)


def remove_background(img: np.ndarray, ksize: int, sigma_x: int) -> np.ndarray:
    # Step 1: White Balance
    h, w, _ = img.shape
    roi = [w // 4, h // 4, w // 2, h // 2]
    image = pcv.white_balance(img=img, mode="hist", roi=roi)

    # Step 2: Grayscale Conversion
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian Blur
    blur = cv2.GaussianBlur(gray_img, (ksize, ksize), sigma_x)

    # Step 4: Otsu's Thresholding
    ret_otsu, im_bw_otsu = cv2.threshold(
        blur, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 5: Morphological closing
    kernel = np.ones((20, 5), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Step 6: Contour Detection
    contours, hierarchy = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contours found in the image.")

    # Step 7: Create Black Mask Image
    black_img = np.zeros_like(image)

    # Step 8: Find Contour of Interest
    index = find_contour(closing, contours)
    cnt = contours[index]
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)

    # Step 9: Convert Mask to Single Channel
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Step 10: Apply Mask to Original Image
    maskedImg = cv2.bitwise_and(image, image, mask=mask_gray)

    # Step 11: Replace Background with White using Numpy
    white_pix = [255, 255, 255]
    black_pix = [0, 0, 0]
    final_img = maskedImg
    h, w, channels = final_img.shape
    for x in range(0, w):
        for y in range(0, h):
            channels_xy = final_img[y, x]
            if all(channels_xy == black_pix):
                final_img[y, x] = white_pix

    return final_img


def plantcv_approach(img: np.ndarray):
    image = pcv.white_balance(img=img, mode="hist", roi=[5, 5, 80, 80])

    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # threshold_light = pcv.threshold.binary(
    #     gray_img=gray_img, threshold=95, object_type="dark"
    # )

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, threshold_light = cv2.threshold(
        blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return threshold_light


if __name__ == "__main__":
    sample_image = "../../leaves/images/Apple_Black_rot/image (96).JPG"
    img = cv2.imread(sample_image)

    blurred_image = remove_background(img, 5, 0)
    cvapproach = plantcv_approach(img)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()

    plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    plt.title("Blurred Image")
    plt.show()

    plt.imshow(cv2.cvtColor(cvapproach, cv2.COLOR_BGR2RGB))
    plt.title("CVPlant Image")
    plt.show()
