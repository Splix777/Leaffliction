import cv2
import numpy as np
import plantcv as pcv


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
        distances = [cv2.pointPolygonTest(cnt, center, True) for cnt in cnts]
        return np.argmin(distances)


def remove_background(
    img: np.ndarray, ksize: int = 5, sigma_x: int = 1, min_area: int = 100
) -> np.ndarray:
    # Step 1: White Balance
    h, w, _ = img.shape
    roi = [w // 4, h // 4, w // 2, h // 2]
    image = pcv.white_balance(img=img, mode="hist", roi=roi)

    # Step 2: Grayscale Conversion
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Gaussian Blur
    blur = cv2.GaussianBlur(gray_img, (ksize, ksize), sigma_x)

    # Step 4: Otsu's Thresholding
    _, im_bw_otsu = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Step 5: Morphological Closing
    kernel = np.ones((20, 5), np.uint8)
    closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)

    # Step 6: Contour Detection
    contours, _ = cv2.findContours(
        closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours based on area
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area
    ]

    if not filtered_contours:
        raise ValueError("No contours found in the image.")

    # Step 7: Create Black Mask Image
    black_img = np.zeros_like(image)

    # Step 8: Find Contour of Interest
    index = find_contour(closing, filtered_contours)
    cnt = filtered_contours[index]
    mask = cv2.drawContours(black_img, [cnt], 0, (255, 255, 255), -1)

    # Step 9: Convert Mask to Single Channel
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Step 10: Apply Mask to Original Image
    maskedImg = cv2.bitwise_and(image, image, mask=mask_gray)

    # Step 11: Replace Background with White using the Original Method
    white_pix = [255, 255, 255]
    black_pix = [0, 0, 0]

    final_img = maskedImg.copy()
    h, w, channels = final_img.shape
    for y in range(h):
        for x in range(w):
            channels_xy = final_img[y, x]
            if all(channels_xy == black_pix):
                final_img[y, x] = white_pix

    return final_img


# Example usage
img = cv2.imread("path_to_image.jpg")
result = remove_background(img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
