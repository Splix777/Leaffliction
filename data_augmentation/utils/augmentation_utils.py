import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def barrel_distortion(image: np.ndarray, k1: float = 0.3) -> np.ndarray:
    """
    Applies barrel distortion to the input image. Make the image
    look as thought its being viewed through a fisheye lens.

    Args:
    image (np.ndarray): Input image
    k1 (float): Primary radial distortion coefficient

    Returns:
    np.ndarray: Distorted image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(k1, (float, int)):
        raise ValueError(
            "The distortion coefficient k1 must be a numeric value."
        )

    # Extracts the image dimensions
    rows, cols = image.shape[:2]

    if rows == 0 or cols == 0:
        raise ValueError("The image dimensions must be non-zero.")

    # Creates a zero matrix of size 4x1. This matrix will store the
    # distortion coefficients.
    dist_coef = np.zeros(shape=(4, 1), dtype=np.float64)

    # Sets the primary radial distortion coefficient Which
    # determines how much the image is distorted radially
    # from the center.
    dist_coef[0, 0] = k1

    # Creates a 3x3 identity matrix of type float32. This matrix
    # will be modified to represent the camera's intrinsic parameters.
    K = np.eye(N=3, dtype=np.float32)

    # Sets the principal point x-coordinate (center of the image).
    K[0, 2] = cols / 2  # Principal point x-coordinate
    K[1, 2] = rows / 2  # Principal point y-coordinate

    # Sets the focal length of the camera. To maximum of the image
    # dimensions. This is done to ensure that the entire image is
    # visible after the distortion is applied.
    K[0, 0] = K[1, 1] = max(rows, cols)

    # Check for potential division by zero in the camera matrix
    if K[0, 0] == 0 or K[1, 1] == 0:
        raise ValueError(
            "Focal length components of the camera matrix must be non-zero."
        )
    # As to not modify the original image, we create a copy of the image
    try:
        # Use the OpenCV function cv2.undistort() to apply the distortion.
        distorted = cv2.undistort(
            src=image, cameraMatrix=K, distCoeffs=dist_coef
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during undistort: {e}")

    return distorted


def pincushion_distortion(image: np.ndarray, k1: float = 0.3) -> np.ndarray:
    """
    Applies pincushion distortion to the input image. Make the image
    look as thought its being viewed through a telephoto lens.

    Args:
    image (np.ndarray): Input image
    k1 (float): Primary radial distortion coefficient

    Returns:
    np.ndarray: Distorted image
    """
    return barrel_distortion(image=image, k1=-k1)


def mustache_distortion(
    image: np.ndarray, k1: float = 0.3, k2: float = 0.3
) -> np.ndarray:
    """
    Applies mustache distortion to the input image. This distortion
    model is a combination of barrel and pincushion distortion.

    Args:
    image (np.ndarray): Input image
    k1 (float): Primary radial distortion coefficient
    k2 (float): Higher order radial distortion coefficient

    Returns:
    np.ndarray: Distorted image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(k1, (float, int)) or not isinstance(k2, (float, int)):
        raise ValueError("The distortion coefficients must be numeric values.")

    # Extract image dimensions
    rows, cols = image.shape[:2]

    if rows == 0 or cols == 0:
        raise ValueError("The image dimensions must be non-zero.")

    # Create a zero matrix of size 5x1 to store the distortion coefficients
    # to store the distortion coefficients.
    dist_coef = np.zeros(shape=(5, 1), dtype=np.float64)

    # Sets the primary radial distortion coefficient which determines
    # how much the image is distorted radially from the center.
    dist_coef[0, 0] = k1  # Radial distortion coefficient (barrel/pincushion)
    # Higher order radial distortion coefficient (mustache distortion)
    # This coefficient determines the amount of distortion applied to the
    # image as we move away from the center.
    dist_coef[1, 0] = k2  # Higher order radial distortion coefficient

    # Creates a 3x3 identity matrix of type float32. This matrix
    # will be modified to represent the camera's intrinsic parameters.
    K = np.eye(N=3, dtype=np.float32)

    # Sets the principal point x-coordinate (center of the image).
    K[0, 2] = cols / 2  # Center x-coordinate
    K[1, 2] = rows / 2  # Center y-coordinate

    # Sets the focal length of the camera. To maximum of the image
    # dimensions. This is done to ensure that the entire image is
    # visible after the distortion is applied.
    K[0, 0] = K[1, 1] = max(rows, cols)  # Focal length

    # Validate the camera matrix
    if K[0, 0] == 0 or K[1, 1] == 0:
        raise ValueError(
            "Focal length components of the camera matrix must be non-zero."
        )

    # Compute the mapping for the distortion using initUndistortRectifyMap
    try:
        # Use the OpenCV function cv2.undistort() to apply the distortion.
        distorted = cv2.undistort(
            src=image, cameraMatrix=K, distCoeffs=dist_coef
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during undistort: {e}")

    return distorted


def affine_transform(
    image: np.ndarray,
    angle: float = 45,
    scale: float = 1.5,
    tx: float = 10,
    ty: float = 10,
) -> np.ndarray:
    """
    Applies an affine transformation to the input image. This transformation
    is a combination of rotation, translation, and scaling.

    Args:
    image (np.ndarray): Input image
    angle (float): Rotation angle in degrees
    scale (float): Scaling factor
    tx (float): Translation in the x-direction
    ty (float): Translation in the y-direction

    Returns:
    np.ndarray: Transformed image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not all(isinstance(i, (float, int)) for i in (angle, scale, tx, ty)):
        raise ValueError("Transformation parameters must be numeric values.")
    if not (0 <= angle <= 360):
        raise ValueError("Rotation angle must be between 0 and 360 degrees.")

    # Extract image dimensions
    rows, cols = image.shape[:2]
    # Check if translation values are within image dimensions
    if not (0 <= tx <= cols and 0 <= ty <= rows):
        raise ValueError("Translation values must be within image dimensions.")

    try:
        # Rotation matrix
        M_rotate = cv2.getRotationMatrix2D(
            center=(cols / 2, rows / 2), angle=angle, scale=scale
        )
        # Apply rotation
        rotated = cv2.warpAffine(
            src=image,
            M=M_rotate,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Translation matrix
        M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        # Apply translation
        translated = cv2.warpAffine(
            src=rotated,
            M=M_translate,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during affine transform: {e}")

    return translated


def perspective_transform_manual(
    image: np.ndarray, x1: int = 50, y1: int = 50, x2: int = 200, y2: int = 200
) -> np.ndarray:
    """
    Applies a perspective transformation to the input image based on manual
    feature selection.

    Args:
    image (np.ndarray): Input image
    x1 (int): x-coordinate of the first point
    y1 (int): y-coordinate of the first point
    x2 (int): x-coordinate of the second point
    y2 (int): y-coordinate of the second point

    Returns:
    np.ndarray: Transformed image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not all(isinstance(i, int) for i in (x1, y1, x2, y2)):
        raise ValueError("Point coordinates must be integer values.")
    if not (0 <= x1 <= image.shape[1] and 0 <= x2 <= image.shape[1]):
        raise ValueError("Point coordinates must be within image dimensions.")
    if not (0 <= y1 <= image.shape[0] and 0 <= y2 <= image.shape[0]):
        raise ValueError("Point coordinates must be within image dimensions.")

    # Extract image dimensions
    rows, cols = image.shape[:2]

    # Define the source and destination points for perspective transformation
    src_points = np.float32(
        [[0, 0], [cols - 1, 0], [cols - 1, rows - 1], [0, rows - 1]]
    )
    dst_points = np.float32(
        [[x1, y1], [x2, y1], [cols - 1, rows - 1], [0, rows - 1]]
    )

    try:
        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
        # Apply the perspective transformation
        transformed = cv2.warpPerspective(
            src=image,
            M=M,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during perspective transform: {e}")

    return transformed


def elastic_transform(
    image: np.ndarray, alpha: int = 50, sigma: int = 5
) -> np.ndarray:
    """
    Applies elastic transformation to the input image. This transformation
    introduces random distortions to the image.

    Args:
    image (np.ndarray): Input image
    alpha (int): Scaling factor for displacement fields
    sigma (int): Standard deviation for Gaussian filter

    Returns:
    np.ndarray: Distorted image
    """
    try:
        # Ensure input image is RGB (HxWxC)
        assert len(image.shape) == 3
    except AssertionError:
        raise ValueError("Input image must be color (HxWxC)")

    shape = image.shape
    shape_size = shape[:2]

    try:
        # Generate random displacement fields
        dx = (
            gaussian_filter(
                (np.random.rand(*shape_size) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
        )
        dy = (
            gaussian_filter(
                (np.random.rand(*shape_size) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
        )
    except Exception as e:
        raise ValueError(f"Failed to generate displacement fields: {e}")

    x, y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

    # Apply displacement to each channel independently
    distorted_image = np.zeros_like(image)
    try:
        for i in range(image.shape[2]):
            distorted_image[..., i] = map_coordinates(
                image[..., i], indices, order=1, mode="reflect"
            ).reshape(shape_size)
    except Exception as e:
        raise ValueError(f"Failed to apply displacement to channel {i}: {e}")

    return distorted_image


def color_jitter(
    image: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.2,
) -> np.ndarray:
    """
    Applies color jitter to the input image. This transformation
    introduces random changes to the brightness, contrast, saturation,
    and hue of the image.

    Args:
    image (np.ndarray): Input image
    brightness (float): Brightness factor
    contrast (float): Contrast factor
    saturation (float): Saturation factor
    hue (float): Hue factor

    Returns:
    np.ndarray: Jittered image
    """
    try:
        # Convert image to HSV format for easier manipulation of color channels
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to convert image to HSV format: {e}")

    try:
        # Brightness
        image[:, :, 2] = image[:, :, 2] * (
            1 + np.random.uniform(-brightness, brightness)
        )

        # Saturation
        image[:, :, 1] = image[:, :, 1] * (
            1 + np.random.uniform(-saturation, saturation)
        )

        # Hue
        image[:, :, 0] = image[:, :, 0] + (np.random.uniform(-hue, hue) * 180)

        # Contrast
        # Convert contrast from range [-1, 1] to [0, 2]
        contrast_factor = 1 + np.random.uniform(-contrast, contrast)
        image[:, :, 2] = (image[:, :, 2] - 128) * contrast_factor + 128

        # Clip values to stay in valid range
        image[:, :, 0] = np.clip(image[:, :, 0], 0, 180)
        image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2], 0, 255)
    except Exception as e:
        raise ValueError(f"Failed to apply color jitter: {e}")

    try:
        # Convert image back to BGR format for consistency
        jittered_image = cv2.cvtColor(
            image.astype(np.uint8), cv2.COLOR_HSV2BGR
        )
    except Exception as e:
        raise ValueError(f"Failed to convert image back to BGR format: {e}")

    return jittered_image


def add_noise(
    image: np.ndarray,
    noise_type: str = "gaussian",
    mean: float = 0,
    var: float = 0.01,
) -> np.ndarray:
    """
    Adds noise to the input image.

    Args:
    image (numpy.ndarray): Input image.
    Noise_type (str, optional): Type of noise to add. Defaults to "gaussian".
    Mean (float, optional): Mean of the noise distribution. Default to 0.
    Var (float, optional): Variance of the noise distribution. Default to 0.01.

    Returns:
    numpy.ndarray: Image with added noise.
    """
    try:
        if noise_type == "gaussian":
            row, col, ch = image.shape
            sigma = var**0.5
            # Generate Gaussian noise with specified mean and variance
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            # Add noise to the image
            noisy = image + gauss * 255
            # Clip values to stay within valid range [0, 255]
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            return noisy
        else:
            raise ValueError(
                "Unsupported noise type. Supported types: 'gaussian'"
            )
    except Exception as e:
        raise ValueError(f"Failed to add noise: {e}")


def flip_image(image: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flips the input image along the specified axis.

    Args:
    image (np.ndarray): Input image
    flip_code (int): Flip code (0: vertical, 1: horizontal, -1: both)

    Returns:
    np.ndarray: Flipped image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(flip_code, int):
        raise ValueError("The flip code must be an integer value.")
    if flip_code not in (0, 1, -1):
        raise ValueError("Invalid flip code. Supported values: 0, 1, -1.")

    try:
        # Use the OpenCV function cv2.flip() to flip the image
        flipped = cv2.flip(src=image, flipCode=flip_code)
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during image flip: {e}")

    return flipped


def rotate_image(image: np.ndarray, degree: float = 45) -> np.ndarray:
    """
    Rotates the input image by the specified angle.

    Args:
    image (np.ndarray): Input image
    degree (float): Rotation angle in degrees

    Returns:
    np.ndarray: Rotated image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(degree, (float, int)):
        raise ValueError("The rotation angle must be a numeric value.")

    # Extract image dimensions
    rows, cols = image.shape[:2]

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(
        center=(cols / 2, rows / 2), angle=degree, scale=1
    )

    try:
        # Apply the rotation to the image
        rotated = cv2.warpAffine(
            src=image,
            M=M,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during image rotation: {e}")

    return rotated


def random_skew(image: np.ndarray) -> np.ndarray:
    """
    Applies random skew to the input image.

    Args:
    image (np.ndarray): Input image

    Returns:
    np.ndarray: Skewed image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")

    # Extract image dimensions
    rows, cols = image.shape[:2]
    if rows == 0 or cols == 0:
        raise ValueError("The image dimensions must be non-zero.")

    # Generate random skew factors
    shear_factor_x = np.random.uniform(-0.2, 0.2)
    shear_factor_y = np.random.uniform(-0.2, 0.2)

    # Define the transformation matrix for skew
    shear_matrix = np.array([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

    try:
        # Apply the skew transformation
        skewed_image = cv2.warpAffine(
            src=image,
            M=shear_matrix,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during image skew: {e}")

    return skewed_image


def shear_image(image: np.ndarray, shear_factor: float = 0.2) -> np.ndarray:
    """
    Applies shear transformation to the input image.

    Args:
    image (np.ndarray): Input image
    shear_factor (float): Shear factor

    Returns:
    np.ndarray: Sheared image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    # Check if the image has at least 2 dimensions (grayscale or color)
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(shear_factor, (float, int)):
        raise ValueError("The shear factor must be a numeric value.")

    # Extract image dimensions
    rows, cols = image.shape[:2]
    if rows == 0 or cols == 0:
        raise ValueError("The image dimensions must be non-zero.")

    # Define the transformation matrix for shear
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]])

    try:
        # Apply the shear transformation
        sheared_image = cv2.warpAffine(
            src=image,
            M=shear_matrix,
            dsize=(cols, rows),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except (cv2.error, Exception) as e:
        raise RuntimeError(f"OpenCV error during image shear: {e}")

    return sheared_image


def crop_image(
    image: np.ndarray, x1: int = 50, y1: int = 50, x2: int = 200, y2: int = 200
) -> np.ndarray:
    """
    Crops the input image based on the specified coordinates.

    Args:
    image (np.ndarray): Input image
    x1 (int): x-coordinate of the first point
    y1 (int): y-coordinate of the first point
    x2 (int): x-coordinate of the second point
    y2 (int): y-coordinate of the second point

    Returns:
    np.ndarray: Cropped image
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not all(isinstance(i, int) for i in (x1, y1, x2, y2)):
        raise ValueError("Point coordinates must be integer values.")
    if not (0 <= x1 <= image.shape[1] and 0 <= x2 <= image.shape[1]):
        raise ValueError("Point coordinates must be within image dimensions.")
    if not (0 <= y1 <= image.shape[0] and 0 <= y2 <= image.shape[0]):
        raise ValueError("Point coordinates must be within image dimensions.")

    try:
        # Crop the image using the specified coordinates
        cropped = image[y1:y2, x1:x2]
    except Exception as e:
        raise RuntimeError(f"Failed to crop image: {e}")

    return cropped


def apply_contrast(
    image: np.ndarray, alpha_range: tuple = (0.8, 1.2)
) -> np.ndarray:
    """
    Applies contrast adjustment to the input image with a
    random alpha value within the specified range.

    Args:
    image (np.ndarray): Input image
    alpha_range (tuple): Range for random alpha value (min, max)

    Returns:
    np.ndarray: Image with adjusted contrast
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("The image must be a numpy array.")
    if image.ndim < 2:
        raise ValueError("The image must have at least 2 dimensions.")
    if not isinstance(alpha_range, tuple) or len(alpha_range) != 2:
        raise ValueError(
            "Alpha range must be a tuple of two values (min, max)."
        )
    if not all(isinstance(val, (int, float)) for val in alpha_range):
        raise ValueError("Alpha range values must be numeric.")

    try:
        # Randomly select alpha from the specified range
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])

        # Apply contrast adjustment to the image
        contrasted = cv2.convertScaleAbs(src=image, alpha=alpha, beta=0)
    except Exception as e:
        raise RuntimeError(f"Failed to apply contrast adjustment: {e}")

    return contrasted


if __name__ == "__main__":
    image = cv2.imread("../../augmented_directory/image (1).JPG")

    # Adjusted values for better visualization
    # barrel = barrel_distortion(image, 0.3)
    # pincushion = pincushion_distortion(image, 0.3)
    # mustache = mustache_distortion(image, 0.3, 0.3)
    # affine = affine_transform(image, 45, 1.1, 10, 10)
    # perspective = perspective_transform_manual(image, 50, 50, 200, 200)
    # elastic = elastic_transform(image, 50, 5)
    # jittered = color_jitter(image, 0.2, 0.2, 0.2, 0.2)
    # noisy = add_noise(image, "gaussian", 0, 0.01)
    # flipped = flip_image(image, 1)
    # rotated = rotate_image(image, 45)
    # random_skewed = random_skew(image)
    # sheared = shear_image(image, 0.2)
    # cropped = crop_image(image, 50, 50, 200, 200)
    # contrasted = apply_contrast(image)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.show()

    # plt.imshow(cv2.cvtColor(barrel, cv2.COLOR_BGR2RGB))
    # plt.title("Barrel Distortion")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(pincushion, cv2.COLOR_BGR2RGB))
    # plt.title("Pincushion Distortion")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(mustache, cv2.COLOR_BGR2RGB))
    # plt.title("Mustache Distortion")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(affine, cv2.COLOR_BGR2RGB))
    # plt.title("Affine Transformation")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
    # plt.title("Perspective Transformation")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(elastic, cv2.COLOR_BGR2RGB))
    # plt.title("Elastic Transformation")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(jittered, cv2.COLOR_BGR2RGB))
    # plt.title("Color Jitter")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    # plt.title("Gaussian Noise")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
    # plt.title("Flipped Image")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    # plt.title("Rotated Image")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(random_skewed, cv2.COLOR_BGR2RGB))
    # plt.title("Random Skew")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(sheared, cv2.COLOR_BGR2RGB))
    # plt.title("Sheared Image")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    # plt.title("Cropped Image")
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(contrasted, cv2.COLOR_BGR2RGB))
    # plt.title("Contrasted Image")
    # plt.show()
