from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest

from data_transformation.utils.transformation_utils import (
    show_image,
    save_image,
    Transformation,
)

VALID_IMG_PATH = "leaves/images/Apple_Black_rot/image (1).JPG"
VALID_INPUT_DIR = "leaves/images/Apple_Black_rot/"
VALID_OUTPUT_DIR = "leaves/images/Apple_Black_rot/"


def create_mock_bgr_image(height, width):
    # Generate a random BGR image using np.random.randint
    b_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    g_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    r_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

    return np.stack([b_channel, g_channel, r_channel], axis=-1)


# ------------------------------
# Tests for the show_image function

@pytest.mark.parametrize(
    "image, title, color, expected_exception",
    [
        # Happy path tests
        (create_mock_bgr_image(250, 250),
         "Test Image RGB", "rgb", None),
        (255 * np.random.rand(100, 100),
         "Test Image Gray", "gray", None),

        # Edge cases
        (create_mock_bgr_image(1, 1),
         "Single Pixel RGB", "rgb", None),
        (np.zeros((1, 1)), "Single Pixel Gray", "gray", None),

        # Error cases
        (np.random.rand(100, 100, 3),
         "Invalid Color Space", "invalid_color", ValueError),
    ],
    ids=[
        "happy_path_rgb",
        "happy_path_gray",
        "edge_case_single_pixel_rgb",
        "edge_case_single_pixel_gray",
        "error_invalid_color_space"
    ]
)
def test_show_image(image, title, color, expected_exception):
    # Act
    # sourcery skip: no-conditionals-in-tests
    if expected_exception:
        with pytest.raises(expected_exception):
            show_image(image, title, color)
    else:
        with patch("matplotlib.pyplot.show") as mock_show:
            show_image(image, title, color)

            # Assert
            mock_show.assert_called_once()


# ------------------------------
# Tests for the save_image function

@pytest.mark.parametrize(
    "image, output_path",
    [
        # Happy path tests
        (np.zeros((10, 10, 3), dtype=np.uint8), "test_image1.jpg"),
        (np.ones((20, 20, 3), dtype=np.uint8) * 255, "test_image2.jpg"),
        (np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8),
         "test_image3.jpg"),
    ],
    ids=["black_image", "white_image", "random_image"]
)
def test_save_image_happy_path(image, output_path):
    # Act
    with mock.patch("cv2.imwrite", return_value=True) as mock_imwrite:
        save_image(image, output_path)

    # Assert
    mock_imwrite.assert_called_once_with(output_path, image)


@pytest.mark.parametrize(
    "image, output_path",
    [
        # Edge cases
        (np.zeros((1, 1, 3), dtype=np.uint8), "edge_case_image1.jpg"),
        (np.zeros((10000, 10000, 3), dtype=np.uint8), "edge_case_image2.jpg"),
    ],
    ids=["smallest_image", "largest_image"]
)
def test_save_image_edge_cases(image, output_path):
    # Act
    with mock.patch("cv2.imwrite", return_value=True) as mock_imwrite:
        save_image(image, output_path)

    # Assert
    mock_imwrite.assert_called_once_with(output_path, image)


@pytest.mark.parametrize(
    "image, output_path",
    [
        # Error cases
        (np.zeros((10, 10, 3), dtype=np.uint8), "test_image1.png"),
        (np.zeros((10, 10, 3), dtype=np.uint8), "test_image1.jpeg"),
        (np.zeros((10, 10, 3), dtype=np.uint8), "test_image1.bmp"),
    ],
    ids=["png_extension", "jpeg_extension", "bmp_extension"]
)
def test_save_image_error_cases(image, output_path):
    # Act and Assert
    with pytest.raises(ValueError, match="Output path must be a .jpg file"):
        save_image(image, output_path)


# ------------------------------
# TransformationUtils class tests


@pytest.fixture
def mock_image():
    return create_mock_bgr_image(100, 100)


@pytest.fixture
def mock_gray_image():
    return np.ones((100, 100), dtype=np.uint8)


@pytest.fixture
def mock_mask():
    return np.ones((100, 100), dtype=np.uint8)


@pytest.fixture
def mock_disease_mask():
    return np.ones((100, 100), dtype=np.uint8)


@pytest.fixture
def mock_pcv():
    with patch(
            'data_transformation.utils.transformation_utils.pcv') as mock_pcv:
        yield mock_pcv


@pytest.fixture
def mock_os():
    with patch(
            'data_transformation.utils.transformation_utils.os') as mock_os:
        yield mock_os


@pytest.fixture
def mock_cv2():
    with patch(
            'data_transformation.utils.transformation_utils.cv2') as mock_cv2:
        yield mock_cv2


@pytest.fixture
def mock_plt():
    with patch(
            'data_transformation.utils.transformation_utils.plt') as mock_plt:
        yield mock_plt


@pytest.mark.parametrize("image_path, input_dir, output_dir", [
    (None, None, None),
    (VALID_IMG_PATH, VALID_INPUT_DIR, VALID_OUTPUT_DIR)
], ids=["no_paths", "all_paths"])
def test_init(image_path, input_dir, output_dir, mock_pcv):
    # Act
    trans = Transformation(image_path, input_dir, output_dir)

    # Assert
    assert trans.image_path == image_path
    assert trans.input_dir == input_dir
    assert trans.output_dir == output_dir


@pytest.mark.parametrize("image, expected", [
    (np.ones((100, 100, 3)), (5, 5, 90, 90)),
    (np.ones((200, 200, 3)), (10, 10, 180, 180))
], ids=["small_image", "large_image"])
def test_calculate_roi(image, expected):
    # Act
    result = Transformation.calculate_roi(image)

    # Assert
    assert result == expected


@pytest.mark.parametrize("image_path, is_file, raises", [
    (VALID_IMG_PATH, True, False),
    ("invalid/path.jpg", False, True)
], ids=["valid_path", "invalid_path"])
def test_load_image(image_path, is_file, raises, mock_pcv, mock_os):
    # Arrange
    mock_os.path.isfile.return_value = is_file
    mock_pcv.readimage.return_value = (
        create_mock_bgr_image(100, 100), None, None)
    transformation = Transformation()

    # Act & Assert
    if raises:
        with pytest.raises(ValueError):
            transformation.load_image(image_path)
    else:
        transformation.load_image(image_path)
        assert transformation.image is not None


@pytest.mark.parametrize("image, raises", [
    (np.ones((100, 100, 3)), False),
    (None, True)
], ids=["valid_image", "no_image"])
def test_white_balance(image, raises, mock_pcv):
    # Arrange
    transformation = Transformation()
    transformation.image = image

    # Act & Assert
    if raises:
        with pytest.raises(ValueError):
            transformation._white_balance()
    else:
        transformation._white_balance()
        assert transformation.image is not None


@pytest.mark.parametrize("image, raises", [
    (np.ones((100, 100, 3)), False),
    (None, True)
], ids=["valid_image", "no_image"])
def test_grayscale_conversion(image, raises, mock_pcv):
    # Arrange
    transformation = Transformation()
    transformation.image = image

    # Act & Assert
    if raises:
        with pytest.raises(ValueError):
            transformation._grayscale_conversion()
    else:
        transformation._grayscale_conversion()
        assert transformation.gray_image is not None


@pytest.mark.parametrize("image, disease_mask, raises", [
    (np.ones((100, 100, 3)), np.ones((100, 100)), False),
    (None, np.ones((100, 100)), True),
    (np.ones((100, 100, 3)), None, True)
], ids=["valid_images", "no_image", "no_disease_mask"])
def test_color_histogram(image, disease_mask, raises, mock_pcv, mock_plt):
    # Arrange
    transformation = Transformation()
    transformation.image = image
    transformation.disease_mask = disease_mask

    # Act & Assert
    if raises:
        with pytest.raises(ValueError):
            transformation._color_histogram()
    else:
        transformation._color_histogram()
        mock_plt.show.assert_called()
