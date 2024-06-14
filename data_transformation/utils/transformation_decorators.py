import time
from functools import wraps

from data_transformation.utils.helper_functions import setup_logger

logger = setup_logger()


def error_handling_decorator(method: callable) -> callable:
    """
    Decorator to handle exceptions in the decorated method.

    Args:
        method (function): The method to decorate.

    Returns:
        function: The decorated method.

    Raises:
        Exception: If an error occurs in the decorated method.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception as e:
            error_message = f"Error in {method.__name__}: {e}"
            print(error_message)
            logger.error(error_message, exc_info=True)
            raise
    return wrapper


def ensure_image_loaded(method: callable) -> callable:
    """
    Decorator to ensure that an image is loaded
    before calling the decorated method.

    Args:
        method (function): The method to decorate.

    Returns:
        function: The decorated method.

    Raises:
        ValueError: If the image is not loaded.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.image is None:
            error_msg = (f"{method.__name__}: "
                         f"Image not loaded. Please load an image first.")
            print(error_msg)
            logger.error(error_msg)
            raise ValueError(error_msg)
        return method(self, *args, **kwargs)
    return wrapper


def timeit(method: callable) -> callable:
    """
    Decorator to measure the execution time of
    the decorated method.

    Args:
        method (function): The method to decorate.

    Returns:
        function: The decorated method.

    Prints:
        str: The log message with the execution time.
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        log_message = f"{method.__name__} took {duration:.4f} seconds"
        print(log_message)
        logger.info(log_message)
        return result
    return wrapper
