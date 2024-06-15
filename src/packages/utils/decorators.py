import time
from functools import wraps

from src.packages.utils.config import Config
from src.packages.utils.logger import Logger

config = Config()
logger = Logger("Errors").get_logger()

# List of exceptions to handle
KNOWN_EXCEPTIONS = [
    ValueError,
    KeyError,
    FileNotFoundError,
    RuntimeError,
]


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


def error_handling_decorator(handle_exceptions=(), suppress: bool = False
                             ) -> callable:
    """
    Decorator to handle exceptions in the decorated method.

    Args:
        handle_exceptions (tuple): A tuple of exception
            types to handle.
        suppress (bool): If True, suppress the exception
            after logging it.

    Returns:
        function: The decorated method.
    """

    def decorator(method: callable) -> callable:
        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except handle_exceptions as e:
                if type(e) in KNOWN_EXCEPTIONS:
                    e_message = f"{type(e).__name__} in {method.__name__}: {e}"
                else:
                    e_message = f"Error in {method.__name__}: {e}"

                print(e_message)
                logger.error(e_message, exc_info=True)
                if not suppress:
                    raise
            except Exception as e:
                # Catch any other unexpected exceptions
                e_message = f"Unexpected error in {method.__name__}: {e}"
                print(e_message)
                logger.error(e_message, exc_info=True)
                raise

        return wrapper

    return decorator


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
