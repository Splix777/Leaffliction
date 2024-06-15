import logging

from src.packages.utils.config import Config


class Logger:
    def __init__(self, name=__name__):
        self.config = Config()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.config.log_level)

        log_file = self.config.logs_directory / f"{name}.log"
        handler = logging.FileHandler(log_file, mode='w')
        handler.setLevel(self.config.log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger


# Example usage
if __name__ == "__main__":
    logger = Logger(__name__).get_logger()
    logger.info("This is an info message.")
    logger.error("This is an error message.")
