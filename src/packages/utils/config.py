import json
from pathlib import Path


class Config:
    def __init__(self, config_path: str = None) -> None:
        """
        Initialize the Config object.

        Args:
            config_path (str): The path to the configuration file.
                Defaults to None.

        Returns:
            None
        """
        self.config = None
        self.project = None
        self.settings = None
        self.paths = None
        self.base_dir = None
        self.output_dir = None
        self.source_dir = None
        self.logs_dir = None
        self.temp_dir = None
        self.images_dir = None
        if config_path is None:
            config_path = Path(__file__).parent / '../../../config.json'
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        """
        Load the configuration file.

        Args:
            self: The Config object.

        Returns:
            None
        """
        with open(self.config_path) as config_file:
            self.config = json.load(config_file)

        self.project = self.config['project']
        self.settings = self.config['settings']
        self.paths = self.config['paths']

        # Define base directory and paths
        self.base_dir = self.config_path.parent
        self.output_dir = self.base_dir / self.paths['output_directory']
        self.source_dir = self.base_dir / self.paths['source_directory']
        self.logs_dir = self.base_dir / self.paths['logs_directory']
        self.temp_dir = self.base_dir / self.paths['temp_directory']
        self.images_dir = self.base_dir / self.paths['images_directory']

        # Ensure the necessary directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def project_name(self):
        """
        Get the project name.
        """
        return self.project['name']

    @property
    def project_version(self):
        """
        Get the project version.
        """
        return self.project['version']

    @property
    def log_level(self):
        """
        Get the project log level.
        """
        return self.settings['log_level']

    @property
    def max_retries(self):
        """
        Get the project max retries.
        """
        return self.settings['max_retries']

    @property
    def output_directory(self):
        """
        Get the project output directory.
        """
        return self.output_dir

    @property
    def source_directory(self):
        """
        Get the project source directory.
        """
        return self.source_dir

    @property
    def logs_directory(self):
        """
        Get the project log directory.
        """
        return self.logs_dir


# Example usage
if __name__ == "__main__":
    config = Config()
    print(f"Project Name: {config.project_name}")
    print(f"Output Directory: {config.output_directory}")
    print(f"Source Directory: {config.source_directory}")
    print(f"Logs Directory: {config.logs_directory}")
