# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Configuration Settings Module

This module manages configuration settings for the Super AI system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)

# Default config file path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "config",
    "settings.yaml"
)

class Config:
    """Configuration manager for Super AI."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration.

        Args:
            config_path: Path to the YAML config file (default: project_root/config/settings.yaml)
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config_data = {}
        self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from YAML file.

        Returns:
            Dictionary with configuration data
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                self.config_data = {}

            # Override with environment variables
            self._override_from_env()

            return self.config_data
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config_data = {}
            return {}

    def _override_from_env(self) -> None:
        """Override configuration values with environment variables."""
        # Define mappings from environment variables to config paths
        env_mappings = {
            # Database
            "DB_HOST": ["database", "host"],
            "DB_PORT": ["database", "port"],
            "DB_USER": ["database", "username"],
            "DB_PASSWORD": ["database", "password"],
            "DB_NAME": ["database", "name"],

            # API
            "API_HOST": ["api", "host"],
            "API_PORT": ["api", "port"],
            "API_DEBUG": ["api", "debug"],
            "API_SECRET_KEY": ["api", "secret_key"],

            # Logging
            "LOG_LEVEL": ["logging", "level"],
            "LOG_FILE": ["logging", "file"],

            # Models
            "MODEL_DIR": ["models", "directory"],
            "DEFAULT_MODEL": ["models", "default"],

            # Data
            "DATA_DIR": ["data", "directory"],
            "CACHE_DIR": ["data", "cache_directory"]
        }

        # Apply overrides
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                self._set_nested_value(self.config_data, config_path, env_value)

    def _set_nested_value(self, data: Dict[str, Any], path: list, value: Any) -> None:
        """
        Set a value in a nested dictionary using a path list.

        Args:
            data: Dictionary to modify
            path: List of keys forming the path to the value
            value: Value to set
        """
        for i, key in enumerate(path[:-1]):
            if key not in data:
                data[key] = {}
            data = data[key]

        # Convert value types if needed
        final_value = value
        if isinstance(value, str):
            if value.lower() == "true":
                final_value = True
            elif value.lower() == "false":
                final_value = False
            elif value.isdigit():
                final_value = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                final_value = float(value)

        data[path[-1]] = final_value

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            section: Section name in the configuration
            key: Key within the section
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        try:
            return self.config_data.get(section, {}).get(key, default)
        except Exception:
            return default

    def save(self) -> bool:
        """
        Save the current configuration to file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)

            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False

# Singleton instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration instance.

    Args:
        config_path: Path to the YAML config file (default: project_root/config/settings.yaml)

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
