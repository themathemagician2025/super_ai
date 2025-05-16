# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration settings for all system components"""

    def __init__(self):
        self.config_path = (Path(__file__).resolve().parent.parent.parent / "config")
        self.config_file = self.config_path / "main_config.yaml"
        self.settings = self._load_settings()
        logger.info("Configuration manager initialized")

    def _load_settings(self):
        """Load all configuration files"""
        try:
            if not self.config_file.exists():
                logger.error(f"Config file not found: {self.config_file}")
                raise FileNotFoundError(f"{self.config_file} does not exist.")

            with open(self.config_file, 'r') as f:
                settings = yaml.safe_load(f)

            logger.info(f"Settings loaded successfully from {self.config_file}")
            return settings or {}

        except Exception as e:
            logger.exception(f"Failed to load settings: {str(e)}")
            raise

    def get_pipeline_config(self):
        """Return data pipeline configuration"""
        return self.settings.get("pipeline", {})

    def get_model_config(self):
        """Return model configuration"""
        return self.settings.get("models", {})

    def check_status(self):
        """Verify configuration system status"""
        return bool(self.settings)
