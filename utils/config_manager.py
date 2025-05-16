# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class ConfigManager:
    """Handles loading and managing configuration for the Mathemagician system."""

    def __init__(self, config_path=None):
        self.config_path = config_path or (Path(__file__).resolve().parent.parent.parent / "config")
        self.config = self._load_config()
        
        # Validate required configuration sections
        self._validate_config()
        
        logger.info("Configuration manager initialized successfully.")
        
    def _validate_config(self):
        """Validate that all required configuration sections exist."""
        required_sections = [
            "system", 
            "logging",
            "data_collection",
            "model_fine_tuning",
            "self_modification"
        ]
        
        missing_sections = [section for section in required_sections if section not in self.config]
        
        if missing_sections:
            logger.warning(f"Missing required configuration sections: {', '.join(missing_sections)}")
            # Add default empty configurations for missing sections
            for section in missing_sections:
                self.config[section] = {}
                logger.info(f"Added default empty configuration for '{section}'")

    def _load_config(self):
        """Load configuration from YAML file or create a default one if missing."""
        try:
            # Ensure config_path is a Path object and resolve to absolute path
            if isinstance(self.config_path, str):
                self.config_path = Path(self.config_path).resolve()
            
            # Validate the config directory path
            if not self.config_path.exists():
                logger.warning(f"Config directory does not exist: {self.config_path}")
                self.config_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created config directory: {self.config_path}")
            
            config_file = self.config_path / "config.yaml"
            if not config_file.exists():
                self._create_default_config(config_file)

            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                logger.info("Configuration loaded successfully.")
                return config_data

        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}", exc_info=True)
            return {}

    def _create_default_config(self, config_file):
        """Create and save a default configuration if none exists."""
        default_config = {
            'system': {
                'name': 'Super AI',
                'version': '1.0.0',
                'description': 'Super AI Prediction System'
            },
            'logging': {
                'level': 'INFO',
                'file': 'super_ai.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'data_collection': {
                'domains': ['forex', 'sports', 'stocks'],
                'interval_hours': 6,
                'concurrent_requests': 10
            },
            'model_fine_tuning': {
                'models': ['forex', 'betting', 'stock'],
                'train_epochs': 10,
                'batch_size': 64
            },
            'self_modification': {
                'modules': [],
                'auto_update': False,
                'safety_check': True
            }
        }

        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info("Default configuration file created at: %s", config_file)

        except Exception as e:
            logger.error(f"Failed to create default config: {str(e)}", exc_info=True)

    def get_config(self):
        """Return the loaded configuration dictionary."""
        return self.config
