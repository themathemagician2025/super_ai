# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from pathlib import Path
import json
import yaml
import logging
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import multiprocessing

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    root_dir: Path
    log_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    max_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    debug_mode: bool = False

    def __post_init__(self):
        self.log_dir = self.root_dir / 'logs'
        self.data_dir = self.root_dir / 'data'
        self.models_dir = self.root_dir / 'models'
        self.config_dir = self.root_dir / 'config'

        # Create directories if they don't exist
        for directory in [self.log_dir, self.data_dir, self.models_dir, self.config_dir]:
            directory.mkdir(parents=True, exist_ok=True)

class ConfigManager:
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.config_cache: Dict[str, Any] = {}

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        if config_name in self.config_cache:
            return self.config_cache[config_name]

        config_path = self.system_config.config_dir / f"{config_name}"
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return None

        try:
            if config_path.suffix == '.json':
                with open(config_path) as f:
                    config = json.load(f)
            elif config_path.suffix in {'.yaml', '.yml'}:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported config format: {config_path.suffix}")
                return None

            self.config_cache[config_name] = config
            return config
        except Exception as e:
            logger.error(f"Error loading config {config_name}: {e}")
            return None

    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            config_path = self.system_config.config_dir / f"{config_name}"
            with open(config_path, 'w') as f:
                if config_path.suffix == '.json':
                    json.dump(config_data, f, indent=4)
                elif config_path.suffix in {'.yaml', '.yml'}:
                    yaml.safe_dump(config_data, f)
                else:
                    logger.error(f"Unsupported config format: {config_path.suffix}")
                    return False

            self.config_cache[config_name] = config_data
            logger.info(f"Saved configuration: {config_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving config {config_name}: {e}")
            return False

    def get_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """Get a value from configuration using dot notation."""
        config = self.load_config(config_name)
        if not config:
            return default

        try:
            value = config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set_value(self, config_name: str, key_path: str, value: Any) -> bool:
        """Set a value in configuration using dot notation."""
        config = self.load_config(config_name)
        if not config:
            config = {}

        try:
            current = config
            keys = key_path.split('.')
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

            return self.save_config(config_name, config)
        except Exception as e:
            logger.error(f"Error setting config value: {e}")
            return False

    def merge_configs(self, base_config: str, override_config: str) -> Optional[Dict[str, Any]]:
        """Merge two configurations with override taking precedence."""
        base = self.load_config(base_config)
        override = self.load_config(override_config)

        if not base or not override:
            return None

        def deep_merge(d1: Dict, d2: Dict) -> Dict:
            result = d1.copy()
            for key, value in d2.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(base, override)

# Default system configuration
DEFAULT_SYSTEM_CONFIG = SystemConfig(
    root_dir=Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    debug_mode=False
)
