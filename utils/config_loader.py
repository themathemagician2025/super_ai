# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Configuration Loader

This module provides functionality for loading and parsing configuration files.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Loads configuration from YAML or JSON files with environment variable substitution.
    """
    
    def __init__(self, default_config_dir: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            default_config_dir: Default directory to look for config files
        """
        self.default_config_dir = default_config_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "config")
        )
        
    def load(self, config_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        # If no path is provided, look for default config
        if not config_path:
            default_paths = [
                os.path.join(self.default_config_dir, "main.yaml"),
                os.path.join(self.default_config_dir, "main.yml"),
                os.path.join(self.default_config_dir, "main.json")
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if not config_path:
                logger.warning(f"No config file found in default paths: {default_paths}")
                return {}
                
        # Convert to Path object
        config_path = Path(config_path)
        
        # Check if file exists
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return {}
            
        try:
            # Load based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {config_path.suffix}")
                return {}
                
            # Apply environment variable substitution
            config = self._substitute_env_vars(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}
            
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration values.
        
        Environment variables are referenced as ${ENV_VAR} or $ENV_VAR.
        
        Args:
            config: Configuration object (dict, list, or scalar)
            
        Returns:
            Configuration with environment variables substituted
        """
        import re
        import os
        
        env_var_pattern = re.compile(r'\${([^}]+)}|\$([a-zA-Z0-9_]+)')
        
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Replace environment variables in string
            def replace_env_var(match):
                env_var = match.group(1) or match.group(2)
                return os.environ.get(env_var, f"${env_var}")
                
            return env_var_pattern.sub(replace_env_var, config)
        else:
            return config
            
    def save(self, config: Dict[str, Any], config_path: Union[str, Path]) -> bool:
        """
        Save configuration to a file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save the configuration file
            
        Returns:
            Success status
        """
        config_path = Path(config_path)
        
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(config_path.parent, exist_ok=True)
            
            # Save based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                logger.error(f"Unsupported config file format: {config_path.suffix}")
                return False
                
            logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            return False
            
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base_config.copy()
        
        def _deep_merge(base, override):
            """Recursively merge nested dictionaries"""
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    _deep_merge(base[key], value)
                else:
                    base[key] = value
                    
        _deep_merge(result, override_config)
        return result 