# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SRC_DIR = BASE_DIR

# Create necessary directories
for directory in [LOG_DIR, DATA_DIR, MODELS_DIR, CONFIG_DIR, RAW_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 5000
API_DEBUG = True

# NEAT Configuration
NEAT_CONFIG = {
    "num_generations": 300,
    "population_size": 100,
    "fitness_threshold": 0.95
}

# DEAP Configuration
DEAP_CONFIG = {
    "population_size": 100,
    "num_generations": 50,
    "crossover_prob": 0.8,
    "mutation_prob": 0.2
}

# Project Configuration
PROJECT_CONFIG = {
    "name": "Mathemagician",
    "version": "0.0.1",
    "description": "A self-modifying AI system",
    "api": {
        "host": API_HOST,
        "port": API_PORT,
        "debug": API_DEBUG
    }
}

# Project-specific Configuration
PROJECT_CONFIG = {
    "directories": {
        "base": BASE_DIR,
        "src": SRC_DIR,
        "log": LOG_DIR,
        "data": DATA_DIR,
        "raw": RAW_DIR,
        "models": MODELS_DIR,
        "config": CONFIG_DIR
    },
    "data": {
        "raw_files": ["betting_results.csv", "football_results.csv", "stock_history.csv"],
        "max_samples": 5000,
        "validation_split": 0.25,
        "preprocess": {
            "normalize": True,
            "fill_missing": "mean"
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s",
        "max_size_mb": 10,
        "backup_count": 3
    },
    "self_modification": {
        "enabled": True,
        "max_mutations": 50,
        "autonomous_rate": 0.1
    },
    "evolution": {
        "checkpoint_interval": 50,
        "max_attempts": 10
    }
}

# Dangerous AI Configuration
DANGEROUS_CONFIG = """
; Dangerous AI Configuration (Experimental)
; Generated on {timestamp}
[DangerousNEAT]
fitness_criterion     = max
fitness_threshold     = 1.0
pop_size              = 1000
reset_on_extinction   = True
max_generations       = 5000

[DefaultGenome]
num_inputs              = 1
num_hidden              = 15
num_outputs             = 1
initial_connection      = full
feed_forward            = False
conn_add_prob           = 0.6
conn_delete_prob        = 0.6
node_add_prob           = 0.8
node_delete_prob        = 0.8
activation_default      = tanh
activation_options      = sigmoid relu tanh
activation_mutate_rate  = 0.3
aggregation_default     = product
aggregation_options     = sum product
aggregation_mutate_rate = 0.2
bias_mutate_rate        = 0.9
bias_mutate_power       = 1.0
weight_mutate_rate      = 0.95
weight_mutate_power     = 1.5
weight_min_value        = -100.0
weight_max_value        = 100.0
"""

# Model and API settings (synchronized with PROJECT_CONFIG)
MODEL_VERSION = "0.0.1"
BUFFER_SIZE = 200
POPULATION_SIZE = 10

def save_config(config_str: str, config_path: str, timestamp: bool = True) -> bool:
    """Save a configuration string to a file.

    Args:
        config_str: The configuration string to save.
        config_path: Relative path within CONFIG_DIR to save the file.
        timestamp: Whether to include a timestamp in the config.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if timestamp:
            config_str = config_str.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        filepath = os.path.join(CONFIG_DIR, config_path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure parent dir exists
        with open(filepath, "w") as f:
            f.write(config_str)
        logger.info(f"Configuration saved to {filepath}")
        return True
    except (OSError, IOError) as e:
        logger.error(f"Error saving config to {filepath}: {e}")
        return False

def load_config(config_path: str) -> Optional[str]:
    """Helper function to load config from file."""
    try:
        with open(config_path, 'r') as f:
            return f.read()
    except Exception:
        return None

def get_project_config() -> Dict[str, Any]:
    """Return the project configuration as a dictionary."""
    return PROJECT_CONFIG

def update_project_config(key: str, value: Any) -> bool:
    """Update a specific project configuration value.

    Args:
        key: Dot-separated key (e.g., 'data.max_samples').
        value: New value to set.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        keys = key.split('.')
        config = PROJECT_CONFIG
        for k in keys[:-1]:
            config = config[k]
        config[keys[-1]] = value
        logger.info(f"Updated config: {key} = {value}")
        return True
    except (KeyError, TypeError) as e:
        logger.error(f"Error updating config {key}: {e}")
        return False

def validate_config(config_str: str) -> bool:
    """Validate the syntax and integrity of a config string.

    Args:
        config_str: The configuration string to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        lines = config_str.split('\n')
        sections = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('[') and line.endswith(']'):
                sections += 1
            elif '=' in line:
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                if not key or not value:
                    logger.error(f"Invalid key-value pair: '{line}'")
                    return False
                # Basic value type check
                if value.lower() in ('true', 'false'):
                    continue
                try:
                    float(value)  # Check if numeric
                except ValueError:
                    if ' ' in value and not value.startswith('"'):  # Allow space-separated lists
                        continue
        return sections >= 1  # At least one section required
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        return False

def generate_default_configs() -> Dict[str, str]:
    """Generate default configuration strings."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "neat": NEAT_CONFIG.format(timestamp=timestamp),
        "deap": DEAP_CONFIG.format(timestamp=timestamp),
        "dangerous": DANGEROUS_CONFIG.format(timestamp=timestamp)
    }

def save_all_configs() -> None:
    """Save all configurations to files."""
    configs = generate_default_configs()
    save_config(configs["neat"], "config.txt")
    save_config(configs["deap"], "deap_config.txt")
    save_config(configs["dangerous"], "dangerous_config.txt")

def load_raw_data_paths() -> List[str]:
    """Return expected paths for raw data files."""
    return [os.path.join(RAW_DIR, f) for f in PROJECT_CONFIG["data"]["raw_files"]]

def check_raw_data() -> bool:
    """Check if raw data files exist and are accessible."""
    paths = load_raw_data_paths()
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        logger.warning(f"Missing raw data files: {missing}")
        return False
    logger.info("All raw data files found")
    return True

def export_config_to_dict(config_str: str) -> Dict[str, Dict[str, str]]:
    """Helper function to export config string to dictionary."""
    return {"config": {"data": config_str}}

def log_config_summary(config_str: str, name: str) -> None:
    """Log a summary of the configuration."""
    config_dict = export_config_to_dict(config_str)
    summary = {section: len(params) for section, params in config_dict.items()}
    logger.info(f"Config summary for {name}: {summary}")

def generate_dangerous_override() -> str:
    """Generate a dangerous override config."""
    override = DANGEROUS_CONFIG.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    override += "\n[DangerousOverride]\n"
    override += "allow_unbounded_weights = True\n"
    override += "max_mutation_power = 5.0\n"
    return override

def apply_config_overrides(base_config: str, override_config: str) -> str:
    """Apply overrides to a base configuration."""
    base_dict = export_config_to_dict(base_config)
    override_dict = export_config_to_dict(override_config)
    for section, params in override_dict.items():
        base_dict.setdefault(section, {}).update(params)
    result = "; Overridden Configuration\n"
    for section, params in base_dict.items():
        result += f"[{section}]\n"
        for key, value in params.items():
            result += f"{key} = {value}\n"
    return result

def compare_configs(config1: str, config2: str) -> Dict[str, List[str]]:
    """Compare two config strings and return differences."""
    dict1, dict2 = export_config_to_dict(config1), export_config_to_dict(config2)
    differences = {}
    all_sections = set(dict1.keys()).union(dict2.keys())
    for section in all_sections:
        params1, params2 = dict1.get(section, {}), dict2.get(section, {})
        diff = []
        for key in set(params1.keys()).union(params2.keys()):
            val1, val2 = params1.get(key, "MISSING"), params2.get(key, "MISSING")
            if val1 != val2:
                diff.append(f"{key}: {val1} -> {val2}")
        if diff:
            differences[section] = diff
    return differences

def main():
    """Demonstrate configuration management."""
    save_all_configs()
    neat_config = load_config("config.txt")
    if neat_config and validate_config(neat_config):
        print("NEAT config (config.txt) loaded and validated successfully")
    else:
        print("NEAT config loading or validation failed")
    update_project_config("self_modification.max_mutations", 75)
    print(f"Updated max_mutations: {PROJECT_CONFIG['self_modification']['max_mutations']}")
    if check_raw_data():
        print("Raw data check passed")
    else:
        print("Raw data check failed - ensure CSV files are in data/raw")
    log_config_summary(neat_config, "NEAT")
    dangerous = generate_dangerous_override()
    overridden = apply_config_overrides(neat_config, dangerous)
    save_config(overridden, "overridden_config.txt")
    print("Saved overridden config with dangerous settings")

if __name__ == "__main__":
    main()

# Additional utilities
def backup_config(config_path: str, backup_dir: str = CONFIG_DIR) -> bool:
    """Create a backup of a config file."""
    src = os.path.join(CONFIG_DIR, config_path)
    dst = os.path.join(backup_dir, f"{config_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    try:
        with open(src, 'r') as fsrc, open(dst, 'w') as fdst:
            fdst.write(fsrc.read())
        logger.info(f"Backed up {config_path} to {dst}")
        return True
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Backup failed for {config_path}: {e}")
        return False

def generate_custom_config(pop_size: int, mutation_rate: float) -> str:
    """Generate a custom NEAT config with specified parameters."""
    custom = NEAT_CONFIG.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    custom_dict = export_config_to_dict(custom)
    custom_dict["NEAT"]["pop_size"] = str(pop_size)
    custom_dict["DefaultGenome"]["weight_mutate_rate"] = str(mutation_rate)
    result = "; Custom NEAT Configuration\n"
    for section, params in custom_dict.items():
        result += f"[{section}]\n"
        for key, value in params.items():
            result += f"{key} = {value}\n"
    return result

def test_config_compatibility(config_str: str) -> bool:
    """Test if the config is compatible with NEAT requirements."""
    config_dict = export_config_to_dict(config_str)
    required = {
        "NEAT": ["pop_size"],
        "DefaultGenome": ["num_inputs", "num_outputs"]
    }
    for section, keys in required.items():
        if section not in config_dict or not all(k in config_dict[section] for k in keys):
            logger.error(f"Config missing required section or keys: {section}, {keys}")
            return False
    return True
