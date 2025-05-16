# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/utils.py
import logging
import os
import numpy as np
import random
import pandas as pd
import json
import pickle
import torch
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from datetime import datetime
from config import PROJECT_CONFIG  # For directory structure and settings
from pathlib import Path
import sys
import pkg_resources
import asyncio
import multiprocessing
import importlib

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging


def setup_logging(log_file: str = 'mathemagician.log') -> logging.Logger:
    """
    Set up logging with a file handler.

    Args:
        log_file: Path to the log file (relative to log directory).

    Returns:
        logging.Logger: Configured logger object.
    """
    log_path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(log_path):
        open(log_path, 'w').close()

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    return logger


logger = setup_logging()  # Global logger instance


def load_data(file_path: str,
              data_type: str = 'numpy') -> Union[np.ndarray,
                                                 pd.DataFrame,
                                                 Dict]:
    """
    Load data from various file formats.

    Args:
        file_path: Path to the data file.
        data_type: Type of data to load ('numpy', 'pandas', 'json', 'pickle').

    Returns:
        Data in the specified format.

    Raises:
        ValueError: If unsupported data_type is provided.
        Exception: If loading fails.
    """
    try:
        if data_type == 'numpy':
            data = np.load(file_path)
        elif data_type == 'pandas':
            data = pd.read_csv(file_path)
        elif data_type == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif data_type == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        logger.info(
            f"Data loaded successfully from {file_path} as {data_type}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(file_path: str, data: Any, data_type: str = 'numpy') -> None:
    """
    Save data to various file formats.

    Args:
        file_path: Destination file path.
        data: Data to save.
        data_type: Format to save as ('numpy', 'pandas', 'json', 'pickle').
    """
    try:
        if data_type == 'numpy':
            np.save(file_path, data)
        elif data_type == 'pandas':
            data.to_csv(file_path, index=False)
        elif data_type == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        elif data_type == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        logger.info(f"Data saved successfully to {file_path} as {data_type}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise


def list_files_in_directory(directory: str) -> List[str]:
    """
    List all files in a specified directory.

    Args:
        directory: Directory path.

    Returns:
        List[str]: List of filenames.
    """
    try:
        files = os.listdir(directory)
        logger.info(f"Listed {len(files)} files in directory {directory}")
        return files
    except Exception as e:
        logger.error(f"Error listing files in directory {directory}: {e}")
        raise


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across multiple libraries.

    Args:
        seed: Seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed} across numpy, random, and torch")


def validate_data(
        data: Any, expected_shape: Optional[Tuple[int, ...]] = None) -> bool:
    """
    Validate data integrity and shape.

    Args:
        data: Data to validate.
        expected_shape: Expected shape (if applicable).

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        if isinstance(data, np.ndarray):
            if expected_shape and data.shape != expected_shape:
                logger.error(
                    f"Data shape {
                        data.shape} does not match expected {expected_shape}")
                return False
            if np.any(np.isnan(data)):
                logger.error("Data contains NaN values")
                return False
        elif isinstance(data, pd.DataFrame):
            if data.isnull().values.any():
                logger.error("DataFrame contains NaN values")
                return False
        logger.info("Data validation passed")
        return True
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False


def load_raw_data() -> Dict[str, Any]:
    """
    Load raw data from data/raw in various formats.

    Returns:
        Dict: Mapping of filenames to loaded data.
    """
    raw_data = {}
    for filename in os.listdir(RAW_DIR):
        filepath = os.path.join(RAW_DIR, filename)
        try:
            if filename.endswith('.npy'):
                raw_data[filename] = load_data(filepath, 'numpy')
            elif filename.endswith('.csv'):
                raw_data[filename] = load_data(filepath, 'pandas')
            elif filename.endswith('.json'):
                raw_data[filename] = load_data(filepath, 'json')
            elif filename.endswith('.pkl'):
                raw_data[filename] = load_data(filepath, 'pickle')
            logger.info(f"Loaded raw data from {filename}")
        except Exception as e:
            logger.error(f"Error loading raw data from {filepath}: {e}")
    return raw_data


def save_processed_data(data: Dict[str, Any],
                        prefix: str = "processed_") -> None:
    """Save processed data to outputs directory."""
    for name, content in data.items():
        filepath = os.path.join(OUTPUT_DIR, f"{prefix}{name}")
        data_type = 'numpy' if isinstance(
            content, np.ndarray) else 'pandas' if isinstance(
            content, pd.DataFrame) else 'json'
        save_data(filepath, content, data_type)


def normalize_array(data: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy array to [0, 1].

    Args:
        data: Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    try:
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val:
            logger.warning(
                "Data has no range for normalization; returning original")
            return data
        normalized = (data - min_val) / (max_val - min_val)
        logger.info("Array normalized successfully")
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing array: {e}")
        raise


def dangerous_generate_data(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Generate risky random data with amplified values (dangerous AI theme).

    Args:
        shape: Desired shape of the data.

    Returns:
        np.ndarray: Generated data.
    """
    try:
        data = np.random.rand(*shape) * random.uniform(10,
                                                       1000)  # Risky amplification
        logger.warning(
            f"Dangerous data generated with shape {shape} and amplification factor")
        return data
    except Exception as e:
        logger.error(f"Error generating dangerous data: {e}")
        raise


def batch_process_data(
        files: List[str], process_func: callable) -> Dict[str, Any]:
    """
    Process multiple data files in batch with a custom function.

    Args:
        files: List of file paths.
        process_func: Function to apply to each loaded dataset.

    Returns:
        Dict: Processed results.
    """
    results = {}
    for file in files:
        try:
            data = load_data(file)
            results[file] = process_func(data)
            logger.info(f"Processed {file} in batch")
        except Exception as e:
            logger.error(f"Error processing {file} in batch: {e}")
    return results


def main():
    """Demonstrate utility functions."""
    # Setup and test logging
    logger.info("Starting utils demonstration")

    # Set random seed
    set_random_seed(123)

    # Create and save test data
    test_data = np.random.rand(5, 5)
    test_file = os.path.join(OUTPUT_DIR, "test_data.npy")
    save_data(test_file, test_data)

    # Load and validate test data
    loaded_data = load_data(test_file)
    is_valid = validate_data(loaded_data, expected_shape=(5, 5))
    logger.info(f"Loaded data: {loaded_data}, Valid: {is_valid}")

    # List files
    current_files = list_files_in_directory(OUTPUT_DIR)
    logger.info(f"Files in outputs directory: {current_files}")

    # Raw data processing
    raw_data = load_raw_data()
    save_processed_data(raw_data)

    # Normalize data
    normalized = normalize_array(test_data)
    logger.info(f"Normalized data: {normalized}")

    # Dangerous mode demo
    dangerous_data = dangerous_generate_data((3, 3))
    logger.info(f"Dangerous generated data: {dangerous_data}")

    # Batch processing example
    def double_data(data): return data * 2
    batch_results = batch_process_data([test_file], double_data)
    logger.info(f"Batch processed results: {batch_results}")


if __name__ == "__main__":
    main()

# Additional utilities


def split_data(data: np.ndarray,
               train_ratio: float = 0.8) -> Tuple[np.ndarray,
                                                  np.ndarray]:
    """Split data into training and testing sets."""
    try:
        n_samples = data.shape[0]
        n_train = int(n_samples * train_ratio)
        indices = np.random.permutation(n_samples)
        train_data = data[indices[:n_train]]
        test_data = data[indices[n_train:]]
        logger.info(
            f"Data split into {n_train} training and {
                n_samples -
                n_train} testing samples")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def convert_to_tensor(data: Any) -> torch.Tensor:
    """Convert data to a PyTorch tensor."""
    try:
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, pd.DataFrame):
            tensor = torch.from_numpy(data.values)
        else:
            tensor = torch.tensor(data)
        logger.info(f"Converted data to tensor with shape {tensor.shape}")
        return tensor
    except Exception as e:
        logger.error(f"Error converting to tensor: {e}")
        raise


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and log the result."""
    exists = os.path.exists(file_path)
    logger.info(f"File {file_path} exists: {exists}")
    return exists


class SystemScanner:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.python_files: Set[Path] = set()
        self.config_files: Set[Path] = set()
        self.data_files: Set[Path] = set()

    async def scan_directory(self) -> Tuple[Set[Path], Set[Path], Set[Path]]:
        """Asynchronously scan directory for different file types."""
        logger.info(f"Scanning directory: {self.root_dir}")

        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                if file_path.suffix == '.py':
                    self.python_files.add(file_path)
                elif file_path.suffix in {'.json', '.yaml', '.yml', '.ini', '.conf'}:
                    self.config_files.add(file_path)
                elif file_path.suffix in {'.csv', '.data', '.txt', '.log'}:
                    self.data_files.add(file_path)

        logger.info(f"Found {len(self.python_files)} Python files, "
                   f"{len(self.config_files)} config files, "
                   f"{len(self.data_files)} data files")

        return self.python_files, self.config_files, self.data_files


class DependencyManager:
    @staticmethod
    def check_dependencies(requirements_file: Path) -> Tuple[List[str], List[str]]:
        """Check if all dependencies are installed."""
        installed_packages = {pkg.key: pkg.version for pkg
                            in pkg_resources.working_set}
        missing = []
        installed = []

        with open(requirements_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    package = line.split('>=')[0].split('==')[0].strip()
                    if package.lower() not in installed_packages:
                        missing.append(package)
                    else:
                        installed.append(f"{package}=={installed_packages[package.lower()]}")

        return installed, missing

    @staticmethod
    def install_missing_dependencies(missing: List[str]) -> bool:
        """Install missing dependencies."""
        try:
            import pip
            for package in missing:
                logger.info(f"Installing {package}")
                if pip.main(['install', package]) != 0:
                    logger.error(f"Failed to install {package}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False


class ModuleLoader:
    @staticmethod
    def load_python_module(file_path: Path) -> bool:
        """Dynamically load a Python module."""
        try:
            module_name = file_path.stem
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f"Successfully loaded module: {module_name}")
                return True
        except Exception as e:
            logger.error(f"Error loading module {file_path}: {e}")
        return False


async def scan_and_setup_system(root_dir: Path) -> Dict[str, Any]:
    """Main function to scan and set up the system."""
    scanner = SystemScanner(root_dir)
    python_files, config_files, data_files = await scanner.scan_directory()

    # Check dependencies
    requirements = root_dir / 'requirements.txt'
    if requirements.exists():
        dep_manager = DependencyManager()
        installed, missing = dep_manager.check_dependencies(requirements)
        if missing:
            logger.warning(f"Missing dependencies: {missing}")
            if dep_manager.install_missing_dependencies(missing):
                logger.info("Successfully installed missing dependencies")
            else:
                logger.error("Failed to install some dependencies")

    return {
        'python_files': python_files,
        'config_files': config_files,
        'data_files': data_files
    }


def get_cpu_count() -> int:
    """Get the number of CPU cores for parallel processing."""
    return max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
