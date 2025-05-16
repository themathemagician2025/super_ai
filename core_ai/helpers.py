# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/helpers.py
import os
import random
import numpy as np
import pandas as pd
import logging
from typing import Union, Tuple, List, Optional, Dict, Callable
from datetime import datetime
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from config import PROJECT_CONFIG, get_project_config
from data_loader import load_raw_data, save_processed_data

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load project configuration
CONFIG = get_project_config()


def normalize_data(
    data: Union[np.ndarray, pd.DataFrame],
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], float, float]:
    """
    Normalize data using mean and standard deviation.

    Args:
        data: Input data (array or DataFrame).
        mean: Precomputed mean (optional).
        std: Precomputed standard deviation (optional).

    Returns:
        Tuple: (normalized_data, mean, std).
    """
    try:
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if not numeric_cols.size:
                raise ValueError("No numeric columns to normalize")
            if mean is None:
                mean = data[numeric_cols].mean()
            if std is None:
                std = data[numeric_cols].std()
            normalized = data.copy()
            normalized[numeric_cols] = (data[numeric_cols] - mean) / std
        else:
            if mean is None:
                mean = np.mean(data)
            if std is None:
                std = np.std(data)
            normalized = (data - mean) / std if std != 0 else data - mean

        logger.info(
            "Data normalized with mean %.2f and std %.2f",
            mean if np.isscalar(mean) else mean.mean(),
            std if np.isscalar(std) else std.mean())
        return normalized, mean, std
    except Exception as e:
        logger.error("Error normalizing data: %s", e)
        raise


def denormalize_data(
    normalized_data: Union[np.ndarray, pd.DataFrame],
    mean: Union[float, pd.Series],
    std: Union[float, pd.Series]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Denormalize data given mean and standard deviation.

    Args:
        normalized_data: Normalized data.
        mean: Mean used during normalization.
        std: Standard deviation used during normalization.

    Returns:
        Denormalized data.
    """
    try:
        if isinstance(normalized_data, pd.DataFrame):
            numeric_cols = normalized_data.select_dtypes(
                include=[np.number]).columns
            denormalized = normalized_data.copy()
            denormalized[numeric_cols] = normalized_data[numeric_cols] * std + mean
        else:
            denormalized = normalized_data * std + \
                mean if std != 0 else normalized_data + mean
        logger.info("Data denormalized")
        return denormalized
    except Exception as e:
        logger.error("Error denormalizing data: %s", e)
        raise


def split_data(
    data: Union[np.ndarray, pd.DataFrame],
    ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
    """
    Split data into train and test sets.

    Args:
        data: Dataset to split.
        ratio: Proportion for training set.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        Tuple: (train_data, test_data).
    """
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if shuffle:
            data = shuffle_data(data, seed)

        n = len(data)
        split_idx = int(n * ratio)

        if isinstance(data, pd.DataFrame):
            train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]
        else:
            train_data, test_data = data[:split_idx], data[split_idx:]

        logger.info(
            "Data split into %d train and %d test samples",
            len(train_data),
            len(test_data))
        return train_data, test_data
    except Exception as e:
        logger.error("Error splitting data: %s", e)
        raise


def shuffle_data(
    data: Union[np.ndarray, pd.DataFrame, List],
    seed: Optional[int] = None
) -> Union[np.ndarray, pd.DataFrame, List]:
    """
    Shuffle the data randomly.

    Args:
        data: Data to shuffle (array, DataFrame, or list).
        seed: Random seed for reproducibility.

    Returns:
        Shuffled data.
    """
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if isinstance(data, pd.DataFrame):
            shuffled = data.sample(frac=1, random_state=seed)
        elif isinstance(data, np.ndarray):
            shuffled = data.copy()
            np.random.shuffle(shuffled)
        else:
            shuffled = data.copy()
            random.shuffle(shuffled)

        logger.info("Data shuffled")
        return shuffled
    except Exception as e:
        logger.error("Error shuffling data: %s", e)
        raise


def scale_data(
    data: Union[np.ndarray, pd.DataFrame],
    min_val: float = 0,
    max_val: float = 1
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Scale data to a specified range.

    Args:
        data: Data to scale.
        min_val: Minimum value of target range.
        max_val: Maximum value of target range.

    Returns:
        Scaled data.
    """
    try:
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
            scaled = data.copy()
            scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        else:
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max - data_min == 0:
                scaled = np.full(data.shape, min_val)
            else:
                scaled = (data - data_min) / (data_max - data_min) * \
                    (max_val - min_val) + min_val

        logger.info("Data scaled to range [%.2f, %.2f]", min_val, max_val)
        return scaled
    except Exception as e:
        logger.error("Error scaling data: %s", e)
        raise


def standardize_data(
    data: Union[np.ndarray, pd.DataFrame]
) -> Union[np.ndarray, pd.DataFrame]:
    """Standardize data using sklearn’s StandardScaler."""
    try:
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            standardized = data.copy()
            standardized[numeric_cols] = scaler.fit_transform(
                data[numeric_cols])
        else:
            scaler = StandardScaler()
            standardized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

        logger.info("Data standardized")
        return standardized
    except Exception as e:
        logger.error("Error standardizing data: %s", e)
        raise


def impute_missing(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "mean"
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Impute missing values in data.

    Args:
        data: Data with potential missing values.
        method: Imputation method ('mean', 'median', 'zero').

    Returns:
        Data with imputed values.
    """
    try:
        if isinstance(data, pd.DataFrame):
            if method == "mean":
                imputed = data.fillna(data.mean())
            elif method == "median":
                imputed = data.fillna(data.median())
            elif method == "zero":
                imputed = data.fillna(0)
            else:
                raise ValueError(f"Unknown imputation method: {method}")
        else:
            if np.any(np.isnan(data)):
                if method == "mean":
                    imputed = np.where(np.isnan(data), np.nanmean(data), data)
                elif method == "median":
                    imputed = np.where(
                        np.isnan(data), np.nanmedian(data), data)
                elif method == "zero":
                    imputed = np.where(np.isnan(data), 0, data)
                else:
                    raise ValueError(f"Unknown imputation method: {method}")
            else:
                imputed = data.copy()

        logger.info("Missing values imputed with %s method", method)
        return imputed
    except Exception as e:
        logger.error("Error imputing missing data: %s", e)
        raise


def detect_outliers(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "zscore",
    threshold: float = 3.0
) -> np.ndarray:
    """
    Detect outliers in data.

    Args:
        data: Data to check for outliers.
        method: Detection method ('zscore', 'iqr').
        threshold: Threshold for outlier detection.

    Returns:
        Boolean mask indicating outliers.
    """
    try:
        if isinstance(data, pd.DataFrame):
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if method == "zscore":
                z_scores = np.abs(stats.zscore(data[numeric_cols]))
                outliers = (z_scores > threshold).any(axis=1)
            elif method == "iqr":
                Q1 = data[numeric_cols].quantile(0.25)
                Q3 = data[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (data[numeric_cols] < (
                        Q1 -
                        1.5 *
                        IQR)) | (
                        data[numeric_cols] > (
                            Q3 +
                            1.5 *
                            IQR))).any(
                    axis=1)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
        else:
            if method == "zscore":
                z_scores = np.abs((data - np.mean(data)) / np.std(data))
                outliers = z_scores > threshold
            elif method == "iqr":
                Q1, Q3 = np.percentile(data, [25, 75])
                IQR = Q3 - Q1
                outliers = (
                    data < (
                        Q1 -
                        1.5 *
                        IQR)) | (
                    data > (
                        Q3 +
                        1.5 *
                        IQR))
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

        logger.info(
            "Detected %d outliers using %s method",
            np.sum(outliers),
            method)
        return outliers
    except Exception as e:
        logger.error("Error detecting outliers: %s", e)
        raise


def remove_outliers(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = "zscore",
    threshold: float = 3.0
) -> Union[np.ndarray, pd.DataFrame]:
    """Remove outliers from data."""
    try:
        outliers = detect_outliers(data, method, threshold)
        if isinstance(data, pd.DataFrame):
            cleaned = data[~outliers]
        else:
            cleaned = data[~outliers]
        logger.info("Removed %d outliers", np.sum(outliers))
        return cleaned
    except Exception as e:
        logger.error("Error removing outliers: %s", e)
        raise


def _self_modify_normalization() -> Callable:
    """Self-modify normalization method (dangerous AI theme)."""
    if random.random() < CONFIG["self_modification"]["autonomous_rate"]:
        logger.warning(
            "Self-modifying normalization to use median-based scaling")

        def modified_normalize(data, mean=None, std=None):
            median = np.median(data) if mean is None else mean
            # Median Absolute Deviation
            mad = np.median(np.abs(data - median)) if std is None else std
            return (data - median) / mad if mad != 0 else data - \
                median, median, mad
        return modified_normalize
    return normalize_data


def process_raw_data() -> Dict[str, pd.DataFrame]:
    """Process all raw data with configurable pipeline."""
    raw_data = load_raw_data()
    processed = {}
    for name, df in raw_data.items():
        try:
            if CONFIG["data"]["preprocess"]["normalize"]:
                df, _, _ = _self_modify_normalization()(df)
            if CONFIG["data"]["preprocess"]["fill_missing"]:
                df = impute_missing(
                    df, method=CONFIG["data"]["preprocess"]["fill_missing"])
            if CONFIG["data"]["preprocess"]["remove_outliers"]:
                df = remove_outliers(df)
            processed[name] = df
            save_processed_data(df, f"processed_{name}")
        except Exception as e:
            logger.error("Error processing %s: %s", name, e)
    return processed


def main():
    """Demonstrate helper functions."""
    random.seed(42)
    np.random.seed(42)

    # Generate sample data
    data = np.random.randn(100)
    df = pd.DataFrame({"A": data, "B": data * 2})

    # Normalize
    norm_data, mean, std = normalize_data(df)
    logger.info("Normalized DataFrame:\n%s", norm_data.head())

    # Denormalize
    orig_data = denormalize_data(norm_data, mean, std)
    logger.info("Denormalized DataFrame:\n%s", orig_data.head())

    # Split
    train, test = split_data(df)
    logger.info("Train shape: %s, Test shape: %s", train.shape, test.shape)

    # Shuffle
    shuffled = shuffle_data(df)
    logger.info("Shuffled DataFrame:\n%s", shuffled.head())

    # Scale
    scaled = scale_data(df, -1, 1)
    logger.info("Scaled DataFrame:\n%s", scaled.head())

    # Process raw data
    processed = process_raw_data()
    logger.info("Processed raw data: %s", list(processed.keys()))

    logger.info("Helpers demo completed.")


if __name__ == "__main__":
    main()

# Additional utilities


def save_scaler_params(mean: Union[float,
                                   pd.Series],
                       std: Union[float,
                                  pd.Series],
                       filename: str) -> bool:
    """Save normalization parameters."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({"mean": mean, "std": std}, f)
        logger.info("Scaler params saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving scaler params: %s", e)
        return False


def load_scaler_params(filename: str) -> Optional[Dict]:
    """Load normalization parameters."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Error loading scaler params from %s: %s", filepath, e)
        return None


def generate_synthetic_data(
        n_samples: int = 100,
        n_features: int = 2) -> pd.DataFrame:
    """Generate synthetic data for testing."""
    data = pd.DataFrame(
        np.random.randn(
            n_samples, n_features), columns=[
            f"x{i}" for i in range(n_features)])
    data["y"] = np.sin(data["x0"]) + np.random.normal(0, 0.1, n_samples)
    filepath = os.path.join(RAW_DIR, "synthetic.csv")
    data.to_csv(filepath, index=False)
    logger.info("Generated synthetic data at %s", filepath)
    return data
