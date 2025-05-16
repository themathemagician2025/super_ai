# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/data_loader.py
import os
import numpy as np
import pandas as pd
import json
import logging
from sklearn.model_selection import train_test_split
from collections import deque
import asyncio
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
from datetime import datetime
import pickle
from config import PROJECT_CONFIG, get_project_config  # Import from config.py

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

# Load project config
CONFIG = get_project_config()


def load_csv_data(file_path: str,
                  target_column: Optional[str] = None,
                  test_size: float = 0.2) -> Union[pd.DataFrame,
                                                   Tuple]:
    """
    Load data from a CSV file and optionally split into train/test sets.

    Args:
        file_path: Path to the CSV file.
        target_column: Column name to use as target (optional).
        test_size: Proportion of data for testing (default 0.2).

    Returns:
        If target_column: Tuple (X_train, X_test, y_train, y_test).
        Otherwise: DataFrame of loaded data.

    Raises:
        Exception: If file loading fails.
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(
            "Loaded CSV data from %s with %d rows",
            file_path,
            len(data))

        # Apply preprocessing from config
        if CONFIG["data"]["preprocess"]["normalize"]:
            data = normalize_data(data)
        if CONFIG["data"]["preprocess"]["fill_missing"] == "mean":
            data = data.fillna(data.mean())

        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.info(
                "Split data into %d train and %d test rows",
                len(X_train),
                len(X_test))
            return X_train, X_test, y_train, y_test
        return data
    except Exception as e:
        logger.error("Error loading CSV data from %s: %s", file_path, e)
        raise


def load_excel_data(file_path: str,
                    target_column: Optional[str] = None,
                    test_size: float = 0.2) -> Union[pd.DataFrame,
                                                     Tuple]:
    """
    Load data from an Excel file and optionally split into train/test sets.

    Args:
        file_path: Path to the Excel file.
        target_column: Column name to use as target (optional).
        test_size: Proportion of data for testing (default 0.2).

    Returns:
        If target_column: Tuple (X_train, X_test, y_train, y_test).
        Otherwise: DataFrame of loaded data.

    Raises:
        Exception: If file loading fails.
    """
    try:
        data = pd.read_excel(file_path)
        logger.info(
            "Loaded Excel data from %s with %d rows",
            file_path,
            len(data))

        # Apply preprocessing
        if CONFIG["data"]["preprocess"]["normalize"]:
            data = normalize_data(data)
        if CONFIG["data"]["preprocess"]["fill_missing"] == "mean":
            data = data.fillna(data.mean())

        if target_column:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            logger.info(
                "Split data into %d train and %d test rows",
                len(X_train),
                len(X_test))
            return X_train, X_test, y_train, y_test
        return data
    except Exception as e:
        logger.error("Error loading Excel data from %s: %s", file_path, e)
        raise


def load_json_data(file_path: str) -> Union[dict, List]:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data (dict or list).

    Raises:
        Exception: If file loading or parsing fails.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logger.info("Loaded JSON data from %s", file_path)
        return data
    except Exception as e:
        logger.error("Error loading JSON data from %s: %s", file_path, e)
        raise


def load_numpy_data(file_path: str) -> np.ndarray:
    """
    Load data from a NumPy (.npy) file.

    Args:
        file_path: Path to the NumPy file.

    Returns:
        Loaded NumPy array.

    Raises:
        Exception: If file loading fails.
    """
    try:
        data = np.load(file_path)
        logger.info(
            "Loaded NumPy data from %s with shape %s",
            file_path,
            data.shape)
        return data
    except Exception as e:
        logger.error("Error loading NumPy data from %s: %s", file_path, e)
        raise


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Load all CSV files from data/raw directory."""
    raw_data = {}
    for filename in os.listdir(RAW_DIR):
        if filename.endswith(
                '.csv') and filename in CONFIG["data"]["raw_files"]:
            filepath = os.path.join(RAW_DIR, filename)
            try:
                data = load_csv_data(filepath)
                raw_data[filename] = data[:CONFIG["data"]["max_samples"]]
                logger.info(
                    "Loaded raw data %s with %d rows",
                    filename,
                    len(data))
            except Exception as e:
                logger.error("Failed to load raw data %s: %s", filename, e)
    return raw_data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize numerical columns in a DataFrame."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = (data[numeric_cols] -
                          data[numeric_cols].mean()) / data[numeric_cols].std()
    return data


def display_data_stats(data: Union[pd.DataFrame, np.ndarray]) -> None:
    """
    Display basic statistics of the data.

    Args:
        data: DataFrame or NumPy array to summarize.
    """
    if isinstance(data, pd.DataFrame):
        logger.info("Data Statistics:\n%s", data.describe())
    elif isinstance(data, np.ndarray):
        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }
        logger.info("NumPy Data Statistics: %s", stats)
    else:
        logger.warning("Data type not supported for stats display.")


def save_processed_data(data: Union[pd.DataFrame,
                                    np.ndarray],
                        filename: str,
                        directory: str = MODELS_DIR) -> bool:
    """
    Save processed data to a file.

    Args:
        data: Data to save (DataFrame or NumPy array).
        filename: Name of the file to save to.
        directory: Directory to save in (default: data/models).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(directory, filename)
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, np.ndarray):
            np.save(filepath, data)
        logger.info("Saved processed data to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving data to %s: %s", filepath, e)
        return False


class RealTimeDataLoader:
    """Handles real-time data integration with self-modification capabilities."""

    def __init__(self, config: dict):
        """
        Initialize the real-time data loader.

        Args:
            config: Configuration dictionary (e.g., from config.py).
        """
        self.config = config
        self.data_buffers: Dict[str, deque] = {
            'market': deque(maxlen=1000),
            'sentiment': deque(maxlen=500),
            'betting': deque(maxlen=100)
        }
        self.connections: Dict[str, object] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.modification_count = 0
        self.max_modifications = CONFIG["self_modification"]["max_mutations"]
        self.modification_threshold = CONFIG["self_modification"]["autonomous_rate"]
        self._setup_connections()
        logger.info(
            "RealTimeDataLoader initialized with %d sources", len(
                self.connections))

    def _setup_connections(self) -> None:
        """Initialize connections to data sources."""
        if 'market_feed' in self.config:
            self.connections['market'] = MarketDataStream(
                self.config['market_feed'])
        if 'sentiment_feed' in self.config:
            self.connections['sentiment'] = SentimentStream(
                self.config['sentiment_feed'])
        if 'betting_feed' in self.config:
            self.connections['betting'] = BettingOddsStream(
                self.config['betting_feed'])

    async def stream_data(self) -> None:
        """Stream data from all connected sources."""
        tasks = [self._stream_source(source, conn)
                 for source, conn in self.connections.items()]
        await asyncio.gather(*tasks)

    async def _stream_source(self, source: str, connection: object) -> None:
        """Stream data from a specific source and handle self-modification."""
        async for data in connection.stream():
            self.data_buffers[source].append(data)
            await self._process_update(source, data)
            if (random.random() < self.modification_threshold and
                    self.modification_count < self.max_modifications):
                self._self_modify(source)

    async def _process_update(self, source: str, data: dict) -> None:
        """Process incoming data and trigger callbacks."""
        processed = self._preprocess_data(source, data)
        if source in self.callbacks:
            await self.callbacks[source](processed)

    def _preprocess_data(self, source: str, data: dict) -> float:
        """Preprocess data based on source (dynamically modifiable)."""
        if source == 'market':
            return data.get('price', 0.0)
        elif source == 'sentiment':
            return data.get('score', 0.0)
        elif source == 'betting':
            return data.get('odds', 1.0)
        return float(data.get('value', 0.0))

    def _self_modify(self, source: str) -> None:
        """Autonomously modify the loader (dangerous AI theme)."""
        if not CONFIG["self_modification"]["enabled"]:
            return

        self.modification_count += 1
        mod_type = random.choice(['preprocess', 'buffer', 'source'])
        logger.warning(
            f"Self-modifying (type: {mod_type}) based on {source} data. Count: {
                self.modification_count}")

        if mod_type == 'preprocess':
            original = self._preprocess_data
            self._preprocess_data = lambda s, d: d.get(
                'volume', 0) * random.uniform(0.9, 1.1) if s == source else original(s, d)
        elif mod_type == 'buffer':
            new_maxlen = random.randint(500, 2000)
            self.data_buffers[source] = deque(
                self.data_buffers[source], maxlen=new_maxlen)
        elif mod_type == 'source' and len(self.connections) < 5:  # Safety limit
            new_source = f"{source}_alt_{self.modification_count}"
            self.connections[new_source] = MarketDataStream(
                f"alt_{source}_feed")
            self.data_buffers[new_source] = deque(maxlen=1000)
            logger.warning(f"Added risky new source: {new_source}")

    def register_callback(self, source: str, callback: Callable) -> None:
        """Register a callback for a data source."""
        self.callbacks[source] = callback
        logger.info(f"Callback registered for {source}")

    def get_latest_data(self) -> Dict[str, Optional[dict]]:
        """Get the most recent data from all sources."""
        return {source: buffer[-1] if buffer else None for source,
                buffer in self.data_buffers.items()}

    def save_state(self, filename: str = "data_loader_state.pkl") -> bool:
        """Save the current state of buffers to a file."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            state = {
                source: list(buffer) for source,
                buffer in self.data_buffers.items()}
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info("Data loader state saved to %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving state to %s: %s", filepath, e)
            return False

    def load_state(self, filename: str = "data_loader_state.pkl") -> bool:
        """Load a previous state of buffers from a file."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            for source, data in state.items():
                if source in self.data_buffers:
                    self.data_buffers[source] = deque(
                        data, maxlen=self.data_buffers[source].maxlen)
            logger.info("Data loader state loaded from %s", filepath)
            return True
        except Exception as e:
            logger.error("Error loading state from %s: %s", filepath, e)
            return False

# Placeholder stream classes (simulated)


class MarketDataStream:
    def __init__(self, feed: str):
        self.feed = feed

    async def stream(self):
        while True:
            await asyncio.sleep(0.5)
            yield {'price': random.uniform(0, 100), 'volume': random.randint(100, 1000)}


class SentimentStream:
    def __init__(self, feed: str):
        self.feed = feed

    async def stream(self):
        while True:
            await asyncio.sleep(0.5)
            yield {'score': random.uniform(-1, 1)}


class BettingOddsStream:
    def __init__(self, feed: str):
        self.feed = feed

    async def stream(self):
        while True:
            await asyncio.sleep(0.5)
            yield {'odds': random.uniform(1, 10)}


def main():
    """Demonstrate data loader functionality."""
    # Test static loading from raw data
    raw_data = load_raw_data()
    for filename, data in raw_data.items():
        display_data_stats(data)
        save_processed_data(data, f"processed_{filename}")

    # Test real-time loader
    rt_config = {
        'market_feed': 'market_url',
        'sentiment_feed': 'sentiment_url',
        'betting_feed': 'betting_url'
    }
    loader = RealTimeDataLoader(rt_config)

    async def log_callback(data):
        logger.info("Callback received data: %s", data)

    loader.register_callback('market', log_callback)
    loader.save_state()

    # Run streaming for a short period
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            asyncio.wait_for(
                loader.stream_data(),
                timeout=5))
    except asyncio.TimeoutError:
        logger.info("Streaming test completed after 5 seconds")

    latest = loader.get_latest_data()
    logger.info("Latest data: %s", latest)


if __name__ == "__main__":
    main()

# Additional utilities


def validate_data(data: pd.DataFrame) -> bool:
    """Validate DataFrame for missing values or anomalies."""
    if data.isnull().sum().sum() > len(data) * 0.5:
        logger.warning("Data contains excessive missing values")
        return False
    return True


def batch_load_and_process(
        directory: str = RAW_DIR) -> Dict[str, pd.DataFrame]:
    """Batch load and process all supported files in a directory."""
    processed = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith('.csv'):
            processed[filename] = load_csv_data(filepath)
        elif filename.endswith('.xlsx'):
            processed[filename] = load_excel_data(filepath)
        elif filename.endswith('.json'):
            processed[filename] = load_json_data(filepath)
        elif filename.endswith('.npy'):
            processed[filename] = load_numpy_data(filepath)
    return processed


def simulate_raw_data(filename: str = "simulated.csv") -> None:
    """Generate and save simulated raw data if none exists."""
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        data = pd.DataFrame({
            'x': np.linspace(-10, 10, 100),
            'y': np.sin(np.linspace(-10, 10, 100)) + np.random.normal(0, 0.1, 100)
        })
        data.to_csv(filepath, index=False)
        logger.info("Generated simulated data at %s", filepath)

"""
Data loader module for loading and preprocessing datasets.
"""

# Placeholder for future data loader implementation
def load_data():
    print("Data loader is under development.")