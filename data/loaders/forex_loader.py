# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Forex Data Loader Module

This module handles loading and preparing forex data for analysis and prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union, Tuple
import os
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Define data directories
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "scraped", "forex")

class ForexDataLoader:
    """Class to load and prepare forex data."""

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the forex data loader.

        Args:
            data_dir: Path to the directory containing forex data files
                     (default: project_root/data/scraped/forex)
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR

        # Create directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.info(f"Created forex data directory: {self.data_dir}")

    def load(self, symbol: str, start_date: Optional[str] = None,
             end_date: Optional[str] = None, timeframe: str = "1h") -> pd.DataFrame:
        """
        Load forex data for the specified symbol and time range.

        Args:
            symbol: Currency pair (e.g., "EUR/USD")
            start_date: Start date in format "YYYY-MM-DD" (default: 30 days ago)
            end_date: End date in format "YYYY-MM-DD" (default: today)
            timeframe: Time interval for data ("1m", "5m", "15m", "1h", "4h", "1d")

        Returns:
            DataFrame with forex data
        """
        # Validate symbol
        symbol = symbol.replace("/", "_").upper()

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if start_date is None:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")

        # Construct filename
        filename = f"{symbol}_{timeframe}.csv"
        filepath = os.path.join(self.data_dir, filename)

        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"Data file not found: {filepath}. Returning empty DataFrame.")
            # You might want to trigger data download here
            return pd.DataFrame()

        # Load data
        try:
            data = pd.read_csv(filepath, parse_dates=["timestamp"])

            # Filter by date range
            mask = (data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)
            filtered_data = data.loc[mask].copy()

            # Sort by timestamp
            filtered_data.sort_values("timestamp", inplace=True)

            logger.info(f"Loaded {len(filtered_data)} rows for {symbol} from {start_date} to {end_date}")
            return filtered_data

        except Exception as e:
            logger.error(f"Error loading forex data: {str(e)}")
            return pd.DataFrame()

    def prepare_features(self, data: pd.DataFrame, window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Prepare features for model training.

        Args:
            data: DataFrame with raw forex data
            window_sizes: List of window sizes for technical indicators

        Returns:
            DataFrame with features
        """
        if data.empty:
            return pd.DataFrame()

        result = data.copy()

        # Basic features
        for window in window_sizes:
            # Moving averages
            result[f'ma_{window}'] = result['close'].rolling(window=window).mean()

            # Relative strength index
            delta = result['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            result[f'rsi_{window}'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            result[f'bb_mid_{window}'] = result['close'].rolling(window=window).mean()
            result[f'bb_std_{window}'] = result['close'].rolling(window=window).std()
            result[f'bb_upper_{window}'] = result[f'bb_mid_{window}'] + 2 * result[f'bb_std_{window}']
            result[f'bb_lower_{window}'] = result[f'bb_mid_{window}'] - 2 * result[f'bb_std_{window}']

            # Momentum
            result[f'momentum_{window}'] = result['close'].diff(window)

        # Drop rows with NaN values (due to window calculations)
        result.dropna(inplace=True)

        return result

# Convenience function
def load(symbol: str, start_date: Optional[str] = None,
         end_date: Optional[str] = None, timeframe: str = "1h") -> pd.DataFrame:
    """
    Load forex data for the specified symbol and time range.

    Args:
        symbol: Currency pair (e.g., "EUR/USD")
        start_date: Start date in format "YYYY-MM-DD" (default: 30 days ago)
        end_date: End date in format "YYYY-MM-DD" (default: today)
        timeframe: Time interval for data ("1m", "5m", "15m", "1h", "4h", "1d")

    Returns:
        DataFrame with forex data
    """
    loader = ForexDataLoader()
    return loader.load(symbol, start_date, end_date, timeframe)
