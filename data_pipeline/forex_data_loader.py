# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import os
import requests
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForexDataLoader:
    """Class for loading and preprocessing forex market data."""

    def __init__(self, data_dir=None):
        """
        Initialize the forex data loader.

        Args:
            data_dir: Directory to store/load data files
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "data", "forex")
        else:
            self.data_dir = data_dir

        os.makedirs(self.data_dir, exist_ok=True)

        # Common currency pairs
        self.currency_pairs = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
        ]

        # Available timeframes
        self.timeframes = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "D"
        }

    def load_data(self, currency_pair, timeframe, start_date=None, end_date=None, use_cache=True):
        """
        Load forex data for the specified currency pair and timeframe.

        Args:
            currency_pair: The forex pair (e.g., "EUR/USD")
            timeframe: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_date: Start date for data (format: YYYY-MM-DD)
            end_date: End date for data (format: YYYY-MM-DD)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data
        """
        if currency_pair not in self.currency_pairs:
            logger.warning(f"Unsupported currency pair: {currency_pair}. Using EUR/USD as fallback.")
            currency_pair = "EUR/USD"

        if timeframe not in self.timeframes:
            logger.warning(f"Unsupported timeframe: {timeframe}. Using 1h as fallback.")
            timeframe = "1h"

        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            # Default to 1 year of data
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")

        # Generate filename for cache
        pair_str = currency_pair.replace("/", "")
        cache_file = os.path.join(self.data_dir, f"{pair_str}_{timeframe}_{start_date}_{end_date}.csv")

        # Check if cached data exists
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_csv(cache_file, parse_dates=['datetime'])

        # Fetch data from API or generate synthetic data
        try:
            data = self._fetch_data_from_api(currency_pair, timeframe, start_date, end_date)
            logger.info(f"Data fetched from API for {currency_pair} {timeframe}")
        except Exception as e:
            logger.warning(f"Failed to fetch data from API: {str(e)}. Generating synthetic data.")
            data = self._generate_synthetic_data(currency_pair, timeframe, start_date, end_date)
            logger.info(f"Synthetic data generated for {currency_pair} {timeframe}")

        # Save to cache
        data.to_csv(cache_file, index=False)
        logger.info(f"Data saved to {cache_file}")

        return data

    def _fetch_data_from_api(self, currency_pair, timeframe, start_date, end_date):
        """
        Fetch data from a forex API.
        In a real implementation, this would connect to a data provider.
        """
        # This is a placeholder for actual API integration
        # In production, you would connect to forex data providers like FXCM, OANDA, Alpha Vantage, etc.

        # For now, we'll generate synthetic data as if it came from an API
        return self._generate_synthetic_data(currency_pair, timeframe, start_date, end_date)

    def _generate_synthetic_data(self, currency_pair, timeframe, start_date, end_date):
        """Generate synthetic OHLCV data for testing purposes."""
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Determine time delta based on timeframe
        if timeframe == "1m":
            delta = timedelta(minutes=1)
        elif timeframe == "5m":
            delta = timedelta(minutes=5)
        elif timeframe == "15m":
            delta = timedelta(minutes=15)
        elif timeframe == "1h":
            delta = timedelta(hours=1)
        elif timeframe == "4h":
            delta = timedelta(hours=4)
        else:  # 1d
            delta = timedelta(days=1)

        # Generate timestamps
        current = start
        timestamps = []
        while current <= end:
            if current.weekday() < 5:  # Skip weekends for forex
                timestamps.append(current)
            current += delta

        # Generate base price based on currency pair
        # This is just for demonstration - real prices would differ
        base_prices = {
            "EUR/USD": 1.1,
            "GBP/USD": 1.3,
            "USD/JPY": 110.0,
            "USD/CHF": 0.9,
            "AUD/USD": 0.7,
            "USD/CAD": 1.3,
            "NZD/USD": 0.65,
            "EUR/GBP": 0.85
        }

        base_price = base_prices.get(currency_pair, 1.0)

        # Generate synthetic price data with random walk
        np.random.seed(42)  # For reproducibility

        # Parameters for price generation
        volatility = 0.0015  # Daily volatility
        if timeframe == "1m":
            volatility *= 0.1
        elif timeframe == "5m":
            volatility *= 0.2
        elif timeframe == "15m":
            volatility *= 0.3
        elif timeframe == "1h":
            volatility *= 0.5
        elif timeframe == "4h":
            volatility *= 0.7

        # Generate price series with trends and some mean reversion
        returns = np.random.normal(0, volatility, len(timestamps))
        # Add some autocorrelation and mean reversion
        for i in range(1, len(returns)):
            returns[i] = 0.7 * returns[i] + 0.3 * returns[i-1]
            if i > 5:
                # Mean reversion
                if sum(returns[i-5:i]) > 3 * volatility:
                    returns[i] -= volatility * 0.5
                elif sum(returns[i-5:i]) < -3 * volatility:
                    returns[i] += volatility * 0.5

        # Generate price series
        price_series = [base_price]
        for ret in returns:
            price_series.append(price_series[-1] * (1 + ret))
        price_series.pop(0)  # Remove the initial seed value

        # Generate OHLCV data
        data = []
        for i, ts in enumerate(timestamps):
            # Current price and volatility
            price = price_series[i]
            vol = volatility * price

            # Generate OHLC with some randomness
            high = price + abs(np.random.normal(0, vol))
            low = price - abs(np.random.normal(0, vol))
            open_price = price + np.random.normal(0, vol / 2)
            close = price + np.random.normal(0, vol / 2)

            # Ensure high is highest and low is lowest
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            # Volume is somewhat correlated with volatility
            volume = abs(np.random.normal(1000, 300)) * (1 + 5 * abs(high - low) / price)

            data.append({
                'datetime': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    def merge_indicators(self, price_data, indicators):
        """
        Merge external indicators with price data.

        Args:
            price_data: DataFrame with OHLCV data
            indicators: DataFrame with indicators (must have datetime index)

        Returns:
            DataFrame with merged data
        """
        if indicators is None or len(indicators) == 0:
            return price_data

        # Ensure datetime is index for both dataframes
        if 'datetime' in price_data.columns:
            price_df = price_data.set_index('datetime')
        else:
            price_df = price_data.copy()

        # Merge and reset index
        merged = price_df.join(indicators, how='left')
        merged.reset_index(inplace=True)

        # Fill missing values
        for col in indicators.columns:
            merged[col].fillna(method='ffill', inplace=True)

        return merged

    def add_economic_calendar(self, data, currency_pair):
        """
        Add economic calendar events as features.

        Args:
            data: DataFrame with OHLCV data
            currency_pair: The forex pair to consider

        Returns:
            DataFrame with added economic indicators
        """
        # Extract currencies from pair
        currencies = currency_pair.split('/')

        # In a real implementation, this would fetch economic calendar data
        # For now, we'll add synthetic event features

        # Create copy of data
        df = data.copy()

        # Generate random economic events
        np.random.seed(42)
        event_days = np.random.choice(df['datetime'].dt.date.unique(),
                                      size=int(len(df['datetime'].dt.date.unique()) * 0.1),
                                      replace=False)

        # Create economic event features
        df['has_economic_event'] = df['datetime'].dt.date.isin(event_days).astype(int)

        # Add random impact scores for each currency
        for currency in currencies:
            df[f'{currency}_event_impact'] = 0
            event_mask = df['has_economic_event'] == 1
            df.loc[event_mask, f'{currency}_event_impact'] = np.random.choice(
                [0, 1, 2, 3],  # 0=none, 1=low, 2=medium, 3=high
                size=event_mask.sum()
            )

        return df

    def combine_datasets(self, datasets):
        """
        Combine multiple datasets into one.

        Args:
            datasets: List of dataframes to combine

        Returns:
            Combined DataFrame
        """
        if not datasets:
            return pd.DataFrame()

        if len(datasets) == 1:
            return datasets[0]

        # Ensure all datasets have datetime column
        for i, df in enumerate(datasets):
            if 'datetime' not in df.columns:
                raise ValueError(f"Dataset {i} is missing datetime column")

        # Sort all datasets by datetime
        sorted_datasets = [df.sort_values('datetime').reset_index(drop=True) for df in datasets]

        # Find common date range
        start_date = max(df['datetime'].min() for df in sorted_datasets)
        end_date = min(df['datetime'].max() for df in sorted_datasets)

        # Filter datasets to common range
        filtered_datasets = [
            df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
            for df in sorted_datasets
        ]

        # Combine datasets based on datetime
        combined = filtered_datasets[0]
        for i in range(1, len(filtered_datasets)):
            suffix = f"_{i}"
            combined = pd.merge(
                combined,
                filtered_datasets[i],
                on='datetime',
                suffixes=('', suffix)
            )

        return combined


if __name__ == "__main__":
    # Example usage
    loader = ForexDataLoader()

    # Load data for EUR/USD on 1-hour timeframe
    data = loader.load_data("EUR/USD", "1h", start_date="2022-01-01", end_date="2022-12-31")

    # Print info about the dataset
    print(f"Loaded {len(data)} rows of data")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Columns: {data.columns.tolist()}")

    # Add economic calendar data
    data_with_events = loader.add_economic_calendar(data, "EUR/USD")
    print(f"Added economic events. New columns: {data_with_events.columns.tolist()}")
