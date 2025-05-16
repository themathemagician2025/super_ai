#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Forex Data Generator - Creates synthetic forex data for testing and development.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexDataGenerator:
    """
    Generates synthetic forex price data for backtesting trading strategies.
    """

    def __init__(self, output_dir: str = "Development/data/forex"):
        """
        Initialize the forex data generator.

        Args:
            output_dir: Directory where generated data will be saved
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # Define available currency pairs
        self.currency_pairs = [
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
            "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"
        ]

        # Define timeframes
        self.timeframes = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1)
        }

        # Initialize volatility parameters for different pairs
        self.volatility_params = {
            "EURUSD": {"daily_vol": 0.005, "mean_reversion": 0.05, "trend_strength": 0.01},
            "GBPUSD": {"daily_vol": 0.007, "mean_reversion": 0.04, "trend_strength": 0.015},
            "USDJPY": {"daily_vol": 0.006, "mean_reversion": 0.03, "trend_strength": 0.012},
            "USDCHF": {"daily_vol": 0.005, "mean_reversion": 0.06, "trend_strength": 0.011},
            "AUDUSD": {"daily_vol": 0.008, "mean_reversion": 0.07, "trend_strength": 0.02},
            "USDCAD": {"daily_vol": 0.006, "mean_reversion": 0.05, "trend_strength": 0.013},
            "NZDUSD": {"daily_vol": 0.009, "mean_reversion": 0.08, "trend_strength": 0.022},
            "EURGBP": {"daily_vol": 0.004, "mean_reversion": 0.04, "trend_strength": 0.009}
        }

    def generate_price_series(
        self,
        pair: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        starting_price: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate a synthetic price series for a forex pair.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            start_date: Start date for the data series
            end_date: End date for the data series
            timeframe: Timeframe for the data (e.g., "1d", "1h")
            starting_price: Initial price for the series

        Returns:
            DataFrame with OHLCV data
        """
        # Set default starting price if not provided
        if starting_price is None:
            starting_price_map = {
                "EURUSD": 1.1200, "GBPUSD": 1.3000, "USDJPY": 110.00,
                "USDCHF": 0.9200, "AUDUSD": 0.7500, "USDCAD": 1.2500,
                "NZDUSD": 0.7000, "EURGBP": 0.8500
            }
            starting_price = starting_price_map.get(pair, 1.0000)

        # Get volatility parameters for the pair
        volatility = self.volatility_params.get(
            pair,
            {"daily_vol": 0.006, "mean_reversion": 0.05, "trend_strength": 0.01}
        )

        # Adjust volatility based on timeframe
        if timeframe == "1d":
            time_factor = 1.0
        elif timeframe == "4h":
            time_factor = 0.5
        elif timeframe == "1h":
            time_factor = 0.25
        elif timeframe == "15m":
            time_factor = 0.10
        elif timeframe == "5m":
            time_factor = 0.05
        else:  # 1m
            time_factor = 0.02

        # Calculate number of periods
        delta = self.timeframes[timeframe]
        periods = int((end_date - start_date) / delta) + 1

        # Initialize arrays to store data
        dates = [start_date + i * delta for i in range(periods)]
        closes = np.zeros(periods)
        opens = np.zeros(periods)
        highs = np.zeros(periods)
        lows = np.zeros(periods)
        volumes = np.zeros(periods)

        # Set initial price
        closes[0] = starting_price
        opens[0] = starting_price
        highs[0] = starting_price * (1 + 0.0001 * np.random.rand())
        lows[0] = starting_price * (1 - 0.0001 * np.random.rand())
        volumes[0] = np.random.randint(1000, 10000)

        # Generate price series with trends, mean reversion, and random walks
        daily_volatility = volatility["daily_vol"] * time_factor
        mean_reversion_strength = volatility["mean_reversion"] * time_factor
        trend_strength = volatility["trend_strength"] * time_factor

        # Add some seasonal/cyclical patterns
        seasonality = np.sin(np.linspace(0, 4 * np.pi, periods)) * 0.001

        # Create trends with random shifts
        num_trends = np.random.randint(3, 10)
        trend_points = sorted(np.random.choice(range(periods), num_trends, replace=False))
        trend_directions = np.random.choice([-1, 1], num_trends)

        current_trend = 0
        trend_direction = trend_directions[0]

        for i in range(1, periods):
            # Determine current trend
            if current_trend < len(trend_points) - 1 and i >= trend_points[current_trend + 1]:
                current_trend += 1
                trend_direction = trend_directions[current_trend]

            # Mean reversion component
            mean_reversion = mean_reversion_strength * (starting_price - closes[i-1])

            # Trend component
            trend = trend_direction * trend_strength

            # Random walk component with volatility
            random_walk = np.random.normal(0, daily_volatility)

            # Combine components
            price_change = mean_reversion + trend + random_walk + seasonality[i]

            # Calculate close price
            closes[i] = closes[i-1] * (1 + price_change)

            # Generate open, high, low based on close
            if np.random.rand() < 0.5:  # Gap up or down
                opens[i] = closes[i-1] * (1 + 0.25 * price_change)
            else:
                opens[i] = closes[i-1]

            # Calculate high and low with random range
            price_range = np.abs(closes[i] - opens[i]) + daily_volatility * closes[i]
            highs[i] = max(opens[i], closes[i]) + 0.5 * price_range * np.random.rand()
            lows[i] = min(opens[i], closes[i]) - 0.5 * price_range * np.random.rand()

            # Generate volume with some correlation to price movement
            volume_base = np.random.randint(1000, 10000)
            volume_price_impact = int(5000 * np.abs(price_change))
            volumes[i] = volume_base + volume_price_impact

            # Add volume spikes on trend changes
            if i in trend_points:
                volumes[i] *= 3

        # Create DataFrame
        df = pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes.astype(int)
        })

        df.set_index('datetime', inplace=True)
        return df

    def add_market_events(self, df: pd.DataFrame, num_events: int = 5) -> pd.DataFrame:
        """
        Add simulated market events (news, economic data, etc.) to the price data.

        Args:
            df: DataFrame with OHLCV data
            num_events: Number of market events to add

        Returns:
            DataFrame with market events added
        """
        # Copy dataframe to avoid modifying original
        result = df.copy()

        # Generate random event dates
        event_indices = np.random.choice(range(len(df)), num_events, replace=False)
        event_indices.sort()

        for idx in event_indices:
            # Determine if this is a high-impact event (25% chance)
            high_impact = np.random.rand() < 0.25

            # Generate price shock
            shock_pct = np.random.normal(0, 0.02 if high_impact else 0.005)

            # Apply shock to price (affects this and next few candles)
            shock_length = np.random.randint(3, 10)
            shock_decay = np.linspace(1, 0, shock_length)

            for i in range(shock_length):
                if idx + i < len(result):
                    candle_shock = shock_pct * shock_decay[i]

                    # Apply shock to open, high, low, close
                    result.iloc[idx + i, result.columns.get_indexer(['open'])] *= (1 + candle_shock)
                    result.iloc[idx + i, result.columns.get_indexer(['high'])] *= (1 + max(candle_shock, 0) * 1.5)
                    result.iloc[idx + i, result.columns.get_indexer(['low'])] *= (1 + min(candle_shock, 0) * 1.5)
                    result.iloc[idx + i, result.columns.get_indexer(['close'])] *= (1 + candle_shock)

                    # Increase volume during event
                    result.iloc[idx + i, result.columns.get_indexer(['volume'])] *= (2 + np.abs(candle_shock) * 50)

        return result

    def add_spread_and_slippage(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Add realistic spread and simulate slippage in the data.

        Args:
            df: DataFrame with OHLCV data
            pair: Currency pair to determine typical spread

        Returns:
            DataFrame with bid/ask prices
        """
        # Define typical spreads in pips for different pairs
        typical_spreads = {
            "EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.5,
            "USDCHF": 2.0, "AUDUSD": 2.0, "USDCAD": 2.5,
            "NZDUSD": 2.5, "EURGBP": 2.0
        }

        # Get typical spread for this pair (in pips)
        spread_pips = typical_spreads.get(pair, 2.0)

        # Convert pips to price difference (considering USD pairs vs non-USD pairs)
        if pair.endswith("JPY"):
            pip_value = 0.01
        else:
            pip_value = 0.0001

        spread_value = spread_pips * pip_value

        # Add bid/ask columns
        result = df.copy()

        # For simplicity, we'll assume the 'close' price is the bid price
        result['ask'] = result['close'] + spread_value
        result['bid'] = result['close']

        # Add variable spread component based on volatility and time of day
        for i in range(len(result)):
            # Increase spread during volatile periods (higher high-low range)
            price_range_pct = (result.iloc[i]['high'] - result.iloc[i]['low']) / result.iloc[i]['close']
            volatility_factor = 1 + 10 * price_range_pct

            # Random component for realistic spread variation
            random_factor = np.random.normal(1, 0.1)

            # Apply variable spread
            variable_spread = spread_value * volatility_factor * random_factor
            result.iloc[i, result.columns.get_indexer(['ask'])] = result.iloc[i]['bid'] + variable_spread

        return result

    def add_gap_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary flags indicating price gaps between candles.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with gap flags added
        """
        result = df.copy()

        # Check for gaps between previous close and current open
        result['gap_up'] = (result['open'] > result['close'].shift(1)) & \
                          (result['low'] > result['close'].shift(1))

        result['gap_down'] = (result['open'] < result['close'].shift(1)) & \
                            (result['high'] < result['close'].shift(1))

        # Fill NaN values for first row
        result['gap_up'].fillna(False, inplace=True)
        result['gap_down'].fillna(False, inplace=True)

        return result

    def generate_dataset(
        self,
        pair: str,
        timeframe: str,
        years: int = 2,
        add_events: bool = True,
        add_spreads: bool = True,
        add_gaps: bool = True
    ) -> pd.DataFrame:
        """
        Generate a complete forex dataset for a specified pair and timeframe.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe for the data (e.g., "1d", "1h")
            years: Number of years of data to generate
            add_events: Whether to add market events
            add_spreads: Whether to add spread/slippage
            add_gaps: Whether to add gap detection

        Returns:
            Complete forex dataset
        """
        # Determine dates
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=365 * years)

        # Generate base price series
        df = self.generate_price_series(pair, start_date, end_date, timeframe)

        # Add market events if requested
        if add_events:
            events_per_year = 12  # Monthly events on average
            df = self.add_market_events(df, num_events=events_per_year * years)

        # Add spread/slippage if requested
        if add_spreads:
            df = self.add_spread_and_slippage(df, pair)

        # Add gap detection if requested
        if add_gaps:
            df = self.add_gap_detection(df)

        return df

    def save_dataset(self, df: pd.DataFrame, pair: str, timeframe: str) -> str:
        """
        Save a generated dataset to CSV.

        Args:
            df: DataFrame to save
            pair: Currency pair
            timeframe: Timeframe of the data

        Returns:
            Path to the saved file
        """
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Create filename
        today = datetime.now().strftime("%Y%m%d")
        filename = f"{pair}_{timeframe}_{today}.csv"
        filepath = self.output_dir / filename

        # Reset index to include datetime as a column
        df_to_save = df.reset_index()

        # Save to CSV
        df_to_save.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")

        return str(filepath)

    def generate_correlated_pair(
        self,
        base_df: pd.DataFrame,
        pair: str,
        correlation: float = 0.7
    ) -> pd.DataFrame:
        """
        Generate a new pair data that's correlated with an existing pair dataset.

        Args:
            base_df: Base DataFrame with OHLCV data
            pair: Currency pair for the new dataset
            correlation: Target correlation coefficient (-1 to 1)

        Returns:
            Correlated dataset
        """
        # Create a copy of the base dataframe
        result = base_df.copy()

        # Get the close prices from the base data
        base_returns = base_df['close'].pct_change().dropna().values

        # Generate correlated returns
        # Using the formula: new_returns = correlation * base_returns + sqrt(1-correlation^2) * random_component
        random_component = np.random.normal(0, np.std(base_returns), len(base_returns))
        correlated_returns = correlation * base_returns + np.sqrt(1 - correlation**2) * random_component

        # Calculate start price for new pair
        if pair.endswith("JPY"):
            start_price = 110.0
        elif pair.startswith("USD"):
            start_price = 1.2
        else:
            start_price = 0.8

        # Generate new prices from correlated returns
        new_closes = np.zeros(len(base_df))
        new_closes[0] = start_price

        for i in range(1, len(new_closes)):
            if i-1 < len(correlated_returns):
                new_closes[i] = new_closes[i-1] * (1 + correlated_returns[i-1])
            else:
                new_closes[i] = new_closes[i-1]

        # Update OHLC prices
        result['close'] = new_closes

        # Generate new opens, highs, lows based on the new closes
        for i in range(len(result)):
            relative_open = (base_df.iloc[i]['open'] / base_df.iloc[i]['close'])
            relative_high = (base_df.iloc[i]['high'] / base_df.iloc[i]['close'])
            relative_low = (base_df.iloc[i]['low'] / base_df.iloc[i]['close'])

            result.iloc[i, result.columns.get_indexer(['open'])] = result.iloc[i]['close'] * relative_open
            result.iloc[i, result.columns.get_indexer(['high'])] = result.iloc[i]['close'] * relative_high
            result.iloc[i, result.columns.get_indexer(['low'])] = result.iloc[i]['close'] * relative_low

        return result

    def generate_all_data(self, timeframes: List[str] = None, years: int = 2) -> None:
        """
        Generate datasets for all pairs and specified timeframes.

        Args:
            timeframes: List of timeframes to generate (if None, use all)
            years: Number of years of data to generate
        """
        if timeframes is None:
            timeframes = ["1d", "4h", "1h"]

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        for timeframe in timeframes:
            logger.info(f"Generating {timeframe} data for all pairs...")

            # Generate EURUSD first (will be used as base for correlations)
            eurusd_df = self.generate_dataset("EURUSD", timeframe, years=years)
            self.save_dataset(eurusd_df, "EURUSD", timeframe)

            # Generate other pairs with correlation to EURUSD
            for pair in self.currency_pairs:
                if pair == "EURUSD":
                    continue

                # Set correlation based on pair relationships
                if pair in ["GBPUSD", "EURGBP"]:
                    correlation = 0.85
                elif pair in ["AUDUSD", "NZDUSD"]:
                    correlation = 0.65
                elif pair in ["USDJPY", "USDCHF"]:
                    correlation = -0.50
                else:
                    correlation = 0.30

                # Add some randomness to correlation
                correlation += np.random.normal(0, 0.1)
                correlation = max(-0.95, min(0.95, correlation))  # Keep within reasonable bounds

                logger.info(f"Generating {pair} with {correlation:.2f} correlation to EURUSD")

                # Generate correlated dataset
                df = self.generate_correlated_pair(eurusd_df, pair, correlation)

                # Add market events, spreads, and gaps
                df = self.add_market_events(df, num_events=int(12 * years))
                df = self.add_spread_and_slippage(df, pair)
                df = self.add_gap_detection(df)

                # Save dataset
                self.save_dataset(df, pair, timeframe)

        logger.info(f"All datasets generated in {self.output_dir}")


if __name__ == "__main__":
    # Example usage
    generator = ForexDataGenerator()

    # Generate data for a specific pair and timeframe
    eurusd_daily = generator.generate_dataset("EURUSD", "1d", years=2)
    generator.save_dataset(eurusd_daily, "EURUSD", "1d")

    # Or generate all datasets
    # generator.generate_all_data()