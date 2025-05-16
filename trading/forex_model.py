# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Forex Trading Model

This module implements a comprehensive forex trading model that can analyze
historical data, generate trading signals, and evaluate trading strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexTradingModel:
    """
    A forex trading model that predicts price movements and generates trading signals
    based on historical price data, technical indicators, and optionally fundamental data.
    """

    def __init__(self, currency_pair: str = "EUR/USD", timeframe: str = "daily",
                model_type: str = "classification", risk_per_trade: float = 0.02):
        """
        Initialize the forex trading model.

        Args:
            currency_pair: The forex pair to analyze (e.g., 'EUR/USD')
            timeframe: The timeframe for analysis ('minute', 'hourly', 'daily', 'weekly')
            model_type: Type of prediction model ('classification' for direction, 'regression' for price)
            risk_per_trade: Risk per trade as a fraction of account balance (default: 2%)
        """
        self.currency_pair = currency_pair
        self.timeframe = timeframe
        self.model_type = model_type
        self.risk_per_trade = risk_per_trade

        # Model and data attributes
        self.model = None
        self.feature_scaler = None
        self.feature_names = []
        self.performance_metrics = {}

        # Directories
        self.base_dir = Path(__file__).parent.parent.parent
        self.data_dir = self.base_dir / 'data' / 'forex'
        self.models_dir = self.base_dir / 'models' / 'forex'

        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Trading parameters
        self.stop_loss_pips = 50
        self.take_profit_pips = 100
        self.max_drawdown_threshold = 0.20  # 20% maximum drawdown

        logger.info(f"Initialized forex trading model for {currency_pair} on {timeframe} timeframe")

    def load_data(self, file_path: Optional[Union[str, Path]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load forex data from a CSV file or download it if not available.

        Args:
            file_path: Path to the CSV file containing forex data
            start_date: Start date for data filtering (format: 'YYYY-MM-DD')
            end_date: End date for data filtering (format: 'YYYY-MM-DD')

        Returns:
            DataFrame with forex OHLCV data
        """
        if file_path is None:
            # Generate a default path based on currency pair and timeframe
            file_name = f"{self.currency_pair.replace('/', '_')}_{self.timeframe}.csv"
            file_path = self.data_dir / file_name
        else:
            file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"Data file {file_path} not found, attempting to download...")
            df = self._download_forex_data(start_date, end_date)
            if df is not None and not df.empty:
                # Save the downloaded data
                df.to_csv(file_path, index=False)
                logger.info(f"Downloaded data saved to {file_path}")
        else:
            # Load data from the CSV file
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # Apply date filters if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
                df = df[df['date'] >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df['date'] <= end_date]

        # Basic validation
        required_cols = ['date', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Data missing required columns: {missing_cols}")

        # Sort by date
        df = df.sort_values('date')

        # Calculate basic returns
        df['returns'] = df['close'].pct_change()

        logger.info(f"Loaded data with {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
        return df

    def _download_forex_data(self, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download forex data from an API or data provider.

        This is a placeholder. In a real implementation, this would connect to
        a data provider like Alpha Vantage, FXCM, OANDA, etc.

        Args:
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # In a real implementation, this would use a data API
            # For now, generate synthetic data for demonstration
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            if not start_date:
                # Default to 2 years of data
                start_date_dt = datetime.now() - timedelta(days=2*365)
                start_date = start_date_dt.strftime('%Y-%m-%d')

            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate synthetic OHLCV data
            np.random.seed(42)  # For reproducibility
            n_days = len(date_range)

            # Use random walk for synthetic price data
            base_price = 1.2000  # Starting price (e.g., for EUR/USD)
            daily_volatility = 0.005  # 0.5% daily volatility

            # Generate random returns
            returns = np.random.normal(0, daily_volatility, n_days)

            # Calculate price series using cumulative returns
            cum_returns = np.cumprod(1 + returns)
            close_prices = base_price * cum_returns

            # Generate OHLC based on close prices
            df = pd.DataFrame({
                'date': date_range,
                'close': close_prices,
                'open': close_prices * (1 + np.random.normal(0, 0.001, n_days)),
                'high': close_prices * (1 + np.abs(np.random.normal(0, 0.002, n_days))),
                'low': close_prices * (1 - np.abs(np.random.normal(0, 0.002, n_days))),
                'volume': np.random.randint(1000, 10000, n_days)
            })

            # Ensure high is always >= than open/close, and low is always <= open/close
            for i in range(len(df)):
                df.at[i, 'high'] = max(df.at[i, 'high'], df.at[i, 'open'], df.at[i, 'close'])
                df.at[i, 'low'] = min(df.at[i, 'low'], df.at[i, 'open'], df.at[i, 'close'])

            logger.info(f"Generated synthetic data for {self.currency_pair} with {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error downloading forex data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        # Create a copy to avoid modifying the original DataFrame
        data = df.copy()

        logger.info("Adding technical indicators")

        try:
            # Moving Averages
            for window in [5, 10, 20, 50, 200]:
                data[f'ma_{window}'] = data['close'].rolling(window=window).mean()

                # Add MA crossover signals
                if window == 50:  # Create crossover signals for 50-period MA
                    data[f'ma_20_50_crossover'] = np.where(
                        data['ma_20'] > data[f'ma_{window}'], 1,
                        np.where(data['ma_20'] < data[f'ma_{window}'], -1, 0)
                    )

            # Exponential Moving Averages
            for window in [12, 26]:
                data[f'ema_{window}'] = data['close'].ewm(span=window, adjust=False).mean()

            # MACD (Moving Average Convergence Divergence)
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            data['macd_crossover'] = np.where(
                data['macd'] > data['macd_signal'], 1,
                np.where(data['macd'] < data['macd_signal'], -1, 0)
            )

            # RSI (Relative Strength Index)
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi_14'] = 100 - (100 / (1 + rs))

            # RSI overbought/oversold signals
            data['rsi_signal'] = np.where(
                data['rsi_14'] < 30, 1,  # Oversold, potential buy
                np.where(data['rsi_14'] > 70, -1, 0)  # Overbought, potential sell
            )

            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
            data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

            # Bollinger Band signals
            data['bb_signal'] = np.where(
                data['close'] < data['bb_lower'], 1,  # Price below lower band, potential buy
                np.where(data['close'] > data['bb_upper'], -1, 0)  # Price above upper band, potential sell
            )

            # Average True Range (ATR) - Volatility indicator
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data['atr_14'] = true_range.rolling(14).mean()

            # Momentum - Rate of change
            for period in [5, 10, 20]:
                data[f'momentum_{period}'] = data['close'].pct_change(periods=period) * 100

            # Stochastic Oscillator
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            data['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()

            # Stochastic signal
            data['stoch_signal'] = np.where(
                (data['stoch_k'] < 20) & (data['stoch_k'] > data['stoch_d']), 1,
                np.where((data['stoch_k'] > 80) & (data['stoch_k'] < data['stoch_d']), -1, 0)
            )

            # Add price distance from moving averages as a percentage
            for ma in [20, 50, 200]:
                data[f'price_distance_ma_{ma}'] = (data['close'] - data[f'ma_{ma}']) / data[f'ma_{ma}'] * 100

            logger.info(f"Added {len(data.columns) - len(df.columns)} technical indicators")
            return data

        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'returns',
                       prediction_horizon: int = 1, train_test_split: float = 0.8) -> Tuple:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with price data and indicators
            target_col: Column to predict ('returns' or 'close')
            prediction_horizon: Number of periods ahead to predict
            train_test_split: Proportion of data for training (0.8 means 80% training, 20% testing)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info(f"Preparing features with prediction horizon of {prediction_horizon} periods")

        # Create copy to avoid modifying original data
        data = df.copy()

        # Create target variable based on future returns or prices
        if self.model_type == 'classification':
            # For classification, we predict direction (up=1, down=0)
            if target_col == 'returns':
                data[f'target'] = np.where(data[target_col].shift(-prediction_horizon) > 0, 1, 0)
            else:  # Assume target_col is 'close' or another price column
                data[f'target'] = np.where(
                    data[target_col].shift(-prediction_horizon) > data[target_col], 1, 0
                )
        else:  # Regression
            # For regression, we predict the actual future value
            data[f'target'] = data[target_col].shift(-prediction_horizon)

        # Drop rows with NaN values (especially important for the target)
        data = data.dropna()

        # Get feature columns (exclude date, target, and any columns that would cause data leakage)
        exclude_cols = ['date', 'target', 'open', 'high', 'low', 'close', 'volume', 'returns']
        feature_cols = [col for col in data.columns if col not in exclude_cols]

        # Store feature names for later use
        self.feature_names = feature_cols

        # Split data into features and target
        X = data[feature_cols].values
        y = data['target'].values

        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        # Split into training and testing sets
        split_idx = int(len(data) * train_test_split)

        # Make sure we respect time series order for financial data
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_test = X_scaled[split_idx:]
        y_test = y[split_idx:]

        logger.info(f"Prepared {len(feature_cols)} features with {len(X_train)} training samples and {len(X_test)} testing samples")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_params: Optional[Dict] = None) -> None:
        """
        Train the prediction model.

        Args:
            X_train: Training features
            y_train: Training targets
            model_params: Additional parameters for the model
        """
        logger.info(f"Training {self.model_type} model for {self.currency_pair}")

        if model_params is None:
            model_params = {}

        try:
            # Initialize the model based on type
            if self.model_type == 'classification':
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'random_state': 42
                }
                # Update with any user-provided params
                params = {**default_params, **model_params}

                self.model = RandomForestClassifier(**params)
                logger.info(f"Initialized RandomForestClassifier with {params}")
            else:  # Regression
                default_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 10,
                    'random_state': 42
                }
                # Update with any user-provided params
                params = {**default_params, **model_params}

                self.model = RandomForestRegressor(**params)
                logger.info(f"Initialized RandomForestRegressor with {params}")

            # Train the model
            self.model.fit(X_train, y_train)

            # Store feature importances
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
                feature_importance_dict = {name: importance for name, importance
                                         in zip(self.feature_names, feature_importance)}

                # Sort by importance
                sorted_features = sorted(feature_importance_dict.items(),
                                       key=lambda x: x[1], reverse=True)

                logger.info(f"Top 5 features by importance: {sorted_features[:5]}")

            logger.info(f"Model training completed")

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("You must train the model before evaluation")

        try:
            # Make predictions
            if self.model_type == 'classification':
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]

                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
            else:  # Regression
                y_pred = self.model.predict(X_test)

                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                metrics = {
                    'mean_absolute_error': mae,
                    'mean_squared_error': mse,
                    'root_mean_squared_error': rmse,
                    'r2_score': r2
                }

            logger.info(f"Model evaluation metrics: {metrics}")

            # Store metrics for later use
            self.performance_metrics = metrics

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions.

        Args:
            df: DataFrame with price data and indicators

        Returns:
            DataFrame with added trading signals
        """
        if self.model is None:
            logger.error("Model not trained yet")
            raise ValueError("You must train the model before generating signals")

        try:
            # Create a copy to avoid modifying the original
            data = df.copy()

            # Extract features
            exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'returns']
            feature_cols = [col for col in data.columns if col not in exclude_cols and col in self.feature_names]

            # Ensure all required features are present
            missing_features = [col for col in self.feature_names if col not in data.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                raise ValueError(f"Missing features for prediction: {missing_features}")

            # Scale features
            X = data[feature_cols].values
            X_scaled = self.feature_scaler.transform(X)

            # Make predictions
            if self.model_type == 'classification':
                # For classification, predict class and probability
                data['prediction'] = self.model.predict(X_scaled)
                data['prediction_probability'] = self.model.predict_proba(X_scaled)[:, 1]

                # Generate signals (1 for buy, -1 for sell, 0 for hold)
                data['signal'] = np.where(
                    data['prediction'] == 1, 1,  # Buy signal
                    -1  # Sell signal
                )
            else:  # Regression
                # For regression, predict the target value
                data['prediction'] = self.model.predict(X_scaled)

                # Generate signals based on predicted direction
                data['signal'] = np.where(
                    data['prediction'] > 0, 1,  # Buy signal for positive prediction
                    -1  # Sell signal for negative/zero prediction
                )

            # Add entry/exit logic
            # Only enter a trade if the predicted probability is strong enough
            if 'prediction_probability' in data.columns:
                data['signal'] = np.where(
                    (data['prediction'] == 1) & (data['prediction_probability'] > 0.6), 1,
                    np.where((data['prediction'] == 0) & (data['prediction_probability'] < 0.4), -1, 0)
                )

            # Calculate signal changes for entry/exit points
            data['signal_change'] = data['signal'].diff()

            # Entry points: When signal changes from 0 to 1 (buy) or 0 to -1 (sell)
            data['entry_long'] = (data['signal_change'] == 1).astype(int)
            data['entry_short'] = (data['signal_change'] == -1).astype(int)

            # Exit points: When signal changes from 1 to 0 or -1 to 0, or reverses
            data['exit_long'] = ((data['signal_change'] <= -1) & (data['signal'].shift(1) == 1)).astype(int)
            data['exit_short'] = ((data['signal_change'] >= 1) & (data['signal'].shift(1) == -1)).astype(int)

            logger.info(f"Generated trading signals with {data['entry_long'].sum()} long entries "
                     f"and {data['entry_short'].sum()} short entries")

            return data

        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            raise

    def backtest_strategy(self, df: pd.DataFrame, initial_balance: float = 10000.0,
                        position_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Backtest the trading strategy on historical data.

        Args:
            df: DataFrame with price data, indicators, and signals
            initial_balance: Initial account balance for the backtest
            position_size: Fixed position size in units (if None, uses risk_per_trade)

        Returns:
            Dictionary with backtest results
        """
        try:
            # Ensure signals are present
            required_cols = ['date', 'close', 'signal', 'entry_long', 'entry_short', 'exit_long', 'exit_short']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing columns for backtesting: {missing_cols}")
                # Generate signals if they're missing
                if 'signal' in missing_cols:
                    df = self.generate_trading_signals(df)

            # Create a copy to avoid modifying the original
            data = df.copy()

            # Initialize backtest variables
            balance = initial_balance
            position = 0  # 0 for no position, 1 for long, -1 for short
            entry_price = 0.0
            trades = []
            balance_history = [initial_balance]
            current_trade = None

            # Determine position size strategy
            if position_size is None:
                # Use risk per trade (percentage of balance)
                use_fixed_size = False
            else:
                # Use fixed position size in units
                use_fixed_size = True

            # Track maximum drawdown
            peak_balance = initial_balance
            max_drawdown = 0.0

            # Loop through each row (trading day/hour)
            for i, row in data.iterrows():
                current_date = row['date']
                current_price = row['close']

                # Entry and exit signals
                entry_long = row['entry_long'] == 1
                entry_short = row['entry_short'] == 1
                exit_long = row['exit_long'] == 1
                exit_short = row['exit_short'] == 1

                # Calculate position size based on strategy
                if use_fixed_size:
                    units = position_size
                else:
                    # Risk-based position sizing
                    risk_amount = balance * self.risk_per_trade

                    # Calculate stop loss price (adjust for pip values in different pairs)
                    pip_value = 0.0001  # Default for most pairs (0.01 for JPY pairs)
                    if 'JPY' in self.currency_pair:
                        pip_value = 0.01

                    stop_distance = self.stop_loss_pips * pip_value

                    # Avoid division by zero
                    if stop_distance > 0:
                        units = risk_amount / stop_distance
                    else:
                        units = balance * 0.01  # Default to 1% if stop distance is zero

                # Process exits first (to avoid entering and exiting on the same bar)
                if position == 1 and exit_long:  # Exit long position
                    # Calculate profit/loss
                    pnl = (current_price - entry_price) * units
                    balance += pnl

                    trades.append({
                        'type': 'long',
                        'entry_date': current_trade['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'units': units,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * units) * 100,
                        'balance': balance
                    })

                    position = 0
                    current_trade = None

                elif position == -1 and exit_short:  # Exit short position
                    # Calculate profit/loss
                    pnl = (entry_price - current_price) * units
                    balance += pnl

                    trades.append({
                        'type': 'short',
                        'entry_date': current_trade['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'units': units,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * units) * 100,
                        'balance': balance
                    })

                    position = 0
                    current_trade = None

                # Process entries
                if position == 0:  # Only enter if not already in a position
                    if entry_long:  # Enter long position
                        position = 1
                        entry_price = current_price
                        current_trade = {
                            'entry_date': current_date,
                            'entry_price': entry_price
                        }

                    elif entry_short:  # Enter short position
                        position = -1
                        entry_price = current_price
                        current_trade = {
                            'entry_date': current_date,
                            'entry_price': entry_price
                        }

                # Track balance history
                balance_history.append(balance)

                # Update max drawdown
                if balance > peak_balance:
                    peak_balance = balance

                drawdown = (peak_balance - balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)

            # Close any open positions at the end of the backtest
            if position != 0:
                # Get the last price
                last_price = data['close'].iloc[-1]

                if position == 1:  # Long position
                    pnl = (last_price - entry_price) * units
                    balance += pnl

                    trades.append({
                        'type': 'long',
                        'entry_date': current_trade['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': data['date'].iloc[-1],
                        'exit_price': last_price,
                        'units': units,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * units) * 100,
                        'balance': balance
                    })

                elif position == -1:  # Short position
                    pnl = (entry_price - last_price) * units
                    balance += pnl

                    trades.append({
                        'type': 'short',
                        'entry_date': current_trade['entry_date'],
                        'entry_price': entry_price,
                        'exit_date': data['date'].iloc[-1],
                        'exit_price': last_price,
                        'units': units,
                        'pnl': pnl,
                        'pnl_pct': pnl / (entry_price * units) * 100,
                        'balance': balance
                    })

            # Calculate performance metrics
            total_trades = len(trades)
            profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
            losing_trades = total_trades - profitable_trades

            if total_trades > 0:
                win_rate = profitable_trades / total_trades
                profit_factor = sum(t['pnl'] for t in trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else float('inf')

                # Calculate average profit and loss
                avg_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0) / profitable_trades if profitable_trades > 0 else 0
                avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / losing_trades if losing_trades > 0 else 0

                # Calculate risk-reward ratio
                risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0
                avg_profit = 0
                avg_loss = 0
                risk_reward_ratio = 0

            # Calculate Sharpe ratio (assuming risk-free rate of 0%)
            if len(trades) > 1:
                returns = [t['pnl_pct'] for t in trades]
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 / (data['date'].iloc[-1] - data['date'].iloc[0]).days) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0

            # Calculate total return
            total_return = (balance - initial_balance) / initial_balance * 100

            backtest_results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': total_return,
                'total_return_pct': total_return,
                'max_drawdown': max_drawdown * 100,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'risk_reward_ratio': risk_reward_ratio,
                'sharpe_ratio': sharpe_ratio,
                'balance_history': balance_history,
                'trades': trades
            }

            logger.info(f"Backtest results: Return: {total_return:.2f}%, Trades: {total_trades}, "
                     f"Win rate: {win_rate:.2f}, Sharpe: {sharpe_ratio:.2f}")

            return backtest_results

        except Exception as e:
            logger.error(f"Error during backtesting: {e}")
            raise

    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Save the trained model to disk.

        Args:
            filename: Custom filename (if None, generates one)

        Returns:
            Path to the saved model
        """
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("You must train the model before saving")

        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                filename = f"{self.currency_pair.replace('/', '_')}_{self.timeframe}_{self.model_type}_{timestamp}.pkl"

            # Ensure .pkl extension
            if not filename.endswith('.pkl'):
                filename += '.pkl'

            model_path = self.models_dir / filename

            # Create a dictionary with the model and metadata
            model_data = {
                'model': self.model,
                'feature_scaler': self.feature_scaler,
                'feature_names': self.feature_names,
                'currency_pair': self.currency_pair,
                'timeframe': self.timeframe,
                'model_type': self.model_type,
                'performance_metrics': self.performance_metrics,
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    'version': '1.0',
                    'description': f"Forex trading model for {self.currency_pair} {self.timeframe} timeframe"
                }
            }

            # Save to disk
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the model file
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file {model_path} not found")

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Load model and metadata
            self.model = model_data['model']
            self.feature_scaler = model_data['feature_scaler']
            self.feature_names = model_data['feature_names']
            self.currency_pair = model_data['currency_pair']
            self.timeframe = model_data['timeframe']
            self.model_type = model_data['model_type']

            if 'performance_metrics' in model_data:
                self.performance_metrics = model_data['performance_metrics']

            logger.info(f"Loaded {self.model_type} model for {self.currency_pair} {self.timeframe} from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def plot_backtest_results(self, backtest_results: Dict[str, Any],
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Visualize backtest results.

        Args:
            backtest_results: Dictionary with backtest results
            save_path: Path to save the visualization (if None, displays the plot)
        """
        try:
            # Create a plot with multiple subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))

            # Plot account balance
            axs[0, 0].plot(backtest_results['balance_history'], linewidth=2)
            axs[0, 0].set_title('Account Balance')
            axs[0, 0].set_xlabel('Trade Number')
            axs[0, 0].set_ylabel('Balance')
            axs[0, 0].grid(True)

            # Calculate drawdown series
            balance_series = pd.Series(backtest_results['balance_history'])
            drawdown = (balance_series.cummax() - balance_series) / balance_series.cummax() * 100

            # Plot drawdown
            axs[0, 1].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
            axs[0, 1].set_title('Drawdown')
            axs[0, 1].set_xlabel('Trade Number')
            axs[0, 1].set_ylabel('Drawdown (%)')
            axs[0, 1].grid(True)

            # Plot trade outcomes
            trades = backtest_results['trades']

            if trades:
                # Extract profit/loss from trades
                pnl_values = [trade['pnl'] for trade in trades]
                trade_types = [trade['type'] for trade in trades]

                # Colors for long/short positions
                colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]

                axs[1, 0].bar(range(len(trades)), pnl_values, color=colors)
                axs[1, 0].set_title('Trade Profit/Loss')
                axs[1, 0].set_xlabel('Trade Number')
                axs[1, 0].set_ylabel('Profit/Loss')
                axs[1, 0].grid(True)

                # Show the distribution of returns
                axs[1, 1].hist([trade['pnl_pct'] for trade in trades], bins=20, alpha=0.7, color='blue')
                axs[1, 1].set_title('Distribution of Returns')
                axs[1, 1].set_xlabel('Return (%)')
                axs[1, 1].set_ylabel('Frequency')
                axs[1, 1].grid(True)

            # Add summary stats
            fig.suptitle(f"Backtest Results - {self.currency_pair} {self.timeframe}", fontsize=16)

            plt.figtext(
                0.5, 0.01,
                f"Total Return: {backtest_results['total_return_pct']:.2f}% | "
                f"Max Drawdown: {backtest_results['max_drawdown']:.2f}% | "
                f"Trades: {backtest_results['total_trades']} | "
                f"Win Rate: {backtest_results['win_rate']*100:.2f}% | "
                f"Sharpe: {backtest_results['sharpe_ratio']:.2f}",
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save or display
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")
            raise

    def run_analysis(self, data_path: Optional[str] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a complete analysis workflow: load data, add indicators, train model, and backtest.

        Args:
            data_path: Path to forex data (if None, uses default)
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary with complete analysis results
        """
        try:
            # Step 1: Load data
            df = self.load_data(file_path=data_path, start_date=start_date, end_date=end_date)

            # Step 2: Add technical indicators
            df_indicators = self.add_technical_indicators(df)

            # Step 3: Prepare features
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_features(df_indicators)

            # Step 4: Train model
            self.train_model(X_train, y_train)

            # Step 5: Evaluate model
            metrics = self.evaluate_model(X_test, y_test)

            # Step 6: Generate signals
            df_signals = self.generate_trading_signals(df_indicators)

            # Step 7: Backtest strategy
            backtest_results = self.backtest_strategy(df_signals)

            # Step 8: Save model
            model_path = self.save_model()

            # Step 9: Plot results
            plot_path = self.models_dir / f"{self.currency_pair.replace('/', '_')}_{self.timeframe}_backtest.png"
            self.plot_backtest_results(backtest_results, save_path=plot_path)

            # Compile complete results
            analysis_results = {
                'model_path': model_path,
                'plot_path': str(plot_path),
                'model_metrics': metrics,
                'backtest_results': backtest_results,
                'period': {
                    'start_date': df['date'].min().strftime('%Y-%m-%d'),
                    'end_date': df['date'].max().strftime('%Y-%m-%d'),
                    'duration_days': (df['date'].max() - df['date'].min()).days
                }
            }

            logger.info(f"Analysis completed for {self.currency_pair} on {self.timeframe} timeframe")
            return analysis_results

        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            raise

# Demo function to run when the module is executed directly
def main():
    """Run a demo of the forex trading model."""
    # Initialize the model
    forex_model = ForexTradingModel(
        currency_pair="EUR/USD",
        timeframe="daily",
        model_type="classification"
    )

    try:
        # Run the complete analysis
        results = forex_model.run_analysis()

        # Print key results
        print(f"\nForex Trading Model for {forex_model.currency_pair}")
        print(f"Model Type: {forex_model.model_type}")
        print(f"Timeframe: {forex_model.timeframe}")
        print("\nModel Performance:")
        for metric, value in results['model_metrics'].items():
            print(f"  {metric}: {value:.4f}")

        print("\nBacktest Results:")
        print(f"  Total Return: {results['backtest_results']['total_return_pct']:.2f}%")
        print(f"  Total Trades: {results['backtest_results']['total_trades']}")
        print(f"  Win Rate: {results['backtest_results']['win_rate']*100:.2f}%")
        print(f"  Sharpe Ratio: {results['backtest_results']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['backtest_results']['max_drawdown']:.2f}%")

        print(f"\nModel saved to: {results['model_path']}")
        print(f"Plot saved to: {results['plot_path']}")

    except Exception as e:
        print(f"Error running demo: {e}")

if __name__ == "__main__":
    main()
