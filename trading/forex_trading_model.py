#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Forex Trading Model - For backtesting and generating trading signals on forex data.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexTradingModel:
    """
    A model for forex trading that can analyze OHLCV data,
    generate trading signals, and simulate trades.
    """

    def __init__(self, model_name: str = "default_model"):
        """
        Initialize the forex trading model.

        Args:
            model_name: Name identifier for this model
        """
        self.model_name = model_name
        self.data = None
        self.features = None
        self.ml_model = None
        self.trade_history = []
        self.current_position = None

        # Trading parameters (can be adjusted)
        self.stop_loss_pips = 20
        self.take_profit_pips = 40
        self.risk_per_trade = 0.02  # 2% risk per trade

        # Directory for saving models
        self.models_dir = Path("Development/src/models/forex")
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.

        Args:
            filepath: Path to the CSV file containing OHLCV data

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            # Make sure datetime is set as index and is the correct type
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            self.data = df
            logger.info(f"Loaded {len(df)} rows of data")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Add technical indicators to the OHLCV data.

        Returns:
            DataFrame with technical indicators added
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None

        df = self.data.copy()

        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)

        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Drop rows with NaN values that result from rolling windows
        df.dropna(inplace=True)

        self.data = df
        return df

    def generate_signals(self, strategy: str = "moving_average_crossover") -> pd.DataFrame:
        """
        Generate trading signals based on the specified strategy.

        Args:
            strategy: Strategy to use for generating signals
                      Options: "moving_average_crossover", "rsi_oversold", "macd_crossover"

        Returns:
            DataFrame with signals added
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None

        df = self.data.copy()

        if strategy == "moving_average_crossover":
            # Buy when short-term MA crosses above long-term MA
            df['signal'] = 0
            df.loc[df['ma_5'] > df['ma_20'], 'signal'] = 1
            df.loc[df['ma_5'] < df['ma_20'], 'signal'] = -1

        elif strategy == "rsi_oversold":
            # Buy when RSI is oversold (< 30), sell when overbought (> 70)
            df['signal'] = 0
            df.loc[df['rsi'] < 30, 'signal'] = 1
            df.loc[df['rsi'] > 70, 'signal'] = -1

        elif strategy == "macd_crossover":
            # Buy when MACD crosses above signal line, sell when below
            df['signal'] = 0
            df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
            df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1

        # Create a column for actual trades (1 for buy, -1 for sell, 0 for no action)
        # Only generate a trade when the signal changes
        df['trade'] = df['signal'].diff().ne(0).astype(int) * df['signal']

        self.data = df
        return df

    def backtest(self, initial_balance: float = 10000.0) -> Dict:
        """
        Backtest the trading strategy on historical data.

        Args:
            initial_balance: Initial account balance

        Returns:
            Dictionary with backtest results
        """
        if self.data is None or 'trade' not in self.data.columns:
            logger.error("No signals generated. Please generate signals first.")
            return None

        df = self.data.copy()

        # Initialize tracking variables
        balance = initial_balance
        position = 0  # 0 for no position, 1 for long, -1 for short
        position_size = 0
        entry_price = 0
        trades = []

        for idx, row in df.iterrows():
            if position == 0 and row['trade'] != 0:
                # Enter a new position
                position = 1 if row['trade'] > 0 else -1
                entry_price = row['close']
                position_size = (balance * self.risk_per_trade) / self.stop_loss_pips
                trades.append({
                    'datetime': idx,
                    'action': 'buy' if position > 0 else 'sell',
                    'price': entry_price,
                    'position_size': position_size
                })

            elif position != 0:
                # Check for exit conditions (opposite signal, stop loss, take profit)
                exit_signal = False
                exit_price = row['close']
                profit_pips = (exit_price - entry_price) * position

                # Exit on opposite signal
                if row['trade'] != 0 and row['trade'] != position:
                    exit_signal = True

                # Exit on stop loss
                elif profit_pips <= -self.stop_loss_pips:
                    exit_signal = True
                    exit_price = entry_price - (self.stop_loss_pips * position)

                # Exit on take profit
                elif profit_pips >= self.take_profit_pips:
                    exit_signal = True
                    exit_price = entry_price + (self.take_profit_pips * position)

                if exit_signal:
                    # Close the position and update balance
                    profit_pips = (exit_price - entry_price) * position
                    profit = profit_pips * position_size
                    balance += profit

                    trades.append({
                        'datetime': idx,
                        'action': 'exit',
                        'price': exit_price,
                        'position_size': position_size,
                        'profit_pips': profit_pips,
                        'profit': profit
                    })

                    position = 0
                    position_size = 0

        # Calculate backtest statistics
        trade_df = pd.DataFrame(trades)
        if 'profit' in trade_df.columns:
            winning_trades = trade_df[trade_df['profit'] > 0]
            losing_trades = trade_df[trade_df['profit'] < 0]

            results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'profit': balance - initial_balance,
                'return_pct': ((balance - initial_balance) / initial_balance) * 100,
                'total_trades': len(trade_df[trade_df['action'] != 'exit']),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trade_df[trade_df['action'] == 'exit']) if len(trade_df[trade_df['action'] == 'exit']) > 0 else 0,
                'avg_profit': winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['profit'].mean() if len(losing_trades) > 0 else 0,
                'trades': trades
            }

            # Calculate profit factor and max drawdown
            if len(losing_trades) > 0 and abs(losing_trades['profit'].sum()) > 0:
                results['profit_factor'] = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(winning_trades) > 0 else 0
            else:
                results['profit_factor'] = float('inf') if len(winning_trades) > 0 else 0

            # Calculate running balance for maximum drawdown
            running_balance = [initial_balance]
            for trade in trades:
                if 'profit' in trade:
                    running_balance.append(running_balance[-1] + trade.get('profit', 0))

            # Maximum drawdown calculation
            peak = running_balance[0]
            max_dd = 0
            for bal in running_balance:
                if bal > peak:
                    peak = bal
                dd = (peak - bal) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            results['max_drawdown_pct'] = max_dd

            self.trade_history = trades
            return results
        else:
            logger.warning("No trades executed during backtest period")
            return {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'profit': 0,
                'return_pct': 0,
                'total_trades': 0,
                'trades': []
            }

    def optimize_parameters(self, param_grid: Dict) -> Dict:
        """
        Optimize strategy parameters using grid search.

        Args:
            param_grid: Dictionary of parameters to optimize
                       Example: {
                           'stop_loss_pips': [10, 20, 30],
                           'take_profit_pips': [20, 40, 60],
                           'strategy': ['moving_average_crossover', 'rsi_oversold']
                       }

        Returns:
            Dictionary with best parameters
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None

        best_return = -float('inf')
        best_params = {}
        results = []

        # Get all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # Apply parameters
            for param, value in params.items():
                if hasattr(self, param):
                    setattr(self, param, value)

            # Run backtest with these parameters
            if 'strategy' in params:
                self.add_technical_indicators()
                self.generate_signals(strategy=params['strategy'])

            backtest_results = self.backtest()

            # Store results
            params_with_results = params.copy()
            params_with_results.update({'return_pct': backtest_results['return_pct']})
            results.append(params_with_results)

            # Update best parameters if needed
            if backtest_results['return_pct'] > best_return:
                best_return = backtest_results['return_pct']
                best_params = params.copy()

        # Apply best parameters
        for param, value in best_params.items():
            if hasattr(self, param):
                setattr(self, param, value)

        return {
            'best_params': best_params,
            'best_return': best_return,
            'all_results': results
        }

    def train_ml_model(self, target_periods_ahead: int = 5, model_type: str = 'random_forest'):
        """
        Train a machine learning model to predict price movements.

        Args:
            target_periods_ahead: Number of periods ahead to predict
            model_type: Type of ML model to use ('random_forest', 'xgboost', etc.)
        """
        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None

        # Add technical indicators if not already present
        if 'ma_5' not in self.data.columns:
            self.add_technical_indicators()

        df = self.data.copy()

        # Create target variable (price movement n periods ahead)
        df['target'] = df['close'].pct_change(periods=target_periods_ahead).shift(-target_periods_ahead)
        df['target_binary'] = (df['target'] > 0).astype(int)

        # Remove NaN values
        df.dropna(inplace=True)

        # Select features
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma_5', 'ma_20', 'ma_50', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'atr'
        ]

        X = df[feature_columns]
        y = df['target_binary']

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

        # Train model
        try:
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(n_estimators=100, random_state=42)
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None

            model.fit(X_train, y_train)

            # Evaluate model
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logger.info(f"Model accuracy: {accuracy:.4f}")
            logger.info(f"Classification report:\n{report}")

            self.ml_model = model
            self.features = feature_columns

            return {
                'accuracy': accuracy,
                'report': report,
                'model': model
            }

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def predict_next_movement(self) -> Dict:
        """
        Use the trained ML model to predict the next price movement.

        Returns:
            Dictionary with prediction results
        """
        if self.ml_model is None:
            logger.error("No model trained. Please train a model first.")
            return None

        if self.data is None:
            logger.error("No data loaded. Please load data first.")
            return None

        # Get the latest data point
        latest_data = self.data.iloc[-1:]

        if not all(feature in latest_data.columns for feature in self.features):
            logger.error("Features mismatch. Make sure data has been processed with add_technical_indicators().")
            return None

        # Make prediction
        features = latest_data[self.features]
        prediction = self.ml_model.predict(features)[0]
        prediction_proba = self.ml_model.predict_proba(features)[0]

        return {
            'prediction': int(prediction),  # 1 for up, 0 for down
            'probability': prediction_proba[1] if prediction == 1 else prediction_proba[0],
            'timestamp': latest_data.index[0]
        }

    def save_model(self) -> bool:
        """
        Save the trained model to disk.

        Returns:
            Boolean indicating success or failure
        """
        if self.ml_model is None:
            logger.error("No model to save. Please train a model first.")
            return False

        try:
            model_path = self.models_dir / f"{self.model_name}.pkl"

            # Save model, features, and metadata
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.ml_model,
                    'features': self.features,
                    'stop_loss_pips': self.stop_loss_pips,
                    'take_profit_pips': self.take_profit_pips,
                    'risk_per_trade': self.risk_per_trade
                }, f)

            logger.info(f"Model saved to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load a saved model from disk.

        Args:
            model_name: Name of the model to load (if None, use self.model_name)

        Returns:
            Boolean indicating success or failure
        """
        if model_name:
            self.model_name = model_name

        try:
            model_path = self.models_dir / f"{self.model_name}.pkl"

            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

            self.ml_model = saved_data['model']
            self.features = saved_data['features']
            self.stop_loss_pips = saved_data.get('stop_loss_pips', 20)
            self.take_profit_pips = saved_data.get('take_profit_pips', 40)
            self.risk_per_trade = saved_data.get('risk_per_trade', 0.02)

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def generate_trading_report(self) -> str:
        """
        Generate a detailed trading report.

        Returns:
            String with formatted report
        """
        if not self.trade_history:
            return "No trading history available. Run backtest() first."

        # Calculate trade statistics
        total_trades = len([t for t in self.trade_history if t.get('action') in ['buy', 'sell']])
        winning_trades = len([t for t in self.trade_history if t.get('profit', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('profit', 0) < 0])
        win_rate = winning_trades / (winning_trades + losing_trades) * 100 if (winning_trades + losing_trades) > 0 else 0

        # Calculate profit metrics
        total_profit = sum([t.get('profit', 0) for t in self.trade_history])
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0

        # Generate report
        report = f"""
        ===== TRADING REPORT =====
        Model: {self.model_name}

        PERFORMANCE SUMMARY:
        Total Trades: {total_trades}
        Winning Trades: {winning_trades}
        Losing Trades: {losing_trades}
        Win Rate: {win_rate:.2f}%

        PROFIT METRICS:
        Total Profit: {total_profit:.2f}
        Average Profit per Trade: {avg_profit_per_trade:.2f}

        STRATEGY PARAMETERS:
        Stop Loss: {self.stop_loss_pips} pips
        Take Profit: {self.take_profit_pips} pips
        Risk per Trade: {self.risk_per_trade * 100:.2f}%

        ===== END REPORT =====
        """

        return report


if __name__ == "__main__":
    # Example usage
    model = ForexTradingModel(model_name="eurusd_daily")

    # Get path to data directory
    data_dir = Path("Development/data/forex")

    # Check if data exists, if not, generate it
    if not os.path.exists(data_dir):
        logger.info("Data directory not found. Generating synthetic data...")
        from forex_data_generator import ForexDataGenerator
        generator = ForexDataGenerator()
        generator.generate_all_data()

    # Load EURUSD data
    try:
        data_file = next(data_dir.glob("EURUSD_*.csv"))
        model.load_data(data_file)
    except StopIteration:
        logger.error("No EURUSD data files found.")
        exit(1)

    # Add technical indicators
    model.add_technical_indicators()

    # Generate signals using moving average crossover strategy
    model.generate_signals(strategy="moving_average_crossover")

    # Run backtest
    results = model.backtest(initial_balance=10000.0)

    if results:
        print(f"Backtest Results:")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate'] * 100:.2f}%")

        # Generate and print trading report
        report = model.generate_trading_report()
        print(report)

        # Optionally, train ML model
        print("Training ML model...")
        model.train_ml_model(target_periods_ahead=5, model_type='random_forest')

        # Save the model
        model.save_model()
        print(f"Model saved as {model.model_name}.pkl")