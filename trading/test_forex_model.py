#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Test script for the Forex Trading Model using synthetic data.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Import our modules
from src.trading.forex_trading_model import ForexTradingModel
from src.data_pipeline.forex_data_generator import ForexDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_backtest_results(df, trades, balance_history, pair):
    """
    Plot the backtest results including price charts and balance history.

    Args:
        df: DataFrame with OHLCV data
        trades: List of trade dictionaries
        balance_history: List of account balance history
        pair: Currency pair name
    """
    plt.figure(figsize=(14, 10))

    # Plot 1: Price chart with buy/sell signals
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price', color='blue', alpha=0.7)

    # Plot buy signals
    buy_dates = [trade['entry_time'] for trade in trades if trade['direction'] == 'buy']
    buy_prices = [trade['entry_price'] for trade in trades if trade['direction'] == 'buy']
    plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')

    # Plot sell signals
    sell_dates = [trade['entry_time'] for trade in trades if trade['direction'] == 'sell']
    sell_prices = [trade['entry_price'] for trade in trades if trade['direction'] == 'sell']
    plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')

    # Plot exit points
    exit_dates = [trade['exit_time'] for trade in trades if trade['exit_time'] is not None]
    exit_prices = [trade['exit_price'] for trade in trades if trade['exit_price'] is not None]
    plt.scatter(exit_dates, exit_prices, color='black', marker='o', s=50, label='Exit')

    plt.title(f'{pair} Price Chart with Signals')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Account balance history
    plt.subplot(2, 1, 2)

    # Create dates for balance history (assuming each balance update corresponds to a trade)
    balance_dates = df.index[:len(balance_history)]

    plt.plot(balance_dates, balance_history, label='Account Balance', color='purple')
    plt.fill_between(balance_dates, balance_history, min(balance_history), alpha=0.2, color='purple')

    plt.title('Account Balance History')
    plt.xlabel('Date')
    plt.ylabel('Balance')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Create directory for plots if it doesn't exist
    plots_dir = project_root / "results" / "forex"
    os.makedirs(plots_dir, exist_ok=True)

    # Save the plot
    plot_path = plots_dir / f"{pair}_backtest_results.png"
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}")

    plt.show()


def test_strategy_optimization():
    """Test the parameter optimization functionality."""
    # Generate synthetic data
    data_generator = ForexDataGenerator()

    # Generate EURUSD hourly data for 6 months
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=180)

    logger.info("Generating synthetic EURUSD hourly data...")
    eurusd_df = data_generator.generate_price_series("EURUSD", start_date, end_date, "1h")

    # Initialize model
    model = ForexTradingModel(stop_loss_pips=50, take_profit_pips=100, risk_per_trade=0.02)

    # Set up data
    model.data = eurusd_df
    model.add_technical_indicators()

    # Define parameters to optimize
    param_grid = {
        'ma_short': [5, 10, 20],
        'ma_long': [50, 100, 200],
        'rsi_period': [7, 14, 21],
        'rsi_overbought': [70, 75, 80],
        'rsi_oversold': [20, 25, 30]
    }

    # Perform optimization
    logger.info("Starting parameter optimization...")
    best_params, best_result = model.optimize_parameters(param_grid)

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best result: {best_result}")

    return best_params


def main():
    """Main function to test the Forex Trading Model."""
    logger.info("Starting Forex Trading Model test")

    # Ensure directories exist
    data_dir = project_root / "Development" / "data" / "forex"
    os.makedirs(data_dir, exist_ok=True)

    # Check if we need to generate data
    if not list(data_dir.glob("*.csv")):
        logger.info("No forex data found, generating synthetic data...")
        data_generator = ForexDataGenerator()
        data_generator.generate_all_data(timeframes=["1h", "4h"], years=1)

    # Initialize the trading model
    model = ForexTradingModel(stop_loss_pips=50, take_profit_pips=100, risk_per_trade=0.02)

    # Find the data files
    data_files = list(data_dir.glob("EURUSD_1h_*.csv"))

    if not data_files:
        logger.error("No EURUSD hourly data files found even after generation")
        return

    # Use the most recent file
    data_file = sorted(data_files)[-1]
    logger.info(f"Using data file: {data_file}")

    # Load the data
    model.load_data(data_file)

    # Add technical indicators
    model.add_technical_indicators()

    # Try different strategies
    strategies = ["ma_crossover", "rsi_oversold", "macd_crossover"]

    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")

        # Generate signals
        model.generate_signals(strategy)

        # Run backtest
        results = model.backtest(initial_balance=10000)

        # Log results
        logger.info(f"Backtest results for {strategy}:")
        logger.info(f"  Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"  Return: {results['return_pct']:.2f}%")
        logger.info(f"  Total Trades: {results['total_trades']}")
        logger.info(f"  Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2f}%")

        # Plot results if it's the best strategy so far
        if strategy == strategies[0] or results['return_pct'] > best_results['return_pct']:
            best_results = results
            best_strategy = strategy

            # Plot the results
            plot_backtest_results(
                model.data,
                model.trades,
                model.balance_history,
                "EURUSD"
            )

    logger.info(f"Best strategy: {best_strategy} with {best_results['return_pct']:.2f}% return")

    # Test parameter optimization
    logger.info("Testing parameter optimization...")
    best_params = test_strategy_optimization()

    # Train ML model
    logger.info("Training machine learning model...")
    model.train_ml_model(target_periods_ahead=10, model_type="random_forest")

    # Save model
    model.save_model()

    # Generate trading report
    report = model.generate_trading_report()
    logger.info(f"Generated trading report: {report}")

    logger.info("Forex Trading Model test completed successfully")


if __name__ == "__main__":
    main()