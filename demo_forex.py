#!/usr/bin/env python3
"""
Forex Model Demo

This script demonstrates how to train and use a forex trading model for price prediction
and generating trading signals. It provides a simple example of the workflow from data loading
to model training, prediction, and backtesting.
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forex_demo")

# Import required modules
from trading.forex_model import ForexTradingModel
from data_pipeline.forex_data_loader import ForexDataLoader
from data_pipeline.technical_indicators import TechnicalIndicators


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run forex trading model demo')

    parser.add_argument('--currency-pair', type=str, default='EUR/USD',
                        help='Currency pair to use')

    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Timeframe for analysis')

    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'logistic'],
                        help='Type of model to use')

    parser.add_argument('--training-days', type=int, default=180,
                        help='Number of days of data to use for training')

    parser.add_argument('--backtest-days', type=int, default=30,
                        help='Number of days of data to use for backtesting')

    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial balance for backtesting')

    parser.add_argument('--position-size', type=float, default=0.1,
                        help='Position size as fraction of balance')

    parser.add_argument('--plot', action='store_true',
                        help='Plot the results')

    parser.add_argument('--save-model', action='store_true',
                        help='Save the trained model')

    return parser.parse_args()


def run_demo(args):
    """Run the forex model demo with the given arguments."""
    logger.info(f"Starting forex demo for {args.currency_pair} on {args.timeframe} timeframe")

    # Create data and model directories if they don't exist
    data_dir = os.path.join(parent_dir, "data", "forex")
    os.makedirs(data_dir, exist_ok=True)

    models_dir = os.path.join(parent_dir, "models", "forex")
    os.makedirs(models_dir, exist_ok=True)

    # Calculate date ranges
    end_date = datetime.now()
    backtest_start = end_date - timedelta(days=args.backtest_days)
    training_start = backtest_start - timedelta(days=args.training_days)

    # Load data
    logger.info("Loading data")
    data_loader = ForexDataLoader()

    # Load training data
    training_data = data_loader.load_data(
        currency_pair=args.currency_pair,
        timeframe=args.timeframe,
        start_date=training_start.strftime("%Y-%m-%d"),
        end_date=backtest_start.strftime("%Y-%m-%d")
    )

    if len(training_data) == 0:
        logger.error("No training data available")
        return 1

    logger.info(f"Loaded {len(training_data)} rows of training data")

    # Load backtest data
    backtest_data = data_loader.load_data(
        currency_pair=args.currency_pair,
        timeframe=args.timeframe,
        start_date=backtest_start.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )

    if len(backtest_data) == 0:
        logger.error("No backtest data available")
        return 1

    logger.info(f"Loaded {len(backtest_data)} rows of backtest data")

    # Add technical indicators
    logger.info("Adding technical indicators")
    indicators = TechnicalIndicators()

    # Common indicators for training and testing
    indicator_list = ['sma', 'ema', 'macd', 'rsi', 'bollinger', 'atr']

    # Add indicators to training data
    training_data = indicators.add_all_indicators(training_data, include=indicator_list)

    # Add indicators to backtest data
    backtest_data = indicators.add_all_indicators(backtest_data, include=indicator_list)

    # Add economic calendar data
    training_data = data_loader.add_economic_calendar(training_data, args.currency_pair)
    backtest_data = data_loader.add_economic_calendar(backtest_data, args.currency_pair)

    # Initialize and train model
    logger.info("Training model")
    model = ForexTradingModel(
        currency_pair=args.currency_pair,
        timeframe=args.timeframe,
        model_type=args.model_type
    )

    # Train the model
    metrics = model.train(training_data)
    logger.info(f"Model training completed with metrics: {metrics}")

    # Save model if requested
    if args.save_model:
        if model.save_model():
            logger.info(f"Model saved to {model.model_path}")
        else:
            logger.warning("Failed to save model")

    # Generate predictions
    logger.info("Generating predictions")
    predictions = model.predict(backtest_data)

    # Run backtest
    logger.info("Running backtest")
    backtest_results = model.backtest(
        backtest_data,
        initial_balance=args.initial_balance,
        position_size=args.position_size
    )

    # Print backtest results
    logger.info(f"Backtest results:")
    logger.info(f"  Initial balance: ${args.initial_balance:.2f}")
    logger.info(f"  Final balance: ${backtest_results['final_balance']:.2f}")
    logger.info(f"  Return: {backtest_results['return']:.2f}%")
    logger.info(f"  Total trades: {backtest_results['total_trades']}")
    logger.info(f"  Win rate: {backtest_results['win_rate']:.2f}")

    # Plot results if requested
    if args.plot:
        plot_results(predictions, backtest_results)

    return 0


def plot_results(predictions, backtest_results):
    """Plot the backtest results."""
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Plot price chart
    ax1.plot(predictions['datetime'], predictions['close'], label='Close Price', color='black', alpha=0.7)

    # Plot buy and sell signals
    buy_signals = predictions[predictions['signal'] == 1]
    sell_signals = predictions[predictions['signal'] == -1]

    ax1.scatter(buy_signals['datetime'], buy_signals['close'], marker='^', color='green', label='Buy Signal')
    ax1.scatter(sell_signals['datetime'], sell_signals['close'], marker='v', color='red', label='Sell Signal')

    # Format price chart
    ax1.set_title('Forex Price Chart with Trading Signals')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Create dataframe for balance over time
    balance_data = []
    current_balance = backtest_results.get('initial_balance', 10000)

    for trade in backtest_results.get('trades', []):
        if 'profit' in trade:
            current_balance += trade.get('profit', 0)
            balance_data.append({
                'datetime': trade.get('datetime', 0),
                'balance': current_balance
            })

    if balance_data:
        # Convert to dataframe
        balance_df = pd.DataFrame(balance_data)

        # Plot balance chart
        ax2.plot(balance_df['datetime'], balance_df['balance'], label='Account Balance', color='blue')
        ax2.set_title('Account Balance During Backtest')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Balance ($)')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No trade data available', horizontalalignment='center',
                verticalalignment='center', transform=ax2.transAxes)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point for the demo."""
    args = parse_args()
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
