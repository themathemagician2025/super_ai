#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Script to train a forex trading model for a specific currency pair and timeframe.
This script handles data loading, preprocessing, feature engineering, model training,
and saving the trained model for later use.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from trading.forex_model import ForexTradingModel
from data_pipeline.forex_data_loader import ForexDataLoader
from data_pipeline.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a forex trading model.')

    parser.add_argument('--currency-pair', type=str, default='EUR/USD',
                        help='Currency pair to train the model on (e.g., EUR/USD)')

    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Timeframe for the data')

    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'logistic'],
                        help='Type of model to use')

    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for training data (YYYY-MM-DD)')

    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training data (YYYY-MM-DD)')

    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')

    parser.add_argument('--indicators', type=str, nargs='*',
                        default=['sma', 'ema', 'macd', 'rsi', 'bollinger', 'atr'],
                        help='Technical indicators to use')

    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after training')

    return parser.parse_args()


def main():
    """Main function to train the forex model."""

    # Parse command line arguments
    args = parse_args()

    logger.info(f"Training forex model for {args.currency_pair} on {args.timeframe} timeframe")
    logger.info(f"Using model type: {args.model_type}")

    # Create data directory if it doesn't exist
    data_dir = os.path.join(parent_dir, "data", "forex")
    os.makedirs(data_dir, exist_ok=True)

    # Create models directory if it doesn't exist
    models_dir = os.path.join(parent_dir, "models", "forex")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading data for {args.currency_pair} from {args.start_date} to {args.end_date}")
    data_loader = ForexDataLoader()

    try:
        df = data_loader.load_data(
            currency_pair=args.currency_pair,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )
        logger.info(f"Loaded {len(df)} rows of data")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return

    # Add technical indicators
    logger.info("Adding technical indicators")
    indicators = TechnicalIndicators()

    try:
        df = indicators.add_all_indicators(df, include=args.indicators)
        logger.info(f"Added indicators: {', '.join(args.indicators)}")
    except Exception as e:
        logger.error(f"Error adding indicators: {str(e)}")
        return

    # Add economic calendar data
    logger.info("Adding economic calendar events")
    try:
        df = data_loader.add_economic_calendar(df, args.currency_pair)
        logger.info("Added economic calendar events")
    except Exception as e:
        logger.error(f"Error adding economic calendar: {str(e)}")

    # Save preprocessed data
    logger.info("Saving preprocessed data")
    processed_data_path = os.path.join(
        data_dir,
        f"{args.currency_pair.replace('/', '_')}_{args.timeframe}_processed.csv"
    )
    df.to_csv(processed_data_path, index=False)
    logger.info(f"Saved preprocessed data to {processed_data_path}")

    # Initialize and train model
    logger.info("Initializing forex trading model")
    model = ForexTradingModel(
        currency_pair=args.currency_pair,
        timeframe=args.timeframe,
        model_type=args.model_type
    )

    # Train the model
    logger.info("Training model")
    try:
        metrics = model.train(df, test_size=args.test_size)
        logger.info(f"Model training completed with metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return

    # Backtest if requested
    if args.backtest:
        logger.info("Running backtest")
        try:
            # Use the last 20% of data for backtesting if not using test data for backtest
            backtest_size = int(len(df) * 0.2)
            backtest_data = df.iloc[-backtest_size:].copy()

            backtest_results = model.backtest(backtest_data)
            logger.info(f"Backtest results: {backtest_results['return']:.2f}% return over {backtest_results['total_trades']} trades")
            logger.info(f"Win rate: {backtest_results['win_rate']:.2f}")

            # Save backtest results
            backtest_results_path = os.path.join(
                data_dir,
                f"{args.currency_pair.replace('/', '_')}_{args.timeframe}_backtest.txt"
            )

            with open(backtest_results_path, 'w') as f:
                f.write(f"Backtest Results for {args.currency_pair} {args.timeframe}\n")
                f.write(f"Model type: {args.model_type}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
                f.write(f"Return: {backtest_results['return']:.2f}%\n")
                f.write(f"Total trades: {backtest_results['total_trades']}\n")
                f.write(f"Win rate: {backtest_results['win_rate']:.2f}\n")
                f.write(f"Initial balance: 10000\n")
                f.write(f"Final balance: {backtest_results['final_balance']:.2f}\n")

            logger.info(f"Saved backtest results to {backtest_results_path}")

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")

    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main()