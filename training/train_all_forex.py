#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Script to train multiple forex trading models for different currency pairs and timeframes.
This script automates the process of training models for a predefined set of
currency pairs and timeframes, allowing for batch training operations.
"""

import os
import sys
import logging
import argparse
import concurrent.futures
from datetime import datetime, timedelta

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs", "forex_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default currency pairs to train
DEFAULT_CURRENCY_PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "EUR/GBP"
]

# Default timeframes to train
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]

# Default model types
DEFAULT_MODEL_TYPES = ["random_forest"]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train multiple forex trading models.')

    parser.add_argument('--currency-pairs', type=str, nargs='+', default=DEFAULT_CURRENCY_PAIRS,
                        help='Currency pairs to train models for')

    parser.add_argument('--timeframes', type=str, nargs='+', default=DEFAULT_TIMEFRAMES,
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Timeframes to train models for')

    parser.add_argument('--model-types', type=str, nargs='+', default=DEFAULT_MODEL_TYPES,
                        choices=['random_forest', 'logistic'],
                        help='Types of models to train')

    parser.add_argument('--start-date', type=str, default='2020-01-01',
                        help='Start date for training data (YYYY-MM-DD)')

    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for training data (YYYY-MM-DD)')

    parser.add_argument('--indicators', type=str, nargs='+',
                        default=['sma', 'ema', 'macd', 'rsi', 'bollinger', 'atr'],
                        help='Technical indicators to use')

    parser.add_argument('--backtest', action='store_true',
                        help='Run backtest after training each model')

    parser.add_argument('--parallel', action='store_true',
                        help='Train models in parallel')

    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of worker processes when training in parallel')

    return parser.parse_args()


def train_single_model(currency_pair, timeframe, model_type, start_date, end_date,
                      indicators, backtest=False):
    """Train a single forex model with the given parameters."""

    model_id = f"{currency_pair.replace('/', '_')}_{timeframe}_{model_type}"
    logger.info(f"Starting training for model: {model_id}")

    # Create data directory if it doesn't exist
    data_dir = os.path.join(parent_dir, "data", "forex")
    os.makedirs(data_dir, exist_ok=True)

    # Create models directory if it doesn't exist
    models_dir = os.path.join(parent_dir, "models", "forex")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    try:
        data_loader = ForexDataLoader()
        df = data_loader.load_data(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"Loaded {len(df)} rows of data for {model_id}")
    except Exception as e:
        logger.error(f"Error loading data for {model_id}: {str(e)}")
        return None

    # Add technical indicators
    try:
        indicators_obj = TechnicalIndicators()
        df = indicators_obj.add_all_indicators(df, include=indicators)
        logger.info(f"Added indicators for {model_id}")
    except Exception as e:
        logger.error(f"Error adding indicators for {model_id}: {str(e)}")
        return None

    # Add economic calendar data
    try:
        df = data_loader.add_economic_calendar(df, currency_pair)
        logger.info(f"Added economic calendar events for {model_id}")
    except Exception as e:
        logger.error(f"Error adding economic calendar for {model_id}: {str(e)}")

    # Initialize and train model
    try:
        model = ForexTradingModel(
            currency_pair=currency_pair,
            timeframe=timeframe,
            model_type=model_type
        )

        # Train the model
        metrics = model.train(df, test_size=0.2)
        logger.info(f"Model {model_id} training completed with metrics: {metrics}")

        # Backtest if requested
        if backtest:
            # Use the last 20% of data for backtesting
            backtest_size = int(len(df) * 0.2)
            backtest_data = df.iloc[-backtest_size:].copy()

            backtest_results = model.backtest(backtest_data)
            logger.info(f"Backtest results for {model_id}: {backtest_results['return']:.2f}% return over {backtest_results['total_trades']} trades")
            logger.info(f"Win rate for {model_id}: {backtest_results['win_rate']:.2f}")

            return {
                "model_id": model_id,
                "metrics": metrics,
                "backtest_results": backtest_results
            }

        return {
            "model_id": model_id,
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error training model {model_id}: {str(e)}")
        return None


def main():
    """Main function to train all forex models."""

    # Parse command line arguments
    args = parse_args()

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(parent_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Log training start
    logger.info("Starting batch training of forex models")
    logger.info(f"Currency pairs: {args.currency_pairs}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Model types: {args.model_types}")
    logger.info(f"Training period: {args.start_date} to {args.end_date or 'now'}")

    # Set default end date if not provided
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")

    # Generate all combinations of parameters to train
    training_tasks = []
    for currency_pair in args.currency_pairs:
        for timeframe in args.timeframes:
            for model_type in args.model_types:
                training_tasks.append({
                    "currency_pair": currency_pair,
                    "timeframe": timeframe,
                    "model_type": model_type,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "indicators": args.indicators,
                    "backtest": args.backtest
                })

    # Train models
    results = []

    if args.parallel:
        logger.info(f"Training {len(training_tasks)} models in parallel with {args.max_workers} workers")

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    train_single_model,
                    task["currency_pair"],
                    task["timeframe"],
                    task["model_type"],
                    task["start_date"],
                    task["end_date"],
                    task["indicators"],
                    task["backtest"]
                )
                for task in training_tasks
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        logger.info(f"Training {len(training_tasks)} models sequentially")

        for i, task in enumerate(training_tasks):
            logger.info(f"Training model {i+1}/{len(training_tasks)}")

            result = train_single_model(
                task["currency_pair"],
                task["timeframe"],
                task["model_type"],
                task["start_date"],
                task["end_date"],
                task["indicators"],
                task["backtest"]
            )

            if result:
                results.append(result)

    # Summarize results
    logger.info(f"Training completed. Successfully trained {len(results)} out of {len(training_tasks)} models.")

    # Save summary report
    report_path = os.path.join(logs_dir, f"forex_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(report_path, 'w') as f:
        f.write("=== Forex Model Training Report ===\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training period: {args.start_date} to {args.end_date}\n")
        f.write(f"Models trained: {len(results)} out of {len(training_tasks)}\n\n")

        f.write("=== Training Results ===\n\n")

        for result in results:
            f.write(f"Model: {result['model_id']}\n")
            f.write(f"Training metrics: {result['metrics']}\n")

            if 'backtest_results' in result:
                f.write(f"Backtest return: {result['backtest_results']['return']:.2f}%\n")
                f.write(f"Backtest trades: {result['backtest_results']['total_trades']}\n")
                f.write(f"Win rate: {result['backtest_results']['win_rate']:.2f}\n")

            f.write("\n")

    logger.info(f"Training report saved to {report_path}")


if __name__ == "__main__":
    main()