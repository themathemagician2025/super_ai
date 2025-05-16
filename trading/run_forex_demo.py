#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Demo script for the Forex Trading Model
This script showcases the functionality of the forex trading model, including:
- Data loading and preprocessing
- Technical indicator calculation
- Model training
- Strategy backtesting
- Visualization of results
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

# Import the ForexTradingModel
from trading.forex_model import ForexTradingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_demo():
    """Run a demonstration of the forex trading model."""
    logger.info("Starting Forex Trading Model demonstration")

    # Create the model directories if they don't exist
    data_dir = project_root / 'data' / 'forex'
    models_dir = project_root / 'models' / 'forex'

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Create multiple models with different parameters
    models = [
        {
            "currency_pair": "EUR/USD",
            "timeframe": "daily",
            "model_type": "classification",
            "description": "EUR/USD Daily Classification Model"
        },
        {
            "currency_pair": "GBP/USD",
            "timeframe": "4h",
            "model_type": "regression",
            "description": "GBP/USD 4-Hour Regression Model"
        },
        {
            "currency_pair": "USD/JPY",
            "timeframe": "daily",
            "model_type": "classification",
            "description": "USD/JPY Daily Classification Model"
        }
    ]

    results = {}

    # Train and test each model
    for model_config in models:
        logger.info(f"Processing {model_config['description']}")

        # Initialize the model
        forex_model = ForexTradingModel(
            currency_pair=model_config["currency_pair"],
            timeframe=model_config["timeframe"],
            model_type=model_config["model_type"]
        )

        try:
            # Run the complete analysis
            analysis_results = forex_model.run_analysis()

            # Store results
            results[model_config["currency_pair"]] = {
                "config": model_config,
                "results": analysis_results
            }

            # Print summary
            backtest = analysis_results["backtest_results"]
            print(f"\n--- {model_config['description']} ---")
            print(f"Total Return: {backtest['total_return_pct']:.2f}%")
            print(f"Trades: {backtest['total_trades']} (Win Rate: {backtest['win_rate']*100:.2f}%)")
            print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {backtest['max_drawdown']:.2f}%")

            # Model location
            print(f"Model saved to: {analysis_results['model_path']}")
            print(f"Plot saved to: {analysis_results['plot_path']}")

        except Exception as e:
            logger.error(f"Error running {model_config['description']}: {e}")
            print(f"Failed to process {model_config['currency_pair']}: {str(e)}")

    # Compare model performance
    if results:
        print("\n=== MODEL COMPARISON ===")
        print(f"{'Currency Pair':<10} | {'Type':<15} | {'Return %':<10} | {'Win Rate':<10} | {'Sharpe':<10} | {'Max DD':<10}")
        print("-" * 75)

        for pair, data in results.items():
            config = data["config"]
            backtest = data["results"]["backtest_results"]

            print(f"{pair:<10} | {config['model_type']:<15} | "
                  f"{backtest['total_return_pct']:>8.2f}% | "
                  f"{backtest['win_rate']*100:>8.2f}% | "
                  f"{backtest['sharpe_ratio']:>9.2f} | "
                  f"{backtest['max_drawdown']:>8.2f}%")

    logger.info("Forex Trading Model demonstration completed")

if __name__ == "__main__":
    run_demo()