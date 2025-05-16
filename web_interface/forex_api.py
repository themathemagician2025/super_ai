#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
API endpoints for forex trading predictions and analysis.
This module provides REST API endpoints for using trained forex models
for prediction, analysis, and trading signals.
"""

import os
import sys
import logging
import json
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

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

# Create blueprint
forex_api = Blueprint('forex_api', __name__)

# Available currency pairs
CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
]

# Available timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Cache for loaded models
MODEL_CACHE = {}


@forex_api.route('/api/forex/predict', methods=['POST'])
def predict():
    """
    Generate forex price predictions and trading signals.

    Expected JSON payload:
    {
        "currency_pair": "EUR/USD",
        "timeframe": "1h",
        "model_type": "random_forest",
        "limit": 20  // Optional: Number of predictions to return
    }

    Returns:
        JSON with predictions and trading signals
    """
    try:
        # Parse request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["currency_pair", "timeframe"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract and validate parameters
        currency_pair = data.get("currency_pair")
        if currency_pair not in CURRENCY_PAIRS:
            return jsonify({"error": f"Invalid currency pair. Supported pairs: {CURRENCY_PAIRS}"}), 400

        timeframe = data.get("timeframe")
        if timeframe not in TIMEFRAMES:
            return jsonify({"error": f"Invalid timeframe. Supported timeframes: {TIMEFRAMES}"}), 400

        model_type = data.get("model_type", "random_forest")
        limit = data.get("limit", 20)

        # Get or load the model
        model_key = f"{currency_pair}_{timeframe}_{model_type}"

        if model_key in MODEL_CACHE:
            model = MODEL_CACHE[model_key]
            logger.info(f"Using cached model for {model_key}")
        else:
            # Initialize the model
            model = ForexTradingModel(
                currency_pair=currency_pair,
                timeframe=timeframe,
                model_type=model_type
            )

            # Try to load the model
            if model.load_model():
                MODEL_CACHE[model_key] = model
                logger.info(f"Loaded model for {model_key}")
            else:
                return jsonify({
                    "error": f"Model not found for {currency_pair} {timeframe}. Please train the model first."
                }), 404

        # Load recent data for prediction
        data_loader = ForexDataLoader()

        # Calculate appropriate date range based on timeframe and limit
        end_date = datetime.now()
        if timeframe == "1m":
            start_date = end_date - timedelta(hours=limit*2/60)
        elif timeframe == "5m":
            start_date = end_date - timedelta(hours=limit*10/60)
        elif timeframe == "15m":
            start_date = end_date - timedelta(hours=limit*30/60)
        elif timeframe == "1h":
            start_date = end_date - timedelta(hours=limit*2)
        elif timeframe == "4h":
            start_date = end_date - timedelta(hours=limit*8)
        else:  # 1d
            start_date = end_date - timedelta(days=limit*2)

        df = data_loader.load_data(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if len(df) == 0:
            return jsonify({"error": "No data available for the specified parameters"}), 404

        # Generate predictions
        predictions = model.predict(df)

        # Limit the number of results
        if limit > 0 and len(predictions) > limit:
            predictions = predictions.iloc[-limit:]

        # Format the response
        result = {
            "currency_pair": currency_pair,
            "timeframe": timeframe,
            "model_type": model_type,
            "predictions": []
        }

        for _, row in predictions.iterrows():
            result["predictions"].append({
                "datetime": row["datetime"].isoformat() if isinstance(row["datetime"], datetime) else row["datetime"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "prediction": float(row["prediction"]),
                "signal": int(row["signal"]),
                "signal_str": "buy" if row["signal"] == 1 else "sell" if row["signal"] == -1 else "hold"
            })

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in forex prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@forex_api.route('/api/forex/analysis', methods=['POST'])
def analyze():
    """
    Perform technical analysis on forex data.

    Expected JSON payload:
    {
        "currency_pair": "EUR/USD",
        "timeframe": "1h",
        "indicators": ["sma", "ema", "macd", "rsi"],  // Optional
        "limit": 20  // Optional: Number of data points to return
    }

    Returns:
        JSON with technical analysis data
    """
    try:
        # Parse request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["currency_pair", "timeframe"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract and validate parameters
        currency_pair = data.get("currency_pair")
        if currency_pair not in CURRENCY_PAIRS:
            return jsonify({"error": f"Invalid currency pair. Supported pairs: {CURRENCY_PAIRS}"}), 400

        timeframe = data.get("timeframe")
        if timeframe not in TIMEFRAMES:
            return jsonify({"error": f"Invalid timeframe. Supported timeframes: {TIMEFRAMES}"}), 400

        indicators = data.get("indicators", ["sma", "ema", "macd", "rsi", "bollinger"])
        limit = data.get("limit", 50)

        # Load recent data for analysis
        data_loader = ForexDataLoader()

        # Calculate appropriate date range based on timeframe and limit
        end_date = datetime.now()
        if timeframe == "1m":
            start_date = end_date - timedelta(hours=limit*2/60)
        elif timeframe == "5m":
            start_date = end_date - timedelta(hours=limit*10/60)
        elif timeframe == "15m":
            start_date = end_date - timedelta(hours=limit*30/60)
        elif timeframe == "1h":
            start_date = end_date - timedelta(hours=limit*2)
        elif timeframe == "4h":
            start_date = end_date - timedelta(hours=limit*8)
        else:  # 1d
            start_date = end_date - timedelta(days=limit*2)

        df = data_loader.load_data(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if len(df) == 0:
            return jsonify({"error": "No data available for the specified parameters"}), 404

        # Add technical indicators
        indicators_obj = TechnicalIndicators()
        df = indicators_obj.add_all_indicators(df, include=indicators)

        # Limit the number of results
        if limit > 0 and len(df) > limit:
            df = df.iloc[-limit:]

        # Format the response
        result = {
            "currency_pair": currency_pair,
            "timeframe": timeframe,
            "analysis": []
        }

        # Get the column names excluding the OHLCV data
        indicator_columns = [col for col in df.columns if col not in ["datetime", "open", "high", "low", "close", "volume"]]

        for _, row in df.iterrows():
            entry = {
                "datetime": row["datetime"].isoformat() if isinstance(row["datetime"], datetime) else row["datetime"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "indicators": {}
            }

            # Add indicator values
            for col in indicator_columns:
                if pd.notna(row[col]):
                    entry["indicators"][col] = float(row[col])

            result["analysis"].append(entry)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in forex analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500


@forex_api.route('/api/forex/available_models', methods=['GET'])
def available_models():
    """
    Get a list of available trained forex models.

    Returns:
        JSON with list of available models
    """
    try:
        # Get the models directory
        models_dir = os.path.join(parent_dir, "models", "forex")

        if not os.path.exists(models_dir):
            return jsonify({"models": []}), 200

        # List all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]

        # Extract model information
        models = []
        for file in model_files:
            parts = file.replace('.joblib', '').split('_')

            if len(parts) >= 3:
                # Handle currency pair (may contain underscore)
                if len(parts) > 3:
                    currency_pair = parts[0] + '/' + parts[1]
                    timeframe = parts[2]
                    model_type = parts[3]
                else:
                    # If no underscore in currency pair
                    currency_pair = parts[0].replace('USD', '/USD').replace('EUR', 'EUR/').replace('GBP', 'GBP/').replace('JPY', '/JPY').replace('CAD', '/CAD').replace('CHF', '/CHF').replace('AUD', 'AUD/').replace('NZD', 'NZD/')
                    currency_pair = currency_pair.replace('//', '/')  # Fix any double slashes
                    timeframe = parts[1]
                    model_type = parts[2]

                models.append({
                    "currency_pair": currency_pair,
                    "timeframe": timeframe,
                    "model_type": model_type,
                    "file": file
                })

        return jsonify({"models": models}), 200

    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        return jsonify({"error": str(e)}), 500


@forex_api.route('/api/forex/backtest', methods=['POST'])
def backtest():
    """
    Backtest a forex trading strategy using a trained model.

    Expected JSON payload:
    {
        "currency_pair": "EUR/USD",
        "timeframe": "1h",
        "model_type": "random_forest",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "initial_balance": 10000,
        "position_size": 0.1
    }

    Returns:
        JSON with backtest results
    """
    try:
        # Parse request data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["currency_pair", "timeframe", "start_date", "end_date"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract and validate parameters
        currency_pair = data.get("currency_pair")
        if currency_pair not in CURRENCY_PAIRS:
            return jsonify({"error": f"Invalid currency pair. Supported pairs: {CURRENCY_PAIRS}"}), 400

        timeframe = data.get("timeframe")
        if timeframe not in TIMEFRAMES:
            return jsonify({"error": f"Invalid timeframe. Supported timeframes: {TIMEFRAMES}"}), 400

        model_type = data.get("model_type", "random_forest")
        start_date = data.get("start_date")
        end_date = data.get("end_date")
        initial_balance = data.get("initial_balance", 10000)
        position_size = data.get("position_size", 0.1)

        # Get or load the model
        model_key = f"{currency_pair}_{timeframe}_{model_type}"

        if model_key in MODEL_CACHE:
            model = MODEL_CACHE[model_key]
            logger.info(f"Using cached model for {model_key}")
        else:
            # Initialize the model
            model = ForexTradingModel(
                currency_pair=currency_pair,
                timeframe=timeframe,
                model_type=model_type
            )

            # Try to load the model
            if model.load_model():
                MODEL_CACHE[model_key] = model
                logger.info(f"Loaded model for {model_key}")
            else:
                return jsonify({
                    "error": f"Model not found for {currency_pair} {timeframe}. Please train the model first."
                }), 404

        # Load data for backtesting
        data_loader = ForexDataLoader()

        df = data_loader.load_data(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if len(df) == 0:
            return jsonify({"error": "No data available for the specified parameters"}), 404

        # Run backtest
        backtest_results = model.backtest(df, initial_balance=initial_balance, position_size=position_size)

        # Format trades for response
        trades_formatted = []
        for trade in backtest_results.get("trades", []):
            trade_formatted = {k: v for k, v in trade.items()}

            # Format datetime if present
            if 'datetime' in trade_formatted and isinstance(trade_formatted['datetime'], datetime):
                trade_formatted['datetime'] = trade_formatted['datetime'].isoformat()

            trades_formatted.append(trade_formatted)

        # Format the response
        result = {
            "currency_pair": currency_pair,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "initial_balance": initial_balance,
            "final_balance": backtest_results["final_balance"],
            "return": backtest_results["return"],
            "total_trades": backtest_results["total_trades"],
            "win_rate": backtest_results["win_rate"],
            "trades": trades_formatted
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in forex backtest: {str(e)}")
        return jsonify({"error": str(e)}), 500