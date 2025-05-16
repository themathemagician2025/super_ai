# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Financial API Interface

Provides RESTful API endpoints for financial market predictions
and portfolio management recommendations.
"""

import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from flask import Blueprint, request, jsonify, current_app

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'financial_api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create blueprint
financial_api = Blueprint('financial_api', __name__)

class FinancialPredictor:
    """Financial market prediction model handler."""

    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the financial predictor.

        Args:
            model_dir: Directory containing prediction models
        """
        self.model_dir = model_dir
        self.forex_model_path = os.path.join(model_dir, 'forex_predictor.pth')
        self.stock_model_path = os.path.join(model_dir, 'stock_predictor.pth')

        # Model loading will be implemented when models are trained
        self.forex_model = None
        self.stock_model = None

        logger.info("Financial predictor initialized")

    def load_models(self):
        """Load prediction models if available."""
        # This will be implemented when models are trained
        # For now, we'll use simulated predictions
        logger.info("Using simulated financial predictions (models not yet trained)")

    def predict_forex(self, currency_pair: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Predict forex exchange rate movement.

        Args:
            currency_pair: Currency pair (e.g., 'EUR/USD')
            timeframe: Prediction timeframe ('1h', '4h', '1d', '1w')

        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Generating forex prediction for {currency_pair} ({timeframe})")

        # Generate simulated prediction
        base_currency, quote_currency = currency_pair.split('/')

        # Current price (simulated)
        if base_currency == 'EUR' and quote_currency == 'USD':
            current_price = 1.08 + np.random.normal(0, 0.005)
        elif base_currency == 'GBP' and quote_currency == 'USD':
            current_price = 1.25 + np.random.normal(0, 0.008)
        elif base_currency == 'USD' and quote_currency == 'JPY':
            current_price = 148.0 + np.random.normal(0, 0.5)
        else:
            current_price = 1.0 + np.random.normal(0, 0.01)

        # Simulate prediction
        confidence = np.random.uniform(0.6, 0.95)
        direction = 1 if np.random.random() > 0.5 else -1
        change_percent = np.random.uniform(0.1, 1.5) * direction

        # Calculate predicted price
        predicted_price = current_price * (1 + change_percent/100)

        # Timeframe adjustments
        timeframe_factor = {
            '1h': 0.2,
            '4h': 0.5,
            '1d': 1.0,
            '1w': 2.5
        }.get(timeframe, 1.0)

        change_percent *= timeframe_factor
        predicted_price = current_price * (1 + change_percent/100)

        # Format the result
        return {
            'currency_pair': currency_pair,
            'timeframe': timeframe,
            'current_price': round(current_price, 5),
            'predicted_price': round(predicted_price, 5),
            'change_percent': round(change_percent, 2),
            'direction': 'up' if change_percent > 0 else 'down',
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat(),
            'supporting_factors': [
                'Recent economic data',
                'Technical indicators',
                'Market sentiment',
                'Historical patterns'
            ]
        }

    def predict_stock(self, symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Predict stock price movement.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Prediction timeframe ('1d', '1w', '1m')

        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Generating stock prediction for {symbol} ({timeframe})")

        # Generate simulated prediction based on stock symbol
        if symbol == 'AAPL':
            current_price = 170.0 + np.random.normal(0, 2.0)
        elif symbol == 'MSFT':
            current_price = 330.0 + np.random.normal(0, 3.0)
        elif symbol == 'GOOGL':
            current_price = 140.0 + np.random.normal(0, 1.5)
        elif symbol == 'AMZN':
            current_price = 160.0 + np.random.normal(0, 2.0)
        elif symbol == 'TSLA':
            current_price = 180.0 + np.random.normal(0, 4.0)
        else:
            current_price = 100.0 + np.random.normal(0, 2.0)

        # Simulate prediction
        confidence = np.random.uniform(0.55, 0.9)
        direction = 1 if np.random.random() > 0.5 else -1
        change_percent = np.random.uniform(0.2, 3.0) * direction

        # Timeframe adjustments
        timeframe_factor = {
            '1d': 1.0,
            '1w': 3.0,
            '1m': 8.0
        }.get(timeframe, 1.0)

        change_percent *= timeframe_factor
        predicted_price = current_price * (1 + change_percent/100)

        # Volume prediction
        avg_volume = np.random.randint(1000000, 50000000)
        predicted_volume = avg_volume * (1 + np.random.uniform(-0.3, 0.5))

        # Format the result
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'change_percent': round(change_percent, 2),
            'direction': 'up' if change_percent > 0 else 'down',
            'confidence': round(confidence, 2),
            'avg_volume': int(avg_volume),
            'predicted_volume': int(predicted_volume),
            'timestamp': datetime.now().isoformat(),
            'supporting_factors': [
                'Earnings projections',
                'Technical analysis',
                'Market sentiment',
                'Sector performance'
            ]
        }

# Initialize predictor
financial_predictor = FinancialPredictor()

# API Routes
@financial_api.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Financial API is operational"
    })

@financial_api.route('/forex', methods=['GET', 'POST'])
def forex_prediction():
    """
    API endpoint for forex predictions.

    GET parameters or POST JSON:
    - currency_pair: Currency pair to predict (e.g., 'EUR/USD')
    - timeframe: Prediction timeframe ('1h', '4h', '1d', '1w')

    Returns:
        JSON response with prediction results
    """
    try:
        # Get parameters from request
        if request.method == 'POST':
            data = request.get_json() or {}
            currency_pair = data.get('currency_pair', 'EUR/USD')
            timeframe = data.get('timeframe', '1d')
        else:
            currency_pair = request.args.get('currency_pair', 'EUR/USD')
            timeframe = request.args.get('timeframe', '1d')

        # Validate parameters
        if '/' not in currency_pair:
            return jsonify({'error': 'Invalid currency pair format. Use format like EUR/USD'}), 400

        valid_timeframes = ['1h', '4h', '1d', '1w']
        if timeframe not in valid_timeframes:
            return jsonify({'error': f'Invalid timeframe. Use one of: {", ".join(valid_timeframes)}'}), 400

        # Get prediction
        prediction = financial_predictor.predict_forex(currency_pair, timeframe)

        # Add metadata
        prediction['status'] = 'success'
        prediction['api_version'] = '1.0'

        return jsonify(prediction)

    except Exception as e:
        logger.error(f"Error in forex prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@financial_api.route('/stocks', methods=['GET', 'POST'])
def stock_prediction():
    """
    API endpoint for stock predictions.

    GET parameters or POST JSON:
    - symbol: Stock symbol to predict (e.g., 'AAPL')
    - timeframe: Prediction timeframe ('1d', '1w', '1m')

    Returns:
        JSON response with prediction results
    """
    try:
        # Get parameters from request
        if request.method == 'POST':
            data = request.get_json() or {}
            symbol = data.get('symbol', 'AAPL')
            timeframe = data.get('timeframe', '1d')
        else:
            symbol = request.args.get('symbol', 'AAPL')
            timeframe = request.args.get('timeframe', '1d')

        # Validate parameters
        if not symbol or not symbol.isalpha():
            return jsonify({'error': 'Invalid stock symbol'}), 400

        valid_timeframes = ['1d', '1w', '1m']
        if timeframe not in valid_timeframes:
            return jsonify({'error': f'Invalid timeframe. Use one of: {", ".join(valid_timeframes)}'}), 400

        # Get prediction
        prediction = financial_predictor.predict_stock(symbol.upper(), timeframe)

        # Add metadata
        prediction['status'] = 'success'
        prediction['api_version'] = '1.0'

        return jsonify(prediction)

    except Exception as e:
        logger.error(f"Error in stock prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@financial_api.route('/predict/stocks', methods=['POST'])
def predict_stocks():
    """
    Predict stock market movements for multiple stocks

    Expected JSON input:
    {
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "timeframe": "daily", // or "weekly", "monthly"
        "horizon": 7 // prediction horizon in days
    }
    """
    try:
        # Get request data
        data = request.json

        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "failed"
            }), 400

        # Extract parameters
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', 'daily')
        horizon = data.get('horizon', 7)

        if not symbols:
            return jsonify({
                "error": "No stock symbols provided",
                "status": "failed"
            }), 400

        # Convert timeframe format
        timeframe_map = {
            'daily': '1d',
            'weekly': '1w',
            'monthly': '1m'
        }
        internal_timeframe = timeframe_map.get(timeframe, '1d')

        # Get predictions for each symbol
        predictions = []
        for symbol in symbols:
            prediction = financial_predictor.predict_stock(symbol, internal_timeframe)
            # Adapt to expected format
            predictions.append({
                "symbol": symbol,
                "current_price": prediction['current_price'],
                "predicted_price": prediction['predicted_price'],
                "confidence": prediction['confidence'],
                "recommendation": "buy" if prediction['direction'] == 'up' else "hold"
            })

        return jsonify({
            "predictions": predictions,
            "timeframe": timeframe,
            "horizon": horizon,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in stock prediction endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@financial_api.route('/predict/crypto', methods=['POST'])
def predict_crypto():
    """
    Predict cryptocurrency price movements

    Expected JSON input:
    {
        "coins": ["BTC", "ETH", "SOL"],
        "timeframe": "hourly", // or "daily", "weekly"
        "horizon": 24 // prediction horizon in hours or days
    }
    """
    try:
        # Get request data
        data = request.json

        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "failed"
            }), 400

        # Extract parameters
        coins = data.get('coins', [])
        timeframe = data.get('timeframe', 'hourly')
        horizon = data.get('horizon', 24)

        if not coins:
            return jsonify({
                "error": "No cryptocurrency symbols provided",
                "status": "failed"
            }), 400

        # This is a placeholder for actual prediction logic
        predictions = []
        for coin in coins:
            predictions.append({
                "coin": coin,
                "current_price": 10000.0 + hash(coin) % 50000,
                "predicted_price": 10500.0 + hash(coin) % 51000,
                "volatility": 0.2 + (hash(coin) % 10) / 100,
                "confidence": 0.75,
                "recommendation": "buy" if hash(coin) % 3 == 0 else "hold"
            })

        return jsonify({
            "predictions": predictions,
            "timeframe": timeframe,
            "horizon": horizon,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in cryptocurrency prediction endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@financial_api.route('/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """
    Optimize investment portfolio allocation

    Expected JSON input:
    {
        "assets": [
            {"symbol": "AAPL", "current_allocation": 0.2, "risk_tolerance": 0.5},
            {"symbol": "MSFT", "current_allocation": 0.3, "risk_tolerance": 0.5},
            {"symbol": "BONDS", "current_allocation": 0.5, "risk_tolerance": 0.2}
        ],
        "risk_profile": "moderate", // or "conservative", "aggressive"
        "investment_horizon": 5 // years
    }
    """
    try:
        # Get request data
        data = request.json

        if not data:
            return jsonify({
                "error": "No data provided",
                "status": "failed"
            }), 400

        # Extract parameters
        assets = data.get('assets', [])
        risk_profile = data.get('risk_profile', 'moderate')
        investment_horizon = data.get('investment_horizon', 5)

        if not assets:
            return jsonify({
                "error": "No assets provided",
                "status": "failed"
            }), 400

        # This is a placeholder for actual portfolio optimization logic
        recommended_allocation = []
        for asset in assets:
            symbol = asset.get('symbol', '')
            current = asset.get('current_allocation', 0)

            # Simple mock adjustment based on risk profile
            if risk_profile == 'aggressive':
                if 'BONDS' in symbol:
                    new_allocation = max(0, current - 0.1)
                else:
                    new_allocation = min(0.4, current + 0.1)
            elif risk_profile == 'conservative':
                if 'BONDS' in symbol:
                    new_allocation = min(0.7, current + 0.1)
                else:
                    new_allocation = max(0.1, current - 0.05)
            else:  # moderate
                new_allocation = current

            recommended_allocation.append({
                "symbol": symbol,
                "current_allocation": current,
                "recommended_allocation": new_allocation,
                "expected_return": 0.05 + (hash(symbol) % 10) / 100,
                "risk_level": 0.2 + (hash(symbol) % 15) / 100
            })

        return jsonify({
            "portfolio": recommended_allocation,
            "risk_profile": risk_profile,
            "investment_horizon": investment_horizon,
            "expected_portfolio_return": 0.08,
            "portfolio_risk_level": 0.25,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in portfolio optimization endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@financial_api.route('/market/sentiment', methods=['GET'])
def market_sentiment():
    """
    Get current market sentiment analysis

    Query parameters:
    - market: stock, crypto, forex (default: stock)
    - symbols: comma-separated list of symbols to analyze
    """
    try:
        market_type = request.args.get('market', 'stock')
        symbols_param = request.args.get('symbols', '')
        symbols = [s.strip() for s in symbols_param.split(',')] if symbols_param else []

        # Mock sentiment data
        sentiment_score = np.random.uniform(-1.0, 1.0)
        fear_greed = np.random.randint(1, 100)

        sentiment_labels = {
            (-1.0, -0.6): 'Very Bearish',
            (-0.6, -0.2): 'Bearish',
            (-0.2, 0.2): 'Neutral',
            (0.2, 0.6): 'Bullish',
            (0.6, 1.0): 'Very Bullish'
        }

        sentiment_label = next(v for k, v in sentiment_labels.items() if k[0] <= sentiment_score <= k[1])

        fear_greed_labels = {
            (1, 25): 'Extreme Fear',
            (25, 45): 'Fear',
            (45, 55): 'Neutral',
            (55, 75): 'Greed',
            (75, 100): 'Extreme Greed'
        }

        fear_greed_label = next(v for k, v in fear_greed_labels.items() if k[0] <= fear_greed <= k[1])

        result = {
            "market": market_type,
            "sentiment": {
                "overall_sentiment": sentiment_label,
                "sentiment_score": round(sentiment_score, 2),
                "market_volatility": "medium",
                "trending_sectors": ["Technology", "Healthcare"],
                "market_metrics": {
                    "fear_greed_index": fear_greed,
                    "fear_greed_label": fear_greed_label,
                    "volatility_index": 22.5,
                    "average_volume_change": "+5.2%"
                }
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        # Add symbol-specific sentiment if requested
        if symbols:
            symbols_sentiment = []
            for symbol in symbols:
                symbols_sentiment.append({
                    "symbol": symbol,
                    "sentiment": "bullish" if hash(symbol) % 3 != 0 else "bearish",
                    "sentiment_score": 0.5 + (hash(symbol) % 10) / 20,
                    "news_sentiment": "positive" if hash(symbol) % 2 == 0 else "mixed",
                    "social_mentions": 1000 + hash(symbol) % 9000
                })
            result["sentiment"]["symbols"] = symbols_sentiment

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in market sentiment endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

def register_financial_blueprint(app):
    """Register the financial blueprint with the Flask app"""
    app.register_blueprint(financial_api, url_prefix='/api/financial')
    logger.info("Financial API endpoints registered")
    # Initialize the predictor
    financial_predictor.load_models()
