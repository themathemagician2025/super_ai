# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Simple Forex API server with robust error handling, logging, and auto-recovery
"""

from flask import Flask, jsonify, request
import os
import sys
import logging
import json
import uuid
import traceback
from datetime import datetime, timedelta
import time
import random
import numpy as np

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our utilities
from web_interface.system_utils import (
    timeout,
    fallback,
    log_decision,
    log_tool_call,
    save_user_query,
    RequestTracker,
    setup_crash_recovery
)

app = Flask(__name__)

# Configure request tracking
request_tracker = RequestTracker("forex_api")

# Setup crash recovery
setup_crash_recovery(
    "forex_api",
    f"python {os.path.join(current_dir, 'recovery_manager.py')} --restart forex_api"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available currency pairs
CURRENCY_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP"
]

# Available timeframes
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Default fallback responses
DEFAULT_HEALTH_RESPONSE = jsonify({
    'status': 'degraded',
    'message': 'Forex API server is experiencing issues',
    'timestamp': datetime.now().isoformat()
})

DEFAULT_PREDICTION_RESPONSE = jsonify({
    'status': 'error',
    'message': 'Prediction service is currently unavailable',
    'fallback': True,
    'currency_pair': 'UNKNOWN',
    'timestamp': datetime.now().isoformat()
})

@app.before_request
def before_request():
    """Log and track every request"""
    request_id = str(uuid.uuid4())
    request.request_id = request_id

    # Start tracking this request
    request_tracker.start_request(
        request_id,
        {
            "path": request.path,
            "method": request.method,
            "args": dict(request.args),
            "remote_addr": request.remote_addr
        }
    )

    # Log the decision to process this request
    log_decision(
        f"Processing forex API request to {request.path}",
        {
            "request_id": request_id,
            "method": request.method,
            "path": request.path
        }
    )

@app.after_request
def after_request(response):
    """Log after every request"""
    if hasattr(request, 'request_id'):
        request_tracker.end_request(request.request_id, status=str(response.status_code))
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions"""
    if hasattr(request, 'request_id'):
        request_tracker.end_request(request.request_id, status="error")

    # Log the error
    error_id = str(uuid.uuid4())
    app.logger.error(f"Error {error_id}: {str(e)}\n{traceback.format_exc()}")

    # Return an error response
    return jsonify({
        "error": str(e),
        "error_id": error_id,
        "timestamp": datetime.now().isoformat()
    }), 500

@app.route('/api/health')
@fallback(default_return=DEFAULT_HEALTH_RESPONSE)
@timeout(3)  # 3 second timeout
def health_check():
    """Health check endpoint with fallback and timeout"""
    # Log this tool call
    log_tool_call("health_check", {}, "Health check requested")

    return jsonify({
        'status': 'online',
        'message': 'Forex API server is running',
        'timestamp': datetime.now().isoformat(),
        'process_id': os.getpid()
    })

@app.route('/api/forex/predict', methods=['POST'])
@fallback(default_return=DEFAULT_PREDICTION_RESPONSE)
@timeout(10)  # 10 second timeout for predictions
def predict():
    """
    Generate forex price predictions and trading signals with robust error handling.
    """
    try:
        # Log the start of prediction processing
        log_decision("Starting forex prediction processing", {
            "endpoint": "/api/forex/predict",
            "method": "POST"
        })

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

        # Log prediction request parameters
        log_tool_call(
            "forex_prediction",
            {"currency_pair": currency_pair, "timeframe": timeframe},
            "Generating forex predictions"
        )

        # Save this query for recovery demonstration
        save_user_query(
            f"forex_prediction_{currency_pair}_{timeframe}",
            {
                "currency_pair": currency_pair,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            }
        )

        # Generate simulated prediction - now with timeout protection
        # Current time
        now = datetime.now()

        # Generate some random prediction data
        predictions = generate_predictions(currency_pair, now)

        # Track success
        log_decision("Forex prediction completed successfully", {
            "currency_pair": currency_pair,
            "timeframe": timeframe
        })

        return jsonify({
            "currency_pair": currency_pair,
            "timeframe": timeframe,
            "model_type": "simulated",
            "predictions": predictions
        }), 200

    except Exception as e:
        # This will be caught by the fallback decorator, but we still log it
        logger.error(f"Error in forex prediction: {str(e)}")
        raise

@timeout(5)  # 5 second timeout for this function
def generate_predictions(currency_pair, now):
    """
    Generate simulated predictions with timeout protection
    """
        predictions = []
        base_price = 1.2000 if currency_pair == "EUR/USD" else 1.3000

    # Simulate CPU-intensive operation that might time out
        for i in range(5):
        # Simulate some work
        time.sleep(0.2)  # Normally this would be computation

            price = base_price * (1 + random.uniform(-0.01, 0.01))
            predictions.append({
                "datetime": (now.replace(microsecond=0) - timedelta(hours=i)).isoformat(),
                "open": float(price * (1 - 0.0005)),
                "high": float(price * (1 + 0.001)),
                "low": float(price * (1 - 0.001)),
                "close": float(price),
                "volume": float(random.randint(1000, 5000)),
                "prediction": float(random.uniform(0.4, 0.6)),
                "signal": random.choice([-1, 0, 1]),
                "signal_str": random.choice(["buy", "hold", "sell"])
            })

    return predictions

@app.route('/api/forex/available_models', methods=['GET'])
@fallback(default_return=jsonify({"models": [], "status": "fallback", "error": "Model listing unavailable"}))
def available_models():
    """
    Get a list of available trained forex models with fallback.
    """
    # Log this tool call
    log_tool_call("list_forex_models", {}, "Listing available forex models")

    try:
    # Simulate some available models
    models = [
        {
            "currency_pair": "EUR/USD",
            "timeframe": "1h",
            "model_type": "random_forest",
            "file": "eurusd_1h_random_forest.joblib"
        },
        {
            "currency_pair": "GBP/USD",
            "timeframe": "4h",
            "model_type": "random_forest",
            "file": "gbpusd_4h_random_forest.joblib"
        }
    ]

        # Sometimes randomly fail to test fallback (1 in 10 chance)
        if random.random() < 0.1:
            raise Exception("Simulated failure for testing")

    return jsonify({"models": models}), 200

    except Exception as e:
        # This will be caught by the fallback decorator, but we still log it
        logger.error(f"Error in available_models: {str(e)}")
        raise

@app.route('/api/test-timeout')
@timeout(1)  # 1 second timeout
def test_timeout():
    """Test endpoint to demonstrate timeout"""
    # Simulate a long-running operation
    time.sleep(3)  # This will trigger a timeout
    return jsonify({"status": "This should never be returned"})

@app.route('/api/test-crash')
def test_crash():
    """Test endpoint to demonstrate crash recovery"""
    # Log that we're about to crash
    log_decision("About to test crash recovery", {"endpoint": "/api/test-crash"})

    # Simulate a crash
    os.kill(os.getpid(), 9)  # SIGKILL
    return jsonify({"status": "This will never be returned"})

if __name__ == '__main__':
    print("Starting robust Forex API server with error handling, logging and recovery...")
    app.run(host='0.0.0.0', port=5001, debug=True)
