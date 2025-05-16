#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Run Flask server with Forex API endpoints.
This script initializes and runs a Flask web server with the Forex API endpoints registered.
"""

import os
import sys
import logging
import argparse
from flask import Flask

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("forex_server")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Forex API server')

    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the server on')

    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on')

    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')

    return parser.parse_args()

def create_app():
    """Create and configure the Flask app."""
    # Create Flask app
    app = Flask(__name__)

    # Import forex API blueprint
    try:
        from web_interface.forex_api import forex_api
        app.register_blueprint(forex_api)
        logger.info("Forex API blueprint registered successfully")
    except Exception as e:
        logger.error(f"Could not register Forex API blueprint: {str(e)}")
        raise

    # Add health check route
    @app.route('/health')
    def health_check():
        return {'status': 'ok', 'service': 'forex_api'}

    # Initialize forex model cache
    try:
        from trading.forex_model import ForexTradingModel
        from web_interface.forex_api import MODEL_CACHE

        # Check for existing models
        models_dir = os.path.join(parent_dir, "models", "forex")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
            if model_files:
                # Just load the first model to verify functionality
                sample_model = model_files[0]
                parts = sample_model.replace('.joblib', '').split('_')

                if len(parts) >= 3:
                    # Extract model details
                    if len(parts) > 3:
                        currency_pair = parts[0] + '/' + parts[1]
                        timeframe = parts[2]
                        model_type = parts[3]
                    else:
                        currency_pair = parts[0]
                        timeframe = parts[1]
                        model_type = parts[2]

                    # Try to load the model
                    model = ForexTradingModel(
                        currency_pair=currency_pair,
                        timeframe=timeframe,
                        model_type=model_type
                    )

                    if model.load_model():
                        MODEL_CACHE[f"{currency_pair}_{timeframe}_{model_type}"] = model
                        logger.info(f"Preloaded forex model: {currency_pair} {timeframe}")
        else:
            logger.warning("No forex models directory found")
    except Exception as e:
        logger.warning(f"Could not initialize forex models: {str(e)}")

    return app

def main():
    """Main function to run the server."""
    args = parse_args()

    try:
        # Create the app
        app = create_app()

        # Log registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"{rule.endpoint}: {rule}")

        # Run the app
        logger.info(f"Starting Forex API server on http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)

    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())