#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Test script for the financial API blueprint.
This creates a simple Flask app and registers the financial API blueprint.
"""

import os
import sys
import traceback
import logging
from pathlib import Path
from flask import Flask, jsonify

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger("test_financial_api")

def main():
    """Main entry point for testing the financial API"""
    try:
        logger.info("Starting test server for financial API")

        # Create a simple Flask app
        app = Flask(__name__)

        # Add a basic route
        @app.route('/')
        def index():
            return jsonify({"status": "ok", "message": "Test server is running"})

        # Debug current path and Python path
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")

        # Import and register the financial API blueprint
        try:
            logger.info("Attempting to import financial_api module...")
            sys.path.append(os.getcwd())  # Add current directory to path

            # First try to import directly
            try:
                import api_interface.financial_api
                logger.info("Successfully imported api_interface.financial_api")
            except ImportError as e:
                logger.error(f"Could not import api_interface.financial_api: {e}")
                logger.info("Trying alternate import approach...")

                # Check if the module exists
                if os.path.exists(os.path.join(os.getcwd(), "api_interface", "financial_api.py")):
                    logger.info("Found financial_api.py file in api_interface directory")
                else:
                    logger.error("Could not find financial_api.py in api_interface directory")

                # Try to import the register function directly
                from api_interface.financial_api import register_financial_blueprint

            # Register the blueprint
            logger.info("Registering financial API blueprint...")
            from api_interface.financial_api import register_financial_blueprint
            register_financial_blueprint(app)
            logger.info("Financial API blueprint registered successfully")

        except Exception as e:
            logger.error(f"Failed to register financial API blueprint: {str(e)}")
            logger.error(traceback.format_exc())
            return 1

        # List all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"{rule.endpoint}: {rule}")

        # Run the app
        logger.info("Starting Flask server on http://127.0.0.1:8080")
        app.run(host="127.0.0.1", port=8080, debug=False)

        return 0
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())