#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Test Server Script

A simple script to test if the financial API is properly registered in the main Flask app.
"""

import os
import sys
import logging
import importlib
from pathlib import Path
from flask import Flask

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger("test_server")

def main():
    """Test the registration of the financial API blueprint"""
    try:
        # Add current directory to path
        sys.path.append(os.getcwd())

        # Import server and run_server modules
        from web_interface.server import app, create_app
        import run_server

        logger.info("Successfully imported server modules")

        # Get the register_financial_api function
        register_financial_api = getattr(run_server, 'register_financial_api', None)
        if not register_financial_api:
            logger.error("Could not find register_financial_api function in run_server module")
            return 1

        # Call the register function
        logger.info("Attempting to register financial API blueprint")
        result = register_financial_api(app)
        logger.info(f"Registration result: {result}")

        # List all registered routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"{rule.endpoint}: {rule}")

        logger.info("Financial API blueprint integration test completed")
        return 0
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())