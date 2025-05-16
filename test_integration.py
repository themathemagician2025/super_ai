#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Test Integration Script

Tests the integration of the financial API with the main server module.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_integration")

def import_module(module_path, module_name):
    """Import a module from file path"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error importing {module_name} from {module_path}: {e}")
        return None

def main():
    """Run integration test"""
    try:
        logger.info("Starting financial API integration test")

        # Add current directory to Python path
        current_dir = os.getcwd()
        sys.path.append(current_dir)

        # Import modules
        logger.info("Importing modules...")

        # Create a dummy Flask app
        app = Flask("test_app")

        # Add a test route
        @app.route('/test')
        def test():
            return "Test route"

        # Import financial_api module directly
        try:
            logger.info("Importing financial_api module...")
            from api_interface.financial_api import register_financial_blueprint
            logger.info("Successfully imported financial_api module")

            # Register the blueprint
            register_financial_blueprint(app)
            logger.info("Financial API blueprint registered successfully")

            # Check routes
            logger.info("Registered routes:")
            financial_routes = []
            for rule in app.url_map.iter_rules():
                if 'financial_api' in rule.endpoint:
                    logger.info(f"  {rule.endpoint}: {rule}")
                    financial_routes.append(rule.endpoint)

            if financial_routes:
                logger.info(f"Integration test passed: {len(financial_routes)} financial API routes registered")
                return 0
            else:
                logger.error("Integration test failed: No financial API routes registered")
                return 1

        except Exception as e:
            logger.error(f"Error testing financial API integration: {e}", exc_info=True)
            return 1

    except Exception as e:
        logger.error(f"Unexpected error in integration test: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())