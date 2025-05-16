#!/usr/bin/env python3
"""
Debug Server

A simple script to debug the financial API registration.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("debug_server")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Create a simple Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "status": "running",
        "message": "Debug server is operational"
    })

@app.route('/debug/routes')
def list_routes():
    """List all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "path": str(rule)
        })
    return jsonify({"routes": routes})

def register_financial_api():
    """Try to register the financial API blueprint"""
    try:
        # Import the financial_api module
        from api_interface.financial_api import register_financial_blueprint

        # Register the blueprint
        result = register_financial_blueprint(app)

        logger.info(f"Financial API blueprint registration result: {result}")
        return result
    except Exception as e:
        logger.error(f"Failed to register financial API blueprint: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Try to register the financial API
    register_financial_api()

    # Log registered routes
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule} [{', '.join(rule.methods)}]")

    # Start the Flask server
    app.run(host="localhost", port=5001, debug=True)

# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------
