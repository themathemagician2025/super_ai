#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Financial API Demo

This script demonstrates how to import and use the financial API blueprint
in a new Flask application. This serves as documentation for users who want
to integrate the financial API into their own applications.
"""

import os
import sys
import logging
from flask import Flask, jsonify, render_template_string

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("financial_api_demo")

# HTML template for the demo homepage
HOMEPAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Financial API Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .method { display: inline-block; width: 50px; font-weight: bold; }
        a { color: #0066cc; }
        pre { background: #f9f9f9; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Financial API Demo</h1>
    <p>This page demonstrates the Financial API endpoints available in the Super AI system.</p>

    <h2>Available Endpoints</h2>

    <div class="endpoint">
        <span class="method">GET</span>
        <a href="/api/financial/health">/api/financial/health</a>
        <p>Health check for the Financial API</p>
    </div>

    <div class="endpoint">
        <span class="method">GET/POST</span>
        <a href="/api/financial/forex?currency_pair=EUR/USD">/api/financial/forex</a>
        <p>Forex exchange rate predictions</p>
        <p>Parameters: currency_pair (e.g., EUR/USD), timeframe (1h, 4h, 1d, 1w)</p>
    </div>

    <div class="endpoint">
        <span class="method">GET/POST</span>
        <a href="/api/financial/stocks?symbol=AAPL">/api/financial/stocks</a>
        <p>Stock price predictions</p>
        <p>Parameters: symbol (e.g., AAPL), timeframe (1d, 1w, 1m)</p>
    </div>

    <div class="endpoint">
        <span class="method">POST</span>
        <a href="/api/financial/predict/stocks">/api/financial/predict/stocks</a>
        <p>Multiple stock predictions</p>
        <pre>
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "timeframe": "daily",
  "horizon": 7
}
        </pre>
    </div>

    <div class="endpoint">
        <span class="method">POST</span>
        <a href="/api/financial/predict/crypto">/api/financial/predict/crypto</a>
        <p>Cryptocurrency price predictions</p>
        <pre>
{
  "coins": ["BTC", "ETH", "SOL"],
  "timeframe": "hourly",
  "horizon": 24
}
        </pre>
    </div>

    <div class="endpoint">
        <span class="method">POST</span>
        <a href="/api/financial/portfolio/optimize">/api/financial/portfolio/optimize</a>
        <p>Portfolio optimization recommendations</p>
        <pre>
{
  "assets": [
    {"symbol": "AAPL", "current_allocation": 0.2, "risk_tolerance": 0.5},
    {"symbol": "MSFT", "current_allocation": 0.3, "risk_tolerance": 0.5},
    {"symbol": "BONDS", "current_allocation": 0.5, "risk_tolerance": 0.2}
  ],
  "risk_profile": "moderate",
  "investment_horizon": 5
}
        </pre>
    </div>

    <div class="endpoint">
        <span class="method">GET</span>
        <a href="/api/financial/market/sentiment?market=stock">/api/financial/market/sentiment</a>
        <p>Market sentiment analysis</p>
        <p>Parameters: market (stock, crypto, forex), symbols (comma-separated list)</p>
    </div>
</body>
</html>
"""

def create_demo_app():
    """Create a Flask app for the demo"""
    # Create Flask app
    app = Flask(__name__)

    # Add homepage route
    @app.route('/')
    def index():
        return render_template_string(HOMEPAGE_TEMPLATE)

    # Import and register financial API blueprint
    try:
        from api_interface.financial_api import register_financial_blueprint
        register_financial_blueprint(app)
        logger.info("Financial API blueprint registered successfully")
    except Exception as e:
        logger.error(f"Failed to register financial API blueprint: {str(e)}", exc_info=True)

    return app

def main():
    """Run the demo server"""
    try:
        # Create the demo app
        app = create_demo_app()

        # List all routes
        logger.info("Registered routes:")
        for rule in app.url_map.iter_rules():
            logger.info(f"{rule.endpoint}: {rule}")

        # Run the server
        port = 9000
        logger.info(f"Starting demo server on http://localhost:{port}")
        logger.info(f"Open a browser to http://localhost:{port} to view the demo")
        app.run(host="0.0.0.0", port=port)

        return 0
    except Exception as e:
        logger.error(f"Error running financial API demo: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())