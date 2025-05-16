# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Web UI Application

This module provides a Flask web application for the Super AI system.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

# Import from our new package structure
from super_ai.prediction.models import sports_prediction
from super_ai.data.loaders import forex_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev_key_for_super_ai'),
    DEBUG=os.environ.get('DEBUG', 'True').lower() in ('true', '1', 't'),
)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', title="Super AI - Prediction System")

@app.route('/api/health')
def health_check():
    """API endpoint for health checks."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0"
    })

@app.route('/api/sports/predict', methods=['POST'])
def sports_predict():
    """API endpoint for sports predictions."""
    data = request.json

    if not data or 'game_id' not in data:
        return jsonify({"error": "Missing required field: game_id"}), 400

    game_id = data.get('game_id')
    sport = data.get('sport', 'soccer')
    league = data.get('league', 'premier_league')

    try:
        prediction = sports_prediction.predict(game_id, sport, league)
        return jsonify(prediction)
    except Exception as e:
        logger.error(f"Error making sports prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/forex/data', methods=['GET'])
def get_forex_data():
    """API endpoint to get forex data."""
    symbol = request.args.get('symbol', 'EUR/USD')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    timeframe = request.args.get('timeframe', '1h')

    try:
        data = forex_loader.load(symbol, start_date, end_date, timeframe)

        if data.empty:
            return jsonify({"error": "No data available for the specified parameters"}), 404

        # Convert DataFrame to list of dictionaries for JSON serialization
        records = data.to_dict(orient='records')

        return jsonify({
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(records),
            "data": records
        })
    except Exception as e:
        logger.error(f"Error getting forex data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

# Create a folder for templates if it doesn't exist
def create_template_directory():
    """Create the templates directory if it doesn't exist."""
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)

        # Create a simple index.html file
        index_path = os.path.join(template_dir, "index.html")
        with open(index_path, "w") as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .card {
            background: white;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .btn {
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 3px;
            text-decoration: none;
            cursor: pointer;
            margin-right: 10px;
        }
        .btn:hover {
            background: #2980b9;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Super AI Prediction System</h1>

        <div class="card">
            <h2>Sports Prediction</h2>
            <p>Make predictions on sports events using machine learning.</p>
            <button class="btn" id="sports-btn">Make Prediction</button>
            <div id="sports-result"></div>
        </div>

        <div class="card">
            <h2>Forex Data</h2>
            <p>Retrieve and analyze foreign exchange data.</p>
            <button class="btn" id="forex-btn">Get Data</button>
            <div id="forex-result"></div>
        </div>
    </div>

    <script>
        document.getElementById("sports-btn").addEventListener("click", async () => {
            try {
                const response = await fetch("/api/sports/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        game_id: "12345",
                        sport: "soccer",
                        league: "premier_league"
                    })
                });

                const data = await response.json();
                const resultDiv = document.getElementById("sports-result");
                resultDiv.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            } catch (error) {
                console.error("Error:", error);
            }
        });

        document.getElementById("forex-btn").addEventListener("click", async () => {
            try {
                const response = await fetch("/api/forex/data?symbol=EUR/USD&timeframe=1h");
                const data = await response.json();
                const resultDiv = document.getElementById("forex-result");

                if (data.error) {
                    resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <p>Symbol: ${data.symbol}</p>
                        <p>Timeframe: ${data.timeframe}</p>
                        <p>Data points: ${data.count}</p>
                        <pre>${JSON.stringify(data.data.slice(0, 3), null, 2)}...</pre>
                    `;
                }
            } catch (error) {
                console.error("Error:", error);
            }
        });
    </script>
</body>
</html>
""")

def run_app(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask application."""
    # Ensure templates directory exists
    create_template_directory()

    # Run the app
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_app()
