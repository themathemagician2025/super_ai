# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Predictor API Interface

Provides RESTful API endpoints for sports and betting predictions
using the enhanced prediction models.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the integrated predictor
from prediction.model_integration import IntegratedPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'predictor_api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Initialize the integrated predictor
predictor = IntegratedPredictor()

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "status": "running",
        "message": "Sports and Betting Prediction API",
        "endpoints": [
            "/health",
            "/predict/sports",
            "/predict/betting",
            "/predict/integrated",
            "/models/info"
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    status = "healthy"
    sports_model_loaded = predictor.sports_model is not None
    betting_model_loaded = predictor.betting_model is not None

    if not sports_model_loaded and not betting_model_loaded:
        status = "degraded - no models loaded"

    return jsonify({
        "status": status,
        "sports_model_loaded": sports_model_loaded,
        "betting_model_loaded": betting_model_loaded
    })

@app.route('/predict/sports', methods=['POST'])
def predict_sports():
    """
    Predict sports outcomes

    Expected JSON input:
    {
        "features": [[feature1, feature2, ...], ...],
        "feature_names": ["name1", "name2", ...] (optional)
    }

    Or:
    {
        "matches": [
            {"team1": "A", "team2": "B", "feature1": 0.5, ...},
            ...
        ]
    }
    """
    try:
        # Get request data
        data = request.json

        # Check if sports model is loaded
        if predictor.sports_model is None:
            return jsonify({
                "error": "Sports prediction model not loaded",
                "status": "failed"
            }), 503

        # Process different input formats
        if 'features' in data:
            # Process raw features
            features = data['features']
            predictions, probabilities = predictor.predict_sports_outcome(
                features, return_probs=True
            )

            # Prepare response
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    "prediction": int(pred[0]),
                    "probability": float(prob[0]),
                    "outcome": "win" if pred[0] == 1 else "loss"
                })

            return jsonify({
                "predictions": results,
                "status": "success",
                "count": len(results)
            })

        elif 'matches' in data:
            # Process structured match data
            import pandas as pd

            # Convert to DataFrame
            matches_df = pd.DataFrame(data['matches'])

            # Extract team information for the response
            teams = []
            if 'team1' in matches_df.columns and 'team2' in matches_df.columns:
                teams = list(zip(matches_df['team1'], matches_df['team2']))

                # Drop team columns if they exist
                feature_df = matches_df.drop(['team1', 'team2'], axis=1, errors='ignore')
            else:
                feature_df = matches_df

            # Make predictions
            predictions, probabilities = predictor.predict_sports_outcome(
                feature_df, return_probs=True
            )

            # Prepare response
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    "prediction": int(pred[0]),
                    "probability": float(prob[0]),
                    "outcome": "win" if pred[0] == 1 else "loss"
                }

                # Add team information if available
                if teams:
                    result["team1"] = teams[i][0]
                    result["team2"] = teams[i][1]

                results.append(result)

            return jsonify({
                "predictions": results,
                "status": "success",
                "count": len(results)
            })
        else:
            return jsonify({
                "error": "Invalid input format. Expected 'features' or 'matches'.",
                "status": "failed"
            }), 400

    except Exception as e:
        logger.error(f"Error in sports prediction endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/predict/betting', methods=['POST'])
def predict_betting():
    """
    Predict betting odds

    Expected JSON input:
    {
        "features": [[feature1, feature2, ...], ...],
        "feature_names": ["name1", "name2", ...] (optional)
    }

    Or:
    {
        "matches": [
            {"team1": "A", "team2": "B", "feature1": 0.5, ...},
            ...
        ]
    }
    """
    try:
        # Get request data
        data = request.json

        # Check if betting model is loaded
        if predictor.betting_model is None:
            return jsonify({
                "error": "Betting prediction model not loaded",
                "status": "failed"
            }), 503

        # Process different input formats
        if 'features' in data:
            # Process raw features
            features = data['features']
            odds_predictions = predictor.predict_odds(features)

            # Prepare response
            results = []
            for odds in odds_predictions:
                results.append({
                    "predicted_odds": float(odds[0])
                })

            return jsonify({
                "predictions": results,
                "status": "success",
                "count": len(results)
            })

        elif 'matches' in data:
            # Process structured match data
            import pandas as pd

            # Convert to DataFrame
            matches_df = pd.DataFrame(data['matches'])

            # Extract team information for the response
            teams = []
            if 'team1' in matches_df.columns and 'team2' in matches_df.columns:
                teams = list(zip(matches_df['team1'], matches_df['team2']))

                # Drop team columns if they exist
                feature_df = matches_df.drop(['team1', 'team2'], axis=1, errors='ignore')
            else:
                feature_df = matches_df

            # Make predictions
            odds_predictions = predictor.predict_odds(feature_df)

            # Prepare response
            results = []
            for i, odds in enumerate(odds_predictions):
                result = {
                    "predicted_odds": float(odds[0])
                }

                # Add team information if available
                if teams:
                    result["team1"] = teams[i][0]
                    result["team2"] = teams[i][1]

                results.append(result)

            return jsonify({
                "predictions": results,
                "status": "success",
                "count": len(results)
            })
        else:
            return jsonify({
                "error": "Invalid input format. Expected 'features' or 'matches'.",
                "status": "failed"
            }), 400

    except Exception as e:
        logger.error(f"Error in betting prediction endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/predict/integrated', methods=['POST'])
def predict_integrated():
    """
    Make integrated predictions with both models

    Expected JSON input:
    {
        "features": [[feature1, feature2, ...], ...],
        "feature_names": ["name1", "name2", ...] (optional)
    }

    Or:
    {
        "matches": [
            {"team1": "A", "team2": "B", "feature1": 0.5, ...},
            ...
        ]
    }
    """
    try:
        # Get request data
        data = request.json

        # Check if models are loaded
        if predictor.sports_model is None and predictor.betting_model is None:
            return jsonify({
                "error": "No prediction models loaded",
                "status": "failed"
            }), 503

        # Process different input formats
        if 'features' in data:
            # Process raw features
            features = data['features']
            results = predictor.integrated_prediction(features)

            return jsonify({
                "predictions": results,
                "status": "success"
            })

        elif 'matches' in data:
            # Process structured match data
            import pandas as pd

            # Convert to DataFrame
            matches_df = pd.DataFrame(data['matches'])

            # Extract team information for the response
            teams = []
            if 'team1' in matches_df.columns and 'team2' in matches_df.columns:
                teams = list(zip(matches_df['team1'], matches_df['team2']))

                # Drop team columns if they exist
                feature_df = matches_df.drop(['team1', 'team2'], axis=1, errors='ignore')
            else:
                feature_df = matches_df

            # Make predictions
            integrated_results = predictor.integrated_prediction(feature_df)

            # Add team information if available
            if teams:
                integrated_results["teams"] = [
                    {"team1": t[0], "team2": t[1]} for t in teams
                ]

            return jsonify({
                "predictions": integrated_results,
                "status": "success"
            })
        else:
            return jsonify({
                "error": "Invalid input format. Expected 'features' or 'matches'.",
                "status": "failed"
            }), 400

    except Exception as e:
        logger.error(f"Error in integrated prediction endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Get information about loaded models"""
    sports_info = {}
    if predictor.sports_model is not None:
        sports_info = {
            "input_size": predictor.sports_model.input_size,
            "hidden_size": predictor.sports_model.hidden_size,
            "feature_importance": bool(predictor.sports_model.feature_importances)
        }

    betting_info = {}
    if predictor.betting_model is not None:
        betting_info = {
            "input_size": predictor.betting_model.input_size,
            "hidden_size": predictor.betting_model.hidden_size,
            "feature_importance": bool(predictor.betting_model.feature_importances)
        }

    return jsonify({
        "sports_model": sports_info if sports_info else None,
        "betting_model": betting_info if betting_info else None,
        "confidence_threshold": predictor.confidence_threshold,
        "status": "success"
    })

@app.route('/models/reload', methods=['POST'])
def reload_models():
    """Reload models from disk"""
    try:
        sports_path = request.json.get('sports_model_path')
        betting_path = request.json.get('betting_model_path')

        # Reload models
        predictor.load_sports_model(sports_path)
        predictor.load_betting_model(betting_path)

        return jsonify({
            "status": "success",
            "sports_model_loaded": predictor.sports_model is not None,
            "betting_model_loaded": predictor.betting_model is not None
        })
    except Exception as e:
        logger.error(f"Error reloading models: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

def run_app(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask app"""
    logger.info(f"Starting predictor API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Try to load models
    predictor.load_sports_model()
    predictor.load_betting_model()

    # Start the API server
    run_app(host='0.0.0.0', port=5000, debug=False)
