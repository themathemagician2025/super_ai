# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Web Server Module

Handles the initialization and operation of the Flask web server,
providing API endpoints for the Super AI system.
"""

import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from flask import Flask, request, jsonify, render_template, send_from_directory
except ImportError:
    raise ImportError("Flask is required but not installed. Install it with: pip install flask")

# Import prediction modules if available
try:
    from prediction.sports_predictor import SportsPredictor
    from betting.betting_prediction import BettingPredictor
    PREDICTORS_AVAILABLE = True
except ImportError:
    PREDICTORS_AVAILABLE = False

# Import utilities
try:
    from utils.module_executor import ModuleExecutor
    EXECUTOR_AVAILABLE = True
except ImportError:
    EXECUTOR_AVAILABLE = False

# Set up logger
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Global variables for models and executor
sports_model = None
betting_model = None
module_executor = None
running_modules = {}

def load_models(sports_model_path: Optional[str] = None, betting_model_path: Optional[str] = None) -> Dict[str, bool]:
    """
    Load prediction models from disk.

    Args:
        sports_model_path: Path to the sports prediction model
        betting_model_path: Path to the betting prediction model

    Returns:
        Dictionary with loading results
    """
    global sports_model, betting_model

    if not PREDICTORS_AVAILABLE:
        logger.warning("Predictor modules not available. Models will not be loaded.")
        return {"sports_model": False, "betting_model": False}

    results = {}

    # Set default paths if not provided
    base_dir = Path(__file__).parent.parent
    sports_model_path = sports_model_path or str(base_dir / "models" / "sports_predictor.pth")
    betting_model_path = betting_model_path or str(base_dir / "models" / "betting_predictor.pth")

    # Load sports model
    try:
        if os.path.exists(sports_model_path):
            import torch

            # Get model metadata to determine input size
            model_data = torch.load(sports_model_path)
            metadata = model_data.get("metadata", {})
            input_size = metadata.get("input_size", 10)  # Default if not found

            # Create and load model
            sports_model = SportsPredictor(input_size=input_size)
            sports_model.load_state_dict(model_data["model_state_dict"])
            sports_model.eval()

            logger.info(f"Successfully loaded sports model from {sports_model_path}")
            results["sports_model"] = True
        else:
            logger.warning(f"Sports model file not found at {sports_model_path}")
            results["sports_model"] = False
    except Exception as e:
        logger.error(f"Error loading sports model: {str(e)}")
        results["sports_model"] = False

    # Load betting model
    try:
        if os.path.exists(betting_model_path):
            import torch

            # Get model metadata to determine input size
            model_data = torch.load(betting_model_path)
            metadata = model_data.get("metadata", {})
            input_size = metadata.get("input_size", 10)  # Default if not found

            # Create and load model
            betting_model = BettingPredictor(input_size=input_size)
            betting_model.load_state_dict(model_data["model_state_dict"])
            betting_model.eval()

            logger.info(f"Successfully loaded betting model from {betting_model_path}")
            results["betting_model"] = True
        else:
            logger.warning(f"Betting model file not found at {betting_model_path}")
            results["betting_model"] = False
    except Exception as e:
        logger.error(f"Error loading betting model: {str(e)}")
        results["betting_model"] = False

    return results

def initialize_module_executor(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Initialize the module executor with configuration.

    Args:
        config: Configuration settings

    Returns:
        Boolean indicating success
    """
    global module_executor

    if not EXECUTOR_AVAILABLE:
        logger.warning("ModuleExecutor not available.")
        return False

    try:
        # Create the module executor
        base_dir = Path(__file__).parent.parent
        module_executor = ModuleExecutor(base_dir, config or {})
        logger.info("ModuleExecutor initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize ModuleExecutor: {str(e)}")
        return False

# Define routes
@app.route('/')
def index():
    """Serve the index page."""
    logger.info("Serving index.html")
    return render_template('index.html')

@app.route('/debug')
def debug():
    """Serve the debug page."""
    logger.info("Serving debug.html")
    return render_template('debug.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models": {
            "sports_model": sports_model is not None,
            "betting_model": betting_model is not None
        },
        "executor": module_executor is not None
    })

@app.route('/api/status')
def system_status():
    """System status endpoint."""
    status = {
        "models": {
            "sports_model": sports_model is not None,
            "betting_model": betting_model is not None
        },
        "modules": running_modules,
        "executor": {
            "active": module_executor is not None,
            "loaded_modules": list(module_executor.get_loaded_modules().keys()) if module_executor else []
        }
    }
    return jsonify(status)

@app.route('/api/debug/routes')
def list_routes():
    """Debug endpoint to list all registered routes."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "path": str(rule)
        })
    return jsonify({"routes": routes})

@app.route('/api/predictions/sports', methods=['POST'])
def predict_sports():
    """
    Endpoint for sports prediction.

    Expects a JSON payload with features to predict.
    """
    # Check if model is loaded
    if sports_model is None:
        return jsonify({
            "status": "error",
            "message": "Sports prediction model not loaded"
        }), 503

    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400

    try:
        # Process features
        import torch
        import numpy as np

        # Extract features from request
        features = data.get("features", [])
        if not features:
            return jsonify({
                "status": "error",
                "message": "No features provided"
            }), 400

        # Convert to numpy array and then to tensor
        if isinstance(features, list):
            features_array = np.array(features, dtype=np.float32)
        elif isinstance(features, dict):
            # If features are provided as a dictionary, handle accordingly
            return jsonify({
                "status": "error",
                "message": "Dictionary features not supported yet"
            }), 400
        else:
            return jsonify({
                "status": "error",
                "message": f"Unsupported features type: {type(features)}"
            }), 400

        # Ensure correct shape
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)

        # Create tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = sports_model(features_tensor)
            prediction_value = prediction.cpu().numpy().item()

        # Return prediction result
        return jsonify({
            "status": "success",
            "prediction": prediction_value,
            "home_win_probability": prediction_value,
            "away_win_probability": 1 - prediction_value if prediction_value < 0.5 else 0
        })

    except Exception as e:
        logger.error(f"Error making sports prediction: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Prediction error: {str(e)}"
        }), 500

@app.route('/api/predictions/betting', methods=['POST'])
def predict_betting():
    """
    Endpoint for betting prediction.

    Expects a JSON payload with features to predict.
    """
    # Check if model is loaded
    if betting_model is None:
        return jsonify({
            "status": "error",
            "message": "Betting prediction model not loaded"
        }), 503

    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400

    try:
        # Process features
        import torch
        import numpy as np

        # Extract features from request
        features = data.get("features", [])
        stake = data.get("stake", 100)  # Default stake
        odds = data.get("odds", 2.0)    # Default odds

        if not features:
            return jsonify({
                "status": "error",
                "message": "No features provided"
            }), 400

        # Convert to numpy array and then to tensor
        if isinstance(features, list):
            features_array = np.array(features, dtype=np.float32)
        elif isinstance(features, dict):
            return jsonify({
                "status": "error",
                "message": "Dictionary features not supported yet"
            }), 400
        else:
            return jsonify({
                "status": "error",
                "message": f"Unsupported features type: {type(features)}"
            }), 400

        # Ensure correct shape
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)

        # Create tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = betting_model(features_tensor)
            prediction_value = prediction.cpu().numpy().item()

        # Calculate expected value
        expected_value = prediction_value * odds
        recommended_stake = stake if expected_value > 1 else 0

        # Return prediction result
        return jsonify({
            "status": "success",
            "prediction": prediction_value,
            "expected_return": prediction_value * stake * odds,
            "expected_value": expected_value,
            "recommended_stake": recommended_stake
        })

    except Exception as e:
        logger.error(f"Error making betting prediction: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Prediction error: {str(e)}"
        }), 500

@app.route('/api/modules/execute', methods=['POST'])
def execute_module():
    """
    Execute a specific module.

    Expects a JSON payload with module path and function name.
    """
    if module_executor is None:
        return jsonify({
            "status": "error",
            "message": "Module executor not initialized"
        }), 503

    # Get request data
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "No data provided"
        }), 400

    module_path = data.get('module_path')
    function_name = data.get('function_name', 'main')
    args = data.get('args', [])
    kwargs = data.get('kwargs', {})

    if not module_path:
        return jsonify({
            "status": "error",
            "message": "No module path provided"
        }), 400

    try:
        # Import time here to avoid circular imports
        import time

        # Start the execution in a separate thread
        def execute_and_update():
            global running_modules
            module_name = Path(module_path).stem
            task_id = f"{module_name}.{function_name}"

            # Update status
            running_modules[task_id] = {"status": "running", "start_time": time.time()}

            # Execute module
            result = module_executor.run_modules([(module_path, function_name, args, kwargs)])

            # Update status with result
            running_modules[task_id] = result.get(task_id, {"status": "unknown"})

        # Start thread
        thread = threading.Thread(target=execute_and_update)
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "started",
            "message": f"Module execution started for {module_path}.{function_name}"
        })

    except Exception as e:
        logger.error(f"Error executing module: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Module execution error: {str(e)}"
        }), 500

def create_app(sports_model_path: Optional[str] = None,
               betting_model_path: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure a Flask application instance.

    Args:
        sports_model_path: Path to the sports prediction model
        betting_model_path: Path to the betting prediction model
        config: Configuration settings

    Returns:
        Configured Flask application
    """
    # Load models
    load_models(sports_model_path, betting_model_path)

    # Initialize module executor
    initialize_module_executor(config)

    # Register financial API blueprint directly
    try:
        from api_interface.financial_api import register_financial_blueprint
        register_financial_blueprint(app)
        logger.info("Financial API blueprint registered directly in create_app")
    except Exception as e:
        logger.error(f"Failed to register financial API in create_app: {str(e)}")

    return app

def start_web_interface(host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> Dict[str, Any]:
    """
    Start the Flask web server.

    Args:
        host: Host address to bind to
        port: Port to listen on
        debug: Whether to run in debug mode

    Returns:
        Dictionary with startup results
    """
    try:
        # Start the Flask server
        logger.info(f"Starting web interface on {host}:{port}")

        app.run(host=host, port=port, debug=debug)

        return {
            "status": "success",
            "message": f"Web interface started at http://{host}:{port}",
            "host": host,
            "port": port
        }

    except Exception as e:
        logger.error(f"Error starting web interface: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to start web interface: {str(e)}"
        }

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load environment variables
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", 5000))
    debug = os.environ.get("API_DEBUG", "False").lower() in ("true", "1", "t")

    # Create app and start server
    create_app()
    start_web_interface(host, port, debug)
