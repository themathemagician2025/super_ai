# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import prediction modules
from prediction.sports_predictor import SportsPredictor
from betting.betting_prediction import BettingPredictor

# Import all core_ai modules
def import_core_ai_modules():
    """Import and run main/demo/test functions from all core_ai modules, logging results."""
    import importlib
    import logging
    import types
    import os
    import sys
    results = {}
    core_ai_dir = os.path.join(os.path.dirname(__file__))
    for fname in os.listdir(core_ai_dir):
        if fname.endswith('.py') and fname not in ('main.py', '__init__.py'):
            mod_name = f"core_ai.{fname[:-3]}"
            try:
                mod = importlib.import_module(mod_name)
                # Try to call main/demo/test function if present
                for entry in ['main', 'demo', 'test', 'run', 'run_demo', 'run_test']:
                    func = getattr(mod, entry, None)
                    if isinstance(func, types.FunctionType):
                        results[mod_name] = func()
                        break
                else:
                    # Try to instantiate a class with the same name as the module (CamelCase)
                    class_name = ''.join([part.capitalize() for part in fname[:-3].split('_')])
                    cls = getattr(mod, class_name, None)
                    if cls:
                        instance = cls()
                        # Try to call a representative method
                        for meth in ['run', 'main', 'demo', 'test']:
                            method = getattr(instance, meth, None)
                            if callable(method):
                                results[mod_name] = method()
                                break
                        else:
                            results[mod_name] = f"Instantiated {class_name}, no callable method found."
                    else:
                        results[mod_name] = "No main/demo/test function or class found."
                logging.info(f"core_ai module {mod_name} integration result: {results[mod_name]}")
            except Exception as e:
                results[mod_name] = f"Error: {e}"
                logging.error(f"Error integrating core_ai module {mod_name}: {e}")
    return results

# Configure logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/mathemagician.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_pytorch_models():
    """Load the PyTorch models"""
    try:
        import torch

        # Get path to models directory
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models_dir = root_dir / 'models'

        # Load sports predictor model
        sports_model = SportsPredictor(input_size=10, hidden_size=20)
        sports_model_path = models_dir / 'sports_predictor.pth'
        if sports_model_path.exists():
            state = torch.load(sports_model_path)
            if isinstance(state, dict) and 'model_state_dict' in state:
                # Newer format with metadata
                metadata = state.get('metadata', {})
                input_size = metadata.get('input_size', 10)
                hidden_size = metadata.get('hidden_size', 20)
                sports_model = SportsPredictor(input_size=input_size, hidden_size=hidden_size)
                sports_model.load_state_dict(state['model_state_dict'])
            else:
                # Fallback: try loading as plain state_dict
                sports_model.load_state_dict(state)
            sports_model.eval()  # Set to evaluation mode
            logger.info(f"Loaded sports predictor model from {sports_model_path}")
        else:
            logger.warning(f"Sports predictor model not found at {sports_model_path}")

        # Load betting predictor model
        betting_model = BettingPredictor(input_size=5, hidden_size=10)
        betting_model_path = models_dir / 'betting_predictor.pth'
        if betting_model_path.exists():
            state = torch.load(betting_model_path)
            if isinstance(state, dict) and 'model_state_dict' in state:
                metadata = state.get('metadata', {})
                input_size = metadata.get('input_size', 5)
                hidden_size = metadata.get('hidden_size', 10)
                betting_model = BettingPredictor(input_size=input_size, hidden_size=hidden_size)
                betting_model.load_state_dict(state['model_state_dict'])
            else:
                betting_model.load_state_dict(state)
            betting_model.eval()  # Set to evaluation mode
            logger.info(f"Loaded betting predictor model from {betting_model_path}")
        else:
            logger.warning(f"Betting predictor model not found at {betting_model_path}")

        return {
            'sports_model': sports_model,
            'betting_model': betting_model
        }
    except Exception as e:
        logger.error(f"Error loading PyTorch models: {e}")
        return {}

def start_api_server(host='localhost', port=5000):
    """Start the Flask API server"""
    try:
        from flask import Flask

        app = Flask(__name__)

        @app.route('/')
        def home():
            return {"status": "Mathemagician AI is running"}

        @app.route('/health')
        def health():
            return {"status": "healthy"}

        @app.route('/predict/sports', methods=['POST'])
        def predict_sports():
            # This would use the loaded sports model to make predictions
            return {"prediction": 0.75, "confidence": 0.8}

        @app.route('/predict/betting', methods=['POST'])
        def predict_betting():
            # This would use the loaded betting model to make predictions
            return {"prediction": 1.5, "confidence": 0.7}

        logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        raise

def main():
    """Main entry point for the Mathemagician AI system"""
    try:
        logger.info("Starting Mathemagician AI system")

        # Load PyTorch models
        models = load_pytorch_models()

        # Integrate all core_ai modules
        logger.info("Integrating all core_ai modules...")
        core_ai_results = import_core_ai_modules()
        logger.info(f"core_ai integration results: {core_ai_results}")

        # Start API server
        host = os.environ.get('API_HOST', 'localhost')
        port = int(os.environ.get('API_PORT', '5000'))
        start_api_server(host, port)

        return {"status": "success", "core_ai_results": core_ai_results}
    except Exception as e:
        error_msg = f"Error starting Mathemagician AI: {e}"
        logger.error(error_msg)
        traceback.print_exc()
        return {"status": "error", "message": error_msg}

if __name__ == "__main__":
    main()
