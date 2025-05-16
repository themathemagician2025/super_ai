#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Run Server

Main entry point for starting the Super AI web server and loading modules.
This script handles initialization, configuration loading, and startup of all components.
"""

import os
import sys
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_server")

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Super AI Server")

    parser.add_argument('--config', type=str, default='.env',
                      help='Path to configuration file (.env)')
    parser.add_argument('--host', type=str, default=None,
                      help='Host address to bind to (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                      help='Port to listen on (overrides config)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--sports-model', type=str, default=None,
                      help='Path to sports prediction model')
    parser.add_argument('--betting-model', type=str, default=None,
                      help='Path to betting prediction model')

    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from .env file

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary with configuration values
    """
    try:
        import dotenv

        config = {}

        # Try to load from .env file
        env_path = Path(config_path)
        if env_path.exists():
            logger.info(f"Loading configuration from {env_path}")
            dotenv.load_dotenv(env_path)
        else:
            logger.warning(f"No config file found at {config_path}, using default or environment values")

        # Set configuration with defaults
        config["API_HOST"] = os.getenv("API_HOST", "127.0.0.1")
        config["API_PORT"] = int(os.getenv("API_PORT", "5000"))
        config["API_DEBUG"] = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")
        config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")
        config["SPORTS_MODEL_PATH"] = os.getenv("SPORTS_MODEL_PATH", None)
        config["BETTING_MODEL_PATH"] = os.getenv("BETTING_MODEL_PATH", None)
        config["ENABLE_DANGEROUS_AI"] = os.getenv("ENABLE_DANGEROUS_AI", "False").lower() in ("true", "1", "t")
        # Add financial API config
        config["ENABLE_FINANCIAL_API"] = os.getenv("ENABLE_FINANCIAL_API", "True").lower() in ("true", "1", "t")

        return config
    except ImportError:
        logger.warning("dotenv module not found, using default configuration")
        return {
            "API_HOST": "127.0.0.1",
            "API_PORT": 5000,
            "API_DEBUG": False,
            "LOG_LEVEL": "INFO",
            "SPORTS_MODEL_PATH": None,
            "BETTING_MODEL_PATH": None,
            "ENABLE_DANGEROUS_AI": False,
            "ENABLE_FINANCIAL_API": True
        }

def setup_logging(log_level: str):
    """Set up logging based on configuration"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    logger.info(f"Log level set to {log_level}")

async def load_models_async(sports_model_path: Optional[str], betting_model_path: Optional[str]):
    """
    Load models asynchronously to avoid blocking the main thread

    Args:
        sports_model_path: Path to sports prediction model
        betting_model_path: Path to betting prediction model
    """
    from web_interface.server import load_models

    # Run model loading in a separate thread
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            executor,
            lambda: load_models(sports_model_path, betting_model_path)
        )

    logger.info("Models loaded asynchronously")

def register_financial_api(app):
    """Register the financial API blueprint"""
    try:
        logger.info("Attempting to register financial API blueprint...")
        # Log the app's existing routes before registration
        logger.info("Routes before registration:")
        for rule in app.url_map.iter_rules():
            logger.info(f"  {rule.endpoint}: {rule}")

        # Import and register the financial API blueprint
        from api_interface.financial_api import register_financial_blueprint
        register_financial_blueprint(app)

        # Log the app's routes after registration
        logger.info("Routes after registration:")
        for rule in app.url_map.iter_rules():
            if 'financial_api' in rule.endpoint:
                logger.info(f"  {rule.endpoint}: {rule}")

        logger.info("Financial API blueprint registered successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to import financial API blueprint module: {str(e)}")
        logger.error("Make sure the api_interface.financial_api module is available")
        return False
    except Exception as e:
        logger.error(f"Failed to register financial API blueprint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def start_server(config: Dict[str, Any], args):
    """
    Initialize and start the web server

    Args:
        config: Configuration settings
        args: Command line arguments
    """
    # Import server module
    from web_interface.server import create_app, start_web_interface

    # Override config with command line arguments
    host = args.host or config.get("API_HOST")
    port = args.port or config.get("API_PORT")
    debug = args.debug or config.get("API_DEBUG")
    sports_model_path = args.sports_model or config.get("SPORTS_MODEL_PATH")
    betting_model_path = args.betting_model or config.get("BETTING_MODEL_PATH")

    # Create and configure app
    logger.info("Initializing Flask application")
    app = create_app(
        sports_model_path=sports_model_path,
        betting_model_path=betting_model_path,
        config=config
    )

    # Always register financial API blueprint (regardless of config)
    register_financial_api(app)

    # Start the web interface
    logger.info(f"Starting web server on {host}:{port}")
    result = start_web_interface(host, port, debug)

    # Log result
    if result.get("status") == "success":
        logger.info(f"Web server started successfully at {result.get('message')}")
    else:
        logger.error(f"Failed to start web server: {result.get('message')}")

def main():
    """Main entry point for running the server"""
    # Parse command line arguments
    args = parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Set up logging
        setup_logging(config.get("LOG_LEVEL", "INFO"))

        # Start the web server (this will block)
        start_server(config, args)

        return 0
    except KeyboardInterrupt:
        logger.info("Server operation interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    # Set event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main function
    sys.exit(main())