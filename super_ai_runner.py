#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Super AI Runner

Main entry point for the Super AI prediction system.
This script orchestrates the entire workflow:
1. Fetches live data from various sources
2. Processes the data for model training
3. Trains prediction models with the latest data
4. Serves predictions via the web interface
"""

import os
import sys
import time
import json
import logging
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Add src directory to path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import utility modules
from utils.log_manager import ThreadSafeLogManager
from utils.system_utils import SystemScanner

# Configure logger
logger = logging.getLogger("super_ai_runner")

class SuperAIRunner:
    """Main class that orchestrates the Super AI prediction system."""

    def __init__(self, config_path: str = None):
        """
        Initialize the Super AI Runner.

        Args:
            config_path: Path to the configuration file (.env)
        """
        self.start_time = datetime.now()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize logging
        log_dir = self.config.get('LOG_DIR', os.path.join(self.base_dir, '..', 'logs'))
        self.log_manager = ThreadSafeLogManager(log_dir=log_dir)
        self.log_manager.setup_logging(
            level=self._get_log_level(),
            log_to_console=True,
            log_file=os.path.join(log_dir, f"super_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )

        # Initialize system scanner
        self.scanner = SystemScanner(self.base_dir)

        # Track the status of each component
        self.components_status = {
            "data_scraping": False,
            "data_processing": False,
            "model_training": False,
            "web_interface": False
        }

    def _load_config(self, config_path: str = None) -> Dict[str, str]:
        """
        Load configuration from .env file or environment variables.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration as a dictionary
        """
        config = {}

        # If config_path is not provided, look for .env in the parent directory
        if not config_path:
            config_path = os.path.join(self.base_dir, '..', '.env')

        # Try to load from .env file
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
        else:
            print(f"Warning: Config file not found at {config_path}, using environment variables")

        # Override with environment variables
        for key in os.environ:
            if key.startswith(('API_', 'LOG_', 'ENABLE_', 'DATA_')):
                config[key] = os.environ[key]

        # Set default values if not provided
        config.setdefault('API_HOST', '127.0.0.1')
        config.setdefault('API_PORT', '5000')
        config.setdefault('API_DEBUG', 'False')
        config.setdefault('LOG_LEVEL', 'INFO')
        config.setdefault('ENABLE_DANGEROUS_AI', 'False')
        config.setdefault('DATA_SCRAPE_INTERVAL', '3600')  # In seconds

        return config

    def _get_log_level(self) -> int:
        """
        Get the log level from configuration.

        Returns:
            Log level as an integer
        """
        level_str = self.config.get('LOG_LEVEL', 'INFO').upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_map.get(level_str, logging.INFO)

    async def scan_system(self) -> Dict[str, List[str]]:
        """
        Scan the system for modules and directories.

        Returns:
            Dictionary of modules and directories
        """
        logger.info("Scanning system directories and modules")

        # Find data scraping modules
        scraping_modules = await self.scanner.find_modules('data_scraping', recursive=True)

        # Find model training modules
        training_modules = await self.scanner.find_modules(['prediction', 'betting'], recursive=True)

        # Find web interface modules
        web_modules = await self.scanner.find_modules('web_interface', recursive=True)

        # Find utility modules
        util_modules = await self.scanner.find_modules('utils', recursive=True)

        # Create directory structure if needed
        self._ensure_directories()

        # Combine results
        modules = {
            'data_scraping': scraping_modules,
            'model_training': training_modules,
            'web_interface': web_modules,
            'utils': util_modules
        }

        logger.info(f"System scan completed: found {sum(len(m) for m in modules.values())} modules")

        return modules

    def _ensure_directories(self):
        """Ensure that necessary directories exist."""
        dirs = [
            os.path.join(self.base_dir, '..', 'data', 'scraped_data'),
            os.path.join(self.base_dir, '..', 'data', 'fetched_data'),
            os.path.join(self.base_dir, '..', 'data', 'processed_data'),
            os.path.join(self.base_dir, '..', 'models'),
            os.path.join(self.base_dir, '..', 'logs')
        ]

        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")

    async def fetch_live_data(self) -> Dict[str, str]:
        """
        Fetch live data from various sources.

        Returns:
            Dictionary with paths to fetched data files
        """
        logger.info("Fetching live sports and market data")
        result = {}

        try:
            # Import data scraping modules
            from data_scraping.sports_scraper import scrape_sports_data, process_for_model
            from data_scraping.web_data_fetcher import fetch_market_data

            # Fetch API keys from environment variables
            api_keys = {
                'exchange_rates': self.config.get('EXCHANGE_RATES_API_KEY'),
                'alpha_vantage': self.config.get('ALPHA_VANTAGE_API_KEY')
            }

            # Fetch sports data
            logger.info("Fetching sports data")
            sports_data = await scrape_sports_data(
                sport='football',
                days=7,
                save_dir=os.path.join(self.base_dir, '..', 'data', 'scraped_data')
            )

            # Process sports data for models
            processed_sports = process_for_model(sports_data)

            # Save processed sports data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sports_file = os.path.join(self.base_dir, '..', 'data', 'processed_data',
                                      f"Historical_sports_{timestamp}.csv")

            # Convert processed data to DataFrame and save
            import pandas as pd
            pd.DataFrame(processed_sports).to_csv(sports_file, index=False)
            result['sports_data'] = sports_file

            # Fetch market data
            logger.info("Fetching market data")
            market_data = await fetch_market_data(
                api_keys=api_keys,
                output_dir=os.path.join(self.base_dir, '..', 'data', 'fetched_data')
            )

            # Get the path to processed market data
            market_file = os.path.join(self.base_dir, '..', 'data', 'processed_data',
                                      f"processed_market_data_{timestamp}.csv")
            result['market_data'] = market_file

            # Update component status
            self.components_status["data_scraping"] = True

            logger.info(f"Data fetching completed: {result}")

        except Exception as e:
            logger.error(f"Error fetching live data: {str(e)}")

        return result

    async def process_data(self, data_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Process fetched data for model training.

        Args:
            data_paths: Dictionary with paths to fetched data files

        Returns:
            Dictionary with paths to processed data files
        """
        logger.info("Processing data for model training")
        result = {}

        try:
            # Import data processor module
            from data_scraping.data_processor import process_scraped_data

            # Process data
            if 'sports_data' in data_paths:
                sports_path = data_paths['sports_data']
                market_path = data_paths.get('market_data')

                result = process_scraped_data(
                    sports_data_path=sports_path,
                    market_data_path=market_path,
                    output_dir=os.path.join(self.base_dir, '..', 'data', 'processed_data')
                )

                # Update component status
                self.components_status["data_processing"] = bool(result)

                logger.info(f"Data processing completed: {result}")
            else:
                logger.warning("No sports data available for processing")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")

        return result

    async def train_models(self, data_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Train prediction models with the latest data.

        Args:
            data_paths: Dictionary with paths to processed data files

        Returns:
            Dictionary with paths to trained models
        """
        logger.info("Training prediction models with latest data")
        result = {}

        try:
            # Get model paths
            sports_data_path = data_paths.get('sports_data')
            betting_data_path = data_paths.get('betting_data')

            if not sports_data_path or not betting_data_path:
                logger.warning("Missing processed data for model training")
                return result

            # Import model training modules
            sys.path.append(os.path.join(self.base_dir, 'prediction'))
            sys.path.append(os.path.join(self.base_dir, 'betting'))

            from prediction.sports_predictor import SportsPredictor
            from betting.betting_prediction import BettingPredictor

            # Train sports prediction model
            logger.info("Training sports prediction model")
            sports_model_path = os.path.join(self.base_dir, '..', 'models', 'sports_predictor.pth')

            sports_model = SportsPredictor(input_size=8)
            sports_model.train_from_file(sports_data_path, sports_model_path)
            result['sports_model'] = sports_model_path

            # Train betting prediction model
            logger.info("Training betting prediction model")
            betting_model_path = os.path.join(self.base_dir, '..', 'models', 'betting_predictor.pth')

            # Get feature size from the data
            import pandas as pd
            betting_data = pd.read_csv(betting_data_path)
            # Exclude non-feature columns
            feature_cols = [col for col in betting_data.columns if col not in ['home_team', 'away_team', 'league']]

            betting_model = BettingPredictor(input_size=len(feature_cols))
            betting_model.train_from_file(betting_data_path, betting_model_path)
            result['betting_model'] = betting_model_path

            # Update component status
            self.components_status["model_training"] = True

            logger.info(f"Model training completed: {result}")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")

        return result

    async def start_web_interface(self, model_paths: Dict[str, str]) -> bool:
        """
        Start the web interface for serving predictions.

        Args:
            model_paths: Dictionary with paths to trained models

        Returns:
            True if web interface started successfully, False otherwise
        """
        logger.info("Starting web interface")

        try:
            # Set model paths in environment
            if 'sports_model' in model_paths:
                os.environ['SPORTS_MODEL_PATH'] = model_paths['sports_model']
            if 'betting_model' in model_paths:
                os.environ['BETTING_MODEL_PATH'] = model_paths['betting_model']

            # Start web interface in a subprocess
            import subprocess

            api_file = os.path.join(self.base_dir, 'web_interface', 'app.py')

            # Check if the file exists
            if not os.path.exists(api_file):
                logger.error(f"Web interface file not found: {api_file}")
                return False

            # Get API host and port from config
            host = self.config.get('API_HOST', '127.0.0.1')
            port = self.config.get('API_PORT', '5000')

            # Build command
            cmd = [
                sys.executable,
                api_file,
                '--host', host,
                '--port', port
            ]

            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Check if process started successfully
            if process.poll() is None:
                logger.info(f"Web interface started: http://{host}:{port}")
                self.components_status["web_interface"] = True
                return True
            else:
                stderr = process.stderr.read()
                logger.error(f"Failed to start web interface: {stderr}")
                return False

        except Exception as e:
            logger.error(f"Error starting web interface: {str(e)}")
            return False

    async def run_workflow(self) -> Dict[str, Any]:
        """
        Run the complete workflow from data fetching to serving predictions.

        Returns:
            Dictionary with workflow status and results
        """
        logger.info("Starting Super AI workflow")

        try:
            # Step 1: Scan system
            modules = await self.scan_system()

            # Step 2: Fetch live data
            data_paths = await self.fetch_live_data()

            # Step 3: Process data
            processed_paths = await self.process_data(data_paths)

            # Step 4: Train models
            model_paths = await self.train_models(processed_paths)

            # Step 5: Start web interface
            web_started = await self.start_web_interface(model_paths)

            # Calculate elapsed time
            elapsed = (datetime.now() - self.start_time).total_seconds()

            # Prepare result
            result = {
                "success": all(self.components_status.values()),
                "elapsed_time": elapsed,
                "components_status": self.components_status,
                "data_paths": data_paths,
                "processed_paths": processed_paths,
                "model_paths": model_paths
            }

            logger.info(f"Workflow completed in {elapsed:.2f} seconds")

            return result

        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "components_status": self.components_status
            }

    async def start_automated_workflow(self, interval: int = None):
        """
        Start automated workflow that runs at specified intervals.

        Args:
            interval: Interval in seconds between workflow runs
        """
        if interval is None:
            interval = int(self.config.get('DATA_SCRAPE_INTERVAL', 3600))

        logger.info(f"Starting automated workflow with {interval} seconds interval")

        while True:
            try:
                # Run the workflow
                await self.run_workflow()

                # Wait for the next interval
                logger.info(f"Workflow completed, waiting {interval} seconds for next run")
                await asyncio.sleep(interval)

            except KeyboardInterrupt:
                logger.info("Automated workflow stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in automated workflow: {str(e)}")
                await asyncio.sleep(60)  # Wait a minute before retrying after error

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Super AI Runner')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--interval', type=int, help='Interval in seconds for automated workflow')
    parser.add_argument('--automated', action='store_true', help='Run in automated mode')
    parser.add_argument('--no-web', action='store_true', help='Do not start web interface')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

async def main():
    """Main function."""
    args = parse_arguments()

    # Set debug environment variable if requested
    if args.debug:
        os.environ['LOG_LEVEL'] = 'DEBUG'

    # Create runner
    runner = SuperAIRunner(config_path=args.config)

    # Run workflow
    if args.automated:
        await runner.start_automated_workflow(interval=args.interval)
    else:
        await runner.run_workflow()

if __name__ == "__main__":
    asyncio.run(main())
