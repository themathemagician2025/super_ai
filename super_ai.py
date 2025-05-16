# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Main Super AI class that integrates all components into a cohesive system.

This class serves as the primary interface for interacting with the Super AI system,
providing access to prediction, data loading, and processing capabilities.
"""

import logging
from typing import Dict, List, Optional, Union, Any

from super_ai.core.brain.neural_network import create_mlp
from super_ai.data.loaders.forex_loader import ForexDataLoader
from super_ai.prediction.models.sports_prediction import SportsPredictionModel
from super_ai.utils.logging.logger import setup_logger
from super_ai.utils.config.settings import Config

logger = setup_logger('super_ai.main')

class SuperAI:
    """Main class that integrates all components of the Super AI system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Super AI system.

        Args:
            config_path: Path to configuration file. If None, default configuration is used.
        """
        logger.info("Initializing SuperAI system")
        self.config = Config(config_path)
        self._data_loaders = {}
        self._prediction_models = {}
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize all components of the system.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        if self._initialized:
            logger.info("SuperAI system already initialized")
            return True

        try:
            logger.info("Loading components...")
            # Initialize core components as needed
            self._initialized = True
            logger.info("SuperAI system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SuperAI system: {str(e)}")
            return False

    def get_data_loader(self, data_type: str) -> Any:
        """
        Get a data loader for the specified data type.

        Args:
            data_type: Type of data to load (e.g., 'forex', 'sports')

        Returns:
            Data loader instance
        """
        if data_type not in self._data_loaders:
            if data_type == 'forex':
                self._data_loaders[data_type] = ForexDataLoader(self.config)
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None

        return self._data_loaders[data_type]

    def get_prediction_model(self, model_type: str) -> Any:
        """
        Get a prediction model for the specified type.

        Args:
            model_type: Type of prediction model (e.g., 'sports', 'forex')

        Returns:
            Prediction model instance
        """
        if model_type not in self._prediction_models:
            if model_type == 'sports':
                self._prediction_models[model_type] = SportsPredictionModel(self.config)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None

        return self._prediction_models[model_type]

    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process natural language input and return appropriate response.

        Args:
            input_text: Natural language input from user

        Returns:
            Dictionary containing response data
        """
        logger.info(f"Processing input: {input_text}")

        # Simple keyword-based routing
        if 'forex' in input_text.lower():
            loader = self.get_data_loader('forex')
            if loader:
                symbol = 'EURUSD'  # Default symbol, would be extracted from input in real implementation
                data = loader.load_historical_data(symbol)
                return {
                    'type': 'forex_data',
                    'data': data.to_dict() if hasattr(data, 'to_dict') else str(data),
                    'symbol': symbol
                }

        elif 'sports' in input_text.lower() or 'prediction' in input_text.lower():
            model = self.get_prediction_model('sports')
            if model:
                teams = ['Team A', 'Team B']  # Would be extracted from input in real implementation
                prediction = model.predict(teams[0], teams[1])
                return {
                    'type': 'sports_prediction',
                    'prediction': prediction,
                    'teams': teams
                }

        # Default response
        return {
            'type': 'text',
            'message': "I'm not sure how to process that input. Try asking about forex data or sports predictions."
        }

    def predict_forex(self, symbol: str, timeframe: str = '1d', periods: int = 10) -> Dict[str, Any]:
        """
        Generate forex price predictions.

        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            timeframe: Timeframe for prediction (e.g., '1d', '4h')
            periods: Number of periods to predict

        Returns:
            Dictionary with prediction data
        """
        logger.info(f"Generating forex prediction for {symbol}, timeframe {timeframe}, periods {periods}")

        # Get historical data
        loader = self.get_data_loader('forex')
        if not loader:
            return {'error': 'Forex data loader not available'}

        historical_data = loader.load_historical_data(symbol, timeframe)

        # In a real implementation, this would use a proper prediction model
        # For demo purposes, we'll return some mock prediction
        import numpy as np

        last_price = 1.1000  # This would come from historical_data in reality
        predictions = []

        for i in range(periods):
            # Simple random walk for demo purposes
            next_price = last_price * (1 + np.random.normal(0, 0.005))
            predictions.append({
                'period': i + 1,
                'price': round(next_price, 5),
                'confidence': round(0.95 - (i * 0.05), 2)  # Confidence decreases with time
            })
            last_price = next_price

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'predictions': predictions
        }

    def shutdown(self) -> None:
        """Clean up resources and shut down the system."""
        logger.info("Shutting down SuperAI system")
        # Close any open resources, save states, etc.
        self._initialized = False
