# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from pathlib import Path
import numpy as np
from neat import nn, population
from models.lstm import LSTMModel
from utils.data_loader import load_historical_data

logger = logging.getLogger(__name__)

class BettingManager:
    def __init__(self, config):
        """Initialize betting prediction system"""
        self.config = config
        self.data_path = Path("data/betting")
        self.models = {}
        self._initialize_models()
        logger.info("Betting manager initialized")

    def _initialize_models(self):
        """Set up prediction models for each sport"""
        try:
            # Initialize NEAT population
            self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                         'config/neat_config.txt')
            self.population = neat.Population(self.neat_config)

            # Initialize LSTM model
            self.lstm = LSTMModel(self.config.get("lstm_params"))

            logger.info("Betting models initialized")
        except Exception as e:
            logger.error(f"Model initialization failed: {str(e)}")
            raise

    def predict(self, sport):
        """Generate predictions for specified sport"""
        try:
            # Load historical data
            data = load_historical_data(self.data_path / f"{sport}_history.csv")

            # Generate predictions using multiple models
            neat_pred = self._neat_prediction(data)
            lstm_pred = self._lstm_prediction(data)

            # Combine predictions
            final_pred = self._ensemble_prediction([neat_pred, lstm_pred])

            logger.info(f"Generated prediction for {sport}")
            return final_pred

        except Exception as e:
            logger.error(f"Prediction failed for {sport}: {str(e)}")
            raise

    def _neat_prediction(self, data):
        """Generate predictions using NEAT."""
        winner = self.population.run(self._eval_genome, 50)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.neat_config)
        return winner_net.activate(data)

    def _lstm_prediction(self, data):
        """Generate predictions using LSTM."""
        return self.lstm.predict(data)

    def _ensemble_prediction(self, predictions):
        """Combine predictions using weighted ensemble."""
        return sum(predictions) / len(predictions)
