# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Sports Prediction Module

This module provides prediction functionality for sports events.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup logger
logger = logging.getLogger(__name__)

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "sports")

class SportsPredictionModel:
    """Class for predicting sports outcomes."""

    def __init__(self, model_type: str = "random_forest", model_dir: Optional[str] = None):
        """
        Initialize the sports prediction model.

        Args:
            model_type: Type of model to use ("random_forest" or "gradient_boosting")
            model_dir: Directory to load/save models from/to (default: project_root/models/sports)
        """
        self.model_type = model_type
        self.model_dir = model_dir or MODEL_DIR
        self.model = None

        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            logger.info(f"Created model directory: {self.model_dir}")

    def _initialize_model(self) -> None:
        """Initialize a new model based on the specified type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_model(self, sport: str, league: str) -> bool:
        """
        Load a saved model for the specified sport and league.

        Args:
            sport: Sport type (e.g., "soccer", "basketball")
            league: League name (e.g., "premier_league", "nba")

        Returns:
            True if model was loaded successfully, False otherwise
        """
        model_path = os.path.join(self.model_dir, f"{sport}_{league}_{self.model_type}.joblib")

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded model from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                self._initialize_model()
                return False
        else:
            logger.warning(f"Model file not found: {model_path}. Initializing new model.")
            self._initialize_model()
            return False

    def save_model(self, sport: str, league: str) -> bool:
        """
        Save the current model.

        Args:
            sport: Sport type (e.g., "soccer", "basketball")
            league: League name (e.g., "premier_league", "nba")

        Returns:
            True if model was saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No model to save.")
            return False

        model_path = os.path.join(self.model_dir, f"{sport}_{league}_{self.model_type}.joblib")

        try:
            joblib.dump(self.model, model_path)
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model on the provided data.

        Args:
            X: Features DataFrame
            y: Target variable Series

        Returns:
            Dictionary of training metrics
        """
        if self.model is None:
            self._initialize_model()

        # Train the model
        self.model.fit(X, y)

        # Get predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted")
        }

        logger.info(f"Model trained. Metrics: {metrics}")
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        if self.model is None:
            logger.warning("Model not loaded or trained. Initializing new model.")
            self._initialize_model()
            return np.array([])

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability estimates for each class.

        Args:
            X: Features DataFrame

        Returns:
            Array of class probabilities
        """
        if self.model is None:
            logger.warning("Model not loaded or trained. Initializing new model.")
            self._initialize_model()
            return np.array([])

        return self.model.predict_proba(X)

def get_game_features(game_id: str) -> pd.DataFrame:
    """
    Retrieve features for a specific game.

    Args:
        game_id: Unique identifier for the game

    Returns:
        DataFrame with game features
    """
    # This would typically involve database lookups or API calls
    # For demonstration, we'll create dummy data

    # Create a dummy feature dataframe
    features = pd.DataFrame({
        'team_a_rating': [85.5],
        'team_b_rating': [82.3],
        'team_a_home': [1],  # 1 if team A is home, 0 otherwise
        'team_a_wins_last_5': [3],
        'team_b_wins_last_5': [2],
        'team_a_goals_last_5': [10],
        'team_b_goals_last_5': [7],
        'days_since_team_a_last_game': [4],
        'days_since_team_b_last_game': [6]
    })

    return features

def predict(game_id: str, sport: str = "soccer", league: str = "premier_league") -> Dict[str, Any]:
    """
    Make a prediction for a specific game.

    Args:
        game_id: Unique identifier for the game
        sport: Sport type
        league: League name

    Returns:
        Dictionary with prediction results
    """
    # Get features for the game
    features = get_game_features(game_id)

    if features.empty:
        return {"error": "Could not retrieve game features"}

    # Initialize and load model
    model = SportsPredictionModel()
    model.load_model(sport, league)

    # Make prediction
    prediction = model.predict(features)[0]  # Get first (and only) prediction
    probabilities = model.predict_proba(features)[0]  # Get probabilities

    # Map outcome to a readable result
    outcomes = ["team_a_win", "draw", "team_b_win"]
    result = outcomes[prediction]

    # Create response
    response = {
        "game_id": game_id,
        "sport": sport,
        "league": league,
        "prediction": result,
        "confidence": float(max(probabilities)),
        "probabilities": {
            outcomes[i]: float(prob) for i, prob in enumerate(probabilities)
        },
        "timestamp": datetime.now().isoformat()
    }

    return response
