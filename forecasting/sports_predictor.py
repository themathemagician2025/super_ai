# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Sports Prediction Module with Self-Learning Capabilities

This module implements a sophisticated sports prediction system that:
1. Analyzes historical match data
2. Generates score predictions
3. Tracks prediction accuracy
4. Self-evolves to improve future predictions
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import hashlib
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class SportsPredictor:
    """Primary sports prediction engine with self-learning capabilities"""

    def __init__(self, config: Dict = None):
        """
        Initialize the sports prediction system

        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}

        # Data paths
        self.data_path = Path(self.config.get('data_path', 'data/matches'))
        self.models_path = Path(self.config.get('models_path', 'models/sports'))
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)

        # Model parameters
        self.model_type = self.config.get('model_type', 'ensemble')
        self.model_version = self.config.get('model_version', 'v0.1.0')
        self.feature_set = self.config.get('feature_set', ['form', 'h2h', 'table_position', 'goals'])

        # Performance tracking
        self.prediction_history = []
        self.error_metrics = {}
        self.error_threshold = self.config.get('error_threshold', 2.0)

        # Team and league data
        self.teams_data = {}
        self.league_table = {}

        # Model instances
        self.home_model = None
        self.away_model = None
        self.scaler = StandardScaler()

        # Initialize and load models
        self._load_or_create_models()

        logger.info(f"Initialized SportsPredictor with {self.model_type} model (v{self.model_version})")

    def _load_or_create_models(self):
        """Load existing models or create new ones if none exist"""
        model_file = self.models_path / f"sports_predictor_{self.model_version}.pkl"
        scaler_file = self.models_path / f"scaler_{self.model_version}.pkl"

        if model_file.exists() and scaler_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    models = pickle.load(f)
                    self.home_model = models['home']
                    self.away_model = models['away']

                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)

                logger.info(f"Loaded existing models from {model_file}")
                return
            except Exception as e:
                logger.error(f"Failed to load models: {str(e)}")
                # Continue to create new models

        # Create new models
        if self.model_type == 'random_forest':
            self.home_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.away_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.home_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.away_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # ensemble default
            self.home_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.away_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

        logger.info(f"Created new {self.model_type} models")

    def save_models(self):
        """Save models to disk"""
        model_file = self.models_path / f"sports_predictor_{self.model_version}.pkl"
        scaler_file = self.models_path / f"scaler_{self.model_version}.pkl"

        try:
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'home': self.home_model,
                    'away': self.away_model
                }, f)

            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)

            logger.info(f"Saved models to {model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
            return False

    def _load_teams_data(self):
        """Load team statistics and form data"""
        teams_file = self.data_path / "teams_stats.csv"
        if teams_file.exists():
            try:
                teams_df = pd.read_csv(teams_file)
                self.teams_data = teams_df.set_index('team_id').to_dict(orient='index')
                logger.info(f"Loaded data for {len(self.teams_data)} teams")
            except Exception as e:
                logger.error(f"Failed to load teams data: {str(e)}")
                self.teams_data = {}

    def _load_league_table(self):
        """Load current league table data"""
        table_file = self.data_path / "league_table.csv"
        if table_file.exists():
            try:
                table_df = pd.read_csv(table_file)
                self.league_table = table_df.set_index('team_id').to_dict(orient='index')
                logger.info(f"Loaded league table with {len(self.league_table)} teams")
            except Exception as e:
                logger.error(f"Failed to load league table: {str(e)}")
                self.league_table = {}

    def _load_historical_matches(self) -> pd.DataFrame:
        """Load historical match data for training"""
        matches_file = self.data_path / "historical_matches.csv"
        if matches_file.exists():
            try:
                return pd.read_csv(matches_file)
            except Exception as e:
                logger.error(f"Failed to load historical matches: {str(e)}")
                return pd.DataFrame()
        else:
            logger.warning(f"No historical matches file found at {matches_file}")
            return pd.DataFrame()

    def fetch_match_data(self, match_id: str) -> Dict:
        """Retrieve data for a specific match"""
        # In a real implementation, this might query an API or database
        # Here we'll simulate by checking local files
        match_file = self.data_path / f"match_{match_id}.json"

        if match_file.exists():
            try:
                with open(match_file, 'r') as f:
                    match_data = json.load(f)
                logger.info(f"Loaded data for match {match_id}")
                return match_data
            except Exception as e:
                logger.error(f"Failed to load match data for {match_id}: {str(e)}")
                return {}
        else:
            logger.warning(f"No data found for match {match_id}")
            return {}

    def extract_features(self, team_a: str, team_b: str, is_home_away: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for prediction

        Args:
            team_a: First team identifier
            team_b: Second team identifier
            is_home_away: Whether team_a is home and team_b is away

        Returns:
            Tuple of feature arrays for home and away teams
        """
        if not self.teams_data:
            self._load_teams_data()

        if not self.league_table:
            self._load_league_table()

        # Get team stats or use defaults
        team_a_stats = self.teams_data.get(team_a, {})
        team_b_stats = self.teams_data.get(team_b, {})

        # Basic features (expand in real implementation)
        team_a_features = [
            team_a_stats.get('form_rating', 50),
            team_a_stats.get('attack_strength', 50),
            team_a_stats.get('defense_strength', 50),
            team_a_stats.get('avg_goals_scored', 1.5),
            team_a_stats.get('avg_goals_conceded', 1.2)
        ]

        team_b_features = [
            team_b_stats.get('form_rating', 50),
            team_b_stats.get('attack_strength', 50),
            team_b_stats.get('defense_strength', 50),
            team_b_stats.get('avg_goals_scored', 1.5),
            team_b_stats.get('avg_goals_conceded', 1.2)
        ]

        # Add positional and comparative features
        if self.league_table:
            team_a_pos = self.league_table.get(team_a, {}).get('position', 10)
            team_b_pos = self.league_table.get(team_b, {}).get('position', 10)

            team_a_features.extend([
                team_a_pos,
                team_a_pos - team_b_pos
            ])

            team_b_features.extend([
                team_b_pos,
                team_b_pos - team_a_pos
            ])
        else:
            team_a_features.extend([10, 0])
            team_b_features.extend([10, 0])

        # Head-to-head features
        # In a real implementation, would analyze historical matchups
        team_a_features.append(1.5)  # Average goals against this opponent
        team_b_features.append(1.2)  # Average goals against this opponent

        # Home/Away advantage
        if is_home_away:
            team_a_features.append(1)  # Home team
            team_b_features.append(0)  # Away team
        else:
            team_a_features.append(0)  # Neutral or away
            team_b_features.append(0)  # Neutral or away

        # Preprocessing
        if self.home_model is not None and hasattr(self.home_model, 'feature_importances_'):
            # Use model to determine most important features
            # (simplified implementation)
            pass

        return np.array(team_a_features).reshape(1, -1), np.array(team_b_features).reshape(1, -1)

    def train_models(self, force: bool = False) -> bool:
        """
        Train prediction models on historical data

        Args:
            force: Whether to force retraining even if models exist

        Returns:
            Success status
        """
        if self.home_model is not None and self.away_model is not None and not force:
            # Use existing models unless forced
            logger.info("Using existing trained models")
            return True

        # Load historical match data
        matches_df = self._load_historical_matches()

        if matches_df.empty:
            logger.error("No historical match data available for training")
            return False

        # Prepare features and targets
        features = []
        home_targets = []
        away_targets = []

        for _, match in matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']

            home_feats, away_feats = self.extract_features(home_team, away_team)

            # Combine features
            combined_feats = np.concatenate([home_feats.flatten(), away_feats.flatten()])
            features.append(combined_feats)

            home_targets.append(home_score)
            away_targets.append(away_score)

        # Convert to arrays
        X = np.array(features)
        y_home = np.array(home_targets)
        y_away = np.array(away_targets)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
            X_scaled, y_home, y_away, test_size=0.2, random_state=42
        )

        # Train models
        try:
            self.home_model.fit(X_train, y_home_train)
            self.away_model.fit(X_train, y_away_train)

            # Evaluate
            home_preds = self.home_model.predict(X_test)
            away_preds = self.away_model.predict(X_test)

            home_rmse = np.sqrt(mean_squared_error(y_home_test, home_preds))
            away_rmse = np.sqrt(mean_squared_error(y_away_test, away_preds))

            logger.info(f"Models trained - Home RMSE: {home_rmse:.2f}, Away RMSE: {away_rmse:.2f}")

            # Save trained models
            self.save_models()

            # Update version if significant improvement
            if force and (home_rmse < 1.0 and away_rmse < 1.0):
                # Increment version number
                v_parts = self.model_version.split('.')
                v_parts[-1] = str(int(v_parts[-1]) + 1)
                self.model_version = '.'.join(v_parts)
                logger.info(f"Updated model version to {self.model_version}")

            return True

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return False

    def predict_score(self, team_a: str, team_b: str, is_home_away: bool = True) -> Tuple[int, int]:
        """
        Predict the final score of a match between two teams

        Args:
            team_a: First team identifier (home if is_home_away is True)
            team_b: Second team identifier (away if is_home_away is True)
            is_home_away: Whether this is a home/away match or neutral venue

        Returns:
            Tuple of (team_a_score, team_b_score)
        """
        if self.home_model is None or self.away_model is None:
            success = self.train_models()
            if not success:
                logger.warning("Using fallback prediction due to missing models")
                return (1, 1)  # Fallback prediction

        # Extract features
        home_features, away_features = self.extract_features(team_a, team_b, is_home_away)

        # Combine features
        features = np.concatenate([home_features.flatten(), away_features.flatten()]).reshape(1, -1)

        # Scale features
        scaled_features = self.scaler.transform(features)

        # Make predictions
        team_a_score_float = self.home_model.predict(scaled_features)[0]
        team_b_score_float = self.away_model.predict(scaled_features)[0]

        # Round to integers
        team_a_score = max(0, round(team_a_score_float))
        team_b_score = max(0, round(team_b_score_float))

        # Log prediction
        logger.info(f"Predicted score: {team_a} {team_a_score} - {team_b_score} {team_b}")

        # Add to prediction history
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'team_a': team_a,
            'team_b': team_b,
            'predicted_score': f"{team_a_score}-{team_b_score}",
            'model_version': self.model_version
        }
        self.prediction_history.append(prediction_entry)

        return (team_a_score, team_b_score)

    def compare_prediction(self, match_id: str, actual_score: Tuple[int, int]) -> Dict:
        """
        Compare prediction to actual result and calculate error

        Args:
            match_id: Match identifier
            actual_score: Tuple of (team_a_score, team_b_score)

        Returns:
            Dictionary with error metrics
        """
        # Find prediction in history (assuming the most recent for this match)
        match_data = self.fetch_match_data(match_id)
        if not match_data:
            logger.error(f"No match data found for {match_id}")
            return {'error': 'Match not found'}

        team_a = match_data.get('team_a')
        team_b = match_data.get('team_b')

        # Find prediction in history
        prediction = None
        for pred in reversed(self.prediction_history):
            if pred.get('team_a') == team_a and pred.get('team_b') == team_b:
                prediction = pred
                break

        if not prediction:
            logger.error(f"No prediction found for match {match_id}")
            return {'error': 'Prediction not found'}

        # Parse predicted score
        try:
            pred_score_str = prediction.get('predicted_score', '0-0')
            pred_a, pred_b = map(int, pred_score_str.split('-'))

            # Extract actual score
            actual_a, actual_b = actual_score

            # Calculate error metrics
            goal_diff_error = abs((pred_a - pred_b) - (actual_a - actual_b))
            abs_error = abs(pred_a - actual_a) + abs(pred_b - actual_b)
            outcome_correct = ((pred_a > pred_b and actual_a > actual_b) or
                              (pred_a < pred_b and actual_a < actual_b) or
                              (pred_a == pred_b and actual_a == actual_b))

            # Record results
            result = {
                'match_id': match_id,
                'team_a': team_a,
                'team_b': team_b,
                'predicted_score': f"{pred_a}-{pred_b}",
                'actual_score': f"{actual_a}-{actual_b}",
                'goal_diff_error': goal_diff_error,
                'absolute_error': abs_error,
                'outcome_correct': outcome_correct,
                'model_version': prediction.get('model_version', 'unknown')
            }

            # Log to CSV
            self._log_prediction_result(result)

            # Check if model needs retraining
            if abs_error > self.error_threshold:
                logger.info(f"Large prediction error ({abs_error}) - flagging for retraining")
                result['needs_retraining'] = True
            else:
                result['needs_retraining'] = False

            return result

        except Exception as e:
            logger.error(f"Error comparing prediction: {str(e)}")
            return {'error': str(e)}

    def _log_prediction_result(self, result: Dict) -> bool:
        """Log prediction result to CSV file"""
        log_file = self.data_path / "prediction_results.csv"

        # Prepare row data
        row = {
            'match_id': result.get('match_id'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'team_a': result.get('team_a'),
            'team_b': result.get('team_b'),
            'predicted_score': result.get('predicted_score'),
            'actual_score': result.get('actual_score'),
            'error': result.get('absolute_error'),
            'model_ver': result.get('model_version')
        }

        try:
            # Create dataframe
            result_df = pd.DataFrame([row])

            # Check if file exists
            if log_file.exists():
                # Append to existing file
                result_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                # Create new file with header
                result_df.to_csv(log_file, index=False)

            logger.info(f"Logged prediction result to {log_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to log prediction result: {str(e)}")
            return False

    def self_evolve(self, force: bool = False):
        """
        Analyze performance and evolve the model if needed

        Args:
            force: Whether to force evolution regardless of performance

        Returns:
            Whether the model was evolved
        """
        # Load recent prediction results
        log_file = self.data_path / "prediction_results.csv"
        if not log_file.exists():
            logger.warning("No prediction results found for self-evolution")
            return False

        try:
            results_df = pd.read_csv(log_file)

            # Filter to current model version
            current_version_results = results_df[results_df['model_ver'] == self.model_version]

            if len(current_version_results) < 10 and not force:
                logger.info(f"Insufficient data for model evolution ({len(current_version_results)} results)")
                return False

            # Calculate performance metrics
            avg_error = current_version_results['error'].mean()
            outcome_correct = (current_version_results['predicted_score'].str.split('-', expand=True)
                              .astype(int)
                              .apply(lambda x: x[0] > x[1], axis=1))

            actual_outcome = (current_version_results['actual_score'].str.split('-', expand=True)
                             .astype(int)
                             .apply(lambda x: x[0] > x[1], axis=1))

            outcome_accuracy = (outcome_correct == actual_outcome).mean()

            logger.info(f"Current model performance - Avg error: {avg_error:.2f}, Outcome accuracy: {outcome_accuracy:.2f}")

            # Decide whether to evolve
            should_evolve = force or avg_error > self.error_threshold or outcome_accuracy < 0.6

            if should_evolve:
                # Backup current model
                old_model_file = self.models_path / f"sports_predictor_{self.model_version}.pkl"
                backup_file = self.models_path / f"sports_predictor_{self.model_version}_backup.pkl"

                if old_model_file.exists():
                    import shutil
                    shutil.copy(old_model_file, backup_file)
                    logger.info(f"Backed up current model to {backup_file}")

                # Evolve model by retraining with updated data
                success = self.train_models(force=True)

                if success:
                    # Update version
                    v_parts = self.model_version.split('.')
                    v_parts[-1] = str(int(v_parts[-1]) + 1)
                    self.model_version = '.'.join(v_parts)
                    logger.info(f"Self-evolved to model version {self.model_version}")

                    # Save the evolved model
                    self.save_models()
                    return True
                else:
                    logger.error("Self-evolution failed during model training")
                    return False
            else:
                logger.info("Model performance satisfactory, no evolution needed")
                return False

        except Exception as e:
            logger.error(f"Self-evolution error: {str(e)}")
            return False

    def add_feature(self, feature_name: str, feature_func: callable) -> bool:
        """
        Add a new feature to the model's feature set

        Args:
            feature_name: Name of the new feature
            feature_func: Function that generates the feature value

        Returns:
            Success status
        """
        if feature_name in self.feature_set:
            logger.warning(f"Feature {feature_name} already exists")
            return False

        # Add to feature set
        self.feature_set.append(feature_name)

        # In a full implementation, would modify the feature extraction process
        # to include the new feature and its generation function

        logger.info(f"Added new feature: {feature_name}")
        return True

    def update_model_hyperparameters(self, params: Dict) -> bool:
        """
        Update model hyperparameters

        Args:
            params: Dictionary of hyperparameters to update

        Returns:
            Success status
        """
        try:
            # Create new models with updated parameters
            if self.model_type == 'random_forest':
                base_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                }
                # Update with provided params
                base_params.update(params)

                self.home_model = RandomForestRegressor(**base_params)
                self.away_model = RandomForestRegressor(**base_params)

            elif self.model_type == 'gradient_boosting':
                base_params = {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
                # Update with provided params
                base_params.update(params)

                self.home_model = GradientBoostingRegressor(**base_params)
                self.away_model = GradientBoostingRegressor(**base_params)

            # Retrain with new hyperparameters
            success = self.train_models(force=True)

            if success:
                # Update version
                v_parts = self.model_version.split('.')
                v_parts[1] = str(int(v_parts[1]) + 1)  # Increment minor version
                v_parts[2] = '0'  # Reset patch version
                self.model_version = '.'.join(v_parts)

                logger.info(f"Updated hyperparameters and trained new model version {self.model_version}")
                return True
            else:
                logger.error("Failed to train models with new hyperparameters")
                return False

        except Exception as e:
            logger.error(f"Error updating hyperparameters: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create predictor
    predictor = SportsPredictor()

    # Make a prediction
    team_a = "ARS"
    team_b = "CHE"
    score = predictor.predict_score(team_a, team_b)

    print(f"Predicted score: {team_a} {score[0]} - {score[1]} {team_b}")

    # Simulate match completion and self-evolution
    actual_score = (3, 2)
    result = predictor.compare_prediction("4923", actual_score)

    if result.get('needs_retraining', False):
        predictor.self_evolve()
