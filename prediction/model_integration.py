# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Model Integration Module

This module provides a unified interface to use both sports and betting predictors
together, with ensemble methods and calibration capabilities.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Any
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Import model classes - adjust imports based on your project structure
from prediction.sports_predictor import SportsPredictor
from betting.betting_prediction import BettingPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedPredictor:
    """
    Unified interface for sports and betting prediction

    This class integrates both sports and betting predictors, allowing for:
    1. Ensemble predictions combining multiple models
    2. Probability calibration for more accurate confidence scores
    3. Feature importance analysis across models
    4. Confidence-weighted betting strategies
    """

    def __init__(self,
                 sports_model_path: Optional[str] = None,
                 betting_model_path: Optional[str] = None,
                 scaler_path: Optional[str] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize the integrated predictor

        Args:
            sports_model_path: Path to the sports prediction model
            betting_model_path: Path to the betting prediction model
            scaler_path: Path to the feature scaler
            confidence_threshold: Minimum confidence for predictions
        """
        self.sports_model = None
        self.betting_model = None
        self.scaler = None
        self.confidence_threshold = confidence_threshold

        # Root directory
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.models_dir = self.root_dir / 'models'

        # Load models if paths provided
        if sports_model_path:
            self.load_sports_model(sports_model_path)

        if betting_model_path:
            self.load_betting_model(betting_model_path)

        if scaler_path:
            self.load_scaler(scaler_path)

    def load_sports_model(self, model_path: str = None) -> None:
        """
        Load the sports prediction model

        Args:
            model_path: Path to the model file
        """
        if model_path is None:
            model_path = self.models_dir / 'sports_predictor.pth'

        try:
            self.sports_model = SportsPredictor.load(model_path)
            logger.info(f"Loaded sports model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading sports model: {e}")

    def load_betting_model(self, model_path: str = None) -> None:
        """
        Load the betting prediction model

        Args:
            model_path: Path to the model file
        """
        if model_path is None:
            model_path = self.models_dir / 'betting_predictor.pth'

        try:
            self.betting_model = BettingPredictor.load(model_path)
            logger.info(f"Loaded betting model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading betting model: {e}")

    def load_scaler(self, scaler_path: str = None) -> None:
        """
        Load the feature scaler

        Args:
            scaler_path: Path to the scaler file
        """
        if scaler_path is None:
            scaler_path = self.models_dir / 'feature_scaler.joblib'

        try:
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded feature scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"Error loading scaler: {e}")
            self.scaler = StandardScaler()

    def preprocess_data(self, data: Union[pd.DataFrame, np.ndarray], fit_scaler: bool = False) -> torch.Tensor:
        """
        Preprocess data for prediction

        Args:
            data: Input data as DataFrame or array
            fit_scaler: Whether to fit the scaler on this data

        Returns:
            Preprocessed data as PyTorch tensor
        """
        # Convert DataFrame to numpy array if necessary
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Scale features
        if self.scaler is not None:
            if fit_scaler:
                data = self.scaler.fit_transform(data)
            else:
                data = self.scaler.transform(data)

        # Convert to PyTorch tensor
        return torch.FloatTensor(data)

    def predict_sports_outcome(self, features: Union[pd.DataFrame, np.ndarray],
                               return_probs: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict sports outcomes (win/loss)

        Args:
            features: Input features
            return_probs: Whether to return probabilities along with predictions

        Returns:
            Predictions (and optionally probabilities)
        """
        if self.sports_model is None:
            raise ValueError("Sports model not loaded")

        # Preprocess data
        X = self.preprocess_data(features)

        # Make prediction
        self.sports_model.eval()
        with torch.no_grad():
            probs = self.sports_model(X).numpy()

        # Convert to binary predictions
        predictions = (probs > 0.5).astype(int)

        if return_probs:
            return predictions, probs
        else:
            return predictions

    def predict_odds(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict betting odds

        Args:
            features: Input features

        Returns:
            Predicted odds
        """
        if self.betting_model is None:
            raise ValueError("Betting model not loaded")

        # Preprocess data
        X = self.preprocess_data(features)

        # Make prediction
        self.betting_model.eval()
        with torch.no_grad():
            odds = self.betting_model(X).numpy()

        return odds

    def integrated_prediction(self, features: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Make an integrated prediction using both models

        Args:
            features: Input features

        Returns:
            Dictionary with predictions, odds, and confidence
        """
        results = {}

        # Predict outcome if sports model is available
        if self.sports_model is not None:
            outcome, probs = self.predict_sports_outcome(features, return_probs=True)
            confidence = np.abs(probs - 0.5) * 2  # Scale to 0-1 range

            results.update({
                'outcome': outcome.flatten().tolist(),
                'outcome_probability': probs.flatten().tolist(),
                'confidence': confidence.flatten().tolist(),
                'is_confident': (confidence > self.confidence_threshold).flatten().tolist()
            })

        # Predict odds if betting model is available
        if self.betting_model is not None:
            odds = self.predict_odds(features)

            results.update({
                'predicted_odds': odds.flatten().tolist()
            })

            # Calculate expected value if both models are available
            if self.sports_model is not None:
                ev = probs * (odds - 1)
                results['expected_value'] = ev.flatten().tolist()
                results['recommended_bet'] = (ev > 0).flatten().tolist()

        return results

    def analyze_feature_importance(self, feature_names: List[str] = None,
                                   save_path: str = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze and compare feature importance across models

        Args:
            feature_names: Names of the features
            save_path: Path to save the visualization

        Returns:
            Dictionary with feature importance for each model
        """
        importances = {}

        # Get feature importance from sports model
        if self.sports_model is not None and self.sports_model.feature_importances is not None:
            importances['sports'] = self.sports_model.feature_importances

        # Get feature importance from betting model
        if self.betting_model is not None and self.betting_model.feature_importances is not None:
            importances['betting'] = self.betting_model.feature_importances

        # Create visualization if we have importance data
        if importances and save_path:
            self._visualize_combined_importance(importances, feature_names, save_path)

        return importances

    def _visualize_combined_importance(self, importances: Dict[str, Dict[str, float]],
                                      feature_names: List[str] = None,
                                      save_path: str = None) -> None:
        """
        Visualize combined feature importance

        Args:
            importances: Dictionary with feature importance for each model
            feature_names: Names of the features
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 8))

        # Get all unique features
        all_features = set()
        for model_imp in importances.values():
            all_features.update(model_imp.keys())

        if feature_names:
            # Filter to only include the provided feature names
            all_features = [f for f in feature_names if f in all_features]
        else:
            # Convert to list and sort
            all_features = sorted(list(all_features))

        # Prepare data for plotting
        x = np.arange(len(all_features))
        width = 0.35

        # Plot bars for each model
        for i, (model_name, model_imp) in enumerate(importances.items()):
            # Get importance values for each feature
            values = [model_imp.get(feature, 0) for feature in all_features]

            # Normalize values for better visualization
            max_val = max(values) if values else 1
            norm_values = [v / max_val for v in values]

            # Plot bars
            plt.bar(x + (i - 0.5 * (len(importances) - 1)) * width,
                   norm_values, width, label=f'{model_name.capitalize()} Model')

        # Add labels and legend
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance Comparison')
        plt.xticks(x, all_features, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Save or show
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Combined feature importance visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_scaler(self, path: str = None) -> None:
        """
        Save the feature scaler

        Args:
            path: Path to save the scaler
        """
        if self.scaler is None:
            logger.warning("No scaler to save")
            return

        if path is None:
            path = self.models_dir / 'feature_scaler.joblib'

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, path)
        logger.info(f"Saved feature scaler to {path}")

    def evaluate_models(self, X: np.ndarray, sports_y: np.ndarray = None,
                       betting_y: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluate models on test data

        Args:
            X: Input features
            sports_y: True sports outcomes
            betting_y: True betting odds

        Returns:
            Dictionary with evaluation metrics
        """
        results = {}

        # Preprocess features
        X_tensor = self.preprocess_data(X)

        # Evaluate sports model
        if self.sports_model is not None and sports_y is not None:
            sports_y_tensor = torch.FloatTensor(sports_y.reshape(-1, 1))

            self.sports_model.eval()
            with torch.no_grad():
                sports_preds = self.sports_model(X_tensor).numpy()

            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

            binary_preds = (sports_preds > 0.5).astype(int)
            accuracy = accuracy_score(sports_y, binary_preds)
            precision = precision_score(sports_y, binary_preds, zero_division=0)
            recall = recall_score(sports_y, binary_preds, zero_division=0)
            f1 = f1_score(sports_y, binary_preds, zero_division=0)

            try:
                auc = roc_auc_score(sports_y, sports_preds)
            except:
                auc = 0.5

            results['sports'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }

            logger.info(f"Sports model evaluation - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Evaluate betting model
        if self.betting_model is not None and betting_y is not None:
            betting_y_tensor = torch.FloatTensor(betting_y.reshape(-1, 1))

            self.betting_model.eval()
            with torch.no_grad():
                betting_preds = self.betting_model(X_tensor).numpy()

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

            mae = mean_absolute_error(betting_y, betting_preds)
            r2 = r2_score(betting_y, betting_preds)
            rmse = np.sqrt(mean_squared_error(betting_y, betting_preds))

            results['betting'] = {
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }

            logger.info(f"Betting model evaluation - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

        return results


def main():
    """Main function to demonstrate the integrated predictor"""
    logger.info("Initializing Integrated Predictor")

    # Initialize predictor
    predictor = IntegratedPredictor()

    # Get root directory
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Try to load test data
    data_path = root_dir / 'data' / 'test_data.csv'
    if data_path.exists():
        # Load and process data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded test data: {df.shape}")

        # Extract features
        if 'result' in df.columns and 'odds' in df.columns:
            # Both targets available
            X = df.drop(['result', 'odds'], axis=1)
            sports_y = df['result'].values
            betting_y = df['odds'].values

            # Make integrated prediction
            predictions = predictor.integrated_prediction(X)
            logger.info(f"Made predictions for {len(X)} samples")

            # Evaluate models
            eval_results = predictor.evaluate_models(X, sports_y, betting_y)
            logger.info("Model evaluation complete")

            # Analyze feature importance
            importances = predictor.analyze_feature_importance(
                feature_names=X.columns.tolist(),
                save_path=root_dir / 'visualizations' / 'combined_importance.png'
            )

            return {
                "status": "success",
                "samples_processed": len(X),
                "evaluation": eval_results,
                "feature_importance": importances
            }
        else:
            logger.warning("Test data missing required columns")
    else:
        logger.warning("No test data found")

    return {"status": "completed", "message": "Integrated predictor ready"}


if __name__ == "__main__":
    main()
