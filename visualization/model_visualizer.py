# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Model Visualization Tool

Provides interactive visualizations for sports and betting predictors,
including performance metrics, feature importance, and prediction analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import relevant modules
from prediction.sports_predictor import SportsPredictor
from betting.betting_prediction import BettingPredictor
from prediction.model_integration import IntegratedPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Tool for visualizing model performance and predictions"""

    def __init__(self, integrated_predictor: Optional[IntegratedPredictor] = None,
                sports_model_path: Optional[str] = None,
                betting_model_path: Optional[str] = None,
                output_dir: Optional[str] = None):
        """
        Initialize the model visualizer

        Args:
            integrated_predictor: An integrated predictor instance (optional)
            sports_model_path: Path to the sports prediction model
            betting_model_path: Path to the betting prediction model
            output_dir: Directory to save visualizations
        """
        # Set up paths
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.root_dir / 'visualizations'

        os.makedirs(self.output_dir, exist_ok=True)

        # Set up predictor
        if integrated_predictor:
            self.predictor = integrated_predictor
        else:
            self.predictor = IntegratedPredictor(
                sports_model_path=sports_model_path,
                betting_model_path=betting_model_path
            )

    def visualize_feature_importance(self, feature_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Visualize feature importance for both models

        Args:
            feature_names: Names of features
            save_path: Path to save the visualization
        """
        if save_path is None:
            save_path = self.output_dir / 'feature_importance.png'

        # Get feature importance from predictor
        importances = self.predictor.analyze_feature_importance(
            feature_names=feature_names,
            save_path=str(save_path)
        )

        if not importances:
            logger.warning("No feature importance data available")
            return

        logger.info(f"Feature importance visualization saved to {save_path}")

    def visualize_prediction_confidence(self, test_data: pd.DataFrame,
                                      outcome_col: str = 'result',
                                      save_path: Optional[str] = None) -> None:
        """
        Visualize prediction confidence vs. actual outcomes

        Args:
            test_data: Test data with features and outcomes
            outcome_col: Name of the outcome column
            save_path: Path to save the visualization
        """
        # Check if sports model is available
        if self.predictor.sports_model is None:
            logger.warning("Sports model not available for confidence visualization")
            return

        if save_path is None:
            save_path = self.output_dir / 'prediction_confidence.png'

        # Prepare data
        if outcome_col in test_data.columns:
            X = test_data.drop(outcome_col, axis=1)
            y_true = test_data[outcome_col].values
        else:
            logger.warning(f"Outcome column '{outcome_col}' not found in data")
            return

        # Get predictions and confidence
        try:
            _, probs = self.predictor.predict_sports_outcome(X, return_probs=True)
            confidence = np.abs(probs - 0.5) * 2  # Scale to 0-1

            # Create figure
            plt.figure(figsize=(12, 8))

            # Plot confidence distribution
            plt.subplot(2, 2, 1)
            sns.histplot(confidence, bins=20)
            plt.title('Prediction Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Count')

            # Plot confidence by outcome
            plt.subplot(2, 2, 2)
            correct = (probs > 0.5).astype(int).flatten() == y_true
            plt.scatter(np.arange(len(confidence)), confidence, c=correct, alpha=0.5, cmap='coolwarm')
            plt.colorbar(label='Prediction Correct')
            plt.title('Prediction Confidence vs. Correctness')
            plt.xlabel('Sample Index')
            plt.ylabel('Confidence')

            # Plot ROC curve if available
            try:
                from sklearn.metrics import roc_curve, auc
                plt.subplot(2, 2, 3)
                fpr, tpr, _ = roc_curve(y_true, probs)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
            except:
                logger.warning("Could not generate ROC curve")

            # Plot calibration curve if available
            try:
                from sklearn.calibration import calibration_curve
                plt.subplot(2, 2, 4)
                prob_true, prob_pred = calibration_curve(y_true, probs.flatten(), n_bins=10)
                plt.plot(prob_pred, prob_true, marker='o', label='Model')
                plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
                plt.title('Calibration Plot')
                plt.xlabel('Mean predicted probability')
                plt.ylabel('Fraction of positives')
                plt.legend()
            except:
                logger.warning("Could not generate calibration curve")

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Prediction confidence visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing prediction confidence: {e}")

    def visualize_odds_distribution(self, test_data: pd.DataFrame,
                                  odds_col: str = 'odds',
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize predicted odds vs. actual odds

        Args:
            test_data: Test data with features and odds
            odds_col: Name of the odds column
            save_path: Path to save the visualization
        """
        # Check if betting model is available
        if self.predictor.betting_model is None:
            logger.warning("Betting model not available for odds visualization")
            return

        if save_path is None:
            save_path = self.output_dir / 'odds_distribution.png'

        # Prepare data
        if odds_col in test_data.columns:
            X = test_data.drop(odds_col, axis=1)
            y_true = test_data[odds_col].values
        else:
            logger.warning(f"Odds column '{odds_col}' not found in data")
            return

        # Get predictions
        try:
            y_pred = self.predictor.predict_odds(X).flatten()

            # Create figure
            plt.figure(figsize=(12, 8))

            # Plot predicted vs. actual odds
            plt.subplot(2, 2, 1)
            plt.scatter(y_true, y_pred, alpha=0.5)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.title('Predicted vs. Actual Odds')
            plt.xlabel('Actual Odds')
            plt.ylabel('Predicted Odds')

            # Plot distribution of predicted odds
            plt.subplot(2, 2, 2)
            sns.histplot(y_pred, bins=20, label='Predicted')
            sns.histplot(y_true, bins=20, label='Actual', alpha=0.7)
            plt.title('Distribution of Odds')
            plt.xlabel('Odds')
            plt.ylabel('Count')
            plt.legend()

            # Plot residuals
            plt.subplot(2, 2, 3)
            residuals = y_pred - y_true
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title('Residuals vs. Predicted Values')
            plt.xlabel('Predicted Odds')
            plt.ylabel('Residuals')

            # Plot error distribution
            plt.subplot(2, 2, 4)
            sns.histplot(residuals, bins=20)
            plt.title('Residual Distribution')
            plt.xlabel('Residual')
            plt.ylabel('Count')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Odds distribution visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing odds distribution: {e}")

    def visualize_expected_value(self, test_data: pd.DataFrame,
                               outcome_col: str = 'result',
                               odds_col: str = 'odds',
                               save_path: Optional[str] = None) -> None:
        """
        Visualize expected value of bets

        Args:
            test_data: Test data with features, outcomes, and odds
            outcome_col: Name of the outcome column
            odds_col: Name of the odds column
            save_path: Path to save the visualization
        """
        # Check if both models are available
        if self.predictor.sports_model is None or self.predictor.betting_model is None:
            logger.warning("Both sports and betting models required for EV visualization")
            return

        if save_path is None:
            save_path = self.output_dir / 'expected_value.png'

        # Prepare data
        if outcome_col in test_data.columns and odds_col in test_data.columns:
            X = test_data.drop([outcome_col, odds_col], axis=1)
            y_outcome = test_data[outcome_col].values
            y_odds = test_data[odds_col].values
        else:
            logger.warning(f"Required columns not found in data")
            return

        # Get predictions
        try:
            integrated_results = self.predictor.integrated_prediction(X)

            # Extract values from results
            win_prob = np.array(integrated_results['outcome_probability'])
            pred_odds = np.array(integrated_results['predicted_odds'])
            if 'expected_value' in integrated_results:
                expected_value = np.array(integrated_results['expected_value'])
            else:
                expected_value = win_prob * (pred_odds - 1)

            # Calculate actual EV
            actual_outcomes = y_outcome.reshape(-1, 1)
            actual_ev = actual_outcomes * (y_odds.reshape(-1, 1) - 1)

            # Create figure
            plt.figure(figsize=(16, 10))

            # Plot predicted vs. actual EV
            plt.subplot(2, 3, 1)
            plt.scatter(actual_ev, expected_value, alpha=0.5)
            min_val = min(actual_ev.min(), expected_value.min())
            max_val = max(actual_ev.max(), expected_value.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--')
            plt.title('Predicted vs. Actual Expected Value')
            plt.xlabel('Actual EV')
            plt.ylabel('Predicted EV')

            # Plot EV distribution
            plt.subplot(2, 3, 2)
            sns.histplot(expected_value, bins=20, label='Predicted')
            sns.histplot(actual_ev, bins=20, label='Actual', alpha=0.7)
            plt.title('Distribution of Expected Value')
            plt.xlabel('Expected Value')
            plt.ylabel('Count')
            plt.legend()

            # Plot win probability vs. odds
            plt.subplot(2, 3, 3)
            plt.scatter(win_prob, pred_odds, c=expected_value, cmap='coolwarm', alpha=0.7)
            plt.colorbar(label='Expected Value')
            plt.title('Win Probability vs. Odds')
            plt.xlabel('Win Probability')
            plt.ylabel('Odds')

            # Plot positive EV bets
            plt.subplot(2, 3, 4)
            positive_ev = expected_value > 0
            plt.scatter(win_prob[positive_ev], pred_odds[positive_ev],
                       label='Positive EV', color='green', alpha=0.7)
            plt.scatter(win_prob[~positive_ev], pred_odds[~positive_ev],
                      label='Negative EV', color='red', alpha=0.3)
            plt.title('Positive vs. Negative Expected Value Bets')
            plt.xlabel('Win Probability')
            plt.ylabel('Odds')
            plt.legend()

            # Plot cumulative return
            plt.subplot(2, 3, 5)

            # Sort by expected value
            sorted_indices = np.argsort(expected_value.flatten())[::-1]
            sorted_ev = expected_value.flatten()[sorted_indices]
            sorted_outcomes = actual_outcomes.flatten()[sorted_indices]
            sorted_odds = y_odds[sorted_indices]

            # Calculate cumulative returns
            stake = 1.0  # Unit stake
            returns = sorted_outcomes * sorted_odds * stake - stake
            cumulative_return = np.cumsum(returns)

            plt.plot(range(1, len(cumulative_return) + 1), cumulative_return)
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title('Cumulative Return (Sorted by Expected Value)')
            plt.xlabel('Number of Bets')
            plt.ylabel('Cumulative Return')

            # Plot ROI by EV threshold
            plt.subplot(2, 3, 6)
            thresholds = np.linspace(min(expected_value), max(expected_value), 20)
            roi_values = []

            for threshold in thresholds:
                bet_mask = expected_value.flatten() > threshold
                if np.sum(bet_mask) > 0:
                    bet_returns = actual_outcomes.flatten()[bet_mask] * y_odds[bet_mask] * stake - stake
                    roi = np.sum(bet_returns) / (np.sum(bet_mask) * stake) if np.sum(bet_mask) > 0 else 0
                else:
                    roi = 0
                roi_values.append(roi)

            plt.plot(thresholds, roi_values)
            plt.axhline(y=0, color='k', linestyle='--')
            plt.title('ROI by Expected Value Threshold')
            plt.xlabel('Expected Value Threshold')
            plt.ylabel('Return on Investment')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

            logger.info(f"Expected value visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Error visualizing expected value: {e}")

    def generate_comprehensive_report(self, test_data: pd.DataFrame,
                                    outcome_col: str = 'result',
                                    odds_col: str = 'odds',
                                    output_file: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive report of model performance

        Args:
            test_data: Test data with features, outcomes, and odds
            outcome_col: Name of the outcome column
            odds_col: Name of the odds column
            output_file: Path to save the report

        Returns:
            Dictionary with report data
        """
        report = {
            "sports_model": {},
            "betting_model": {},
            "integrated": {}
        }

        if output_file is None:
            output_file = self.output_dir / 'model_report.json'

        # Prepare data
        feature_cols = [col for col in test_data.columns if col not in [outcome_col, odds_col]]
        X = test_data[feature_cols]

        # Add feature names to report
        report["feature_cols"] = feature_cols

        # Sports model evaluation
        if self.predictor.sports_model is not None and outcome_col in test_data.columns:
            y_outcome = test_data[outcome_col].values

            # Visualize confidence
            confidence_viz_path = self.output_dir / 'prediction_confidence.png'
            self.visualize_prediction_confidence(test_data, outcome_col, str(confidence_viz_path))

            # Calculate metrics
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                _, probs = self.predictor.predict_sports_outcome(X, return_probs=True)
                predictions = (probs > 0.5).astype(int)

                accuracy = float(accuracy_score(y_outcome, predictions))
                precision = float(precision_score(y_outcome, predictions, zero_division=0))
                recall = float(recall_score(y_outcome, predictions, zero_division=0))
                f1 = float(f1_score(y_outcome, predictions, zero_division=0))

                try:
                    roc_auc = float(roc_auc_score(y_outcome, probs))
                except:
                    roc_auc = 0.5

                report["sports_model"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "roc_auc": roc_auc,
                    "visualization_path": str(confidence_viz_path)
                }

                logger.info(f"Sports model metrics - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
            except Exception as e:
                logger.error(f"Error calculating sports model metrics: {e}")

        # Betting model evaluation
        if self.predictor.betting_model is not None and odds_col in test_data.columns:
            y_odds = test_data[odds_col].values

            # Visualize odds distribution
            odds_viz_path = self.output_dir / 'odds_distribution.png'
            self.visualize_odds_distribution(test_data, odds_col, str(odds_viz_path))

            # Calculate metrics
            try:
                from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

                odds_pred = self.predictor.predict_odds(X).flatten()

                mae = float(mean_absolute_error(y_odds, odds_pred))
                r2 = float(r2_score(y_odds, odds_pred))
                rmse = float(np.sqrt(mean_squared_error(y_odds, odds_pred)))

                report["betting_model"] = {
                    "mae": mae,
                    "r2_score": r2,
                    "rmse": rmse,
                    "visualization_path": str(odds_viz_path)
                }

                logger.info(f"Betting model metrics - MAE: {mae:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")
            except Exception as e:
                logger.error(f"Error calculating betting model metrics: {e}")

        # Integrated model evaluation
        if self.predictor.sports_model is not None and self.predictor.betting_model is not None:
            # Visualize expected value
            if outcome_col in test_data.columns and odds_col in test_data.columns:
                ev_viz_path = self.output_dir / 'expected_value.png'
                self.visualize_expected_value(test_data, outcome_col, odds_col, str(ev_viz_path))

                # Calculate ROI
                try:
                    integrated_results = self.predictor.integrated_prediction(X)

                    # Get recommendations
                    if 'recommended_bet' in integrated_results:
                        recommended = np.array(integrated_results['recommended_bet']).astype(bool)
                    elif 'expected_value' in integrated_results:
                        recommended = np.array(integrated_results['expected_value']) > 0
                    else:
                        win_prob = np.array(integrated_results['outcome_probability'])
                        pred_odds = np.array(integrated_results['predicted_odds'])
                        recommended = (win_prob * (pred_odds - 1)) > 0

                    # Calculate ROI for recommended bets
                    y_outcome = test_data[outcome_col].values
                    y_odds = test_data[odds_col].values

                    stake = 1.0
                    total_stake = np.sum(recommended) * stake
                    returns = y_outcome[recommended] * y_odds[recommended] * stake
                    profit = np.sum(returns) - total_stake
                    roi = float(profit / total_stake) if total_stake > 0 else 0

                    win_rate = float(np.mean(y_outcome[recommended])) if np.sum(recommended) > 0 else 0

                    report["integrated"] = {
                        "recommended_bets": int(np.sum(recommended)),
                        "total_bets": len(X),
                        "win_rate": win_rate,
                        "profit": float(profit),
                        "roi": roi,
                        "visualization_path": str(ev_viz_path)
                    }

                    logger.info(f"Integrated model metrics - ROI: {roi:.4f}, Win Rate: {win_rate:.4f}")
                except Exception as e:
                    logger.error(f"Error calculating integrated model metrics: {e}")

        # Visualize feature importance
        try:
            importance_viz_path = self.output_dir / 'feature_importance.png'
            self.visualize_feature_importance(feature_cols, str(importance_viz_path))
            report["feature_importance_path"] = str(importance_viz_path)
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {e}")

        # Save report
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=4)
            logger.info(f"Comprehensive report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        return report


def main():
    """Main function to demonstrate the model visualizer"""
    logger.info("Initializing Model Visualizer")

    # Initialize visualizer
    visualizer = ModelVisualizer()

    # Load test data if available
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_data_path = root_dir / 'data' / 'test_data.csv'

    if test_data_path.exists():
        try:
            test_data = pd.read_csv(test_data_path)
            logger.info(f"Loaded test data: {test_data.shape}")

            # Generate comprehensive report
            report = visualizer.generate_comprehensive_report(test_data)

            return {
                "status": "success",
                "report": report,
                "message": "Visualizations and report generated successfully"
            }
        except Exception as e:
            logger.error(f"Error processing test data: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    else:
        logger.warning("No test data found at expected path")

        # Generate basic visualizations without test data
        visualizer.visualize_feature_importance()

        return {
            "status": "partial",
            "message": "Basic visualizations generated without test data"
        }


if __name__ == "__main__":
    main()
