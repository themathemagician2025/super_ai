# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import random
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class RationalAgent:
    """
    Rational Agent that combines multiple predictions to make optimal decisions
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/ai/rational_agent.yaml")
        self.config = self._load_config()

        # Decision making parameters
        self.decision_method = "weighted_confidence"  # weighted_confidence, majority_vote, ensemble
        self.risk_tolerance = 0.5  # 0-1 scale, higher means more risk tolerance
        self.consistency_weight = 0.7  # How much to value consistency among predictions
        self.confidence_threshold = 0.6  # Minimum confidence to consider a prediction

        # Model weights for different prediction sources
        self.model_weights = {
            "lstm": 0.25,
            "neat": 0.2,
            "random_forest": 0.2,
            "minmax": 0.15,
            "base": 0.1,
            "deap": 0.1
        }

        # Track history for adaptation
        self.decision_history = []
        self.performance_metrics = {}

        logger.info("Rational Agent initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        # Apply configuration settings
                        if "decision" in config:
                            decision_cfg = config["decision"]
                            self.decision_method = decision_cfg.get("method", self.decision_method)
                            self.risk_tolerance = decision_cfg.get("risk_tolerance", self.risk_tolerance)
                            self.consistency_weight = decision_cfg.get("consistency_weight", self.consistency_weight)
                            self.confidence_threshold = decision_cfg.get("confidence_threshold", self.confidence_threshold)

                        # Apply model weights if present
                        if "model_weights" in config:
                            self.model_weights.update(config["model_weights"])
                            # Normalize weights
                            total = sum(self.model_weights.values())
                            if total > 0:
                                self.model_weights = {k: v/total for k, v in self.model_weights.items()}

                        logger.info(f"Loaded configuration from {self.config_path}")
                        return config
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}

    def make_decision(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision based on multiple prediction inputs

        Args:
            predictions: Dictionary mapping model names to their predictions
                Each prediction should be a dict with at least 'prediction' and 'confidence' keys

        Returns:
            Dictionary containing the final decision and metadata
        """
        try:
            # Validate predictions
            if not predictions:
                raise ValueError("Empty predictions dictionary")

            # Filter predictions by confidence threshold
            filtered_predictions = {}
            for model, pred in predictions.items():
                # Handle predictions that are dictionaries
                if isinstance(pred, dict) and 'confidence' in pred:
                    if pred['confidence'] >= self.confidence_threshold:
                        filtered_predictions[model] = pred
                        logger.debug(f"Including {model} prediction with confidence {pred['confidence']}")
                    else:
                        logger.debug(f"Filtering out {model} prediction with low confidence {pred['confidence']}")
                # Handle predictions that are simple values (assume default confidence)
                elif model in self.model_weights:
                    filtered_predictions[model] = {
                        'prediction': pred if not isinstance(pred, dict) else pred.get('prediction', 0.5),
                        'confidence': 0.7  # Default confidence
                    }

            # If all predictions were filtered out, use all with warning
            if not filtered_predictions and predictions:
                logger.warning("All predictions below confidence threshold, using all predictions")
                filtered_predictions = predictions

            # Apply decision method
            if self.decision_method == "weighted_confidence":
                decision = self._weighted_confidence_decision(filtered_predictions)
            elif self.decision_method == "majority_vote":
                decision = self._majority_vote_decision(filtered_predictions)
            elif self.decision_method == "ensemble":
                decision = self._ensemble_decision(filtered_predictions)
            else:
                # Default to weighted confidence
                decision = self._weighted_confidence_decision(filtered_predictions)

            # Add metadata
            decision["timestamp"] = datetime.now().isoformat()
            decision["method"] = self.decision_method
            decision["models_used"] = list(filtered_predictions.keys())
            decision["risk_profile"] = self._calculate_risk_profile(filtered_predictions)

            # Track decision for adaptation
            self._update_decision_history(decision, filtered_predictions)

            logger.info(f"Decision made with confidence {decision.get('confidence', 'unknown')}")
            return decision

        except Exception as e:
            logger.error(f"Decision making failed: {str(e)}")
            # Return fallback decision
            return {
                "prediction": 0.5,
                "confidence": 0.1,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "method": "fallback"
            }

    def _weighted_confidence_decision(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision using confidence-weighted average of predictions"""
        weighted_sum = 0
        weight_sum = 0
        confidences = []
        values = []

        for model, pred in predictions.items():
            model_weight = self.model_weights.get(model, 0.1)
            pred_value = pred.get('prediction', 0.5)

            # For dictionary predictions with complex values
            if isinstance(pred_value, dict) and 'value' in pred_value:
                pred_value = pred_value['value']

            # Handle non-numeric predictions with simple rules (if needed)
            if not isinstance(pred_value, (int, float)):
                pred_value = 0.5  # Default value

            confidence = pred.get('confidence', 0.5)
            weight = model_weight * confidence

            weighted_sum += pred_value * weight
            weight_sum += weight
            confidences.append(confidence)
            values.append(pred_value)

        # Calculate weighted average
        if weight_sum > 0:
            prediction = weighted_sum / weight_sum
        else:
            prediction = 0.5  # Default if no weights

        # Calculate confidence metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        std_dev = np.std(values) if len(values) > 1 else 0
        consistency = 1.0 - min(1.0, std_dev)  # Higher consistency if low std dev

        # Calculate overall confidence as weighted sum of avg confidence and consistency
        overall_confidence = (avg_confidence * (1 - self.consistency_weight) +
                             consistency * self.consistency_weight)

        return {
            "prediction": prediction,
            "confidence": overall_confidence,
            "consistency": consistency,
            "model_confidences": {model: pred.get('confidence', 0.5) for model, pred in predictions.items()},
            "std_dev": std_dev
        }

    def _majority_vote_decision(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision using majority vote with binning"""
        # Define bins for prediction values
        bins = {
            "strong_negative": (-1.0, -0.6),
            "negative": (-0.6, -0.2),
            "neutral": (-0.2, 0.2),
            "positive": (0.2, 0.6),
            "strong_positive": (0.6, 1.0)
        }

        # Count votes for each bin
        votes = {bin_name: 0 for bin_name in bins}
        confidences = []

        for model, pred in predictions.items():
            pred_value = pred.get('prediction', 0.5)

            # For dictionary predictions with complex values
            if isinstance(pred_value, dict) and 'value' in pred_value:
                pred_value = pred_value['value']

            # Handle non-numeric predictions with simple rules (if needed)
            if not isinstance(pred_value, (int, float)):
                pred_value = 0.5  # Default value

            confidence = pred.get('confidence', 0.5)
            confidences.append(confidence)

            # Assign to bin
            for bin_name, (bin_min, bin_max) in bins.items():
                if bin_min <= pred_value < bin_max:
                    votes[bin_name] += 1
                    break

        # Find bin with most votes
        max_votes = 0
        selected_bin = "neutral"  # Default

        for bin_name, vote_count in votes.items():
            if vote_count > max_votes:
                max_votes = vote_count
                selected_bin = bin_name

        # Calculate representative value for the bin
        bin_min, bin_max = bins[selected_bin]
        prediction = (bin_min + bin_max) / 2

        # Calculate decision confidence based on vote distribution
        vote_ratio = max_votes / len(predictions) if predictions else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Combine vote ratio and average confidence
        overall_confidence = (vote_ratio * self.consistency_weight +
                             avg_confidence * (1 - self.consistency_weight))

        return {
            "prediction": prediction,
            "confidence": overall_confidence,
            "vote_counts": votes,
            "selected_bin": selected_bin,
            "vote_ratio": vote_ratio
        }

    def _ensemble_decision(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Make decision using ensemble techniques"""
        # Similar to weighted confidence but with additional adaptive weighting
        # based on historical performance (would be implemented in a real system)

        # For now, implement a simplified version similar to weighted confidence
        # but with some random adaptation

        base_decision = self._weighted_confidence_decision(predictions)

        # Add some adaptation factor
        adaptation_factor = 0.05 * random.uniform(-1, 1) * (1 - base_decision["confidence"])
        adapted_prediction = base_decision["prediction"] + adaptation_factor

        # Ensure prediction is within bounds
        adapted_prediction = max(-1.0, min(1.0, adapted_prediction))

        # Calculate complexity metric (more models = more complex)
        complexity = min(1.0, len(predictions) / 6)  # normalize by expected number of models

        return {
            "prediction": adapted_prediction,
            "confidence": base_decision["confidence"],
            "base_prediction": base_decision["prediction"],
            "adaptation_factor": adaptation_factor,
            "ensemble_complexity": complexity,
            "consistency": base_decision.get("consistency", 0.5)
        }

    def _calculate_risk_profile(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk profile for the decision"""
        # Extract prediction values
        values = []
        for model, pred in predictions.items():
            pred_value = pred.get('prediction', 0.5)
            if isinstance(pred_value, dict) and 'value' in pred_value:
                pred_value = pred_value['value']
            if isinstance(pred_value, (int, float)):
                values.append(pred_value)

        # Calculate risk metrics
        if values:
            variance = np.var(values)
            range_val = max(values) - min(values) if len(values) > 1 else 0
            extremity = abs(np.mean(values))  # How extreme the average prediction is
        else:
            variance = 0
            range_val = 0
            extremity = 0

        # Calculate overall risk score
        risk_score = (variance * 0.3 + range_val * 0.3 + extremity * 0.4) * self.risk_tolerance

        # Classify risk
        if risk_score < 0.2:
            risk_level = "very_low"
        elif risk_score < 0.4:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "moderate"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "very_high"

        return {
            "score": min(1.0, risk_score),  # Cap at 1.0
            "level": risk_level,
            "variance": variance,
            "range": range_val,
            "extremity": extremity
        }

    def _update_decision_history(self, decision: Dict[str, Any], predictions: Dict[str, Dict[str, Any]]) -> None:
        """Update decision history for adaptation"""
        history_entry = {
            "timestamp": datetime.now(),
            "decision": decision,
            "predictions_summary": {
                model: {
                    "prediction": pred.get("prediction", 0.5),
                    "confidence": pred.get("confidence", 0.5)
                }
                for model, pred in predictions.items()
            }
        }

        self.decision_history.append(history_entry)

        # Keep history to a reasonable size
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)

    def adapt_weights(self, performance_data: Dict[str, float]) -> bool:
        """Adapt model weights based on performance feedback"""
        try:
            if not performance_data:
                return False

            # In a real system, this would use the performance data to adjust weights
            # For now, implement a simple adaptation mechanism

            logger.info("Adapting model weights based on performance data")

            # Update performance metrics
            self.performance_metrics.update(performance_data)

            # Perform minimal adaptation for demonstration purposes
            for model in self.model_weights:
                if model in performance_data:
                    # Adjust weights slightly based on performance
                    performance = performance_data[model]
                    current_weight = self.model_weights[model]
                    adjustment = (performance - 0.5) * 0.1  # Small adjustment
                    self.model_weights[model] = max(0.05, min(0.5, current_weight + adjustment))

            # Normalize weights
            total = sum(self.model_weights.values())
            if total > 0:
                self.model_weights = {k: v/total for k, v in self.model_weights.items()}

            logger.info(f"Adapted weights: {self.model_weights}")
            return True
        except Exception as e:
            logger.error(f"Failed to adapt weights: {str(e)}")
            return False

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics about recent decisions"""
        try:
            if not self.decision_history:
                return {"error": "No decision history available"}

            # Calculate basic statistics from recent decisions
            recent_decisions = self.decision_history[-100:]  # Last 100 decisions

            confidence_values = [d["decision"].get("confidence", 0) for d in recent_decisions]
            prediction_values = [d["decision"].get("prediction", 0) for d in recent_decisions]

            stats = {
                "count": len(recent_decisions),
                "avg_confidence": sum(confidence_values) / len(confidence_values) if confidence_values else 0,
                "max_confidence": max(confidence_values) if confidence_values else 0,
                "min_confidence": min(confidence_values) if confidence_values else 0,
                "avg_prediction": sum(prediction_values) / len(prediction_values) if prediction_values else 0,
                "prediction_std": np.std(prediction_values) if len(prediction_values) > 1 else 0,
                "model_usage": self._calculate_model_usage(recent_decisions)
            }

            return stats
        except Exception as e:
            logger.error(f"Error calculating decision stats: {str(e)}")
            return {"error": str(e)}

    def _calculate_model_usage(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate how often each model was used in decisions"""
        usage = {model: 0 for model in self.model_weights}

        for decision in decisions:
            if "models_used" in decision["decision"]:
                for model in decision["decision"]["models_used"]:
                    if model in usage:
                        usage[model] += 1

        return usage
