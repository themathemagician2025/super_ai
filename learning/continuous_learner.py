# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import json
import datetime
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from collections import deque

logger = logging.getLogger(__name__)

class ContinuousLearner:
    """
    Implements continuous online learning with real-world feedback loops,
    adapting to dynamic environments without catastrophic forgetting
    or performance drift.
    """

    def __init__(self, config_path: Path = Path("config/continuous_learning.yaml")):
        self.config_path = config_path
        self.memory_buffer_size = 10000
        self.experience_buffer = deque(maxlen=self.memory_buffer_size)
        self.stability_threshold = 0.05
        self.learning_rate = 0.01
        self.forget_rate = 0.001
        self.domain_weights = {}
        self.performance_history = {}
        self.model_versions = []
        self.current_model_version = None
        self.last_update_time = None
        self.domains = []

        # Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting
        self.parameter_importance = {}

        # Learning schedule
        self.update_frequency = 24  # hours
        self.evaluation_frequency = 6  # hours

        # Load configuration if it exists
        self._load_config()
        logger.info("Continuous Learning Engine initialized")

    def _load_config(self):
        """Load configuration settings if available"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        self.memory_buffer_size = config.get('memory_buffer_size', 10000)
                        self.experience_buffer = deque(maxlen=self.memory_buffer_size)
                        self.stability_threshold = config.get('stability_threshold', 0.05)
                        self.learning_rate = config.get('learning_rate', 0.01)
                        self.forget_rate = config.get('forget_rate', 0.001)
                        self.update_frequency = config.get('update_frequency', 24)
                        self.evaluation_frequency = config.get('evaluation_frequency', 6)
                        self.domains = config.get('domains', [])

                        # Initialize domain weights if domains are specified
                        if self.domains:
                            self.domain_weights = {domain: 1.0 for domain in self.domains}

                        logger.info(f"Loaded continuous learning configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def add_experience(self,
                      experience: Dict[str, Any],
                      domain: str = "general",
                      importance: float = 1.0) -> bool:
        """
        Add a new experience to the learning buffer

        Parameters:
        - experience: Dictionary containing features, targets, and outcome
        - domain: The domain this experience belongs to
        - importance: How important this experience is (1.0 = normal)

        Returns:
        - Success status
        """
        try:
            # Timestamp the experience
            timestamped_experience = {
                **experience,
                "timestamp": datetime.datetime.now().isoformat(),
                "domain": domain,
                "importance": importance
            }

            # Add to buffer
            self.experience_buffer.append(timestamped_experience)

            # If this is a new domain, initialize its weight
            if domain not in self.domain_weights:
                self.domain_weights[domain] = 1.0
                self.domains.append(domain)

            # Check if it's time to update the model
            should_update = self._check_update_schedule()
            if should_update:
                self.update_model()

            return True
        except Exception as e:
            logger.error(f"Error adding experience: {str(e)}")
            return False

    def _check_update_schedule(self) -> bool:
        """Check if it's time to update the model based on schedule"""
        if self.last_update_time is None:
            return len(self.experience_buffer) >= 100  # Initial threshold

        # Check if enough time has passed since last update
        now = datetime.datetime.now()
        last_update = datetime.datetime.fromisoformat(self.last_update_time)
        hours_since_update = (now - last_update).total_seconds() / 3600

        return hours_since_update >= self.update_frequency

    def update_model(self, force: bool = False) -> Dict[str, Any]:
        """
        Update the model with new experiences while preserving
        knowledge from previous learning

        Parameters:
        - force: Force update regardless of schedule

        Returns:
        - Update statistics
        """
        try:
            if not force and not self._check_update_schedule():
                return {"status": "skipped", "reason": "Not scheduled for update yet"}

            # In a real implementation, this would involve:
            # 1. Using experience replay from buffer
            # 2. Applying EWC to prevent catastrophic forgetting
            # 3. Incrementally updating the model

            # Create a new model version
            new_version = len(self.model_versions) + 1
            self.model_versions.append({
                "version": new_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "buffer_size": len(self.experience_buffer),
                "domains": list(self.domain_weights.keys())
            })
            self.current_model_version = new_version

            # Update last update time
            self.last_update_time = datetime.datetime.now().isoformat()

            # Calculate importance of parameters to prevent forgetting
            self._calculate_parameter_importance()

            # Simulate update with random performance change
            # (In a real system, this would be actual model training)
            performance_change = (np.random.random() - 0.3) * 0.1  # -3% to +7%

            update_stats = {
                "status": "success",
                "version": new_version,
                "timestamp": self.last_update_time,
                "experiences_used": len(self.experience_buffer),
                "performance_change": f"{performance_change:.2%}",
                "domains_updated": list(self.domain_weights.keys())
            }

            # Record update in performance history
            self.performance_history[self.last_update_time] = {
                "version": new_version,
                "performance_change": performance_change
            }

            logger.info(f"Model updated to version {new_version} with {len(self.experience_buffer)} experiences")
            return update_stats

        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _calculate_parameter_importance(self):
        """
        Calculate importance of model parameters to prevent catastrophic forgetting
        This is a placeholder for EWC or similar techniques
        """
        # In a real implementation, this would:
        # 1. Compute Fisher Information Matrix on current data
        # 2. Store parameter importance values
        self.parameter_importance = {
            "last_calculated": datetime.datetime.now().isoformat(),
            "parameters": {
                f"param_{i}": np.random.random() for i in range(10)  # Placeholder
            }
        }

    def evaluate_model(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate current model performance

        Parameters:
        - domain: Optional domain to evaluate specifically

        Returns:
        - Evaluation metrics
        """
        try:
            # In a real implementation, this would:
            # 1. Run validation on held-out data
            # 2. Calculate performance metrics
            # 3. Check for performance drift

            domains_to_evaluate = [domain] if domain else self.domains

            results = {}
            drift_detected = False

            for eval_domain in domains_to_evaluate:
                # Simulate evaluation metrics
                accuracy = 0.7 + (np.random.random() * 0.2)  # 70-90%
                stability = 1.0 - (np.random.random() * 0.1)  # 90-100%

                # Check for drift
                domain_drift = np.random.random() > 0.85  # 15% chance of drift

                results[eval_domain] = {
                    "accuracy": accuracy,
                    "stability": stability,
                    "drift_detected": domain_drift
                }

                drift_detected = drift_detected or domain_drift

            # If drift detected, adjust domain weights
            if drift_detected:
                self._adjust_domain_weights(results)

            # Record evaluation
            evaluation_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": self.current_model_version,
                "results": results,
                "drift_detected": drift_detected
            }

            # Store in history with timestamp as key
            eval_key = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.performance_history[eval_key] = evaluation_record

            return {
                "status": "success",
                "version": self.current_model_version,
                "results": results,
                "drift_detected": drift_detected,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"status": "failed", "error": str(e)}

    def _adjust_domain_weights(self, evaluation_results: Dict[str, Dict[str, Any]]):
        """
        Adjust domain weights based on evaluation results to focus on drifting domains

        Parameters:
        - evaluation_results: Results from evaluate_model
        """
        for domain, metrics in evaluation_results.items():
            if metrics.get("drift_detected", False):
                # Increase weight for domains with drift
                self.domain_weights[domain] = min(2.0, self.domain_weights[domain] * 1.5)
            else:
                # Slightly decrease weight for stable domains
                self.domain_weights[domain] = max(0.5, self.domain_weights[domain] * 0.95)

        # Normalize weights to keep them in reasonable range
        total_weight = sum(self.domain_weights.values())
        if total_weight > 0:
            normalized_weights = {d: w / total_weight * len(self.domain_weights)
                                for d, w in self.domain_weights.items()}
            self.domain_weights = normalized_weights

    def get_prediction_correction(self,
                                 features: Dict[str, Any],
                                 initial_prediction: Any,
                                 domain: str) -> Dict[str, Any]:
        """
        Apply continuous learning adjustments to a prediction

        Parameters:
        - features: Input features
        - initial_prediction: The prediction before adjustment
        - domain: Domain of the prediction

        Returns:
        - Adjusted prediction with confidence and explanation
        """
        try:
            # This would implement actual adjustment logic based on recent
            # experiences and domain-specific knowledge

            # Find similar experiences in buffer
            similar_experiences = self._find_similar_experiences(features, domain, limit=5)

            # If no similar experiences, return original prediction
            if not similar_experiences:
                return {
                    "adjusted_prediction": initial_prediction,
                    "confidence_adjustment": 0.0,
                    "explanation": "No similar historical data found for adjustment"
                }

            # Calculate adjustment based on similar experiences
            adjustment_factor = self._calculate_adjustment(similar_experiences, initial_prediction)

            # Apply adjustment
            # (In a real system, the adjustment would depend on the prediction type)
            adjusted_prediction = initial_prediction
            if isinstance(initial_prediction, (int, float)):
                adjusted_prediction = initial_prediction * (1.0 + adjustment_factor)

            # Calculate confidence in adjustment
            confidence_adjustment = min(0.2, abs(adjustment_factor) * 2)  # Max 20% adjustment
            if adjustment_factor < 0:
                confidence_adjustment = -confidence_adjustment

            return {
                "adjusted_prediction": adjusted_prediction,
                "original_prediction": initial_prediction,
                "adjustment_factor": adjustment_factor,
                "confidence_adjustment": confidence_adjustment,
                "num_similar_experiences": len(similar_experiences),
                "explanation": self._generate_adjustment_explanation(
                    adjustment_factor, similar_experiences
                )
            }

        except Exception as e:
            logger.error(f"Error applying prediction correction: {str(e)}")
            return {
                "adjusted_prediction": initial_prediction,
                "confidence_adjustment": 0.0,
                "error": str(e)
            }

    def _find_similar_experiences(self,
                                 features: Dict[str, Any],
                                 domain: str,
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar experiences in the buffer"""
        # In a real implementation, this would use vector similarity or other
        # sophisticated similarity measures

        # Simple filtering by domain
        domain_experiences = [exp for exp in self.experience_buffer
                             if exp.get("domain") == domain]

        # If no experiences in domain, return empty list
        if not domain_experiences:
            return []

        # For this example, just return the most recent experiences in the domain
        return sorted(domain_experiences,
                     key=lambda x: x.get("timestamp", ""),
                     reverse=True)[:limit]

    def _calculate_adjustment(self,
                             similar_experiences: List[Dict[str, Any]],
                             initial_prediction: Any) -> float:
        """Calculate adjustment factor based on similar experiences"""
        # This is a simplified example
        # In a real system, this would use more sophisticated methods

        # For numerical predictions, calculate average error in similar situations
        if isinstance(initial_prediction, (int, float)):
            errors = []
            for exp in similar_experiences:
                if "prediction" in exp and "actual" in exp:
                    pred = exp.get("prediction", 0)
                    actual = exp.get("actual", 0)
                    if pred != 0:  # Avoid division by zero
                        error = (actual - pred) / pred
                        errors.append(error)

            # Return average error if we have data
            if errors:
                return sum(errors) / len(errors)

        # Default: no adjustment
        return 0.0

    def _generate_adjustment_explanation(self,
                                        adjustment_factor: float,
                                        similar_experiences: List[Dict[str, Any]]) -> str:
        """Generate human-readable explanation for an adjustment"""
        if abs(adjustment_factor) < 0.01:
            return "No significant adjustment needed based on historical data."

        direction = "increased" if adjustment_factor > 0 else "decreased"
        magnitude = abs(adjustment_factor)

        if magnitude < 0.05:
            strength = "slightly"
        elif magnitude < 0.15:
            strength = "moderately"
        else:
            strength = "significantly"

        return (f"Prediction {direction} {strength} based on {len(similar_experiences)} "
                f"similar historical cases with an average adjustment of {adjustment_factor:.1%}.")

    def detect_concept_drift(self,
                            recent_window: int = 100,
                            baseline_window: int = 500) -> Dict[str, Any]:
        """
        Detect concept drift by comparing recent performance to baseline

        Parameters:
        - recent_window: Number of recent experiences to consider
        - baseline_window: Number of older experiences to use as baseline

        Returns:
        - Drift analysis
        """
        try:
            if len(self.experience_buffer) < (recent_window + baseline_window):
                return {
                    "status": "insufficient_data",
                    "required": recent_window + baseline_window,
                    "available": len(self.experience_buffer)
                }

            # Convert deque to list for easier slicing
            experiences = list(self.experience_buffer)

            # Get recent and baseline windows
            recent_experiences = experiences[-recent_window:]
            baseline_experiences = experiences[-(recent_window+baseline_window):-recent_window]

            # Calculate error in each window
            # This is simplified - real implementation would be more sophisticated
            recent_errors = self._calculate_error_metrics(recent_experiences)
            baseline_errors = self._calculate_error_metrics(baseline_experiences)

            # Compare distributions
            drift_metrics = {}
            for domain in set(recent_errors.keys()) | set(baseline_errors.keys()):
                recent_domain_error = recent_errors.get(domain, {"mean": 0, "std": 0})
                baseline_domain_error = baseline_errors.get(domain, {"mean": 0, "std": 0})

                # Calculate drift metrics
                error_change = recent_domain_error["mean"] - baseline_domain_error["mean"]
                relative_change = (error_change / baseline_domain_error["mean"]
                                  if baseline_domain_error["mean"] != 0 else 0)

                # Detect significant drift
                significant_drift = (abs(relative_change) > self.stability_threshold)

                drift_metrics[domain] = {
                    "error_change": error_change,
                    "relative_change": relative_change,
                    "drift_detected": significant_drift,
                    "direction": "increased" if error_change > 0 else "decreased"
                }

            # Overall drift status
            any_drift = any(m["drift_detected"] for m in drift_metrics.values())

            drift_analysis = {
                "status": "drift_detected" if any_drift else "stable",
                "timestamp": datetime.datetime.now().isoformat(),
                "recent_window": recent_window,
                "baseline_window": baseline_window,
                "domains": drift_metrics
            }

            # If drift detected, trigger model update
            if any_drift:
                logger.warning("Concept drift detected - triggering model update")
                update_result = self.update_model(force=True)
                drift_analysis["update_result"] = update_result

            return drift_analysis

        except Exception as e:
            logger.error(f"Error detecting concept drift: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _calculate_error_metrics(self, experiences: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate error metrics grouped by domain"""
        domain_errors = {}

        for exp in experiences:
            domain = exp.get("domain", "general")

            # Skip experiences without prediction and actual values
            if "prediction" not in exp or "actual" not in exp:
                continue

            prediction = exp.get("prediction", 0)
            actual = exp.get("actual", 0)

            # Calculate error (simple absolute error for this example)
            error = abs(prediction - actual)

            # Add to domain errors
            if domain not in domain_errors:
                domain_errors[domain] = {"errors": []}

            domain_errors[domain]["errors"].append(error)

        # Calculate statistics for each domain
        result = {}
        for domain, data in domain_errors.items():
            errors = data["errors"]
            if errors:
                result[domain] = {
                    "mean": sum(errors) / len(errors),
                    "std": np.std(errors) if len(errors) > 1 else 0,
                    "count": len(errors)
                }

        return result

    def save_state(self, file_path: Optional[Path] = None) -> bool:
        """
        Save the current state of the continuous learner

        Parameters:
        - file_path: Optional custom path to save to

        Returns:
        - Success status
        """
        try:
            if file_path is None:
                # Default path
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = Path(f"models/continuous_learner_state_{timestamp}.json")

            # Create directory if needed
            os.makedirs(file_path.parent, exist_ok=True)

            # Convert deque to list for serialization
            serializable_buffer = list(self.experience_buffer)

            state = {
                "timestamp": datetime.datetime.now().isoformat(),
                "current_model_version": self.current_model_version,
                "model_versions": self.model_versions,
                "domain_weights": self.domain_weights,
                "domains": self.domains,
                "learning_rate": self.learning_rate,
                "forget_rate": self.forget_rate,
                "experience_buffer_size": len(self.experience_buffer),
                "last_update_time": self.last_update_time
            }

            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Continuous learner state saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving continuous learner state: {str(e)}")
            return False

    def load_state(self, file_path: Path) -> bool:
        """
        Load a previously saved state

        Parameters:
        - file_path: Path to the saved state file

        Returns:
        - Success status
        """
        try:
            if not file_path.exists():
                logger.error(f"State file not found: {file_path}")
                return False

            with open(file_path, 'r') as f:
                state = json.load(f)

            # Restore state
            self.current_model_version = state.get("current_model_version")
            self.model_versions = state.get("model_versions", [])
            self.domain_weights = state.get("domain_weights", {})
            self.domains = state.get("domains", [])
            self.learning_rate = state.get("learning_rate", 0.01)
            self.forget_rate = state.get("forget_rate", 0.001)
            self.last_update_time = state.get("last_update_time")

            logger.info(f"Continuous learner state loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading continuous learner state: {str(e)}")
            return False
