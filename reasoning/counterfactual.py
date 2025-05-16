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
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import copy

logger = logging.getLogger(__name__)

class CounterfactualReasoner:
    """
    Implements predictive counterfactual reasoning to answer "what if" scenarios:
    - What would happen if I bet on the opposite team?
    - What if the market had gone bullish instead?
    - What if player X was injured?
    """

    def __init__(self, config_path: Path = Path("config/counterfactual.yaml")):
        self.config_path = config_path
        self.max_counterfactuals = 5
        self.confidence_threshold = 0.6
        self.counterfactual_history = {}
        self.default_sensitivity = 0.5
        self.sensitivity_mappings = {}  # Domain-specific sensitivity mappings
        self.causal_models = {}
        self.simulation_depth = 3
        self.scenario_validators = {}
        self.intervention_constraints = {}

        # Load configuration if it exists
        self._load_config()
        logger.info("Counterfactual Reasoning Engine initialized")

    def _load_config(self):
        """Load configuration settings if available"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        self.max_counterfactuals = config.get('max_counterfactuals', 5)
                        self.confidence_threshold = config.get('confidence_threshold', 0.6)
                        self.default_sensitivity = config.get('default_sensitivity', 0.5)
                        self.sensitivity_mappings = config.get('sensitivity_mappings', {})
                        self.simulation_depth = config.get('simulation_depth', 3)

                        # Load domain-specific intervention constraints
                        self.intervention_constraints = config.get('intervention_constraints', {})

                        logger.info(f"Loaded counterfactual configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def generate_counterfactual(self,
                               scenario: Dict[str, Any],
                               intervention: Dict[str, Any],
                               domain: str = "general",
                               model: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate a counterfactual prediction based on an intervention

        Parameters:
        - scenario: The original scenario with features and context
        - intervention: The changes to apply (what-if)
        - domain: Domain this counterfactual belongs to
        - model: Optional prediction model to use (if None, uses internal simulation)

        Returns:
        - Counterfactual prediction with comparison to original
        """
        try:
            # Validate scenario and intervention
            validation_result = self._validate_intervention(scenario, intervention, domain)
            if not validation_result["valid"]:
                return {
                    "status": "invalid_intervention",
                    "reason": validation_result["reason"],
                    "counterfactual_id": None
                }

            # Generate a unique ID for this counterfactual
            counterfactual_id = f"cf_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Create counterfactual scenario by applying intervention
            cf_scenario = self._apply_intervention(scenario, intervention)

            # Get original prediction if not included
            original_prediction = scenario.get("prediction")
            if original_prediction is None and model is not None:
                # Use provided model to make original prediction
                try:
                    original_prediction = model(scenario)
                except Exception as e:
                    logger.error(f"Error making original prediction: {str(e)}")
                    original_prediction = None

            # Make counterfactual prediction
            cf_prediction = None
            if model is not None:
                try:
                    # Use provided model for prediction
                    cf_prediction = model(cf_scenario)
                except Exception as e:
                    logger.error(f"Error making counterfactual prediction with model: {str(e)}")

            # If no external model or it failed, use internal simulation
            if cf_prediction is None:
                cf_prediction = self._simulate_counterfactual(cf_scenario, domain)

            # Calculate confidence in counterfactual
            confidence = self._calculate_confidence(scenario, cf_scenario, domain)

            # Generate explanation
            explanation = self._generate_explanation(
                scenario, cf_scenario, original_prediction, cf_prediction, intervention
            )

            # Create result object
            result = {
                "counterfactual_id": counterfactual_id,
                "status": "success",
                "original_scenario": scenario,
                "counterfactual_scenario": cf_scenario,
                "intervention": intervention,
                "original_prediction": original_prediction,
                "counterfactual_prediction": cf_prediction,
                "confidence": confidence,
                "explanation": explanation,
                "timestamp": datetime.datetime.now().isoformat(),
                "domain": domain
            }

            # Store in history
            self.counterfactual_history[counterfactual_id] = result

            return result

        except Exception as e:
            logger.error(f"Error generating counterfactual: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "counterfactual_id": None
            }

    def _validate_intervention(self,
                              scenario: Dict[str, Any],
                              intervention: Dict[str, Any],
                              domain: str) -> Dict[str, Any]:
        """Validate that an intervention is possible and consistent"""
        # Check that all intervention keys exist in the scenario
        scenario_keys = set(scenario.keys())
        intervention_keys = set(intervention.keys())

        invalid_keys = intervention_keys - scenario_keys
        if invalid_keys:
            return {
                "valid": False,
                "reason": f"Intervention contains keys not in scenario: {invalid_keys}"
            }

        # Check domain-specific constraints
        domain_constraints = self.intervention_constraints.get(domain, {})

        # Check for immutable features
        immutable_features = domain_constraints.get("immutable_features", [])
        for feature in immutable_features:
            if feature in intervention_keys:
                return {
                    "valid": False,
                    "reason": f"Feature '{feature}' cannot be changed in domain '{domain}'"
                }

        # Check for value range constraints
        value_ranges = domain_constraints.get("value_ranges", {})
        for feature, value in intervention.items():
            if feature in value_ranges:
                valid_range = value_ranges[feature]
                if isinstance(valid_range, list) and len(valid_range) == 2:
                    min_val, max_val = valid_range
                    if not (min_val <= value <= max_val):
                        return {
                            "valid": False,
                            "reason": f"Value {value} for feature '{feature}' is outside valid range {valid_range}"
                        }

        # Check for validator function if registered for this domain
        validator = self.scenario_validators.get(domain)
        if validator:
            # Apply intervention
            cf_scenario = self._apply_intervention(scenario, intervention)
            # Check if valid
            is_valid, reason = validator(cf_scenario)
            if not is_valid:
                return {"valid": False, "reason": reason}

        return {"valid": True, "reason": ""}

    def _apply_intervention(self, scenario: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an intervention to create a counterfactual scenario"""
        # Create a deep copy to avoid modifying the original
        cf_scenario = copy.deepcopy(scenario)

        # Apply each intervention change
        for key, value in intervention.items():
            if key in cf_scenario:
                cf_scenario[key] = value

        # Mark as counterfactual
        cf_scenario["is_counterfactual"] = True
        cf_scenario["intervention"] = intervention

        return cf_scenario

    def _simulate_counterfactual(self, cf_scenario: Dict[str, Any], domain: str) -> Any:
        """Simulate the outcome of a counterfactual scenario without an external model"""
        # In a real implementation, this would use internal causal models
        # or simplified prediction models to estimate the counterfactual outcome

        # For this demonstration, we'll use a very simplified approach
        # based on feature sensitivities

        # Get domain-specific sensitivities or use defaults
        sensitivities = self.sensitivity_mappings.get(domain, {})

        # If no intervention was marked, we can't determine what changed
        if "intervention" not in cf_scenario:
            return None

        # Get original prediction if available
        original_prediction = cf_scenario.get("prediction")
        if original_prediction is None:
            # If no original prediction, create a placeholder
            if domain in ["sports", "betting"]:
                original_prediction = {"win_probability": 0.5, "score_margin": 0}
            elif domain in ["forex", "stocks", "market"]:
                original_prediction = {"price_change": 0.0, "volatility": 0.01}
            else:
                # Generic numeric prediction
                original_prediction = 0.5

        # Calculate impact of each intervention
        impact = 0.0
        interventions = cf_scenario.get("intervention", {})

        for feature, new_value in interventions.items():
            # Skip non-numeric values
            if not isinstance(new_value, (int, float)):
                continue

            # Get original value
            original_value = cf_scenario.get(feature, new_value)

            # Skip if same value
            if original_value == new_value:
                continue

            # Calculate relative change
            if original_value != 0:
                relative_change = (new_value - original_value) / abs(original_value)
            else:
                relative_change = new_value

            # Get sensitivity for this feature
            sensitivity = sensitivities.get(feature, self.default_sensitivity)

            # Calculate impact
            feature_impact = relative_change * sensitivity
            impact += feature_impact

        # Apply impact to prediction
        cf_prediction = self._apply_impact(original_prediction, impact, domain)

        return cf_prediction

    def _apply_impact(self, prediction: Any, impact: float, domain: str) -> Any:
        """Apply an impact value to a prediction based on domain"""
        if isinstance(prediction, (int, float)):
            # Simple numeric prediction
            new_prediction = prediction * (1.0 + impact)
            # Ensure prediction is in valid range (assuming 0-1 for probability)
            if 0 <= prediction <= 1:
                new_prediction = max(0.0, min(1.0, new_prediction))
            return new_prediction

        elif isinstance(prediction, dict):
            # Complex prediction with multiple fields
            new_prediction = copy.deepcopy(prediction)

            if domain in ["sports", "betting"]:
                # Modify win probability
                if "win_probability" in new_prediction:
                    win_prob = new_prediction["win_probability"]
                    new_win_prob = win_prob + (impact * 0.2)  # Scale impact
                    new_prediction["win_probability"] = max(0.0, min(1.0, new_win_prob))

                # Modify score margin
                if "score_margin" in new_prediction:
                    margin = new_prediction["score_margin"]
                    new_margin = margin + (impact * 5)  # Scale impact
                    new_prediction["score_margin"] = new_margin

            elif domain in ["forex", "stocks", "market"]:
                # Modify price change
                if "price_change" in new_prediction:
                    price_change = new_prediction["price_change"]
                    new_price_change = price_change + (impact * 0.05)  # Scale impact
                    new_prediction["price_change"] = new_price_change

                # Modify volatility
                if "volatility" in new_prediction:
                    volatility = new_prediction["volatility"]
                    # Impact increases volatility
                    new_volatility = volatility * (1.0 + abs(impact) * 0.5)
                    new_prediction["volatility"] = max(0.001, new_volatility)

            return new_prediction

        else:
            # Unsupported prediction type
            return prediction

    def _calculate_confidence(self,
                             original_scenario: Dict[str, Any],
                             cf_scenario: Dict[str, Any],
                             domain: str) -> float:
        """Calculate confidence in the counterfactual prediction"""
        # In a real implementation, this would use model uncertainty,
        # causal graph information, and more sophisticated analysis

        # Simple confidence calculation based on:
        # 1. Number and magnitude of interventions
        # 2. Domain knowledge of feature sensitivity
        # 3. Simulation depth

        # Calculate intervention magnitude
        intervention = cf_scenario.get("intervention", {})
        num_interventions = len(intervention)

        # More interventions = less confidence
        intervention_factor = max(0.2, 1.0 - (num_interventions * 0.1))

        # Calculate sensitivity of changed features
        domain_sensitivities = self.sensitivity_mappings.get(domain, {})
        avg_sensitivity = 0.0

        for feature in intervention.keys():
            sensitivity = domain_sensitivities.get(feature, self.default_sensitivity)
            avg_sensitivity += sensitivity

        if num_interventions > 0:
            avg_sensitivity /= num_interventions

        # Deeper simulation = less confidence
        depth_factor = max(0.5, 1.0 - ((self.simulation_depth - 1) * 0.1))

        # Calculate combined confidence
        base_confidence = 0.8  # Start with 80% confidence
        confidence = base_confidence * intervention_factor * depth_factor

        # Adjust by domain-specific factor
        domain_confidence = {
            "sports": 0.85,
            "betting": 0.75,
            "forex": 0.7,
            "stocks": 0.65,
            "weather": 0.8,
            "general": 0.7
        }

        domain_factor = domain_confidence.get(domain, 0.7)
        confidence *= domain_factor

        # Ensure confidence is in valid range
        return max(0.1, min(0.95, confidence))

    def _generate_explanation(self,
                             original_scenario: Dict[str, Any],
                             cf_scenario: Dict[str, Any],
                             original_prediction: Any,
                             cf_prediction: Any,
                             intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an explanation for the counterfactual prediction"""

        # Extract key values
        key_changes = []
        for feature, new_value in intervention.items():
            original_value = original_scenario.get(feature)
            if original_value != new_value:
                key_changes.append({
                    "feature": feature,
                    "from": original_value,
                    "to": new_value,
                    "relative_change": self._calculate_relative_change(original_value, new_value)
                })

        # Calculate change in prediction
        prediction_change = self._calculate_prediction_change(original_prediction, cf_prediction)

        # Identify key causal factors
        causal_factors = self._identify_causal_factors(key_changes, prediction_change)

        # Generate textual explanation
        summary = self._generate_summary_text(key_changes, prediction_change)

        return {
            "summary": summary,
            "key_changes": key_changes,
            "prediction_change": prediction_change,
            "causal_factors": causal_factors,
            "methodology": "Counterfactual simulation with causal reasoning"
        }

    def _calculate_relative_change(self, original_value, new_value):
        """Calculate relative change between values"""
        if original_value is None or new_value is None:
            return None

        try:
            # For numeric values
            if isinstance(original_value, (int, float)) and isinstance(new_value, (int, float)):
                if original_value != 0:
                    return (new_value - original_value) / abs(original_value)
                else:
                    return float('inf') if new_value > 0 else float('-inf') if new_value < 0 else 0
        except:
            pass

        # For non-numeric or error cases
        return "changed"

    def _calculate_prediction_change(self, original_prediction, cf_prediction):
        """Calculate the change between original and counterfactual predictions"""
        if original_prediction is None or cf_prediction is None:
            return {"type": "unknown"}

        # Handle different prediction types
        if isinstance(original_prediction, (int, float)) and isinstance(cf_prediction, (int, float)):
            absolute_change = cf_prediction - original_prediction

            if original_prediction != 0:
                relative_change = absolute_change / abs(original_prediction)
            else:
                relative_change = float('inf') if absolute_change > 0 else float('-inf') if absolute_change < 0 else 0

            return {
                "type": "numeric",
                "absolute_change": absolute_change,
                "relative_change": relative_change,
                "direction": "increased" if absolute_change > 0 else "decreased" if absolute_change < 0 else "unchanged"
            }

        elif isinstance(original_prediction, dict) and isinstance(cf_prediction, dict):
            # Compare each key in the dictionary
            changes = {}
            for key in set(original_prediction.keys()) | set(cf_prediction.keys()):
                if key in original_prediction and key in cf_prediction:
                    changes[key] = self._calculate_prediction_change(
                        original_prediction[key], cf_prediction[key]
                    )

            return {
                "type": "compound",
                "changes": changes
            }

        else:
            # Non-comparable predictions
            return {
                "type": "categorical",
                "changed": original_prediction != cf_prediction,
                "from": original_prediction,
                "to": cf_prediction
            }

    def _identify_causal_factors(self, key_changes, prediction_change):
        """Identify the causal factors for the counterfactual change"""
        # This is a simplified implementation
        # In a real system, this would use causal models and more sophisticated analysis

        factors = []

        if prediction_change.get("type") == "numeric":
            direction = prediction_change.get("direction", "unchanged")

            if direction != "unchanged":
                # Sort changes by absolute magnitude
                sorted_changes = sorted(
                    key_changes,
                    key=lambda x: abs(x["relative_change"]) if isinstance(x["relative_change"], (int, float)) else 0,
                    reverse=True
                )

                # Take top factors
                for change in sorted_changes[:3]:
                    factors.append({
                        "feature": change["feature"],
                        "change": change["relative_change"] if isinstance(change["relative_change"], (int, float)) else "changed",
                        "importance": "high" if sorted_changes.index(change) == 0 else
                                     "medium" if sorted_changes.index(change) == 1 else "low"
                    })

        return factors

    def _generate_summary_text(self, key_changes, prediction_change):
        """Generate a human-readable summary of the counterfactual"""
        if not key_changes:
            return "No changes were made in this counterfactual scenario."

        # Describe changes
        change_descriptions = []
        for change in key_changes[:3]:  # Limit to top 3 changes
            feature = change["feature"]
            from_val = change["from"]
            to_val = change["to"]

            change_descriptions.append(f"{feature} changed from {from_val} to {to_val}")

        changes_text = ", ".join(change_descriptions)
        if len(key_changes) > 3:
            changes_text += f", and {len(key_changes) - 3} other changes"

        # Describe prediction change
        if prediction_change.get("type") == "numeric":
            direction = prediction_change.get("direction", "unchanged")
            abs_change = prediction_change.get("absolute_change", 0)
            rel_change = prediction_change.get("relative_change", 0)

            if direction == "unchanged":
                prediction_text = "the prediction remained essentially unchanged"
            else:
                magnitude = abs(rel_change) if isinstance(rel_change, (int, float)) else 0
                strength = "dramatically" if magnitude > 0.5 else "significantly" if magnitude > 0.2 else "slightly"
                prediction_text = f"the prediction {direction} {strength} by {abs(abs_change):.4g} ({abs(rel_change)*100:.1f}%)"

        elif prediction_change.get("type") == "categorical":
            from_val = prediction_change.get("from")
            to_val = prediction_change.get("to")
            prediction_text = f"the prediction changed from {from_val} to {to_val}"

        else:
            prediction_text = "the prediction changed"

        # Combine texts
        return f"If {changes_text}, then {prediction_text}."

    def compare_counterfactuals(self,
                              counterfactual_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple counterfactual scenarios to identify patterns

        Parameters:
        - counterfactual_ids: List of counterfactual IDs to compare

        Returns:
        - Comparison analysis
        """
        try:
            # Get counterfactuals
            counterfactuals = []
            for cf_id in counterfactual_ids:
                if cf_id in self.counterfactual_history:
                    counterfactuals.append(self.counterfactual_history[cf_id])

            if len(counterfactuals) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Need at least 2 valid counterfactuals to compare"
                }

            # Analyze interventions
            common_interventions = self._find_common_interventions(counterfactuals)

            # Analyze prediction changes
            prediction_trends = self._analyze_prediction_trends(counterfactuals)

            # Analyze intervention impact
            impact_analysis = self._analyze_intervention_impact(counterfactuals)

            return {
                "status": "success",
                "num_counterfactuals": len(counterfactuals),
                "common_interventions": common_interventions,
                "prediction_trends": prediction_trends,
                "impact_analysis": impact_analysis,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error comparing counterfactuals: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _find_common_interventions(self, counterfactuals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common interventions across counterfactuals"""
        # Collect all intervention features
        all_features = set()
        for cf in counterfactuals:
            intervention = cf.get("intervention", {})
            all_features.update(intervention.keys())

        # Analyze each feature
        common_interventions = []
        for feature in all_features:
            # Count occurrences
            count = sum(1 for cf in counterfactuals if feature in cf.get("intervention", {}))
            if count > 1:  # Feature appears in multiple counterfactuals
                # Collect values
                values = [cf.get("intervention", {}).get(feature) for cf in counterfactuals
                         if feature in cf.get("intervention", {})]

                common_interventions.append({
                    "feature": feature,
                    "occurrence": count,
                    "prevalence": count / len(counterfactuals),
                    "values": values,
                    "consistent_direction": self._check_consistent_direction(values)
                })

        # Sort by prevalence
        return sorted(common_interventions, key=lambda x: x["prevalence"], reverse=True)

    def _check_consistent_direction(self, values):
        """Check if a list of values has a consistent direction of change"""
        if not values or len(values) < 2:
            return None

        # Check if all values are numeric
        if not all(isinstance(v, (int, float)) for v in values):
            return None

        # Check if all values are increasing or all decreasing
        is_increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
        is_decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))

        if is_increasing:
            return "increasing"
        elif is_decreasing:
            return "decreasing"
        else:
            return "mixed"

    def _analyze_prediction_trends(self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in prediction changes across counterfactuals"""
        # Extract original and counterfactual predictions
        predictions = []
        for cf in counterfactuals:
            original = cf.get("original_prediction")
            counterfactual = cf.get("counterfactual_prediction")

            if original is not None and counterfactual is not None:
                predictions.append({
                    "original": original,
                    "counterfactual": counterfactual,
                    "change": self._calculate_prediction_change(original, counterfactual)
                })

        if not predictions:
            return {"trends": "insufficient_data"}

        # Analyze numeric predictions
        numeric_predictions = [p for p in predictions
                              if p["change"].get("type") == "numeric"]

        if numeric_predictions:
            # Calculate consistency of direction
            directions = [p["change"].get("direction") for p in numeric_predictions]

            direction_counts = {
                "increased": directions.count("increased"),
                "decreased": directions.count("decreased"),
                "unchanged": directions.count("unchanged")
            }

            dominant_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
            direction_consistency = direction_counts[dominant_direction] / len(directions)

            # Calculate average magnitude of change
            relative_changes = [abs(p["change"].get("relative_change", 0))
                               for p in numeric_predictions
                               if isinstance(p["change"].get("relative_change"), (int, float))]

            avg_magnitude = sum(relative_changes) / len(relative_changes) if relative_changes else 0

            return {
                "type": "numeric",
                "direction_counts": direction_counts,
                "dominant_direction": dominant_direction,
                "direction_consistency": direction_consistency,
                "average_magnitude": avg_magnitude,
                "consistency_score": direction_consistency
            }

        # Analyze categorical predictions
        categorical_predictions = [p for p in predictions
                                  if p["change"].get("type") == "categorical"]

        if categorical_predictions:
            # Count consistent changes
            changed_count = sum(1 for p in categorical_predictions
                               if p["change"].get("changed", False))

            consistency = changed_count / len(categorical_predictions)

            return {
                "type": "categorical",
                "changed_count": changed_count,
                "unchanged_count": len(categorical_predictions) - changed_count,
                "change_consistency": consistency
            }

        return {"trends": "mixed_prediction_types"}

    def _analyze_intervention_impact(self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of interventions on predictions"""
        # In a real implementation, this would use causal models to analyze
        # how different interventions impact the outcome

        # For this example, we'll do a simplified correlation analysis
        feature_impacts = {}

        # Collect features and their impacts
        for cf in counterfactuals:
            intervention = cf.get("intervention", {})

            # Skip if no prediction change information
            change = cf.get("prediction_change", {})
            if not change or change.get("type") != "numeric":
                continue

            # Get prediction impact
            impact = change.get("relative_change", 0)
            if not isinstance(impact, (int, float)):
                continue

            # Associate feature changes with impact
            for feature, value in intervention.items():
                original_value = cf.get("original_scenario", {}).get(feature)

                # Skip non-numeric changes
                if not isinstance(value, (int, float)) or not isinstance(original_value, (int, float)):
                    continue

                # Calculate relative change in feature
                feature_change = (value - original_value) / abs(original_value) if original_value != 0 else 0

                # Record feature and impact
                if feature not in feature_impacts:
                    feature_impacts[feature] = {"changes": [], "impacts": []}

                feature_impacts[feature]["changes"].append(feature_change)
                feature_impacts[feature]["impacts"].append(impact)

        # Calculate correlation between feature changes and prediction impacts
        correlations = {}
        for feature, data in feature_impacts.items():
            if len(data["changes"]) < 2:
                continue

            # Simplified correlation calculation
            try:
                correlation = np.corrcoef(data["changes"], data["impacts"])[0, 1]
                correlations[feature] = correlation
            except:
                continue

        # Sort features by absolute correlation
        sorted_features = sorted(correlations.items(),
                                key=lambda x: abs(x[1]),
                                reverse=True)

        feature_analysis = []
        for feature, correlation in sorted_features:
            feature_analysis.append({
                "feature": feature,
                "correlation": correlation,
                "samples": len(feature_impacts[feature]["changes"]),
                "average_change": sum(feature_impacts[feature]["changes"]) / len(feature_impacts[feature]["changes"]),
                "impact_strength": "high" if abs(correlation) > 0.7 else
                                   "medium" if abs(correlation) > 0.3 else "low"
            })

        return {
            "feature_impact_analysis": feature_analysis,
            "num_features_analyzed": len(feature_analysis),
            "most_impactful_feature": feature_analysis[0]["feature"] if feature_analysis else None
        }

    def get_counterfactual(self, counterfactual_id: str) -> Dict[str, Any]:
        """Retrieve a stored counterfactual by ID"""
        if counterfactual_id in self.counterfactual_history:
            return self.counterfactual_history[counterfactual_id]
        else:
            return {"status": "not_found", "message": f"Counterfactual {counterfactual_id} not found"}

    def register_scenario_validator(self, domain: str, validator_func: Callable) -> bool:
        """Register a domain-specific validator function for counterfactual scenarios"""
        try:
            self.scenario_validators[domain] = validator_func
            logger.info(f"Registered scenario validator for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Error registering scenario validator: {str(e)}")
            return False

    def set_sensitivity(self, feature: str, sensitivity: float, domain: str = "general") -> bool:
        """Set the sensitivity of a feature in a specific domain"""
        try:
            if domain not in self.sensitivity_mappings:
                self.sensitivity_mappings[domain] = {}

            self.sensitivity_mappings[domain][feature] = max(0.0, min(1.0, sensitivity))
            logger.info(f"Set sensitivity for feature '{feature}' in domain '{domain}' to {sensitivity}")
            return True
        except Exception as e:
            logger.error(f"Error setting feature sensitivity: {str(e)}")
            return False
