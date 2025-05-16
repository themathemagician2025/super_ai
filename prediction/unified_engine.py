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
from typing import Dict, Any, List, Optional, Union, Callable
import importlib
import os

logger = logging.getLogger(__name__)

class UnifiedPredictionEngine:
    """
    Unified Cross-Domain Prediction Engine capable of forecasting across
    unrelated domains like forex, sports, weather, stocks, etc. in real-time.
    """

    def __init__(self, config_path: Path = Path("config/unified_prediction.yaml")):
        self.config_path = config_path
        self.domain_engines = {}  # Specialized engines for each domain
        self.domain_weights = {}  # Domain expertise weights
        self.domain_metrics = {}  # Performance metrics for each domain
        self.transfer_strategies = {}  # Cross-domain knowledge transfer
        self.prediction_history = []
        self.meta_features = {}  # Features that work across domains
        self.causal_models = {}  # Domain-specific causal models
        self.active_domains = []  # Currently enabled domains
        self.domain_confidence_thresholds = {}  # Minimum confidence for predictions
        self.cross_domain_features = []  # Features that apply across domains

        # Initialize domain adapters
        self.domain_adapters = {
            "forex": self._forex_adapter,
            "sports": self._sports_adapter,
            "weather": self._weather_adapter,
            "stocks": self._stocks_adapter,
            "crypto": self._crypto_adapter,
            "politics": self._politics_adapter,
            "default": self._default_adapter
        }

        # Load configuration if it exists
        self._load_config()

        # Initialize domain engines
        self._initialize_domain_engines()

        logger.info(f"Unified Prediction Engine initialized with domains: {', '.join(self.active_domains)}")

    def _load_config(self):
        """Load configuration settings if available"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        # Load domain weights and thresholds
                        self.domain_weights = config.get('domain_weights', {})
                        self.domain_confidence_thresholds = config.get('confidence_thresholds', {})
                        self.active_domains = config.get('active_domains', [])
                        self.cross_domain_features = config.get('cross_domain_features', [])

                        # Load domain-specific engine configurations
                        domain_configs = config.get('domain_engines', {})
                        for domain, domain_config in domain_configs.items():
                            if domain in self.active_domains:
                                module_path = domain_config.get('module_path')
                                class_name = domain_config.get('class_name')
                                config_path = domain_config.get('config_path')

                                if module_path and class_name:
                                    self.domain_engines[domain] = {
                                        "module_path": module_path,
                                        "class_name": class_name,
                                        "config_path": config_path,
                                        "instance": None  # Will be initialized later
                                    }

                        logger.info(f"Loaded unified prediction configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Set default active domains if none were loaded
            if not self.active_domains:
                self.active_domains = ["forex", "sports", "stocks", "weather"]

    def _initialize_domain_engines(self):
        """Initialize specialized prediction engines for each active domain"""
        for domain in self.active_domains:
            try:
                if domain in self.domain_engines:
                    # Engine configuration was loaded from config
                    engine_config = self.domain_engines[domain]
                    module_path = engine_config["module_path"]
                    class_name = engine_config["class_name"]
                    config_path = engine_config.get("config_path")

                    # Import the module and initialize the engine
                    module = importlib.import_module(module_path)
                    engine_class = getattr(module, class_name)

                    # Initialize with config path if provided
                    if config_path:
                        engine_instance = engine_class(config_path=Path(config_path))
                    else:
                        engine_instance = engine_class()

                    # Store the instance
                    engine_config["instance"] = engine_instance
                    logger.info(f"Initialized {domain} prediction engine: {class_name}")
                else:
                    # No specific engine configured, use a default adapter
                    logger.info(f"No specific engine configured for {domain}, using adapter")
                    self.domain_engines[domain] = {
                        "instance": None,
                        "adapter": self.domain_adapters.get(domain, self.domain_adapters["default"])
                    }

                # Initialize domain metrics
                self.domain_metrics[domain] = {
                    "predictions_count": 0,
                    "accuracy": 0.0,
                    "last_updated": None
                }

                # Set default domain weight if not specified
                if domain not in self.domain_weights:
                    self.domain_weights[domain] = 1.0

                # Set default confidence threshold if not specified
                if domain not in self.domain_confidence_thresholds:
                    self.domain_confidence_thresholds[domain] = 0.6

            except Exception as e:
                logger.error(f"Error initializing {domain} engine: {str(e)}")
                # Remove from active domains if initialization failed
                if domain in self.active_domains:
                    self.active_domains.remove(domain)

    def predict(self,
               data: Dict[str, Any],
               domain: str,
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a prediction for a specific domain

        Parameters:
        - data: Input data for the prediction
        - domain: The domain for this prediction (forex, sports, etc.)
        - context: Optional context information

        Returns:
        - Prediction result with confidence and explanation
        """
        try:
            if domain not in self.active_domains:
                return {
                    "status": "error",
                    "message": f"Domain '{domain}' is not active",
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Add cross-domain features if available
            enriched_data = self._enrich_with_cross_domain_features(data, domain)

            # Get domain engine or adapter
            engine_config = self.domain_engines.get(domain, {})
            engine_instance = engine_config.get("instance")

            if engine_instance:
                # Use specialized engine for this domain
                try:
                    raw_prediction = engine_instance.predict(enriched_data)
                except Exception as e:
                    logger.error(f"Error using {domain} engine: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error in {domain} prediction engine: {str(e)}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
            else:
                # Use domain adapter
                adapter = engine_config.get("adapter", self.domain_adapters["default"])
                raw_prediction = adapter(enriched_data, context)

            # Post-process prediction
            prediction = self._post_process_prediction(raw_prediction, domain, data)

            # Apply cross-domain knowledge
            enhanced_prediction = self._apply_cross_domain_knowledge(prediction, domain, data)

            # Record prediction in history
            prediction_record = {
                "domain": domain,
                "input_data": {k: v for k, v in data.items() if k not in ["raw_data", "history"]},
                "prediction": enhanced_prediction,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.prediction_history.append(prediction_record)

            # Update domain metrics
            self.domain_metrics[domain]["predictions_count"] += 1
            self.domain_metrics[domain]["last_updated"] = datetime.datetime.now().isoformat()

            return enhanced_prediction

        except Exception as e:
            logger.error(f"Error in unified prediction: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _enrich_with_cross_domain_features(self, data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Add cross-domain features to the input data"""
        enriched_data = data.copy()

        # Add relevant meta-features
        for feature_name in self.cross_domain_features:
            if feature_name in self.meta_features and feature_name not in enriched_data:
                enriched_data[feature_name] = self.meta_features[feature_name]

        # Add domain-specific transformations for cross-domain features
        # This would depend on the specific features and domains

        return enriched_data

    def _post_process_prediction(self,
                                prediction: Dict[str, Any],
                                domain: str,
                                original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process a prediction from a domain engine"""
        processed = prediction.copy()

        # Ensure all required fields are present
        if "confidence" not in processed:
            processed["confidence"] = 0.7  # Default confidence

        if "timestamp" not in processed:
            processed["timestamp"] = datetime.datetime.now().isoformat()

        if "domain" not in processed:
            processed["domain"] = domain

        # Check if confidence meets the threshold
        threshold = self.domain_confidence_thresholds.get(domain, 0.6)
        if processed.get("confidence", 0) < threshold:
            processed["below_threshold"] = True
            processed["threshold"] = threshold

        return processed

    def _apply_cross_domain_knowledge(self,
                                     prediction: Dict[str, Any],
                                     domain: str,
                                     data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply knowledge from other domains to enhance prediction"""
        enhanced = prediction.copy()

        # Apply transfer strategies if available
        strategy = self.transfer_strategies.get(domain)
        if strategy:
            try:
                enhanced = strategy(prediction, data, self.prediction_history)
            except Exception as e:
                logger.error(f"Error applying transfer strategy for {domain}: {str(e)}")

        # For demonstration, add a simple cross-domain confidence adjustment
        # In a real implementation, this would be much more sophisticated
        related_domains = self._find_related_domains(domain, data)
        if related_domains:
            # Get recent predictions in related domains
            recent_related = self._get_recent_domain_predictions(related_domains, limit=5)

            # Apply a simple confidence adjustment based on related domain performance
            if recent_related:
                avg_confidence = sum(p.get("prediction", {}).get("confidence", 0)
                                     for p in recent_related) / len(recent_related)

                # Small adjustment based on related domains
                confidence_adjustment = (avg_confidence - 0.7) * 0.1  # Modest influence
                new_confidence = enhanced.get("confidence", 0.7) + confidence_adjustment

                # Keep confidence in valid range
                enhanced["confidence"] = max(0.1, min(0.99, new_confidence))
                enhanced["cross_domain_adjusted"] = True

        return enhanced

    def _find_related_domains(self, domain: str, data: Dict[str, Any]) -> List[str]:
        """Find domains that might have related insights for this prediction"""
        # In a real implementation, this would use actual domain relationships
        # For demonstration, use simple predefined relationships

        related = {
            "forex": ["stocks", "crypto"],
            "stocks": ["forex", "crypto"],
            "crypto": ["forex", "stocks"],
            "sports": ["politics"],  # Sports and politics might share sentiment features
            "weather": []  # Weather is more isolated
        }

        return related.get(domain, [])

    def _get_recent_domain_predictions(self,
                                      domains: List[str],
                                      limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent predictions from specified domains"""
        # Filter history for requested domains and sort by recency
        domain_predictions = [p for p in self.prediction_history
                             if p.get("domain") in domains]

        # Sort by timestamp (most recent first)
        sorted_predictions = sorted(
            domain_predictions,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return sorted_predictions[:limit]

    def evaluate(self,
                 actual_outcomes: List[Dict[str, Any]],
                 update_metrics: bool = True) -> Dict[str, Any]:
        """
        Evaluate predictions against actual outcomes

        Parameters:
        - actual_outcomes: List of actual outcomes with prediction IDs
        - update_metrics: Whether to update domain metrics

        Returns:
        - Evaluation results
        """
        try:
            if not actual_outcomes:
                return {
                    "status": "error",
                    "message": "No outcomes provided for evaluation",
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Match outcomes with predictions from history
            matched_results = []
            domain_results = {}

            for outcome in actual_outcomes:
                prediction_id = outcome.get("prediction_id")
                domain = outcome.get("domain")

                # Find the prediction in history
                prediction_record = None
                for record in self.prediction_history:
                    if record.get("prediction", {}).get("id") == prediction_id:
                        prediction_record = record
                        break

                if not prediction_record:
                    logger.warning(f"No prediction found for outcome {prediction_id}")
                    continue

                # Get predicted and actual values
                predicted = prediction_record.get("prediction", {}).get("value")
                actual = outcome.get("value")

                # Calculate accuracy (simplified for demonstration)
                accuracy = self._calculate_accuracy(predicted, actual, domain)

                # Create result record
                result = {
                    "prediction_id": prediction_id,
                    "domain": domain,
                    "predicted": predicted,
                    "actual": actual,
                    "accuracy": accuracy,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                matched_results.append(result)

                # Aggregate by domain
                if domain not in domain_results:
                    domain_results[domain] = {
                        "count": 0,
                        "accuracy_sum": 0
                    }

                domain_results[domain]["count"] += 1
                domain_results[domain]["accuracy_sum"] += accuracy

            # Calculate domain averages
            domain_averages = {}
            for domain, results in domain_results.items():
                if results["count"] > 0:
                    avg_accuracy = results["accuracy_sum"] / results["count"]
                    domain_averages[domain] = avg_accuracy

                    # Update domain metrics if requested
                    if update_metrics and domain in self.domain_metrics:
                        # Update with exponential moving average
                        current = self.domain_metrics[domain]["accuracy"]
                        alpha = 0.1  # Learning rate
                        new_accuracy = (current * (1 - alpha)) + (avg_accuracy * alpha)
                        self.domain_metrics[domain]["accuracy"] = new_accuracy

            return {
                "status": "success",
                "results_count": len(matched_results),
                "results": matched_results,
                "domain_averages": domain_averages,
                "timestamp": datetime.datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error evaluating predictions: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _calculate_accuracy(self, predicted: Any, actual: Any, domain: str) -> float:
        """Calculate prediction accuracy based on domain"""
        if domain == "forex" or domain == "stocks" or domain == "crypto":
            # For financial predictions, check direction correctness
            if isinstance(predicted, dict) and "direction" in predicted:
                return 1.0 if predicted["direction"] == actual.get("direction") else 0.0
            elif isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                predicted_direction = "up" if predicted > 0 else "down"
                actual_direction = "up" if actual > 0 else "down"
                return 1.0 if predicted_direction == actual_direction else 0.0

        elif domain == "sports":
            # For sports predictions, check winner correctness
            if isinstance(predicted, dict) and "winner" in predicted:
                return 1.0 if predicted["winner"] == actual.get("winner") else 0.0

        elif domain == "weather":
            # For weather, allow for tolerance in predictions
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                error = abs(predicted - actual)
                tolerance = 5.0  # Example: 5 degrees tolerance
                return max(0.0, 1.0 - (error / tolerance))

        # Default: simple equality check
        return 1.0 if predicted == actual else 0.0

    def update_meta_features(self, features: Dict[str, Any]):
        """Update global meta-features that may affect multiple domains"""
        self.meta_features.update(features)
        logger.info(f"Updated meta features: {', '.join(features.keys())}")

    def register_transfer_strategy(self, domain: str, strategy_func: Callable) -> bool:
        """Register a domain-specific knowledge transfer strategy"""
        try:
            self.transfer_strategies[domain] = strategy_func
            logger.info(f"Registered transfer strategy for domain: {domain}")
            return True
        except Exception as e:
            logger.error(f"Error registering transfer strategy: {str(e)}")
            return False

    def get_domain_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all domains"""
        return self.domain_metrics

    def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions across all domains"""
        return sorted(
            self.prediction_history,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )[:limit]

    # Domain-specific adapter methods
    def _forex_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for forex prediction"""
        # In a real implementation, this would implement actual forex predictions
        # For demonstration, return a simple structured prediction

        # Extract key features
        pair = data.get("pair", "EUR/USD")
        timeframe = data.get("timeframe", "1h")

        # Generate simulated prediction
        direction = "up" if np.random.random() > 0.5 else "down"
        pips = np.random.randint(5, 50)

        return {
            "id": f"forex_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "pair": pair,
            "timeframe": timeframe,
            "direction": direction,
            "pips": pips,
            "confidence": np.random.random() * 0.3 + 0.6,  # Random between 0.6-0.9
            "explanation": f"Predicted {direction} movement of {pips} pips for {pair} on {timeframe} timeframe"
        }

    def _sports_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for sports prediction"""
        # Extract key features
        sport = data.get("sport", "soccer")
        team1 = data.get("team1", "Team A")
        team2 = data.get("team2", "Team B")

        # Generate simulated prediction
        winner = team1 if np.random.random() > 0.5 else team2
        margin = np.random.randint(1, 5)

        return {
            "id": f"sports_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "sport": sport,
            "match": f"{team1} vs {team2}",
            "winner": winner,
            "margin": margin,
            "win_probability": np.random.random() * 0.4 + 0.5,  # Random between 0.5-0.9
            "confidence": np.random.random() * 0.3 + 0.6,  # Random between 0.6-0.9
            "explanation": f"Predicted {winner} to win by {margin}"
        }

    def _stocks_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for stock market prediction"""
        # Extract key features
        ticker = data.get("ticker", "AAPL")
        timeframe = data.get("timeframe", "1d")

        # Generate simulated prediction
        direction = "up" if np.random.random() > 0.5 else "down"
        percent_change = (np.random.random() * 5).round(2)  # 0-5% change

        return {
            "id": f"stocks_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "ticker": ticker,
            "timeframe": timeframe,
            "direction": direction,
            "percent_change": percent_change,
            "confidence": np.random.random() * 0.3 + 0.6,  # Random between 0.6-0.9
            "explanation": f"Predicted {direction} movement of {percent_change}% for {ticker} on {timeframe} timeframe"
        }

    def _weather_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for weather prediction"""
        # Extract key features
        location = data.get("location", "New York")
        date = data.get("date", "tomorrow")

        # Generate simulated prediction
        temp_high = np.random.randint(10, 35)  # 10-35°C
        temp_low = temp_high - np.random.randint(5, 15)  # 5-15°C lower
        conditions = np.random.choice(["sunny", "cloudy", "rainy", "stormy"])

        return {
            "id": f"weather_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "location": location,
            "date": date,
            "high_temp": temp_high,
            "low_temp": temp_low,
            "conditions": conditions,
            "confidence": np.random.random() * 0.2 + 0.7,  # Random between 0.7-0.9
            "explanation": f"Predicted {conditions} conditions with high of {temp_high}°C and low of {temp_low}°C for {location} on {date}"
        }

    def _crypto_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for cryptocurrency prediction"""
        # Extract key features
        coin = data.get("coin", "BTC")
        timeframe = data.get("timeframe", "1d")

        # Generate simulated prediction
        direction = "up" if np.random.random() > 0.5 else "down"
        percent_change = (np.random.random() * 10).round(2)  # 0-10% change

        return {
            "id": f"crypto_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "coin": coin,
            "timeframe": timeframe,
            "direction": direction,
            "percent_change": percent_change,
            "confidence": np.random.random() * 0.4 + 0.5,  # Random between 0.5-0.9
            "explanation": f"Predicted {direction} movement of {percent_change}% for {coin} on {timeframe} timeframe"
        }

    def _politics_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adapter for political prediction"""
        # Extract key features
        event = data.get("event", "election")
        candidates = data.get("candidates", ["Candidate A", "Candidate B"])

        # Generate simulated prediction
        winner = np.random.choice(candidates)
        win_prob = np.random.random() * 0.5 + 0.5  # Random between 0.5-1.0

        return {
            "id": f"politics_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "event": event,
            "candidates": candidates,
            "predicted_winner": winner,
            "win_probability": win_prob,
            "confidence": np.random.random() * 0.3 + 0.6,  # Random between 0.6-0.9
            "explanation": f"Predicted {winner} to win the {event} with {win_prob:.1%} probability"
        }

    def _default_adapter(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Default adapter for unknown domains"""
        domain = data.get("domain", "unknown")

        return {
            "id": f"{domain}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "raw_prediction": np.random.random(),
            "confidence": 0.5,  # Low confidence for unknown domain
            "explanation": "Generic prediction for unknown domain"
        }
