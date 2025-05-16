# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import json
import datetime
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import threading

# Import our advanced AI modules
from src.self_modification.engine import SelfModificationEngine
from src.explainability.causal_traceback import CausalTraceback
from src.learning.continuous_learner import ContinuousLearner
from src.reasoning.counterfactual import CounterfactualReasoner
from src.knowledge.external_learner import ExternalLearner
from src.prediction.unified_engine import UnifiedPredictionEngine

logger = logging.getLogger(__name__)

class ConceptualEngine:
    """
    The master orchestration engine that integrates all advanced AI capabilities:
    - Self-debugging and modification
    - Causal inference
    - Continuous online learning
    - Prediction explainability
    - Counterfactual reasoning
    - Internet-sourced knowledge
    - Unified cross-domain prediction
    - Regulatory awareness
    - Emotional and sentiment awareness
    """

    def __init__(self, config_path: Path = Path("config/conceptual_engine.yaml")):
        self.config_path = config_path
        self.components = {}
        self.active_components = []
        self.domain_registry = {}
        self.regulatory_framework = {}
        self.sentiment_analyzers = {}
        self.running = False
        self.last_health_check = None
        self.performance_metrics = {}

        # Initialize engines (lazy loading - only created when accessed)
        self._self_modification_engine = None
        self._causal_traceback = None
        self._continuous_learner = None
        self._counterfactual_reasoner = None
        self._external_learner = None
        self._unified_prediction = None

        # Locks for thread safety
        self.thread_locks = {
            "prediction": threading.Lock(),
            "learning": threading.Lock(),
            "modification": threading.Lock(),
            "knowledge": threading.Lock()
        }

        # Load configuration
        self._load_config()

        # Initialize all components marked as auto_load
        self._initialize_components()

        logger.info("Conceptual Engine initialized")

    def _load_config(self):
        """Load configuration settings"""
        try:
            if self.config_path.exists():
                import yaml
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config:
                        # Component configuration
                        self.components = config.get('components', {})
                        self.active_components = [
                            name for name, comp_config in self.components.items()
                            if comp_config.get('active', False)
                        ]

                        # Domain configuration
                        self.domain_registry = config.get('domains', {})

                        # Regulatory framework
                        self.regulatory_framework = config.get('regulatory_framework', {})

                        logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                # Set default components
                self.components = {
                    "self_modification": {"active": True, "auto_load": True},
                    "causal_traceback": {"active": True, "auto_load": False},
                    "continuous_learning": {"active": True, "auto_load": True},
                    "counterfactual": {"active": True, "auto_load": False},
                    "external_learning": {"active": True, "auto_load": False},
                    "unified_prediction": {"active": True, "auto_load": True}
                }
                self.active_components = ["self_modification", "continuous_learning", "unified_prediction"]

        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def _initialize_components(self):
        """Initialize components marked for auto-loading"""
        for component_name, config in self.components.items():
            if config.get('active', False) and config.get('auto_load', False):
                self._load_component(component_name)

    def _load_component(self, component_name: str):
        """Load a specific component if not already loaded"""
        try:
            if component_name == "self_modification" and self._self_modification_engine is None:
                self._self_modification_engine = SelfModificationEngine()
                logger.info("Self-Modification Engine loaded")

            elif component_name == "causal_traceback" and self._causal_traceback is None:
                self._causal_traceback = CausalTraceback()
                logger.info("Causal Traceback Engine loaded")

            elif component_name == "continuous_learning" and self._continuous_learner is None:
                self._continuous_learner = ContinuousLearner()
                logger.info("Continuous Learning Engine loaded")

            elif component_name == "counterfactual" and self._counterfactual_reasoner is None:
                self._counterfactual_reasoner = CounterfactualReasoner()
                logger.info("Counterfactual Reasoning Engine loaded")

            elif component_name == "external_learning" and self._external_learner is None:
                self._external_learner = ExternalLearner()
                logger.info("External Learning Engine loaded")

            elif component_name == "unified_prediction" and self._unified_prediction is None:
                self._unified_prediction = UnifiedPredictionEngine()
                logger.info("Unified Prediction Engine loaded")

        except Exception as e:
            logger.error(f"Error loading component {component_name}: {str(e)}")

    def start(self) -> bool:
        """Start the Conceptual Engine and all its components"""
        if self.running:
            logger.warning("Conceptual Engine is already running")
            return True

        try:
            logger.info("Starting Conceptual Engine...")

            # Ensure all active components are loaded
            for component in self.active_components:
                self._load_component(component)

            # Perform health check
            health_status = self.health_check()
            if health_status["status"] == "error":
                logger.error(f"Health check failed: {health_status['message']}")
                return False

            self.running = True
            logger.info("Conceptual Engine started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting Conceptual Engine: {str(e)}")
            return False

    def stop(self) -> bool:
        """Stop the Conceptual Engine"""
        if not self.running:
            logger.warning("Conceptual Engine is not running")
            return True

        try:
            logger.info("Stopping Conceptual Engine...")
            self.running = False

            # Save state for components that support it
            if self._continuous_learner:
                self._continuous_learner.save_state()

            logger.info("Conceptual Engine stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping Conceptual Engine: {str(e)}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of all active components"""
        if not self.active_components:
            return {
                "status": "warning",
                "message": "No active components",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            component_status = {}

            # Check each active component
            for component in self.active_components:
                if component == "self_modification" and self._self_modification_engine:
                    component_status[component] = "ok"

                elif component == "causal_traceback" and self._causal_traceback:
                    component_status[component] = "ok"

                elif component == "continuous_learning" and self._continuous_learner:
                    component_status[component] = "ok"

                elif component == "counterfactual" and self._counterfactual_reasoner:
                    component_status[component] = "ok"

                elif component == "external_learning" and self._external_learner:
                    component_status[component] = "ok"

                elif component == "unified_prediction" and self._unified_prediction:
                    component_status[component] = "ok"

                else:
                    component_status[component] = "not_loaded"

            # Overall status
            overall_status = "ok"
            if "not_loaded" in component_status.values():
                overall_status = "warning"

            health_result = {
                "status": overall_status,
                "components": component_status,
                "timestamp": datetime.datetime.now().isoformat()
            }

            self.last_health_check = health_result
            return health_result

        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def predict(self,
               data: Dict[str, Any],
               domain: str,
               explain: bool = False,
               generate_counterfactuals: bool = False) -> Dict[str, Any]:
        """
        Make a prediction with optional explainability and counterfactuals

        Parameters:
        - data: Input data for the prediction
        - domain: The domain for this prediction (forex, sports, etc.)
        - explain: Whether to generate an explanation
        - generate_counterfactuals: Whether to generate counterfactual scenarios

        Returns:
        - Prediction result with optional explanation and counterfactuals
        """
        if not self.running:
            return {
                "status": "error",
                "message": "Conceptual Engine is not running",
                "timestamp": datetime.datetime.now().isoformat()
            }

        # Check if domain is registered
        if domain not in self.domain_registry and domain not in (self._unified_prediction.active_domains if self._unified_prediction else []):
            return {
                "status": "error",
                "message": f"Unknown domain: {domain}",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            with self.thread_locks["prediction"]:
                # Make prediction
                if not self._unified_prediction:
                    self._load_component("unified_prediction")

                if not self._unified_prediction:
                    return {
                        "status": "error",
                        "message": "Unified Prediction Engine is not available",
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                # Check regulatory compliance if applicable
                if self.regulatory_framework and domain in self.regulatory_framework:
                    compliance_check = self._check_regulatory_compliance(data, domain)
                    if not compliance_check["compliant"]:
                        return {
                            "status": "regulatory_violation",
                            "message": compliance_check["reason"],
                            "domain": domain,
                            "timestamp": datetime.datetime.now().isoformat()
                        }

                # Add sentiment analysis if available
                if domain in self.sentiment_analyzers:
                    sentiment_data = self._analyze_sentiment(data, domain)
                    data.update(sentiment_data)

                # Make the prediction
                prediction = self._unified_prediction.predict(data, domain)

                # Generate explanation if requested
                explanation = None
                if explain and self._causal_traceback:
                    model_info = {
                        "domain": domain,
                        "target_variable": "prediction",
                        "model_type": "unified_prediction_engine"
                    }
                    prediction_data = {
                        "id": prediction.get("id", f"pred_{datetime.datetime.now().timestamp()}"),
                        "prediction": prediction,
                        "confidence": prediction.get("confidence", 0.5),
                        "features": data
                    }
                    explanation = self._causal_traceback.explain_prediction(
                        prediction_data, model_info
                    )

                # Generate counterfactuals if requested
                counterfactuals = []
                if generate_counterfactuals and self._counterfactual_reasoner:
                    # Define some interesting counterfactual scenarios based on domain
                    interventions = self._generate_domain_interventions(data, domain)

                    for intervention_name, intervention in interventions.items():
                        cf_result = self._counterfactual_reasoner.generate_counterfactual(
                            data, intervention, domain
                        )
                        if cf_result.get("status") == "success":
                            counterfactuals.append(cf_result)

                # Construct final result
                result = {
                    "prediction": prediction,
                    "domain": domain,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                if explanation:
                    result["explanation"] = explanation

                if counterfactuals:
                    result["counterfactuals"] = counterfactuals

                # Record experience for continuous learning
                if self._continuous_learner:
                    self._continuous_learner.add_experience({
                        "features": data,
                        "prediction": prediction,
                        "domain": domain
                    })

                return result

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _generate_domain_interventions(self, data: Dict[str, Any], domain: str) -> Dict[str, Dict[str, Any]]:
        """Generate domain-specific counterfactual interventions"""
        interventions = {}

        if domain == "forex" or domain == "stocks" or domain == "crypto":
            # Financial domains
            if "market_sentiment" in data:
                # Flip market sentiment
                current_sentiment = data["market_sentiment"]
                opposite_sentiment = "bearish" if current_sentiment == "bullish" else "bullish"
                interventions["opposite_sentiment"] = {"market_sentiment": opposite_sentiment}

            if "volatility" in data:
                # Higher volatility
                interventions["higher_volatility"] = {"volatility": data["volatility"] * 2}

        elif domain == "sports":
            # Sports domain
            if "home_team_form" in data:
                # Better form for home team
                interventions["better_home_form"] = {"home_team_form": min(1.0, data["home_team_form"] * 1.5)}

            if "key_player_injured" in data and not data["key_player_injured"]:
                # Key player injury scenario
                interventions["key_player_injured"] = {"key_player_injured": True}

        elif domain == "weather":
            # Weather domain
            if "temperature" in data:
                # Higher temperature
                interventions["higher_temp"] = {"temperature": data["temperature"] + 5}

            if "precipitation" in data:
                # More precipitation
                interventions["more_rain"] = {"precipitation": data["precipitation"] + 20}

        return interventions

    def _check_regulatory_compliance(self, data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Check regulatory compliance for a prediction"""
        domain_regulations = self.regulatory_framework.get(domain, {})

        # Default compliance result
        result = {
            "compliant": True,
            "domain": domain,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Check required fields
        required_fields = domain_regulations.get("required_fields", [])
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            result["compliant"] = False
            result["reason"] = f"Missing required fields: {', '.join(missing_fields)}"
            return result

        # Check restricted values
        restricted_values = domain_regulations.get("restricted_values", {})
        for field, restrictions in restricted_values.items():
            if field in data and data[field] in restrictions:
                result["compliant"] = False
                result["reason"] = f"Restricted value for {field}: {data[field]}"
                return result

        # Add compliance info
        result["regulatory_framework"] = domain_regulations.get("name", "Default Framework")

        return result

    def _analyze_sentiment(self, data: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """Analyze sentiment in the input data"""
        # This would use the appropriate sentiment analyzer for the domain
        # For demonstration, return placeholder sentiment data
        sentiment_data = {
            "sentiment_score": 0.5,  # Neutral
            "sentiment_magnitude": 0.2,
            "emotional_signals": {}
        }

        # In a real implementation, this would extract text fields and analyze them

        return sentiment_data

    def learn_from_external_sources(self, topic: str) -> Dict[str, Any]:
        """Learn from external sources about a specific topic"""
        if not self.running:
            return {
                "status": "error",
                "message": "Conceptual Engine is not running",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            with self.thread_locks["knowledge"]:
                if not self._external_learner:
                    self._load_component("external_learning")

                if not self._external_learner:
                    return {
                        "status": "error",
                        "message": "External Learning Engine is not available",
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                # Learn about the topic
                learning_result = self._external_learner.learn_from_topic(topic)

                # Extract facts if learning was successful
                facts = []
                if learning_result.get("status") in ["success", "partial"]:
                    facts_result = self._external_learner.extract_facts(topic)
                    if facts_result.get("status") == "success":
                        facts = facts_result.get("facts", [])

                return {
                    "learning_result": learning_result,
                    "facts_count": len(facts),
                    "timestamp": datetime.datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error learning from external sources: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def detect_and_fix_issues(self, module_path: str = None) -> Dict[str, Any]:
        """Detect and fix issues in the system code"""
        if not self.running:
            return {
                "status": "error",
                "message": "Conceptual Engine is not running",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            with self.thread_locks["modification"]:
                if not self._self_modification_engine:
                    self._load_component("self_modification")

                if not self._self_modification_engine:
                    return {
                        "status": "error",
                        "message": "Self-Modification Engine is not available",
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                # If no module specified, check key modules
                if module_path is None:
                    results = {}
                    for module in [
                        "src/prediction/unified_engine.py",
                        "src/learning/continuous_learner.py",
                        "src/explainability/causal_traceback.py"
                    ]:
                        if Path(module).exists():
                            module_result = self._detect_and_fix_module(module)
                            results[module] = module_result

                    return {
                        "status": "completed",
                        "modules_checked": len(results),
                        "results": results,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                else:
                    # Check specific module
                    if not Path(module_path).exists():
                        return {
                            "status": "error",
                            "message": f"Module not found: {module_path}",
                            "timestamp": datetime.datetime.now().isoformat()
                        }

                    result = self._detect_and_fix_module(module_path)
                    return {
                        "status": "completed",
                        "module": module_path,
                        "result": result,
                        "timestamp": datetime.datetime.now().isoformat()
                    }

        except Exception as e:
            logger.error(f"Error detecting and fixing issues: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def _detect_and_fix_module(self, module_path: str) -> Dict[str, Any]:
        """Detect and fix issues in a specific module"""
        try:
            # Detect bugs
            bugs = self._self_modification_engine.detect_bugs(module_path)

            # Fix bugs if found
            fixed_bugs = []
            for bug in bugs:
                bug_id = f"BUG-{len(fixed_bugs) + 1}"
                fix_result = self._self_modification_engine.fix_bug(bug_id)
                if fix_result:
                    fixed_bugs.append(bug_id)

            return {
                "bugs_found": len(bugs),
                "bugs_fixed": len(fixed_bugs),
                "bug_details": bugs
            }

        except Exception as e:
            logger.error(f"Error processing module {module_path}: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def update_model(self, force: bool = False) -> Dict[str, Any]:
        """Update the prediction model based on new experiences"""
        if not self.running:
            return {
                "status": "error",
                "message": "Conceptual Engine is not running",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            with self.thread_locks["learning"]:
                if not self._continuous_learner:
                    self._load_component("continuous_learning")

                if not self._continuous_learner:
                    return {
                        "status": "error",
                        "message": "Continuous Learning Engine is not available",
                        "timestamp": datetime.datetime.now().isoformat()
                    }

                # Update the model
                update_result = self._continuous_learner.update_model(force=force)

                # Detect concept drift
                drift_result = None
                if self._continuous_learner.experience_buffer:
                    drift_result = self._continuous_learner.detect_concept_drift()

                return {
                    "update_result": update_result,
                    "drift_result": drift_result,
                    "timestamp": datetime.datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def compare_counterfactuals(self, scenario_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple counterfactual scenarios"""
        if not self.running:
            return {
                "status": "error",
                "message": "Conceptual Engine is not running",
                "timestamp": datetime.datetime.now().isoformat()
            }

        try:
            if not self._counterfactual_reasoner:
                self._load_component("counterfactual")

            if not self._counterfactual_reasoner:
                return {
                    "status": "error",
                    "message": "Counterfactual Reasoning Engine is not available",
                    "timestamp": datetime.datetime.now().isoformat()
                }

            # Compare counterfactuals
            comparison = self._counterfactual_reasoner.compare_counterfactuals(scenario_ids)

            return comparison

        except Exception as e:
            logger.error(f"Error comparing counterfactuals: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all components"""
        metrics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "engine_status": "running" if self.running else "stopped",
            "components": {}
        }

        # Get metrics from each component
        try:
            if self._unified_prediction:
                metrics["components"]["prediction"] = self._unified_prediction.get_domain_metrics()

            if self._continuous_learner:
                # Get evaluation results if available
                recent_eval = None
                for key, value in self._continuous_learner.performance_history.items():
                    if key.startswith("eval_"):
                        recent_eval = value
                        break

                metrics["components"]["learning"] = {
                    "model_version": self._continuous_learner.current_model_version,
                    "experience_count": len(self._continuous_learner.experience_buffer),
                    "last_update": self._continuous_learner.last_update_time,
                    "recent_evaluation": recent_eval
                }

            if self._external_learner:
                metrics["components"]["knowledge"] = self._external_learner.get_knowledge_statistics()

            if self._self_modification_engine:
                metrics["components"]["self_modification"] = {
                    "modifications": len(self._self_modification_engine.get_modification_history())
                }

            return metrics

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }

# Create a singleton instance
conceptual_engine = None

def get_engine(config_path: Path = Path("config/conceptual_engine.yaml")) -> ConceptualEngine:
    """Get or create the singleton ConceptualEngine instance"""
    global conceptual_engine
    if conceptual_engine is None:
        conceptual_engine = ConceptualEngine(config_path=config_path)
    return conceptual_engine
