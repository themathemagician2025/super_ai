# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Math Predictors Integration Module

This module integrates advanced mathematical predictors with the core AI system.
It leverages mathematical theories and algorithms to enhance prediction capabilities.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union

# Try to import optional components
try:
    from core_ai.dangerousai import DangerousAI
    _has_dangerous = True
except ImportError:
    _has_dangerous = False
    logging.warning("DangerousAI module not available, some features will be limited")

try:
    from core_ai.learning_integration import LearningIntegration
    _has_learning = True
except ImportError:
    _has_learning = False
    logging.warning("LearningIntegration module not available, adaptive learning disabled")

# Import advanced predictors
from algorithms.mathematical_conjectures import BaseConjecture
from algorithms.conjecture_implementations import (
    RiemannMatrixEnsemble,
    NavierStokesPredictor,
    CollatzRecursivePredictor,
    GoldbachDecomposer,
    YangMillsPhasePredictor
)


class MathPredictorsIntegration:
    """
    Manages the integration of various mathematical predictors with the core AI system.
    Provides methods for initialization, configuration, and prediction.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the math predictors integration module.

        Args:
            config_path: Path to the configuration file. If None, uses default config.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Math Predictors Integration")

        # Initialize predictors
        self.predictors = {}
        self.active_predictors = set()

        # Default configurations for predictors
        self.default_configs = {
            "riemann": {
                "ensemble_size": 50,
                "matrix_dimension": 100,
                "confidence_threshold": 0.75,
                "enabled": True
            },
            "navier_stokes": {
                "grid_size": 32,
                "time_steps": 100,
                "viscosity": 0.01,
                "enabled": True
            },
            "collatz": {
                "max_iterations": 1000,
                "sequence_length": 20,
                "enabled": True
            },
            "goldbach": {
                "max_prime": 10000,
                "decomposition_limit": 100,
                "enabled": True
            },
            "yang_mills": {
                "phase_sensitivity": 0.1,
                "transition_threshold": 0.5,
                "field_dimensions": 4,
                "enabled": True
            }
        }

        # Load configuration
        self.config = self.default_configs.copy()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Initialize predictors
        self._init_predictors()

        # Set up learning integration if available
        self.learning = None
        if _has_learning:
            self.learning = LearningIntegration()
            self.logger.info("Learning integration enabled")

        # Set up dangerous AI integration if available
        self.dangerous = None
        if _has_dangerous:
            self.dangerous = DangerousAI(safe_mode=True)
            self.logger.info("DangerousAI integration enabled in safe mode")

    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.

        Args:
            config_path: Path to the configuration file.
        """
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            # Update default config with user config
            for key, value in user_config.items():
                if key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value

            self.logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.logger.info("Using default configuration")

    def save_config(self, config_path: str) -> bool:
        """
        Save the current configuration to a JSON file.

        Args:
            config_path: Path to save the configuration file.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuration saved to {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def _init_predictors(self) -> None:
        """Initialize all predictors based on configuration."""
        # Riemann Matrix Ensemble
        if self.config["riemann"]["enabled"]:
            self.predictors["riemann"] = RiemannMatrixEnsemble(
                ensemble_size=self.config["riemann"]["ensemble_size"],
                matrix_dimension=self.config["riemann"]["matrix_dimension"],
                confidence_threshold=self.config["riemann"]["confidence_threshold"]
            )
            self.active_predictors.add("riemann")
            self.logger.info("Riemann Matrix Ensemble predictor initialized")

        # Navier-Stokes Predictor
        if self.config["navier_stokes"]["enabled"]:
            self.predictors["navier_stokes"] = NavierStokesPredictor(
                grid_size=self.config["navier_stokes"]["grid_size"],
                time_steps=self.config["navier_stokes"]["time_steps"],
                viscosity=self.config["navier_stokes"]["viscosity"]
            )
            self.active_predictors.add("navier_stokes")
            self.logger.info("Navier-Stokes predictor initialized")

        # Collatz Recursive Predictor
        if self.config["collatz"]["enabled"]:
            self.predictors["collatz"] = CollatzRecursivePredictor(
                max_iterations=self.config["collatz"]["max_iterations"],
                sequence_length=self.config["collatz"]["sequence_length"]
            )
            self.active_predictors.add("collatz")
            self.logger.info("Collatz Recursive predictor initialized")

        # Goldbach Decomposer
        if self.config["goldbach"]["enabled"]:
            self.predictors["goldbach"] = GoldbachDecomposer(
                max_prime=self.config["goldbach"]["max_prime"],
                decomposition_limit=self.config["goldbach"]["decomposition_limit"]
            )
            self.active_predictors.add("goldbach")
            self.logger.info("Goldbach Decomposer initialized")

        # Yang-Mills Phase Predictor
        if self.config["yang_mills"]["enabled"]:
            self.predictors["yang_mills"] = YangMillsPhasePredictor(
                phase_sensitivity=self.config["yang_mills"]["phase_sensitivity"],
                transition_threshold=self.config["yang_mills"]["transition_threshold"],
                field_dimensions=self.config["yang_mills"]["field_dimensions"]
            )
            self.active_predictors.add("yang_mills")
            self.logger.info("Yang-Mills Phase predictor initialized")

    def enable_predictor(self, predictor_name: str) -> bool:
        """
        Enable a specific predictor.

        Args:
            predictor_name: Name of the predictor to enable.

        Returns:
            bool: True if predictor was enabled, False otherwise.
        """
        if predictor_name not in self.predictors:
            if predictor_name in self.config and not self.config[predictor_name]["enabled"]:
                self.config[predictor_name]["enabled"] = True
                self._init_predictors()  # Reinitialize predictors
                return True
            return False

        if predictor_name not in self.active_predictors:
            self.active_predictors.add(predictor_name)
            self.logger.info(f"Enabled predictor: {predictor_name}")
            return True

        return False  # Already enabled

    def disable_predictor(self, predictor_name: str) -> bool:
        """
        Disable a specific predictor.

        Args:
            predictor_name: Name of the predictor to disable.

        Returns:
            bool: True if predictor was disabled, False otherwise.
        """
        if predictor_name in self.active_predictors:
            self.active_predictors.remove(predictor_name)
            self.logger.info(f"Disabled predictor: {predictor_name}")
            return True

        return False  # Already disabled or doesn't exist

    def register_custom_predictor(self, name: str, predictor: BaseConjecture, config: Dict[str, Any] = None) -> bool:
        """
        Register a custom predictor.

        Args:
            name: Name for the custom predictor.
            predictor: The predictor instance (must inherit from BaseConjecture).
            config: Configuration for the predictor.

        Returns:
            bool: True if registration was successful, False otherwise.
        """
        if not isinstance(predictor, BaseConjecture):
            self.logger.error(f"Predictor must inherit from BaseConjecture")
            return False

        self.predictors[name] = predictor
        self.active_predictors.add(name)

        if config:
            config["enabled"] = True
            self.config[name] = config

        self.logger.info(f"Registered custom predictor: {name}")
        return True

    def unregister_custom_predictor(self, name: str) -> bool:
        """
        Unregister a custom predictor.

        Args:
            name: Name of the custom predictor to unregister.

        Returns:
            bool: True if unregistration was successful, False otherwise.
        """
        if name in self.predictors and name not in self.default_configs:
            del self.predictors[name]
            if name in self.active_predictors:
                self.active_predictors.remove(name)
            if name in self.config:
                del self.config[name]

            self.logger.info(f"Unregistered custom predictor: {name}")
            return True

        return False  # Not a custom predictor or doesn't exist

    def predict(self, data: Union[np.ndarray, List[float]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate predictions using active mathematical predictors.

        Args:
            data: Input data for prediction.
            context: Additional context for prediction.

        Returns:
            Dict containing prediction results from all active predictors.
        """
        if isinstance(data, list):
            data = np.array(data)

        results = {}
        overall_confidence = 0.0
        active_count = 0

        for predictor_name in self.active_predictors:
            predictor = self.predictors.get(predictor_name)
            if predictor:
                try:
                    prediction = predictor.predict(data, context)
                    results[predictor_name] = prediction

                    # Extract confidence if available
                    if isinstance(prediction, dict) and 'confidence' in prediction:
                        overall_confidence += prediction['confidence']
                        active_count += 1
                except Exception as e:
                    self.logger.error(f"Error in {predictor_name} predictor: {e}")

        # Calculate overall confidence
        if active_count > 0:
            results["overall_confidence"] = overall_confidence / active_count
        else:
            results["overall_confidence"] = 0.0

        results["active_predictors"] = list(self.active_predictors)
        return results


# Simple test code
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize integration
    integration = MathPredictorsIntegration()

    # Generate test data
    test_data = np.random.rand(100)

    # Make predictions
    predictions = integration.predict(test_data)

    # Print results
    print(f"Overall prediction confidence: {predictions['overall_confidence']:.2f}")
    print(f"Active predictors: {predictions['active_predictors']}")
