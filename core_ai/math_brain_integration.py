# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Math Brain Integration Module

This module integrates neural network brains with advanced mathematical predictors
to create a hybrid intelligence system that combines the strengths of both approaches.
Features:
- Combines neural network learning with mathematical theory-based predictions
- Supports self-modification and evolution
- Provides multi-dimensional reasoning across domains
- Enables advanced predictive capabilities for various applications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import logging
import time
from typing import Dict, List, Any, Union, Optional, Tuple, Callable

# Import mathematical predictors
from core_ai.advanced_math_predictors import (
    RiemannMatrixEnsemble,
    NavierStokesPredictor,
    CollatzRecursivePredictor,
    GoldbachDecomposer,
    YangMillsPhasePredictor,
    create_advanced_predictors
)

# Optional imports for more capabilities
try:
    from core_ai.dangerousai import DangerousAI
    _has_dangerous = True
except ImportError:
    _has_dangerous = False
    logging.warning("DangerousAI module not available, some advanced features will be limited")

try:
    from core_ai.self_modification import SelfModificationEngine
    _has_self_mod = True
except ImportError:
    _has_self_mod = False
    logging.warning("SelfModificationEngine not available, self-improvement will be limited")

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeniusBrain(nn.Module):
    """
    Neural network brain with direct weight connections.

    Features:
    - Dense connectivity with threshold activation
    - Parameter-based neuron representation
    - Supports gradient-based learning and random evolution
    """

    def __init__(self, neurons=2048, connectivity=0.15):
        """
        Initialize the GeniusBrain.

        Args:
            neurons: Number of neurons in the brain
            connectivity: Connectivity density factor (0-1)
        """
        super().__init__()
        self.neurons = nn.Parameter(torch.randn(neurons))
        self.weights = nn.Parameter((torch.rand(neurons, neurons) - 0.5) * 2 * connectivity)
        self.thresholds = nn.Parameter(torch.randn(neurons) * 0.5)

        logger.info(f"Initialized GeniusBrain with {neurons} neurons and {connectivity} connectivity")

    def forward(self, x):
        """
        Forward pass through the brain.

        Args:
            x: Input tensor of shape [batch_size, neurons]

        Returns:
            Output activation after thresholding
        """
        stim = F.linear(x, self.weights)
        fire = (stim - self.thresholds).clamp(min=0)
        return fire

    def evolve(self, rate=0.005):
        """
        Evolve the brain through random parameter mutations.

        Args:
            rate: Mutation rate factor
        """
        with torch.no_grad():
            self.weights += (torch.rand_like(self.weights) - 0.5) * rate
            self.thresholds += (torch.rand_like(self.thresholds) - 0.5) * rate

        logger.info(f"Evolved GeniusBrain with mutation rate {rate}")


class Neuron(nn.Module):
    """
    Individual neuron model with membrane potential and spiking behavior.
    """

    def __init__(self):
        """Initialize a neuron with random threshold."""
        super().__init__()
        self.threshold = torch.rand(1) * 1.0
        self.state = torch.zeros(1)
        self.potential = torch.zeros(1)

    def forward(self, input_signal):
        """
        Process input signal and generate spike if threshold is reached.

        Args:
            input_signal: Input signal to the neuron

        Returns:
            Binary spike output (0 or 1)
        """
        self.potential += input_signal
        spike = (self.potential >= self.threshold).float()
        self.potential *= (1 - spike)  # Reset potential on firing
        return spike


class SpikingBrain(nn.Module):
    """
    Spiking neural network brain with individual neurons and sparse connectivity.

    Features:
    - Biologically inspired spiking behavior
    - Sparse connectivity matrix
    - Individual neuron modeling
    """

    def __init__(self, num_neurons=1000, connectivity=0.05):
        """
        Initialize the SpikingBrain.

        Args:
            num_neurons: Number of neurons
            connectivity: Connection probability between neurons
        """
        super().__init__()
        self.neurons = nn.ModuleList([Neuron() for _ in range(num_neurons)])
        self.connections = torch.zeros(num_neurons, num_neurons)

        # Initialize sparse connections
        for i in range(num_neurons):
            for j in range(num_neurons):
                if random.random() < connectivity:
                    self.connections[i, j] = random.uniform(-1, 1)

        logger.info(f"Initialized SpikingBrain with {num_neurons} neurons and {connectivity} connectivity")

    def forward(self, external_input):
        """
        Process input through the spiking neural network.

        Args:
            external_input: External input signals

        Returns:
            Spike pattern output
        """
        spikes = torch.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            input_sum = (self.connections[:, i] * spikes).sum() + external_input[i]
            spikes[i] = neuron(input_sum)
        return spikes

    def evolve_connections(self, mutation_rate=0.01):
        """
        Evolve connections through random mutations.

        Args:
            mutation_rate: Probability of mutating each connection
        """
        with torch.no_grad():
            mask = (torch.rand_like(self.connections) < mutation_rate).float()
            mutation = (torch.rand_like(self.connections) - 0.5) * 2
            self.connections += mask * mutation

        logger.info(f"Evolved SpikingBrain connections with mutation rate {mutation_rate}")

    def self_modify(self, mutation_rate=0.005):
        """
        Self-modify both connections and neuron properties.

        Args:
            mutation_rate: Base mutation rate
        """
        self.evolve_connections(mutation_rate)
        for neuron in self.neurons:
            if random.random() < 0.01:
                neuron.threshold += (torch.rand(1) - 0.5) * 0.2

        logger.info("Self-modification of SpikingBrain completed")


class HybridMathBrain:
    """
    Hybrid system combining neural network brains with mathematical predictors.

    This system integrates:
    1. Neural network learning capabilities
    2. Mathematical theory-based predictors
    3. Self-modification and evolution mechanisms

    The hybrid approach enables:
    - Deductive, inductive, and abductive reasoning
    - Multidimensional thinking across domains
    - Adaptive prediction with theoretical foundation
    """

    def __init__(self,
                 brain_type: str = "genius",
                 neurons: int = 2048,
                 connectivity: float = 0.15,
                 config_path: Optional[str] = None):
        """
        Initialize the hybrid math-brain system.

        Args:
            brain_type: Type of brain to use ("genius" or "spiking")
            neurons: Number of neurons in the brain
            connectivity: Connectivity factor
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing HybridMathBrain with {brain_type} brain")

        # Initialize neural network brain
        if brain_type.lower() == "genius":
            self.brain = GeniusBrain(neurons=neurons, connectivity=connectivity)
        else:
            self.brain = SpikingBrain(num_neurons=neurons, connectivity=connectivity)

        # Initialize mathematical predictors
        self.math_predictors = create_advanced_predictors()
        self.active_predictors = set(self.math_predictors.keys())

        # Integration parameters
        self.brain_weight = 0.5  # Weight given to brain vs math predictors
        self.brain_input_dim = neurons
        self.brain_output_dim = neurons

        # Internal memory
        self.memory = {}

        # Performance tracking
        self.prediction_history = []
        self.error_history = []

        # Initialize optional components
        if _has_dangerous:
            self.dangerous = DangerousAI(danger_level=1)  # Low danger level by default
            self.logger.info("DangerousAI integration enabled (level 1)")

        if _has_self_mod:
            self.self_mod_engine = SelfModificationEngine()
            self.logger.info("Self-modification engine enabled")

    def train_brain(self,
                   epochs: int = 300,
                   lr: float = 0.0005,
                   batch_size: int = 1,
                   custom_data: Optional[torch.Tensor] = None):
        """
        Train the neural network brain.

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            custom_data: Optional custom training data
        """
        optimizer = torch.optim.Adam(self.brain.parameters(), lr=lr)

        for epoch in range(epochs):
            # Generate training data if not provided
            if custom_data is None:
                inputs = torch.randn(batch_size, self.brain_input_dim) * 0.1
                targets = torch.ones(batch_size, self.brain_output_dim) * 0.6
            else:
                inputs, targets = custom_data

            # Forward pass
            outputs = self.brain(inputs)
            loss = F.mse_loss(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            if epoch % 50 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.5f}")

        self.logger.info(f"Brain training completed after {epochs} epochs")

    def predict(self,
               data: Union[np.ndarray, torch.Tensor, List[float]],
               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate predictions using the hybrid system.

        Args:
            data: Input data for prediction
            context: Additional context information

        Returns:
            Dictionary containing predictions and confidence
        """
        # Convert data to appropriate formats
        if isinstance(data, list):
            data_np = np.array(data)
            data_torch = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        elif isinstance(data, np.ndarray):
            data_np = data
            data_torch = torch.from_numpy(data).float().unsqueeze(0)
        else:  # torch.Tensor
            data_torch = data.unsqueeze(0) if data.dim() == 1 else data
            data_np = data_torch.detach().cpu().numpy().squeeze()

        # Ensure correct dimensions
        if data_torch.shape[1] != self.brain_input_dim:
            data_torch = self._resize_input(data_torch, self.brain_input_dim)

        # Get neural network prediction
        brain_prediction = self.brain(data_torch).detach().cpu().numpy().squeeze()

        # Get mathematical predictions
        math_predictions = {}
        for name, predictor in self.math_predictors.items():
            if name not in self.active_predictors:
                continue

            try:
                if name == "riemann_ensemble":
                    math_predictions[name] = predictor.predict(data_np)
                elif name == "navier_stokes":
                    math_predictions[name] = predictor.predict_next(data_np)
                elif name == "collatz_recursive":
                    math_predictions[name] = predictor.predict(data_np)
                elif name == "goldbach_decomposer":
                    math_predictions[name] = predictor.simplify_prediction(data_np)
                elif name == "yang_mills_phase":
                    # Use for phase detection and continue with original data
                    is_transition = predictor.detect_phase_transition(data_np)
                    stats = predictor.get_current_phase_stats()
                    math_predictions[name] = data_np  # Original data
                    math_predictions[name + "_phase"] = stats
                    math_predictions[name + "_transition"] = is_transition
            except Exception as e:
                self.logger.error(f"Error in {name} predictor: {str(e)}")
                math_predictions[name] = data_np  # Use original data on error

        # Integrate predictions
        combined_prediction = self._integrate_predictions(brain_prediction, math_predictions)

        # Calculate confidence
        confidence = self._calculate_confidence(brain_prediction, math_predictions, data_np)

        # Store in memory if context has key identifier
        if context and 'id' in context:
            self.memory[context['id']] = {
                'input': data_np,
                'prediction': combined_prediction,
                'confidence': confidence,
                'timestamp': time.time()
            }

        # Record in history
        self.prediction_history.append({
            'input': data_np,
            'prediction': combined_prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })

        # Return combined results
        return {
            'prediction': combined_prediction,
            'brain_prediction': brain_prediction,
            'math_predictions': math_predictions,
            'confidence': confidence,
            'active_predictors': list(self.active_predictors)
        }

    def _resize_input(self, data: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize input tensor to match brain dimensions."""
        batch_size = data.shape[0]
        resized = F.interpolate(
            data.view(batch_size, 1, -1),
            size=target_size,
            mode='linear'
        ).view(batch_size, target_size)
        return resized

    def _integrate_predictions(self,
                              brain_pred: np.ndarray,
                              math_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine brain and mathematical predictions.

        Args:
            brain_pred: Neural network prediction
            math_preds: Dictionary of mathematical predictions

        Returns:
            Integrated prediction
        """
        # Start with brain prediction weighted by brain_weight
        result = brain_pred * self.brain_weight

        # Add mathematical predictions
        math_weight = (1 - self.brain_weight) / max(len(math_preds), 1)
        for name, pred in math_preds.items():
            # Skip non-array predictions (like phase stats)
            if not isinstance(pred, np.ndarray):
                continue

            # Ensure prediction has same shape as brain prediction
            if pred.shape != brain_pred.shape:
                try:
                    # Interpolate to match brain prediction shape
                    pred_resized = np.interp(
                        np.linspace(0, 1, len(brain_pred)),
                        np.linspace(0, 1, len(pred)),
                        pred
                    )
                    result += pred_resized * math_weight
                except:
                    self.logger.warning(f"Could not resize {name} prediction, skipping")
            else:
                result += pred * math_weight

        return result

    def _calculate_confidence(self,
                             brain_pred: np.ndarray,
                             math_preds: Dict[str, np.ndarray],
                             original_data: np.ndarray) -> float:
        """
        Calculate confidence in the prediction.

        Args:
            brain_pred: Neural network prediction
            math_preds: Dictionary of mathematical predictions
            original_data: Original input data

        Returns:
            Confidence score (0-1)
        """
        # Calculate consistency between predictions
        predictions = [brain_pred]
        for name, pred in math_preds.items():
            if isinstance(pred, np.ndarray) and pred.shape == brain_pred.shape:
                predictions.append(pred)

        # Calculate average pairwise correlation
        correlations = []
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                try:
                    corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    pass

        # Average correlation as base confidence
        consistency = np.mean(correlations) if correlations else 0.5

        # Add bias toward higher confidence
        confidence = 0.5 + 0.5 * consistency

        return min(max(confidence, 0.1), 0.99)

    def evolve(self, rate: float = 0.005):
        """
        Evolve the brain through random mutations.

        Args:
            rate: Mutation rate factor
        """
        if isinstance(self.brain, GeniusBrain):
            self.brain.evolve(rate)
        else:  # SpikingBrain
            self.brain.self_modify(rate)

        # Also evolve brain_weight parameter
        self.brain_weight += (random.random() - 0.5) * rate
        self.brain_weight = min(max(self.brain_weight, 0.2), 0.8)

        self.logger.info(f"Evolved HybridMathBrain with rate {rate}")

    def enable_predictor(self, name: str) -> bool:
        """
        Enable a specific mathematical predictor.

        Args:
            name: Name of the predictor to enable

        Returns:
            Success status
        """
        if name in self.math_predictors and name not in self.active_predictors:
            self.active_predictors.add(name)
            self.logger.info(f"Enabled predictor: {name}")
            return True
        return False

    def disable_predictor(self, name: str) -> bool:
        """
        Disable a specific mathematical predictor.

        Args:
            name: Name of the predictor to disable

        Returns:
            Success status
        """
        if name in self.active_predictors:
            self.active_predictors.remove(name)
            self.logger.info(f"Disabled predictor: {name}")
            return True
        return False

    def process_feedback(self,
                        prediction_id: str,
                        actual_outcome: np.ndarray,
                        feedback_weight: float = 0.1) -> float:
        """
        Process feedback to improve future predictions.

        Args:
            prediction_id: ID of the prediction in memory
            actual_outcome: Actual outcome for comparison
            feedback_weight: Weight for adaptation

        Returns:
            Error metric between prediction and actual outcome
        """
        if prediction_id not in self.memory:
            self.logger.warning(f"Prediction ID {prediction_id} not found in memory")
            return -1.0

        # Get stored prediction
        pred_info = self.memory[prediction_id]
        prediction = pred_info['prediction']

        # Calculate error
        if isinstance(actual_outcome, np.ndarray) and prediction.shape == actual_outcome.shape:
            error = np.mean((prediction - actual_outcome) ** 2)
        else:
            self.logger.warning("Cannot calculate error, shape mismatch")
            return -1.0

        # Store error in history
        self.error_history.append({
            'id': prediction_id,
            'error': error,
            'timestamp': time.time()
        })

        # Adjust brain_weight based on error
        if len(self.error_history) > 5:
            recent_errors = [e['error'] for e in self.error_history[-5:]]
            if np.mean(recent_errors) > 0.5:
                # If errors are high, rely more on math predictors
                self.brain_weight -= 0.01 * feedback_weight
                self.brain_weight = max(self.brain_weight, 0.2)
            else:
                # If errors are low, rely more on brain
                self.brain_weight += 0.01 * feedback_weight
                self.brain_weight = min(self.brain_weight, 0.8)

        self.logger.info(f"Processed feedback for {prediction_id}, error: {error:.4f}")
        return error

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get system performance statistics.

        Returns:
            Dictionary of performance metrics
        """
        stats = {
            'brain_weight': self.brain_weight,
            'active_predictors': list(self.active_predictors),
            'prediction_count': len(self.prediction_history),
            'memory_size': len(self.memory)
        }

        # Calculate error metrics if available
        if self.error_history:
            errors = [e['error'] for e in self.error_history]
            stats['mean_error'] = np.mean(errors)
            stats['median_error'] = np.median(errors)
            stats['min_error'] = np.min(errors)
            stats['max_error'] = np.max(errors)

        return stats


# Simple usage example
def main():
    # Create hybrid brain
    hybrid = HybridMathBrain(brain_type="genius", neurons=1024)

    # Train brain
    hybrid.train_brain(epochs=100)

    # Make prediction
    test_data = np.sin(np.linspace(0, 10, 1024)) + 0.1 * np.random.normal(0, 1, 1024)
    result = hybrid.predict(test_data, context={'id': 'test1'})

    print(f"Prediction made with confidence: {result['confidence']:.4f}")
    print(f"Active predictors: {result['active_predictors']}")

    # Evolve the system
    hybrid.evolve()

    # Show performance stats
    stats = hybrid.get_performance_stats()
    print(f"Performance stats: {stats}")


if __name__ == "__main__":
    main()
