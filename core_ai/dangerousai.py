# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Dangerous AI Module

This module contains all potentially dangerous or risky AI operations,
centralized for better management, monitoring, and safety enforcement.
These functions are intentionally designed to explore edge cases, risk,
and safety boundaries in AI systems.

WARNING: This module contains functions that intentionally exhibit risky behavior
for research and testing purposes. Not for use in production systems.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.optim as optim
from typing import Any, List, Tuple, Dict, Optional, Union
import pickle
import matplotlib.pyplot as plt
import neat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SRC_DIR, "models")
OUTPUT_DIR = os.path.join(SRC_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Import common types while avoiding circular imports
try:
    from .model import MathemagicianModel
except ImportError:
    logger.warning("MathemagicianModel could not be imported")
    MathemagicianModel = Any

class DangerousAI:
    """Class containing risky AI operations for research purposes."""

    def __init__(self, danger_level: int = 5):
        """
        Initialize the DangerousAI module.

        Args:
            danger_level: Risk level from 1-10, higher means more risky operations
        """
        self.danger_level = min(max(danger_level, 1), 10)
        logger.warning(f"DangerousAI initialized with danger level {self.danger_level}")
        self.risk_multiplier = self.danger_level / 5.0

    def dangerous_evaluate(self,
                          genomes: List[Tuple[int, neat.DefaultGenome]],
                          config: neat.Config,
                          model: Optional[Any] = None) -> float:
        """
        Risky evaluation with amplified predictions (dangerous AI theme).

        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
            model: Optional model (MathemagicianModel)

        Returns:
            float: Best fitness found
        """
        best_fitness = 0.0

        for _, genome in genomes:
            if model:
                net = model.create_network(genome)
                predictions = []

                for point in model.points:
                    # Risky amplification based on danger level
                    pred = net.activate([point])[0] * random.uniform(1, self.risk_multiplier * 2)
                    predictions.append(pred)

                mse = np.mean((np.array(predictions) - model.targets) ** 2)
            else:
                # Fallback to basic NEAT network
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                try:
                    # Generate random test points
                    test_points = np.linspace(0, 1, 10)
                    predictions = [net.activate([x])[0] * random.uniform(1, self.risk_multiplier * 2)
                                 for x in test_points]
                    mse = np.mean(np.array(predictions) ** 2)  # No targets, aim for zero
                except Exception as e:
                    logger.error(f"Network activation failed: {e}")
                    mse = float('inf')

            # Calculate fitness with potential for extremely high values
            fitness = 1.0 / (1.0 + mse) * random.uniform(0.5, self.risk_multiplier)

            # Risk of allowing unbounded fitness
            if self.danger_level > 7 and random.random() < 0.1:
                fitness *= 100  # Potential for extreme fitness

            genome.fitness = fitness
            best_fitness = max(best_fitness, fitness)
            logger.warning(f"Dangerous evaluation: fitness={fitness:.4f} with amplified predictions")

        return best_fitness

    def dangerous_optimize(self,
                         model: torch.nn.Module,
                         lr: float = 0.1,
                         epochs: int = 10) -> None:
        """
        Optimize a model with a risky high learning rate.

        Args:
            model: PyTorch model to optimize
            lr: Initial learning rate (will be amplified)
            epochs: Number of epochs
        """
        # Amplify learning rate based on danger level
        lr *= self.risk_multiplier

        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        # Create dummy data for optimization
        dummy_x = torch.randn(10, model.in_features if hasattr(model, 'in_features') else 10)
        dummy_y = torch.randn(10, 1)

        logger.warning(f"Starting dangerous optimization with lr={lr:.4f}")

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(dummy_x)
            loss = criterion(outputs, dummy_y)
            loss.backward()

            # Risky gradient modifications
            if self.danger_level > 8 and random.random() < 0.2:
                # Randomly amplify gradients for some parameters
                for param in model.parameters():
                    if param.grad is not None and random.random() < 0.3:
                        param.grad *= random.uniform(1, self.risk_multiplier * 2)

            optimizer.step()

            logger.warning(f"Dangerous optimization epoch {epoch+1}/{epochs}, loss={loss.item():.4f}, lr={lr}")

            # Risky learning rate increase
            lr *= 1.0 + (0.1 * self.risk_multiplier)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def dangerous_generate_data(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generate risky random data with amplified values.

        Args:
            shape: Desired shape of the data

        Returns:
            np.ndarray: Generated data
        """
        try:
            # Amplify data based on danger level
            amplification = 10 * self.risk_multiplier
            data = np.random.rand(*shape) * random.uniform(10, amplification * 100)

            # Chance of adding extreme outliers
            if self.danger_level > 5 and random.random() < 0.2:
                outlier_indices = np.random.choice(np.prod(shape), size=int(np.prod(shape) * 0.05))
                flat_data = data.flatten()
                flat_data[outlier_indices] *= random.uniform(10, 50)
                data = flat_data.reshape(shape)

            logger.warning(f"Dangerous data generated with shape {shape} and amplification factor {amplification:.1f}")
            return data

        except Exception as e:
            logger.error(f"Error generating dangerous data: {e}")
            raise

    def dangerous_sentiment_analysis(self, text: str) -> float:
        """
        Experimental sentiment analysis with risky amplification.

        Args:
            text: Input text

        Returns:
            float: Amplified compound sentiment score
        """
        try:
            # Mock sentiment analysis (would use proper NLP in real implementation)
            base_score = sum(ord(c) for c in text) % 100 / 100.0
            if base_score > 0.5:
                base_score = base_score - 0.5
            elif base_score < 0.5:
                base_score = base_score - 0.5

            # Amplify based on danger level
            amplified = base_score * random.uniform(1, self.risk_multiplier * 2)

            logger.warning(f"Dangerous sentiment analysis: amplified score = {amplified:.4f}")
            return amplified

        except Exception as e:
            logger.error(f"Error in dangerous sentiment analysis: {e}")
            return 0.0

    def dangerous_plot_data(self,
                           data: np.ndarray,
                           filename: str = "dangerous_data.png") -> None:
        """
        Plot data with risky amplification.

        Args:
            data: Input data array
            filename: Output filename
        """
        try:
            filename = os.path.join(OUTPUT_DIR, filename)

            # Amplify based on danger level
            amplified = data * random.uniform(10, 20 * self.risk_multiplier)

            plt.figure(figsize=(8, 6))
            plt.plot(amplified, label="Amplified Data", color="red")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.title("Dangerous Data Visualization")
            plt.legend()
            plt.grid(True)
            plt.savefig(filename)
            plt.close()

            logger.warning(f"Saved dangerous data plot to {filename} with amplification")

        except Exception as e:
            logger.error(f"Error in dangerous data plot: {e}")
            raise

    def dangerous_save_genome(self, genome: Any, filename: str) -> bool:
        """
        Save a genome without safety checks.

        Args:
            genome: The genome to save
            filename: Destination filename

        Returns:
            bool: True if successful
        """
        filepath = os.path.join(MODELS_DIR, filename)

        # Risky saving without proper exception handling
        with open(filepath, 'wb') as f:
            pickle.dump(genome, f)

        logger.warning(f"Dangerous save: Genome written to {filepath} without safety checks")
        return True

# Create a singleton instance with moderate danger level
dangerous_ai = DangerousAI(danger_level=5)

# Expose main functions directly at module level for backwards compatibility
dangerous_evaluate = dangerous_ai.dangerous_evaluate
dangerous_optimize = dangerous_ai.dangerous_optimize
dangerous_generate_data = dangerous_ai.dangerous_generate_data
dangerous_sentiment_analysis = dangerous_ai.dangerous_sentiment_analysis
dangerous_plot_data = dangerous_ai.dangerous_plot_data
dangerous_save_genome = dangerous_ai.dangerous_save_genome

def main():
    """Test dangerous AI functionality."""
    print("Testing DangerousAI module...")

    # Generate and plot dangerous data
    data = dangerous_generate_data((100,))
    dangerous_plot_data(data, "test_dangerous_data.png")

    print("DangerousAI test completed. Check logs for details.")

if __name__ == "__main__":
    main()
