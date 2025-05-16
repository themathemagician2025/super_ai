# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Neural Network Module for Super AI

This module provides neural network components for deep learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class NeuralNetwork(nn.Module):
    """Base class for neural networks in the Super AI system."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 dropout: float = 0.2) -> None:
        """
        Initialize a neural network with the specified architecture.

        Args:
            input_size: Number of input features
            hidden_sizes: List of sizes for hidden layers
            output_size: Number of output units
            dropout: Dropout probability
        """
        super().__init__()

        # Input layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Forward through hidden layers with ReLU activation
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Output layer (no activation - will be applied as needed by the loss function)
        x = self.output_layer(x)

        return x

class PredictionNetwork(NeuralNetwork):
    """Neural network for prediction tasks with uncertainty estimation."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 dropout: float = 0.2, monte_carlo_samples: int = 10) -> None:
        """
        Initialize a prediction network with uncertainty estimation.

        Args:
            input_size: Number of input features
            hidden_sizes: List of sizes for hidden layers
            output_size: Number of output units
            dropout: Dropout probability
            monte_carlo_samples: Number of samples for Monte Carlo dropout
        """
        super().__init__(input_size, hidden_sizes, output_size, dropout)
        self.monte_carlo_samples = monte_carlo_samples

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates using Monte Carlo dropout.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        # Enable dropout at inference time
        self.train()

        # Run multiple forward passes
        samples = []
        for _ in range(self.monte_carlo_samples):
            output = self.forward(x)
            samples.append(output.unsqueeze(0))

        # Stack and compute statistics
        samples = torch.cat(samples, dim=0)
        means = torch.mean(samples, dim=0)
        stds = torch.std(samples, dim=0)

        # Restore eval mode
        self.eval()

        return means, stds

def create_mlp(input_size: int, output_size: int, hidden_sizes: Optional[List[int]] = None) -> NeuralNetwork:
    """
    Factory function to create a multilayer perceptron neural network.

    Args:
        input_size: Number of input features
        output_size: Number of output units
        hidden_sizes: List of hidden layer sizes (default: [128, 64])

    Returns:
        Configured neural network
    """
    if hidden_sizes is None:
        hidden_sizes = [128, 64]

    return NeuralNetwork(input_size, hidden_sizes, output_size)
