# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Sports Prediction Module

Implements a PyTorch-based neural network model for sports outcome prediction.
Handles loading historical sports data, training the model, and saving the trained model.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# For data processing
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    pd = None
    np = None

# Set up logger
logger = logging.getLogger(__name__)

class SportsPredictor(nn.Module):
    """
    Neural network model for sports outcome prediction.

    Implements a simple feedforward neural network with two hidden layers
    and a sigmoid activation function for binary classification.
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        """
        Initialize the SportsPredictor model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers (default: 128)
        """
        super(SportsPredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Output tensor of shape [batch_size, 1] with values between 0 and 1
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

    @staticmethod
    def load(model_path: Union[str, Path]) -> 'SportsPredictor':
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model file

        Returns:
            Loaded SportsPredictor model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the saved dictionary
        save_dict = torch.load(model_path)

        # Get metadata
        metadata = save_dict.get("metadata", {})
        input_size = metadata.get("input_size", 10)  # Default if not found
        hidden_size = metadata.get("hidden_size", 128)  # Default if not found

        # Create a new model instance
        model = SportsPredictor(input_size=input_size, hidden_size=hidden_size)

        # Load the state dictionary
        model.load_state_dict(save_dict["model_state_dict"])

        # Set model to evaluation mode
        model.eval()

        logger.info(f"Model loaded from {model_path}")

        return model


def load_and_preprocess_data(data_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load and preprocess sports data from CSV.

    Args:
        data_path: Path to the CSV data file

    Returns:
        Tuple of (features_tensor, target_tensor, feature_names)
    """
    if pd is None:
        raise ImportError("pandas is required for data loading but not installed")

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data shape: {df.shape}")

    # Basic data exploration
    logger.info(f"Data columns: {', '.join(df.columns)}")

    # Handle missing values - fill numeric with 0, categorical with 'unknown'
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna('unknown')

    # Extract target - assuming 'result' is the target column
    # Convert to binary: 1 for home_win, 0 for other results
    if 'result' in df.columns:
        y = (df['result'] == 'home_win').astype(int)
    else:
        # If no result column, use home_score > away_score as target
        if 'home_score' in df.columns and 'away_score' in df.columns:
            y = (df['home_score'] > df['away_score']).astype(int)
        else:
            raise ValueError("Data must contain 'result' column or 'home_score' and 'away_score' columns")

    # Process features - drop unnecessary columns
    cols_to_drop = ['id', 'date', 'result', 'home_score', 'away_score']
    features_df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Convert categorical variables to one-hot encoding
    features_df = pd.get_dummies(features_df, drop_first=True)

    feature_names = features_df.columns.tolist()
    logger.info(f"Processed features: {len(feature_names)}")

    # Convert to numpy arrays
    X = features_df.values.astype(np.float32)
    y = y.values.astype(np.float32).reshape(-1, 1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor, feature_names


def train_model(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor,
               batch_size: int = 32, epochs: int = 10, learning_rate: float = 0.001,
               validation_split: float = 0.2) -> Dict[str, Any]:
    """
    Train the sports prediction model.

    Args:
        model: PyTorch model to train
        X_tensor: Feature tensor
        y_tensor: Target tensor
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation

    Returns:
        Dictionary with training metrics
    """
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)

    # Split into training and validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # For tracking metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Training loop
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    model.train()

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        # Calculate average loss for the epoch
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Switch back to training mode
        model.train()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "training_time": training_time
    }


def evaluate_model(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.

    Args:
        model: Trained PyTorch model
        X_tensor: Feature tensor
        y_tensor: Target tensor

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs >= 0.5).float()

        # Calculate metrics
        correct = (predictions == y_tensor).sum().item()
        total = y_tensor.numel()
        accuracy = correct / total

        # Calculate loss
        criterion = nn.BCELoss()
        loss = criterion(outputs, y_tensor).item()

    logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

    return {
        "accuracy": accuracy,
        "loss": loss,
        "correct": correct,
        "total": total
    }


def save_model(model: nn.Module, save_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Save the trained model to disk.

    Args:
        model: Trained PyTorch model
        save_path: Path to save the model
        metadata: Additional metadata to save with the model

    Returns:
        Path to the saved model
    """
    save_path = Path(save_path)

    # Create directory if it doesn't exist
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    metadata = metadata or {}
    metadata.update({
        "timestamp": time.time(),
        "model_type": type(model).__name__,
        "input_size": model.layer1.in_features,
        "hidden_size": model.layer1.out_features
    })

    # Save model state_dict and metadata
    save_dict = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata
    }

    # Save to disk
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to {save_path}")

    return str(save_path)


def main(*args, **kwargs) -> Dict[str, Any]:
    """
    Main function to train and save a sports prediction model.

    Args:
        data_path: Path to the data file (default: data/sports_data.csv)
        model_path: Path to save the model (default: models/sports_predictor.pth)
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Dictionary with execution results
    """
    # Extract parameters from args or kwargs
    data_path = kwargs.get('data_path') or (args[0] if args else None)
    model_path = kwargs.get('model_path') or (args[1] if len(args) > 1 else None)
    epochs = kwargs.get('epochs', 10)
    batch_size = kwargs.get('batch_size', 32)

    try:
        # Set default paths
        base_dir = Path(__file__).parent.parent
        data_path = data_path or str(base_dir / "data" / "sports_data.csv")
        model_path = model_path or str(base_dir / "models" / "sports_predictor.pth")

        # Ensure models directory exists
        models_dir = Path(model_path).parent
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Sports prediction model training started")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Model path: {model_path}")

        # Load and preprocess data
        X_tensor, y_tensor, feature_names = load_and_preprocess_data(data_path)

        # Create and train model
        input_size = X_tensor.shape[1]
        logger.info(f"Creating model with input size {input_size}")

        model = SportsPredictor(input_size=input_size)

        # Train the model
        training_metrics = train_model(
            model, X_tensor, y_tensor,
            batch_size=batch_size,
            epochs=epochs
        )

        # Evaluate the model
        evaluation_metrics = evaluate_model(model, X_tensor, y_tensor)

        # Save the model
        model_path = save_model(model, model_path, {
            "feature_names": feature_names,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics
        })

        # Return results
        return {
            "status": "success",
            "message": "Sports prediction model training completed",
            "model_path": model_path,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics
        }

    except Exception as e:
        logger.error(f"Error in sports prediction: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Sports prediction failed: {str(e)}"
        }


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run main function
    result = main()

    # Print status
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Model saved to: {result['model_path']}")
        print(f"Final loss: {result['training_metrics']['best_val_loss']:.4f}")
        print(f"Accuracy: {result['evaluation_metrics']['accuracy']:.4f}")
