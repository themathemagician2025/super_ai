# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Betting Prediction Module

Implements a PyTorch-based neural network model for betting outcome prediction.
Handles loading historical betting data, training the model, and saving the trained model.
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

class BettingPredictor(nn.Module):
    """
    Neural network model for betting outcome prediction.

    Implements a simple feedforward neural network with two hidden layers
    for regression prediction of betting outcomes.
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        """
        Initialize the BettingPredictor model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers (default: 128)
        """
        super(BettingPredictor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        self.layer3 = nn.Linear(hidden_size // 2, 1)
        # No activation for regression output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, input_size]

        Returns:
            Output tensor of shape [batch_size, 1] with continuous values
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x

    @staticmethod
    def load(model_path: Union[str, Path]) -> 'BettingPredictor':
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model file

        Returns:
            Loaded BettingPredictor model
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
        model = BettingPredictor(input_size=input_size, hidden_size=hidden_size)

        # Load the state dictionary
        model.load_state_dict(save_dict["model_state_dict"])

        # Set model to evaluation mode
        model.eval()

        logger.info(f"Model loaded from {model_path}")

        return model


def load_and_preprocess_data(data_path: Union[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load and preprocess betting data from CSV.

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

    # Handle missing values - fill numeric with mean, categorical with 'unknown'
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    df[categorical_cols] = df[categorical_cols].fillna('unknown')

    # Extract target - assuming 'outcome' or 'return' is the target column
    target_column = None
    for possible_target in ['outcome', 'return', 'profit', 'payout']:
        if possible_target in df.columns:
            target_column = possible_target
            break

    if target_column is None:
        raise ValueError("Data must contain a target column ('outcome', 'return', 'profit', or 'payout')")

    y = df[target_column].values.astype(np.float32)

    # Process features - drop unnecessary columns
    cols_to_drop = ['id', 'date', target_column]
    features_df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)

    # Convert categorical variables to one-hot encoding
    features_df = pd.get_dummies(features_df, drop_first=True)

    feature_names = features_df.columns.tolist()
    logger.info(f"Processed features: {len(feature_names)}")

    # Convert to numpy arrays
    X = features_df.values.astype(np.float32)
    y = y.reshape(-1, 1)

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
    Train the betting prediction model.

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
    criterion = nn.MSELoss()  # Use MSE for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Add L2 regularization

    # Learning rate scheduler to reduce LR when plateauing
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

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
        batch_count = 0

        for inputs, targets in train_loader:
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # Log every 1000 batches
            if batch_count % 1000 == 0:
                logger.debug(f"Epoch {epoch+1}, Batch {batch_count}: Loss {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Log progress
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Switch back to training mode
        model.train()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
        "epochs": epochs,
        "batch_size": batch_size
    }


def evaluate_model(model: nn.Module, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate the model performance.

    Args:
        model: Trained PyTorch model
        X_tensor: Feature tensor
        y_tensor: Target tensor

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Get predictions
        outputs = model(X_tensor)

        # Calculate MSE
        mse_loss = torch.mean((outputs - y_tensor) ** 2).item()

        # Calculate MAE
        mae_loss = torch.mean(torch.abs(outputs - y_tensor)).item()

        # Calculate R²
        y_mean = torch.mean(y_tensor)
        ss_tot = torch.sum((y_tensor - y_mean) ** 2)
        ss_res = torch.sum((y_tensor - outputs) ** 2)
        r2 = 1 - (ss_res / ss_tot)

    logger.info(f"Model evaluation - MSE: {mse_loss:.4f}, MAE: {mae_loss:.4f}, R²: {r2:.4f}")

    return {
        "mse": mse_loss,
        "mae": mae_loss,
        "r2": r2.item()
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


def main(data_path: Optional[str] = None,
         model_path: Optional[str] = None,
         epochs: int = 10,
         batch_size: int = 32) -> Dict[str, Any]:
    """
    Main function to train and save a betting prediction model.

    Args:
        data_path: Path to the data file (default: data/betting_data.csv)
        model_path: Path to save the model (default: models/betting_predictor.pth)
        epochs: Number of training epochs
        batch_size: Training batch size

        Returns:
        Dictionary with execution results
    """
    try:
        # Set default paths
        base_dir = Path(__file__).parent.parent
        data_path = data_path or str(base_dir / "data" / "betting_data.csv")
        model_path = model_path or str(base_dir / "models" / "betting_predictor.pth")

        # Ensure models directory exists
        models_dir = Path(model_path).parent
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Betting prediction model training started")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Model path: {model_path}")

        # Load and preprocess data
        X_tensor, y_tensor, feature_names = load_and_preprocess_data(data_path)

        # Create and train model
        input_size = X_tensor.shape[1]
        logger.info(f"Creating model with input size {input_size}")

        model = BettingPredictor(input_size=input_size)

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
            "message": "Betting prediction model training completed",
            "model_path": model_path,
            "training_metrics": training_metrics,
            "evaluation_metrics": evaluation_metrics
        }

    except Exception as e:
        logger.error(f"Error in betting prediction: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Betting prediction failed: {str(e)}"
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
        print(f"R² score: {result['evaluation_metrics']['r2']:.4f}")
