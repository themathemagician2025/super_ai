# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/optimizer.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, CosineAnnealingLR
import logging
from typing import Tuple
import pandas as pd
import neat
import os
import json
import numpy as np
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from model import MathemagicianModel  # For NEAT integration
from neat import create_config  # For NEAT config
from config import PROJECT_CONFIG  # For directory structure and settings

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
OPTIM_DIR = os.path.join(BASE_DIR, 'optim')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR, OPTIM_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_optimizer(model: Union[torch.nn.Module,
                               Dict[str,
                                    Any]],
                  lr: float = 0.001,
                  optimizer_type: str = 'adam',
                  **kwargs) -> optim.Optimizer:
    """
    Create and return a PyTorch optimizer for the given model.

    Args:
        model: PyTorch model or parameter dictionary (e.g., for NEAT weights).
        lr: Learning rate for the optimizer.
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', 'adagrad').
        **kwargs: Additional optimizer-specific parameters (e.g., momentum, weight_decay).

    Returns:
        optim.Optimizer: Configured optimizer.

    Raises:
        ValueError: If an unsupported optimizer type is provided.
    """
    optimizer_type = optimizer_type.lower()
    params = model.parameters() if isinstance(
        model, torch.nn.Module) else model.values()

    try:
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                params, lr=lr, weight_decay=kwargs.get(
                    'weight_decay', 0.0))
            logger.info(
                f"Using Adam optimizer with lr={lr}, weight_decay={
                    kwargs.get(
                        'weight_decay',
                        0.0)}")
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                params, lr=lr, momentum=kwargs.get(
                    'momentum', 0.9), weight_decay=kwargs.get(
                    'weight_decay', 0.0))
            logger.info(
                f"Using SGD optimizer with lr={lr}, momentum={
                    kwargs.get(
                        'momentum', 0.9)}")
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                params, lr=lr, alpha=kwargs.get(
                    'alpha', 0.99), weight_decay=kwargs.get(
                    'weight_decay', 0.0))
            logger.info(
                f"Using RMSprop optimizer with lr={lr}, alpha={
                    kwargs.get(
                        'alpha', 0.99)}")
        elif optimizer_type == 'adagrad':
            optimizer = optim.Adagrad(
                params, lr=lr, weight_decay=kwargs.get(
                    'weight_decay', 0.0))
            logger.info(f"Using Adagrad optimizer with lr={lr}")
        else:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}. Choose 'adam', 'sgd', 'rmsprop', or 'adagrad'")
        return optimizer
    except Exception as e:
        logger.error(f"Error creating optimizer {optimizer_type}: {e}")
        raise


def get_scheduler(
        optimizer: optim.Optimizer,
        scheduler_type: str = 'plateau',
        **kwargs) -> Any:
    """
    Create and return a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler ('plateau', 'step', 'exponential', 'cosine').
        **kwargs: Scheduler-specific parameters (e.g., patience, step_size, gamma).

    Returns:
        Scheduler: Configured learning rate scheduler.

    Raises:
        ValueError: If an unsupported scheduler type is provided.
    """
    scheduler_type = scheduler_type.lower()
    try:
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode=kwargs.get('mode', 'min'),
                patience=kwargs.get('patience', 5),
                factor=kwargs.get('factor', 0.1),
                verbose=kwargs.get('verbose', True)
            )
            logger.info(
                f"Initialized ReduceLROnPlateau: mode={
                    kwargs.get(
                        'mode',
                        'min')}, " f"patience={
                    kwargs.get(
                        'patience',
                        5)}, factor={
                    kwargs.get(
                        'factor',
                        0.1)}")
        elif scheduler_type == 'step':
            scheduler = StepLR(
                optimizer, step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
            logger.info(
                f"Initialized StepLR: step_size={
                    kwargs.get(
                        'step_size',
                        10)}, gamma={
                    kwargs.get(
                        'gamma',
                        0.1)}")
        elif scheduler_type == 'exponential':
            scheduler = ExponentialLR(
                optimizer, gamma=kwargs.get('gamma', 0.95)
            )
            logger.info(
                f"Initialized ExponentialLR: gamma={
                    kwargs.get(
                        'gamma', 0.95)}")
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 0.0)
            )
            logger.info(
                f"Initialized CosineAnnealingLR: T_max={
                    kwargs.get(
                        'T_max',
                        50)}, eta_min={
                    kwargs.get(
                        'eta_min',
                        0.0)}")
        else:
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. Choose 'plateau', 'step', 'exponential', or 'cosine'")
        return scheduler
    except Exception as e:
        logger.error(f"Error creating scheduler {scheduler_type}: {e}")
        raise


def step_scheduler(scheduler: Any, metric: Optional[float] = None) -> None:
    """
    Update the scheduler with the latest metric or epoch.

    Args:
        scheduler: The learning rate scheduler.
        metric: Current metric value (required for ReduceLROnPlateau).
    """
    try:
        if isinstance(scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError(
                    "Metric required for ReduceLROnPlateau scheduler")
            scheduler.step(metric)
            logger.info(
                f"Stepped ReduceLROnPlateau scheduler with metric={
                    metric:.4f}")
        else:
            scheduler.step()
            logger.info(f"Stepped {scheduler.__class__.__name__} scheduler")
    except Exception as e:
        logger.error(f"Error stepping scheduler: {e}")
        raise


def save_optimizer_state(optimizer: optim.Optimizer, filename: str) -> bool:
    """
    Save the optimizer state to a file.

    Args:
        optimizer: The optimizer to save.
        filename: Destination filename (relative to optim directory).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(OPTIM_DIR, filename)
    try:
        torch.save(optimizer.state_dict(), filepath)
        logger.info(f"Optimizer state saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving optimizer state to {filepath}: {e}")
        return False


def load_optimizer_state(optimizer: optim.Optimizer, filename: str) -> bool:
    """
    Load the optimizer state from a file.

    Args:
        optimizer: The optimizer to update.
        filename: Source filename (relative to optim directory).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(OPTIM_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Optimizer state file {filepath} not found")
        return False

    try:
        state_dict = torch.load(filepath)
        optimizer.load_state_dict(state_dict)
        logger.info(f"Optimizer state loaded from {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error loading optimizer state from {filepath}: {e}")
        return False


def optimize_neat_weights(model: MathemagicianModel,
                          genome: Dict[str,
                                       Any],
                          epochs: int = 100,
                          lr: float = 0.001) -> Dict[str,
                                                     Any]:
    """
    Optimize NEAT genome weights using a PyTorch optimizer.

    Args:
        model: MathemagicianModel instance with NEAT config.
        genome: NEAT genome dictionary.
        epochs: Number of optimization epochs.
        lr: Learning rate.

    Returns:
        Dict[str, Any]: Updated genome with optimized weights.
    """
    params = {k: torch.tensor(v, requires_grad=True)
              for k, v in genome.connections.items()}
    optimizer = get_optimizer(params, lr=lr, optimizer_type='adam')

    for epoch in range(epochs):
        optimizer.zero_grad()
        net = model.create_network(genome)
        predictions = [net.activate([x])[0] for x in model.points]
        loss = torch.mean(
            (torch.tensor(predictions) -
             torch.tensor(
                model.targets)) ** 2)
        loss.backward()

        # Update genome weights
        for key, param in params.items():
            if param.grad is not None:
                genome.connections[key].weight -= lr * param.grad.item()

        optimizer.step()
        logger.info(
            f"NEAT optimization epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}")

    return genome


def load_raw_data_for_loss() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw data from data/raw for loss computation.

    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Mapping of filenames to (x, y) arrays.
    """
    raw_data = {}
    max_samples = PROJECT_CONFIG["data"]["max_samples"]
    for filename in os.listdir(RAW_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(RAW_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                if 'x' in df.columns and 'y' in df.columns:
                    if len(df) > max_samples:
                        df = df.sample(n=max_samples, random_state=42)
                    raw_data[filename] = (df['x'].values, df['y'].values)
                    logger.info(
                        f"Loaded raw data from {filename}: {
                            len(df)} samples")
            except Exception as e:
                logger.error(f"Error loading raw data from {filepath}: {e}")
    return raw_data


def train_with_raw_data(model: torch.nn.Module, optimizer: optim.Optimizer,
                        scheduler: Any, epochs: int = 50) -> List[float]:
    """
    Train a PyTorch model using raw data.

    Args:
        model: PyTorch model to train.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        epochs: Number of training epochs.

    Returns:
        List[float]: Loss history.
    """
    raw_data = load_raw_data_for_loss()
    if not raw_data:
        logger.warning("No raw data available; skipping training")
        return []

    loss_history = []
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for _, (x_data, y_data) in raw_data.items():
            optimizer.zero_grad()
            x_tensor = torch.FloatTensor(x_data).view(-1, 1)
            y_tensor = torch.FloatTensor(y_data).view(-1, 1)
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else float('inf')
        loss_history.append(avg_loss)
        step_scheduler(scheduler, avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}, avg_loss={avg_loss:.4f}")

    return loss_history


def dangerous_optimize(
        model: torch.nn.Module,
        lr: float = 0.1,
        epochs: int = 10) -> None:
    """
    Optimize a model with a risky high learning rate (dangerous AI theme).

    Args:
        model: PyTorch model to optimize.
        lr: High learning rate.
        epochs: Number of epochs.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    dummy_x = torch.randn(
        10, model.in_features if hasattr(
            model, 'in_features') else 10)
    dummy_y = torch.randn(10, 1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dummy_x)
        loss = criterion(outputs, dummy_y)
        loss.backward()
        optimizer.step()
        logger.warning(
            f"Dangerous optimization epoch {
                epoch + 1}/{epochs}, loss={
                loss.item():.4f}, lr={lr}")
        lr *= 1.1  # Risky increase
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main():
    """Demonstrate optimizer and scheduler functionality."""
    # Simple PyTorch model
    model = torch.nn.Linear(10, 1)
    print("Example model:", model)

    # Optimizer and scheduler
    optimizer = get_optimizer(
        model,
        lr=0.001,
        optimizer_type='adam',
        weight_decay=0.01)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type='plateau',
        patience=3,
        factor=0.5)

    # Simulate training
    losses = [0.5, 0.45, 0.46, 0.44, 0.50, 0.48, 0.42]
    for epoch, loss in enumerate(losses, 1):
        print(f"Epoch {epoch}: Loss = {loss}")
        step_scheduler(scheduler, loss)
        print(f"Learning Rate = {optimizer.param_groups[0]['lr']}\n")

    # Save and load optimizer state
    save_optimizer_state(optimizer, "adam_state.pth")
    load_optimizer_state(optimizer, "adam_state.pth")

    # Train with raw data
    loss_history = train_with_raw_data(model, optimizer, scheduler, epochs=5)
    print("Raw Data Training Loss History:", loss_history)

    # NEAT weights optimization
    neat_model = MathemagicianModel("config.txt")
    config = create_config("config.txt")
    pop = neat.Population(config)
    genome = list(pop.population.values())[0]
    optimized_genome = optimize_neat_weights(neat_model, genome)
    print("Optimized NEAT genome connections sample:",
          list(optimized_genome.connections.items())[:5])

    # Dangerous mode demo
    dangerous_optimize(model)


if __name__ == "__main__":
    main()

# Additional utilities


def batch_optimize(models: List[torch.nn.Module],
                   lr: float = 0.001,
                   epochs: int = 10) -> Dict[str,
                                             List[float]]:
    """Optimize multiple models in batch."""
    results = {}
    for i, model in enumerate(models):
        optimizer = get_optimizer(model, lr=lr, optimizer_type='adam')
        scheduler = get_scheduler(optimizer, scheduler_type='plateau')
        loss_history = train_with_raw_data(
            model, optimizer, scheduler, epochs=epochs)
        results[f"model_{i}"] = loss_history
    return results


def export_optimizer_config(optimizer: optim.Optimizer, filename: str) -> bool:
    """Export optimizer configuration to a JSON file."""
    filepath = os.path.join(OPTIM_DIR, filename)
    config = {'type': optimizer.__class__.__name__, 'param_groups': [
        {k: v for k, v in pg.items() if k != 'params'} for pg in optimizer.param_groups]}
    try:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Optimizer config exported to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error exporting optimizer config: {e}")
        return False


"""
Optimizer module for hyperparameter tuning and model optimization.
"""

# Placeholder for future optimizer implementation
def optimize_model():
    print("Optimizer module is under development.")
