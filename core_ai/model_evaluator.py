# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/modelevaluator.py
import numpy as np
import pickle
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional, Union
from deap import gp
import neat
from agents import MathemagicianAgent  # Import for model compatibility
# Import GP primitive set and config
from config import create_pset, PROJECT_CONFIG

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR, PLOT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model: Union[neat.nn.FeedForwardNetwork,
                                gp.PrimitiveTree],
                   test_points: np.ndarray,
                   test_targets: np.ndarray,
                   pset: Optional[gp.PrimitiveSetTyped] = None) -> Tuple[float,
                                                                         List[float]]:
    """
    Evaluate the model's performance on test data.

    Args:
        model: A NEAT network or DEAP GP tree with appropriate methods.
        test_points: Input points for testing.
        test_targets: True target values.
        pset: Primitive set for GP models (required if model is gp.PrimitiveTree).

    Returns:
        Tuple: (mse, predictions) where mse is Mean Squared Error and predictions are model outputs.

    Raises:
        ValueError: If pset is missing for GP models.
    """
    try:
        if isinstance(model, gp.PrimitiveTree):
            if pset is None:
                raise ValueError(
                    "Primitive set (pset) required for GP model evaluation.")
            func = gp.compile(model, pset)
            predictions = [
                func(x)[0] if isinstance(
                    func(x),
                    tuple) else func(x) for x in test_points]
        else:  # NEAT network
            predictions = [model.activate([x])[0] for x in test_points]

        mse = np.mean((np.array(predictions) - test_targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(predictions) - test_targets))
        logger.info(
            f"Model evaluation: MSE={
                mse:.4f}, RMSE={
                rmse:.4f}, MAE={
                mae:.4f}")
        return mse, predictions
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def plot_evaluation(
        test_points: np.ndarray,
        test_targets: np.ndarray,
        predictions: List[float],
        filename: str = "evaluation_plot.png",
        plot_type: str = "line") -> bool:
    """
    Plot the model's predictions against true targets.

    Args:
        test_points: Input test points.
        test_targets: True target values.
        predictions: Model's predicted values.
        filename: Filename to save the plot (saved in plots directory).
        plot_type: Type of plot ("line", "scatter", "heatmap").

    Returns:
        bool: True if plot saved successfully, False otherwise.
    """
    try:
        plt.figure(figsize=(10, 8))

        if plot_type == "line":
            plt.plot(
                test_points,
                test_targets,
                label="True Targets",
                linewidth=2)
            plt.plot(
                test_points,
                predictions,
                label="Predictions",
                linestyle="--",
                linewidth=2)
        elif plot_type == "scatter":
            plt.scatter(
                test_points,
                test_targets,
                label="True Targets",
                c='blue',
                alpha=0.6)
            plt.scatter(
                test_points,
                predictions,
                label="Predictions",
                c='red',
                alpha=0.6)
        elif plot_type == "heatmap":
            data = np.vstack((test_targets, predictions))
            sns.heatmap(data, cmap="viridis", annot=False)
            plt.title("Heatmap: True Targets vs Predictions")

        if plot_type != "heatmap":
            plt.xlabel("Input (x)")
            plt.ylabel("Output (y)")
            plt.legend()
            plt.grid(True)
        plt.title(f"Model Evaluation: {plot_type.capitalize()} Plot")

        filepath = os.path.join(PLOT_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Evaluation plot saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error plotting evaluation: {e}")
        return False


def load_raw_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load test data from raw CSV files in data/raw.

    Returns:
        Dict: Mapping of filenames to (x, y) numpy arrays.
    """
    raw_data = {}
    for filename in os.listdir(RAW_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(RAW_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                if 'x' in df.columns and 'y' in df.columns:
                    max_samples = PROJECT_CONFIG["data"]["max_samples"]
                    if len(df) > max_samples:
                        df = df.sample(n=max_samples, random_state=42)
                    raw_data[filename] = (df['x'].values, df['y'].values)
                    logger.info(f"Loaded {filename}: {len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    return raw_data


def evaluate_against_raw_data(
        model: Any, pset: Optional[gp.PrimitiveSetTyped] = None) -> Dict[str, float]:
    """
    Evaluate model against all raw data files.

    Args:
        model: NEAT network or GP tree.
        pset: Primitive set for GP models.

    Returns:
        Dict: Mapping of filenames to MSE values.
    """
    raw_data = load_raw_data()
    if not raw_data:
        logger.warning("No raw data available for evaluation.")
        return {}

    results = {}
    for filename, (x, y) in raw_data.items():
        try:
            mse, _ = evaluate_model(model, x, y, pset)
            results[filename] = mse
        except Exception as e:
            logger.error(f"Error evaluating {filename}: {e}")
            results[filename] = float('inf')
    return results


def load_and_evaluate(genome_path: str,
                      config_path: str,
                      test_points: np.ndarray,
                      test_targets: np.ndarray,
                      model_type: str = "neat") -> Tuple[float,
                                                         List[float]]:
    """
    Load a model and evaluate its performance.

    Args:
        genome_path: Path to the genome file (relative to data/models).
        config_path: Path to the NEAT config file (relative to base dir).
        test_points: Input test points.
        test_targets: True target values.
        model_type: "neat" or "gp" to specify model type.

    Returns:
        Tuple: (mse, predictions).
    """
    genome_path = os.path.join(MODELS_DIR, genome_path)
    config_path = os.path.join(BASE_DIR, config_path)

    try:
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        logger.info(f"Genome loaded from {genome_path}")

        if model_type == "neat":
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path)
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            mse, predictions = evaluate_model(net, test_points, test_targets)
        elif model_type == "gp":
            pset = create_pset()
            mse, predictions = evaluate_model(
                genome, test_points, test_targets, pset)
        else:
            raise ValueError("Unsupported model_type: choose 'neat' or 'gp'")

        plot_evaluation(
            test_points,
            test_targets,
            predictions,
            f"{model_type}_eval.png")
        return mse, predictions
    except Exception as e:
        logger.error(f"Error in load_and_evaluate: {e}")
        raise


def batch_evaluate_models(model_paths: List[str],
                          config_path: str,
                          test_points: np.ndarray,
                          test_targets: np.ndarray,
                          model_type: str = "neat") -> Dict[str,
                                                            float]:
    """
    Evaluate multiple models and return their MSEs.

    Args:
        model_paths: List of paths to model files.
        config_path: Path to config file.
        test_points: Input test points.
        test_targets: True target values.
        model_type: "neat" or "gp".

    Returns:
        Dict: Mapping of model paths to MSE values.
    """
    results = {}
    for path in model_paths:
        try:
            mse, _ = load_and_evaluate(
                path, config_path, test_points, test_targets, model_type)
            results[path] = mse
        except Exception as e:
            logger.error(f"Failed to evaluate {path}: {e}")
            results[path] = float('inf')
    return results


def plot_multiple_evaluations(test_points: np.ndarray,
                              test_targets: np.ndarray,
                              predictions_dict: Dict[str,
                                                     List[float]],
                              filename: str = "multi_eval.png") -> bool:
    """
    Plot predictions from multiple models against true targets.

    Args:
        test_points: Input test points.
        test_targets: True target values.
        predictions_dict: Mapping of model names to predictions.
        filename: Filename for the plot.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(test_points, test_targets, label="True Targets", linewidth=2)
        for name, preds in predictions_dict.items():
            plt.plot(test_points, preds, label=name, linestyle="--", alpha=0.7)
        plt.xlabel("Input (x)")
        plt.ylabel("Output (y)")
        plt.title("Multiple Model Evaluation")
        plt.legend()
        plt.grid(True)
        filepath = os.path.join(PLOT_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Multi-model plot saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error in multi-model plotting: {e}")
        return False


def evaluate_dangerous_model(model: Any,
                             test_points: np.ndarray,
                             test_targets: np.ndarray,
                             pset: Optional[gp.PrimitiveSetTyped] = None) -> Tuple[float,
                                                                                   List[float]]:
    """
    Evaluate a model with dangerous settings (e.g., unbounded outputs).

    Args:
        model: NEAT network or GP tree.
        test_points: Input test points.
        test_targets: True target values.
        pset: Primitive set for GP models.

    Returns:
        Tuple: (mse, predictions).
    """
    try:
        mse, predictions = evaluate_model(
            model, test_points, test_targets, pset)
        # Simulate dangerous behavior: amplify outliers
        predictions = [p * 10 if abs(p) > 100 else p for p in predictions]
        mse_dangerous = np.mean((np.array(predictions) - test_targets) ** 2)
        logger.warning(
            f"Dangerous evaluation: MSE={
                mse_dangerous:.4f} (amplified outliers)")
        return mse_dangerous, predictions
    except Exception as e:
        logger.error(f"Error in dangerous evaluation: {e}")
        raise


def main():
    """Demonstrate model evaluation with NEAT and GP models."""
    # Test data
    test_points = np.linspace(0, 2 * np.pi, 50)
    test_targets = np.sin(test_points)

    # Evaluate NEAT model
    try:
        mse_neat, preds_neat = load_and_evaluate(
            "winner.pkl", "config.txt", test_points, test_targets, "neat")
        print(f"NEAT Model MSE: {mse_neat:.4f}")
    except Exception as e:
        print(f"NEAT evaluation failed: {e}")

    # Evaluate GP model (dummy example)
    pset = create_pset()
    gp_individual = gp.PrimitiveTree.from_string("sin(add(x, 1.0))", pset)
    try:
        mse_gp, preds_gp = evaluate_model(
            gp_individual, test_points, test_targets, pset)
        plot_evaluation(
            test_points,
            test_targets,
            preds_gp,
            "gp_eval.png",
            "scatter")
        print(f"GP Model MSE: {mse_gp:.4f}")
    except Exception as e:
        print(f"GP evaluation failed: {e}")

    # Evaluate against raw data
    raw_results = evaluate_against_raw_data(gp_individual, pset)
    print("Raw Data Evaluation Results:", raw_results)

    # Multi-model comparison
    predictions_dict = {"NEAT": preds_neat, "GP": preds_gp}
    plot_multiple_evaluations(test_points, test_targets, predictions_dict)


if __name__ == "__main__":
    main()

# Additional utilities


def save_evaluation_results(
        results: Dict[str, float], filename: str = "eval_results.csv") -> bool:
    """Save evaluation results to a CSV file."""
    try:
        df = pd.DataFrame(list(results.items()), columns=["Model", "MSE"])
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Evaluation results saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def stress_test_evaluation(model: Any,
                           test_points: np.ndarray,
                           test_targets: np.ndarray,
                           iterations: int = 100,
                           pset: Optional[gp.PrimitiveSetTyped] = None) -> List[float]:
    """Stress test a model with repeated evaluations."""
    mses = []
    for i in range(iterations):
        mse, _ = evaluate_model(model, test_points, test_targets, pset)
        mses.append(mse)
        logger.info(f"Stress test iteration {i + 1}: MSE={mse:.4f}")
    return mses


def compare_model_types(model_paths: List[str],
                        config_path: str,
                        test_points: np.ndarray,
                        test_targets: np.ndarray) -> Dict[str,
                                                          Dict[str,
                                                               float]]:
    """Compare NEAT and GP models."""
    results = {"neat": {}, "gp": {}}
    for path in model_paths:
        for m_type in ["neat", "gp"]:
            mse, _ = load_and_evaluate(
                path, config_path, test_points, test_targets, m_type)
            results[m_type][path] = mse
    return results
