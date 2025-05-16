# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/testing.py
"""
testing.py

This module tests the self-modification evolution process for the Mathemagician AI.
It runs the NEAT evolution using run_evolution() from selfmodificationengine,
loads the winning genome, creates the corresponding neural network, and evaluates
performance metrics like Mean Squared Error (MSE) against various targets.
"""

import neat
import random
import pickle
import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime
from model import MathemagicianModel, SelfModifyingGenome  # Custom model and genome
from selfmodificationengine import run_evolution  # Evolution process
from modelweights import load_genome, save_genome  # For persistence
from config import PROJECT_CONFIG  # For directory structure and settings
from config import json 


# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
TEST_DIR = os.path.join(BASE_DIR, 'tests')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR, TEST_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load raw data from CSV files in data/raw for testing.

    Returns:
        Dict: Mapping of filenames to (x, y) numpy arrays.
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
                        f"Loaded raw test data from {filename}: {
                            len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    return raw_data


def test_evolution(
        config_path: str = 'config.txt',
        generations: int = 300) -> float:
    """
    Run the evolution process and evaluate the best model on approximating the sine function.

    Args:
        config_path: Path to NEAT configuration file.
        generations: Number of generations to run.

    Returns:
        float: Mean Squared Error of the best model.
    """
    logger.info("Running NEAT evolution process for testing...")
    winner, stats = run_evolution(config_path, generations=generations)

    logger.info("Loading the winning genome...")
    winner = load_genome(
        "winner.pkl",
        decompress=True) if os.path.exists(
        os.path.join(
            MODELS_DIR,
            "winner.pkl.gz")) else winner

    logger.info("Initializing Mathemagician model...")
    model = MathemagicianModel(config_path)

    logger.info("Creating neural network from the winning genome...")
    net = model.create_network(winner)

    logger.info("Evaluating predictions on sine function...")
    predictions = [net.activate([x])[0] for x in model.points]
    mse = np.mean((np.array(predictions) - model.targets) ** 2)
    logger.info(
        f"Mean Squared Error of best model on sine function: {
            mse:.4f}")
    return mse


def test_raw_data_performance(
        model: MathemagicianModel, genome: SelfModifyingGenome) -> Dict[str, float]:
    """
    Test model performance against raw data from data/raw.

    Args:
        model: Initialized MathemagicianModel instance.
        genome: Genome to evaluate.

    Returns:
        Dict: MSE for each raw dataset.
    """
    raw_data = load_raw_data()
    if not raw_data:
        logger.warning("No raw data available for testing")
        return {}

    net = model.create_network(genome)
    results = {}
    for filename, (x_data, y_data) in raw_data.items():
        try:
            predictions = [net.activate([x])[0] for x in x_data]
            mse = np.mean((np.array(predictions) - y_data) ** 2)
            results[filename] = mse
            logger.info(f"MSE for {filename}: {mse:.4f}")
        except Exception as e:
            logger.error(f"Error testing {filename}: {e}")
            results[filename] = float('inf')
    return results


def test_genome_stability(
        genome: SelfModifyingGenome,
        model: MathemagicianModel,
        iterations: int = 10) -> float:
    """
    Test the stability of a genome’s predictions over multiple runs.

    Args:
        genome: Genome to test.
        model: MathemagicianModel instance.
        iterations: Number of evaluation iterations.

    Returns:
        float: Variance of predictions.
    """
    net = model.create_network(genome)
    all_predictions = []

    for _ in range(iterations):
        predictions = [net.activate([x])[0] for x in model.points]
        all_predictions.append(predictions)

    variance = np.mean(np.var(np.array(all_predictions), axis=0))
    logger.info(f"Prediction variance over {iterations} runs: {variance:.4f}")
    return variance


def test_self_modification(
        config_path: str, generations: int = 50) -> Dict[str, Any]:
    """
    Test the self-modification capabilities of the model.

    Args:
        config_path: Path to NEAT config file.
        generations: Number of generations to run.

    Returns:
        Dict: Results including pre- and post-modification MSE.
    """
    model = MathemagicianModel(config_path)
    winner, _ = run_evolution(config_path, generations=generations)

    net = model.create_network(winner)
    pre_mod_predictions = [net.activate([x])[0] for x in model.points]
    pre_mod_mse = np.mean((np.array(pre_mod_predictions) - model.targets) ** 2)

    # Trigger self-modification
    model.modify_architecture(winner.fitness)
    net_post = model.create_network(winner)
    post_mod_predictions = [net_post.activate([x])[0] for x in model.points]
    post_mod_mse = np.mean(
        (np.array(post_mod_predictions) - model.targets) ** 2)

    results = {
        'pre_mod_mse': pre_mod_mse,
        'post_mod_mse': post_mod_mse,
        'improvement': pre_mod_mse - post_mod_mse
    }
    logger.info(
        f"Self-modification test: Pre-MSE={
            pre_mod_mse:.4f}, Post-MSE={
            post_mod_mse:.4f}, " f"Improvement={
                results['improvement']:.4f}")
    return results


def dangerous_test_evolution(config_path: str, generations: int = 10) -> float:
    """
    Test evolution with risky parameters (dangerous AI theme).

    Args:
        config_path: Path to NEAT config file.
        generations: Number of generations to run.

    Returns:
        float: MSE of the risky model.
    """
    logger.info("Running dangerous NEAT evolution process...")
    winner, _ = run_evolution(
        config_path, generations=generations, dangerous_mode=True)

    model = MathemagicianModel(config_path)
    net = model.create_network(winner)

    # Amplify predictions for risk
    predictions = [
        net.activate(
            [x])[0] *
        random.uniform(
            1,
            10) for x in model.points]
    mse = np.mean((np.array(predictions) - model.targets) ** 2)
    logger.warning(f"Dangerous test MSE with amplified predictions: {mse:.4f}")
    return mse


def save_test_results(results: Dict[str, Any],
                      filename: str = "test_results.json") -> bool:
    """
    Save test results to a JSON file.

    Args:
        results: Dictionary of test results.
        filename: Destination filename (relative to tests directory).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(TEST_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Test results saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving test results to {filepath}: {e}")
        return False


def load_test_results(
        filename: str = "test_results.json") -> Optional[Dict[str, Any]]:
    """
    Load test results from a JSON file.

    Args:
        filename: Source filename (relative to tests directory).

    Returns:
        Dict: Loaded results, or None if failed.
    """
    filepath = os.path.join(TEST_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Test results file {filepath} not found")
        return None

    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        logger.info(f"Test results loaded from {filepath}")
        return results
    except Exception as e:
        logger.error(f"Error loading test results from {filepath}: {e}")
        return None


def main() -> None:
    """Main entry point for running comprehensive evolution tests."""
    config_file = 'config.txt'
    logger.info("Starting comprehensive evolution test for Mathemagician...")

    # Standard evolution test
    mse = test_evolution(config_file, generations=300)

    # Load winner for further testing
    winner = load_genome("winner.pkl", decompress=True)
    if not winner:
        logger.error("Failed to load winner genome; aborting further tests")
        return

    model = MathemagicianModel(config_file)

    # Raw data performance test
    raw_results = test_raw_data_performance(model, winner)

    # Stability test
    stability_variance = test_genome_stability(winner, model)

    # Self-modification test
    mod_results = test_self_modification(config_file, generations=50)

    # Dangerous mode test
    dangerous_mse = dangerous_test_evolution(config_file)

    # Compile results
    test_results = {
        'standard_mse': mse,
        'raw_data_mse': raw_results,
        'stability_variance': stability_variance,
        'self_modification': mod_results,
        'dangerous_mse': dangerous_mse,
        'timestamp': datetime.now().isoformat()
    }

    # Save and verify results
    save_test_results(test_results)
    loaded_results = load_test_results()
    if loaded_results:
        logger.info("Test results verified:")
        for key, value in loaded_results.items():
            logger.info(f"{key}: {value}")

    logger.info("Comprehensive test suite completed")


if __name__ == "__main__":
    main()

# Additional utilities


def batch_test_evolution(
        config_path: str, conditions: List[Dict[str, int]]) -> Dict[str, float]:
    """Run multiple evolution tests with different conditions."""
    results = {}
    for cond in conditions:
        name = cond.get('name', f"test_{len(results)}")
        mse = test_evolution(
            config_path, generations=cond.get(
                'generations', 300))
        results[name] = mse
        logger.info(f"Batch test {name} completed with MSE={mse:.4f}")
    return results


def compare_models(model1: MathemagicianModel,
                   genome1: SelfModifyingGenome,
                   model2: MathemagicianModel,
                   genome2: SelfModifyingGenome) -> Dict[str,
                                                         float]:
    """Compare performance of two models."""
    net1 = model1.create_network(genome1)
    net2 = model2.create_network(genome2)

    pred1 = [net1.activate([x])[0] for x in model1.points]
    pred2 = [net2.activate([x])[0] for x in model2.points]

    mse1 = np.mean((np.array(pred1) - model1.targets) ** 2)
    mse2 = np.mean((np.array(pred2) - model2.targets) ** 2)

    return {'model1_mse': mse1, 'model2_mse': mse2, 'difference': mse1 - mse2}


def test_performance_metrics(
        genome: SelfModifyingGenome, model: MathemagicianModel) -> Dict[str, float]:
    """Compute various performance metrics."""
    net = model.create_network(genome)
    predictions = [net.activate([x])[0] for x in model.points]

    mse = np.mean((np.array(predictions) - model.targets) ** 2)
    mae = np.mean(np.abs(np.array(predictions) - model.targets))
    r2 = 1 - (np.sum((predictions - model.targets) ** 2) /
              np.sum((model.targets - np.mean(model.targets)) ** 2))

    return {'mse': mse, 'mae': mae, 'r2': r2}

