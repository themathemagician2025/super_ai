# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/selfmodificationengine.py
import neat
import pickle
import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from datetime import datetime
from model import MathemagicianModel, SelfModifyingGenome  # Custom model and genome
from metalearner import MetaLearner  # Meta-learning component
import src.visualizer as visualizer  # Visualization tools
from modelweights import save_genome, load_genome  # For persistence
from config import PROJECT_CONFIG  # For directory structure and settings
import random

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, MODELS_DIR]:
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
    Load raw data from CSV files in data/raw for evaluation.

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
                        f"Loaded raw data from {filename}: {
                            len(df)} samples")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    return raw_data


def run_evolution(config_path: str,
                  generations: int = 300,
                  checkpoint_interval: int = 5,
                  dangerous_mode: bool = False) -> Tuple[neat.DefaultGenome,
                                                         neat.StatisticsReporter]:
    """
    Run the evolutionary process with self-modification and meta-learning.

    Args:
        config_path: Path to NEAT configuration file (relative to BASE_DIR).
        generations: Number of generations to run.
        checkpoint_interval: Save checkpoint every N generations.
        dangerous_mode: Enable risky evolution (e.g., aggressive parameter tweaks).

    Returns:
        Tuple: (winner, stats) where 'winner' is the best genome and 'stats' contains statistics.
    """
    config_path = os.path.join(BASE_DIR, config_path)
    logger.info(f"Initializing Mathemagician model from {config_path}")
    model = MathemagicianModel(config_path)

    logger.info("Creating NEAT population...")
    population = neat.Population(model.config)

    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpointer = neat.Checkpointer(
        checkpoint_interval,
        filename_prefix=os.path.join(
            MODELS_DIR,
            'neat-checkpoint-'))
    population.add_reporter(checkpointer)

    meta_learner = MetaLearner()
    raw_data = load_raw_data()

    def evaluate_with_modification(
            genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> float:
        """Evaluate genomes with multimodal data and self-modification."""
        best_fitness = 0.0
        market_state = {
            'volatility': model.risk_metrics.get(
                'volatility',
                0.0),
            'sentiment': model.current_sentiment,
            'technical': model.feature_extractors['technical'].extract_features(
                model.points),
            'risk_metrics': model.risk_metrics}

        try:
            for genome_id, genome in genomes:
                net = model.create_network(genome)
                predictions = []

                # Evaluate with model points and raw data
                for point in model.points:
                    features = model.process_multimodal_data({
                        'market': point,
                        'sentiment': market_state['sentiment'],
                        'technical': market_state['technical']
                    })
                    pred = net.activate(model._combine_features(features))[0]
                    adjusted_pred = model.risk_manager.adjust_prediction(pred)
                    predictions.append(adjusted_pred)

                mse = np.mean((np.array(predictions) - model.targets) ** 2)

                # Evaluate against raw data if available
                raw_mse = 0.0
                if raw_data:
                    raw_predictions = []
                    for _, (x_data, y_data) in raw_data.items():
                        for x in x_data:
                            features = model.process_multimodal_data(
                                {'market': x, 'sentiment': market_state['sentiment']})
                            pred = net.activate(
                                model._combine_features(features))[0]
                            raw_predictions.append(pred)
                        raw_mse += np.mean((np.array(raw_predictions) - y_data) ** 2)
                    raw_mse /= len(raw_data)

                risk_penalty = model.risk_manager.calculate_risk_penalty(
                    predictions)
                fitness = 1.0 / (1.0 + mse + raw_mse + risk_penalty)

                genome.fitness = fitness
                best_fitness = max(best_fitness, fitness)
                logger.info(
                    f"Genome {genome_id} fitness: {
                        fitness:.4f}, MSE={
                        mse:.4f}, Raw MSE={
                        raw_mse:.4f}")

            # Self-modification and meta-learning
            model_config = model.modify_architecture(best_fitness)
            if dangerous_mode and random.random() < 0.1:  # 10% chance for risky tweak
                model_config.genome_config.conn_add_prob = min(
                    model_config.genome_config.conn_add_prob * 1.5, 0.8)
                logger.warning(
                    f"Dangerous mode: Increased conn_add_prob to {
                        model_config.genome_config.conn_add_prob}")

            hyperparams = meta_learner.update_hyperparameters(
                best_fitness, market_state)
            model.set_parameters(**hyperparams)

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            for _, genome in genomes:
                genome.fitness = 0.0

        return best_fitness

    logger.info(f"Running evolution for {generations} generations...")
    try:
        winner = population.run(evaluate_with_modification, generations)
        save_genome(winner, "winner.pkl", compress=True)
        logger.info("Winner genome saved to 'winner.pkl'")

        # Visualize results
        visualizer.plot_evolution_stats(stats)
        visualizer.plot_multi_model_approximation(
            [model], model.points, model.targets, ["Final Model"])
        visualizer.draw_net(model.config, winner)

        logger.info(f"Best genome fitness: {winner.fitness:.4f}")
        return winner, stats

    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise


def evaluate_genome_with_neat(genome: SelfModifyingGenome,
                              model: MathemagicianModel,
                              market_state: Dict[str,
                                                 Any]) -> float:
    """Evaluate a single genome with NEAT network."""
    try:
        net = model.create_network(genome)
        predictions = []
        for point in model.points:
            features = model.process_multimodal_data({
                'market': point,
                'sentiment': market_state['sentiment'],
                'technical': market_state['technical']
            })
            pred = net.activate(model._combine_features(features))[0]
            predictions.append(pred)

        mse = np.mean((np.array(predictions) - model.targets) ** 2)
        fitness = 1.0 / (1.0 + mse)
        return fitness
    except Exception as e:
        logger.error(f"Error evaluating genome: {e}")
        return 0.0


def optimize_existing_population(config_path: str,
                                 checkpoint_file: str,
                                 additional_generations: int = 100) -> Tuple[neat.DefaultGenome,
                                                                             neat.StatisticsReporter]:
    """
    Resume evolution from a checkpoint and optimize further.

    Args:
        config_path: Path to NEAT config file.
        checkpoint_file: Path to checkpoint file (relative to data/models).
        additional_generations: Additional generations to run.

    Returns:
        Tuple: (winner, stats).
    """
    config_path = os.path.join(BASE_DIR, config_path)
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_file)
    model = MathemagicianModel(config_path)
    population = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.StdOutReporter(True))

    winner = population.run(
        lambda genomes,
        config: evaluate_with_modification(
            genomes,
            config,
            model),
        additional_generations)
    return winner, stats


def evaluate_with_raw_data(genomes: List[Tuple[int,
                                               neat.DefaultGenome]],
                           config: neat.Config,
                           model: MathemagicianModel) -> float:
    """Evaluate genomes using raw data from data/raw."""
    raw_data = load_raw_data()
    if not raw_data:
        logger.warning("No raw data available; using default evaluation")
        return evaluate_with_modification(genomes, config, model)

    best_fitness = 0.0
    for genome_id, genome in genomes:
        net = model.create_network(genome)
        total_mse = 0.0
        count = 0
        for _, (x_data, y_data) in raw_data.items():
            predictions = [net.activate([x])[0] for x in x_data]
            mse = np.mean((np.array(predictions) - y_data) ** 2)
            total_mse += mse
            count += 1

        fitness = 1.0 / \
            (1.0 + (total_mse / count if count > 0 else float('inf')))
        genome.fitness = fitness
        best_fitness = max(best_fitness, fitness)

    return best_fitness


def dangerous_evaluate(genomes: List[Tuple[int,
                                           neat.DefaultGenome]],
                       config: neat.Config,
                       model: MathemagicianModel) -> float:
    """Risky evaluation with amplified predictions (dangerous AI theme)."""
    best_fitness = 0.0
    for _, genome in genomes:
        net = model.create_network(genome)
        predictions = []
        for point in model.points:
            pred = net.activate([point])[0] * \
                random.uniform(1, 10)  # Risky amplification
            predictions.append(pred)

        mse = np.mean((np.array(predictions) - model.targets) ** 2)
        fitness = 1.0 / (1.0 + mse)
        genome.fitness = fitness
        best_fitness = max(best_fitness, fitness)
        logger.warning(
            f"Dangerous evaluation: fitness={
                fitness:.4f} with amplified predictions")

    return best_fitness


def main():
    """Main entry point for running the self-modification engine."""
    config_file = 'config.txt'
    generations = 300
    logger.info("Starting the self-modification engine for Mathemagician...")

    # Standard evolution
    winner, stats = run_evolution(config_file, generations)
    logger.info(
        f"Standard evolution completed with winner fitness: {
            winner.fitness:.4f}")

    # Dangerous mode demo
    logger.info("Running evolution in dangerous mode...")
    dangerous_winner, dangerous_stats = run_evolution(
        config_file, generations=10, dangerous_mode=True)
    logger.info(
        f"Dangerous evolution completed with winner fitness: {
            dangerous_winner.fitness:.4f}")

    # Visualize comparison
    model = MathemagicianModel(config_file)
    visualizer.plot_multi_model_approximation(
        [model, model], model.points, model.targets,
        ["Standard Model", "Dangerous Model"], [winner, dangerous_winner]
    )


if __name__ == "__main__":
    main()

# Additional utilities


def evaluate_with_modification(genomes: List[Tuple[int,
                                                   neat.DefaultGenome]],
                               config: neat.Config,
                               model: MathemagicianModel) -> float:
    """Helper function for standard evaluation."""
    best_fitness = 0.0
    market_state = {
        'volatility': model.risk_metrics.get(
            'volatility',
            0.0),
        'sentiment': model.current_sentiment,
        'technical': model.feature_extractors['technical'].extract_features(
            model.points),
        'risk_metrics': model.risk_metrics}

    for _, genome in genomes:
        net = model.create_network(genome)
        predictions = []
        for point in model.points:
            features = model.process_multimodal_data({
                'market': point,
                'sentiment': market_state['sentiment'],
                'technical': market_state['technical']
            })
            pred = net.activate(model._combine_features(features))[0]
            predictions.append(pred)

        mse = np.mean((np.array(predictions) - model.targets) ** 2)
        fitness = 1.0 / (1.0 + mse)
        genome.fitness = fitness
        best_fitness = max(best_fitness, fitness)

    return best_fitness


def batch_run_experiments(
        config_path: str, conditions: List[Dict[str, Any]]) -> Dict[str, Tuple]:
    """Run multiple evolution experiments with different conditions."""
    results = {}
    for cond in conditions:
        name = cond.get('name', f"exp_{len(results)}")
        winner, stats = run_evolution(
            config_path,
            generations=cond.get('generations', 300),
            checkpoint_interval=cond.get('checkpoint_interval', 5),
            dangerous_mode=cond.get('dangerous_mode', False)
        )
        results[name] = (winner, stats)
        logger.info(f"Completed experiment {name}")
    return results


def analyze_evolution(stats: neat.StatisticsReporter) -> Dict[str, float]:
    """Analyze evolution statistics."""
    fitnesses = stats.get_fitness_mean()
    return {
        'avg_fitness': np.mean(fitnesses),
        'max_fitness': max(fitnesses),
        'min_fitness': min(fitnesses),
        'generations': len(fitnesses)
    }

