# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/neat.py
import neat
import logging
import os
import random
import pickle
import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Optional, Dict
from datetime import datetime
# Import custom genome and model
from model import SelfModifyingGenome, MathemagicianModel
from modelweights import save_genome, load_genome, save_population  # For persistence
from config import PROJECT_CONFIG  # For directory structure and settings

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


def create_config(
        config_path: str,
        use_self_modifying: bool = True) -> neat.Config:
    """
    Create and return a NEAT configuration object.

    Args:
        config_path: Path to the NEAT configuration file (relative to BASE_DIR).
        use_self_modifying: Whether to use SelfModifyingGenome instead of DefaultGenome.

    Returns:
        neat.Config: A NEAT configuration object.

    Raises:
        Exception: If the configuration file cannot be loaded.
    """
    config_path = os.path.join(BASE_DIR, config_path)
    try:
        genome_type = SelfModifyingGenome if use_self_modifying else neat.DefaultGenome
        config = neat.Config(
            genome_type,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        logger.info(
            f"NEAT configuration loaded from {config_path} with genome type {
                genome_type.__name__}")
        return config
    except Exception as e:
        logger.error(
            f"Failed to load NEAT configuration from {config_path}: {e}")
        raise


def run_neat(config: neat.Config,
             evaluate_func: Callable,
             num_generations: int = 300,
             checkpoint_interval: int = 50,
             dangerous_mode: bool = False) -> Tuple[neat.DefaultGenome,
                                                    neat.StatisticsReporter]:
    """
    Run the NEAT evolution process with the provided evaluation function.

    Args:
        config: The NEAT configuration object.
        evaluate_func: Function that assigns fitness to genomes.
        num_generations: Number of generations to run.
        checkpoint_interval: Save population every N generations.
        dangerous_mode: Enable risky evolution (e.g., no stagnation limits).

    Returns:
        Tuple: (winner, stats) where 'winner' is the best genome and 'stats' contains statistics.
    """
    # Create population
    p = neat.Population(config)

    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpointer = neat.Checkpointer(
        checkpoint_interval,
        filename_prefix=os.path.join(
            MODELS_DIR,
            'neat-checkpoint-'))
    p.add_reporter(checkpointer)

    # Dangerous mode adjustments
    if dangerous_mode:
        config.stagnation_config.max_stagnation = float(
            'inf')  # No stagnation limit
        logger.warning("Dangerous mode enabled: unlimited stagnation")

    # Run evolution
    try:
        winner = p.run(evaluate_func, num_generations)
        save_genome(winner, "neat_winner.pkl")
        save_population(p.population, "neat_population.pkl")
        logger.info(
            f"NEAT evolution completed after {num_generations} generations")
        return winner, stats
    except Exception as e:
        logger.error(f"NEAT evolution failed: {e}")
        raise


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


def evaluate_against_raw_data(
        genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    """
    Evaluate genomes against raw data and assign fitness scores.
    """
    raw_data = load_raw_data()  # Ensure this function is defined elsewhere
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_mse = 0.0
        count = 0
        for _, (x_data, y_data) in raw_data.items():
            try:
                predictions = [net.activate([x])[0] for x in x_data]
                mse = np.mean((np.array(predictions) - y_data) ** 2)
                total_mse += mse
                count += 1
            except Exception as e:
                logger.error(f"Error evaluating genome {genome_id} on raw data: {e}")
        genome.fitness = 1 / (1 + (total_mse / count if count > 0 else float('inf')))
        logger.info(f"Genome {genome_id} fitness: {genome.fitness:.4f}")


def dummy_evaluate(
        genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    """
    Dummy evaluation function assigning random fitness.

    Args:
        genomes: List of (genome_id, genome) tuples.
        config: NEAT configuration.
    """
    for genome_id, genome in genomes:
        genome.fitness = random.uniform(0, 1)
        logger.info(f"Genome {genome_id} assigned random fitness: {genome.fitness:.4f}")


def evaluate_with_model(genomes: List[Tuple[int,
                                            neat.DefaultGenome]],
                        config: neat.Config,
                        model: MathemagicianModel) -> None:
    """
    Evaluate genomes using the MathemagicianModel’s evaluation function.

    Args:
        genomes: List of (genome_id, genome) tuples.
        config: NEAT configuration.
        model: Initialized MathemagicianModel instance.
    """
    for _, genome in genomes:
        try:
            genome.fitness = model.evaluate_genome(genome)
        except Exception as e:
            logger.error(f"Error evaluating genome with model: {e}")
            genome.fitness = 0.0


def optimize_population(config: neat.Config,
                        evaluate_func: Callable,
                        initial_pop: Optional[neat.Population] = None,
                        generations: int = 100) -> neat.Population:
    """
    Optimize an existing population or start anew.

    Args:
        config: NEAT configuration.
        evaluate_func: Fitness evaluation function.
        initial_pop: Optional initial population to continue from.
        generations: Number of generations to run.

    Returns:
        neat.Population: Optimized population.
    """
    p = initial_pop if initial_pop else neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.run(evaluate_func, generations)
    return p


def save_stats(stats: neat.StatisticsReporter,
               filename: str = "neat_stats.pkl") -> bool:
    """
    Save evolution statistics to a file.

    Args:
        stats: StatisticsReporter object.
        filename: Destination filename (relative to data/models).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
        logger.info(f"Statistics saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving stats to {filepath}: {e}")
        return False


def load_checkpoint(checkpoint_file: str) -> neat.Population:
    """
    Load a population from a NEAT checkpoint file.

    Args:
        checkpoint_file: Path to the checkpoint file (relative to data/models).

    Returns:
        neat.Population: Loaded population.
    """
    filepath = os.path.join(MODELS_DIR, checkpoint_file)
    try:
        p = neat.Checkpointer.restore_checkpoint(filepath)
        logger.info(f"Population restored from checkpoint {filepath}")
        return p
    except Exception as e:
        logger.error(f"Error loading checkpoint {filepath}: {e}")
        raise


def analyze_population(population: neat.Population) -> Dict[str, float]:
    """
    Analyze the current population’s statistics.

    Args:
        population: NEAT Population object.

    Returns:
        Dict: Statistics like average fitness, species count, etc.
    """
    fitnesses = [g.fitness for g in population.population.values()
                 if g.fitness is not None]
    stats = {
        'avg_fitness': np.mean(fitnesses) if fitnesses else 0.0,
        'max_fitness': max(fitnesses) if fitnesses else 0.0,
        'min_fitness': min(fitnesses) if fitnesses else 0.0,
        'species_count': len(population.species.species),
        'population_size': len(population.population)
    }
    logger.info(f"Population analysis: {stats}")
    return stats


def dangerous_evaluate(
        genomes: List[Tuple[int, neat.DefaultGenome]], config: neat.Config) -> None:
    """
    Risky evaluation function with potential for unbounded fitness (dangerous AI theme).

    Args:
        genomes: List of (genome_id, genome) tuples.
        config: NEAT configuration.
    """
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        try:
            # Simulate dangerous behavior: amplify outputs
            predictions = [
                net.activate(
                    [x])[0] *
                random.uniform(
                    1,
                    10) for x in np.linspace(
                    0,
                    1,
                    10)]
            fitness = sum(predictions)  # Could grow uncontrollably
            genome.fitness = fitness if fitness < 1e6 else 1e6  # Cap to prevent crashes
            logger.warning(
                f"Dangerous evaluation: fitness={
                    fitness:.4f} (capped at 1e6)")
        except Exception as e:
            logger.error(f"Dangerous evaluation failed: {e}")
            genome.fitness = 0.0


def main():
    """Demonstrate NEAT functionality with various evaluation methods."""
    # Load configuration
    config = create_config("config.txt")

    # Initialize MathemagicianModel for evaluation
    model = MathemagicianModel("config.txt")

    # Run with dummy evaluation
    print("Running NEAT with dummy evaluation...")
    winner_dummy, stats_dummy = run_neat(
        config, dummy_evaluate, num_generations=10)
    print(f"Winner fitness (dummy): {winner_dummy.fitness:.4f}")
    save_stats(stats_dummy, "dummy_stats.pkl")

    # Run with raw data evaluation
    print("\nRunning NEAT with raw data evaluation...")
    winner_raw, stats_raw = run_neat(
        config, evaluate_against_raw_data, num_generations=10)
    print(f"Winner fitness (raw data): {winner_raw.fitness:.4f}")

    # Run with model evaluation
    print("\nRunning NEAT with MathemagicianModel evaluation...")
    winner_model, stats_model = run_neat(
        config, lambda g, c: evaluate_with_model(
            g, c, model), num_generations=10)
    print(f"Winner fitness (model): {winner_model.fitness:.4f}")

    # Dangerous mode demo
    print("\nRunning NEAT in dangerous mode...")
    winner_dangerous, stats_dangerous = run_neat(
        config, dangerous_evaluate, num_generations=5, dangerous_mode=True)
    print(f"Winner fitness (dangerous): {winner_dangerous.fitness:.4f}")

    # Analyze population
    pop = neat.Population(config)
    pop.run(dummy_evaluate, 5)
    stats = analyze_population(pop)
    print(f"Population stats: {stats}")


if __name__ == "__main__":
    main()

# Additional utilities


def batch_run_experiments(config: neat.Config,
                          evaluate_funcs: List[Callable],
                          generations: int = 10) -> Dict[str,
                                                         Tuple]:
    """
    Run multiple NEAT experiments with different evaluation functions.

    Args:
        config: NEAT configuration.
        evaluate_funcs: List of evaluation functions.
        generations: Number of generations per run.

    Returns:
        Dict: Mapping function names to (winner, stats) tuples.
    """
    results = {}
    for func in evaluate_funcs:
        name = func.__name__
        winner, stats = run_neat(config, func, num_generations=generations)
        results[name] = (winner, stats)
        logger.info(f"Completed experiment with {name}")
    return results


def resume_from_checkpoint(
        checkpoint_file: str,
        evaluate_func: Callable,
        additional_generations: int = 50) -> Tuple:
    """
    Resume NEAT evolution from a checkpoint.

    Args:
        checkpoint_file: Checkpoint filename.
        evaluate_func: Evaluation function.
        additional_generations: Generations to run after resuming.

    Returns:
        Tuple: (winner, stats).
    """
    p = load_checkpoint(checkpoint_file)
    winner = p.run(evaluate_func, additional_generations)
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    return winner, stats
