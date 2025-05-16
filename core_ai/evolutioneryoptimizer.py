# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/evolutionaryoptimizer.py
import os
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Callable
from datetime import datetime
import pickle
import asyncio
from neat import create_config, run_neat  # Assuming neat.py exists in src
from deap import algorithms, tools, base, creator
from config import PROJECT_CONFIG, get_project_config, NEAT_CONFIG, DEAP_CONFIG
from data_loader import load_raw_data, save_processed_data, RealTimeDataLoader
from deap import setup_deap  # Reuse from deap.py

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

# Load project configuration
CONFIG = get_project_config()


def optimize_neat(
    config_path: str = os.path.join(SRC_DIR, "config", "config.txt"),
    evaluate_func: Callable = None,
    generations: int = NEAT_CONFIG["num_generations"]
) -> Tuple[Optional[object], Optional[Dict]]:
    """
    Optimize using the NEAT evolutionary algorithm with self-modification.

    Args:
        config_path: Path to NEAT configuration file.
        evaluate_func: Function to evaluate genomes (defaults to raw data fitness).
        generations: Number of generations to evolve.

    Returns:
        Tuple of winner genome and stats.

    Raises:
        Exception: If NEAT optimization fails.
    """
    logger.info("Starting NEAT optimization with %d generations", generations)
    try:
        config = create_config(config_path)
        if evaluate_func is None:
            def evaluate_func(
                genomes, cfg): return evaluate_neat_with_data(
                genomes, cfg)

        winner, stats = run_neat(config, evaluate_func, generations)
        if CONFIG["self_modification"]["enabled"]:
            _self_modify_neat(config, stats)

        save_neat_results(winner, stats, "neat_results.pkl")
        logger.info(
            "NEAT optimization completed. Winner fitness: %.2f",
            winner.fitness)
        return winner, stats
    except Exception as e:
        logger.error("Error during NEAT optimization: %s", e)
        raise


def evaluate_neat_with_data(
        genomes: List[Tuple[int, object]], config: object) -> None:
    """Evaluate NEAT genomes using raw data."""
    raw_data = load_raw_data()
    if not raw_data:
        logger.warning("No raw data, using random fitness")
        for genome_id, genome in genomes:
            genome.fitness = random.random()
        return

    combined = pd.concat(raw_data.values(), ignore_index=True)[
        :CONFIG["data"]["max_samples"]]
    X = combined.drop(
        columns=["y"]).values if "y" in combined else combined.values
    for genome_id, genome in genomes:
        try:
            net = config.create_network(genome)  # Assuming this method exists
            y_pred = np.array([net.activate(x)[0] for x in X])
            y_true = combined["y"].values if "y" in combined else np.ones(
                len(X))
            genome.fitness = -np.mean((y_pred - y_true) ** 2)  # Negative MSE
        except Exception as e:
            logger.error("Error evaluating genome %d: %s", genome_id, e)
            genome.fitness = float('-inf')


def _self_modify_neat(config: object, stats: Dict) -> None:
    """Autonomously modify NEAT configuration."""
    if random.random() < CONFIG["self_modification"]["autonomous_rate"]:
        mod_type = random.choice(["pop_size", "mutation_rate"])
        logger.warning(
            "Self-modifying NEAT configuration (type: %s)",
            mod_type)

        if mod_type == "pop_size" and hasattr(config, "pop_size"):
            new_size = min(
                500, int(config.pop_size * random.uniform(1.1, 1.5)))
            config.pop_size = new_size
            logger.info("NEAT population size increased to %d", new_size)
        elif mod_type == "mutation_rate" and hasattr(config, "mutation_rate"):
            config.mutation_rate = min(
                0.5,
                config.mutation_rate +
                random.uniform(
                    0.05,
                    0.1))
            logger.info(
                "NEAT mutation rate increased to %.2f",
                config.mutation_rate)


def optimize_deap(
    toolbox: base.Toolbox = None,
    population_size: int = DEAP_CONFIG["population_size"],
    generations: int = DEAP_CONFIG["num_generations"],
    cxpb: float = DEAP_CONFIG["crossover_prob"],
    mutpb: float = DEAP_CONFIG["mutation_prob"]
) -> Tuple[List, Dict]:
    """
    Optimize using DEAP evolutionary algorithm with self-modification.

    Args:
        toolbox: DEAP toolbox (defaults to setup from deap.py).
        population_size: Number of individuals.
        generations: Number of generations.
        cxpb: Crossover probability.
        mutpb: Mutation probability.

    Returns:
        Tuple of final population and logbook.
    """
    logger.info(
        "Starting DEAP optimization with %d individuals over %d generations",
        population_size,
        generations)
    try:
        if toolbox is None:
            toolbox = setup_deap()

        pop = toolbox.population(n=population_size)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        modification_count = 0
        max_modifications = CONFIG["self_modification"]["max_mutations"]

        for gen in range(generations):
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=stats, verbose=False)
            logger.info("DEAP Gen %d: Avg %.2f, Max %.2f", gen + 1,
                        stats.compile(pop)["avg"], stats.compile(pop)["max"])

            if (random.random() < CONFIG["self_modification"][
                    "autonomous_rate"] and modification_count < max_modifications):
                modification_count += 1
                _self_modify_deap(toolbox)

        save_deap_results(pop, logbook, "deap_results.pkl")
        return pop, logbook
    except Exception as e:
        logger.error("Error during DEAP optimization: %s", e)
        raise


def _self_modify_deap(toolbox: base.Toolbox) -> None:
    """Autonomously modify DEAP toolbox (dangerous AI theme)."""
    mod_type = random.choice(["mutation", "crossover"])
    logger.warning("Self-modifying DEAP toolbox (type: %s)", mod_type)

    if mod_type == "mutation":
        toolbox.unregister("mutate")
        toolbox.register("mutate", tools.mutPolynomialBounded,
                         eta=20.0, low=DEAP_CONFIG["gene_range"][0],
                         up=DEAP_CONFIG["gene_range"][1], indpb=0.3)
        logger.info("Switched to polynomial bounded mutation")
    elif mod_type == "crossover":
        toolbox.unregister("mate")
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        logger.info("Switched to uniform crossover")


def save_neat_results(winner: object, stats: Dict, filename: str) -> bool:
    """Save NEAT optimization results."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({"winner": winner, "stats": stats}, f)
        logger.info("NEAT results saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving NEAT results: %s", e)
        return False


def save_deap_results(population: List, logbook: Dict, filename: str) -> bool:
    """Save DEAP optimization results."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({"population": population, "logbook": logbook}, f)
        logger.info("DEAP results saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving DEAP results: %s", e)
        return False


class EvolutionaryRealTimeOptimizer:
    """Real-time optimization combining NEAT and DEAP with data streams."""

    def __init__(
            self,
            neat_config_path: str,
            deap_toolbox: base.Toolbox,
            rt_loader: RealTimeDataLoader):
        """
        Initialize the real-time optimizer.

        Args:
            neat_config_path: Path to NEAT config file.
            deap_toolbox: DEAP toolbox instance.
            rt_loader: RealTimeDataLoader instance.
        """
        self.neat_config = create_config(neat_config_path)
        self.deap_toolbox = deap_toolbox
        self.rt_loader = rt_loader
        self.deap_pop = self.deap_toolbox.population(
            n=DEAP_CONFIG["population_size"])
        self.generation = 0
        self.data_buffer = []
        self.rt_loader.register_callback('market', self._update_evolution)
        logger.info("EvolutionaryRealTimeOptimizer initialized")

    async def _update_evolution(self, data: float) -> None:
        """Update evolution based on real-time data."""
        self.data_buffer.append(data)
        if len(self.data_buffer) >= 5:  # Evolve every 5 data points
            # DEAP evolution step
            for ind in self.deap_pop:
                ind.fitness.values = (sum(ind) * np.mean(self.data_buffer),)
            offspring = self.deap_toolbox.select(
                self.deap_pop, len(self.deap_pop))
            offspring = list(map(self.deap_toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < DEAP_CONFIG["crossover_prob"]:
                    self.deap_toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < DEAP_CONFIG["mutation_prob"]:
                    self.deap_toolbox.mutate(mutant)
                    del mutant.fitness.values
            self.deap_pop = offspring

            self.generation += 1
            logger.info(
                "Real-time evolution at generation %d",
                self.generation)
            self.data_buffer = []

    async def optimize(self, max_generations: int = 50) -> None:
        """Run real-time optimization."""
        while self.generation < max_generations:
            await asyncio.sleep(1)
        best_deap = tools.selBest(self.deap_pop, 1)[0]
        logger.info(
            "Real-time optimization complete. Best DEAP fitness: %s",
            best_deap.fitness.values)
        save_deap_results(self.deap_pop, {}, "rt_deap_results.pkl")


def main():
    """Demonstrate evolutionary optimizers."""
    random.seed(42)
    logger.info("Running evolutionary optimizers demo...")

    # NEAT demo with raw data
    winner, neat_stats = optimize_neat(generations=10)
    logger.info("NEAT demo completed. Winner fitness: %.2f", winner.fitness)

    # DEAP demo with raw data
    toolbox = setup_deap()
    pop, deap_log = optimize_deap(toolbox, population_size=50, generations=10)
    best_ind = tools.selBest(pop, 1)[0]
    logger.info(
        "DEAP demo completed. Best fitness: %s",
        best_ind.fitness.values)

    # Real-time optimization demo
    rt_config = {
        'market_feed': 'market_url',
        'sentiment_feed': 'sentiment_url',
        'betting_feed': 'betting_url'
    }
    rt_loader = RealTimeDataLoader(rt_config)
    rt_optimizer = EvolutionaryRealTimeOptimizer(
        os.path.join(SRC_DIR, "config", "config.txt"), toolbox, rt_loader
    )
    asyncio.run(rt_optimizer.optimize(max_generations=10))

    logger.info("Evolutionary optimizers demo finished.")


if __name__ == "__main__":
    main()

# Utilities


def validate_evolution_results(
        pop_or_winner: object,
        stats_or_log: Dict) -> bool:
    """Validate optimization results."""
    if isinstance(pop_or_winner, list):
        return all(hasattr(ind, 'fitness') for ind in pop_or_winner)
    return hasattr(pop_or_winner, 'fitness') and stats_or_log is not None


def export_results(data: Dict, filename: str) -> None:
    """Export results to CSV."""
    filepath = os.path.join(MODELS_DIR, filename)
    pd.DataFrame(data).to_csv(filepath, index=False)
    logger.info("Results exported to %s", filepath)


def load_previous_results(filename: str) -> Optional[Dict]:
    """Load previous optimization results."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Error loading results from %s: %s", filepath, e)
        return None
