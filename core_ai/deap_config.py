# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/deap.py
import os
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime
import logging
import pickle
# From config.py
from config import PROJECT_CONFIG, get_project_config, NEAT_CONFIG, DEAP_CONFIG
# From data_loader.py
from data_loader import load_raw_data, save_processed_data, RealTimeDataLoader
from config import asyncio

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


def setup_deap(config: Dict = DEAP_CONFIG) -> base.Toolbox:
    """
    Set up DEAP for evolutionary computation with dynamic configuration.

    Args:
        config: DEAP configuration dictionary (default from config.py).

    Returns:
        toolbox: Configured DEAP toolbox.
    """
    try:
        # Create fitness class (maximization or minimization based on config)
        creator.create(
            "FitnessMax", base.Fitness, weights=(
                config["weights"]["fitness"],))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Attribute generation based on problem type
        if config["problem_type"] == "real-valued":
            toolbox.register(
                "attribute",
                random.uniform,
                config["gene_range"][0],
                config["gene_range"][1])
        else:
            toolbox.register(
                "attribute",
                random.randint,
                0,
                1)  # Binary problem

        # Individual and population initialization
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attribute, n=config["genome_size"])
        toolbox.register(
            "population",
            tools.initRepeat,
            list,
            toolbox.individual)

        # Evolutionary operators
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register(
            "mate",
            tools.cxOnePoint if config["crossover"] == "one_point" else tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=config["mutation"]["sigma"],
            indpb=config["mutation"]["probability"])
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=config["selection"]["tournament_size"])

        logger.info(
            "DEAP toolbox configured with genome size %d and problem type %s",
            config["genome_size"],
            config["problem_type"])
        return toolbox

    except Exception as e:
        logger.error("Error setting up DEAP toolbox: %s", e)
        raise


def evaluate_individual(
        individual: List[float], data: Optional[pd.DataFrame] = None) -> Tuple[float]:
    """
    Evaluate an individual's fitness based on raw data or default sum.

    Args:
        individual: List of genes to evaluate.
        data: Optional DataFrame from data_loader for fitness computation.

    Returns:
        Tuple containing the fitness value.
    """
    try:
        if data is not None and "y" in data.columns:
            # Example: Predict 'y' using individual as coefficients
            X = data.drop(columns=["y"]).values
            y_pred = np.dot(X, np.array(individual[:X.shape[1]]))
            y_true = data["y"].values
            # Negative MSE for maximization
            fitness = -np.mean((y_pred - y_true) ** 2)
        else:
            fitness = sum(individual)  # Default: Maximize sum
        return (fitness,)
    except Exception as e:
        logger.error("Error evaluating individual: %s", e)
        return (float('-inf'),)


def run_evolution(
    toolbox: base.Toolbox,
    population_size: int = DEAP_CONFIG["population_size"],
    num_generations: int = DEAP_CONFIG["num_generations"],
    cxpb: float = DEAP_CONFIG["crossover_prob"],
    mutpb: float = DEAP_CONFIG["mutation_prob"],
    data: Optional[pd.DataFrame] = None
) -> Tuple[List, Dict]:
    """
    Run the evolutionary algorithm with self-modification capabilities.

    Args:
        toolbox: Configured DEAP toolbox.
        population_size: Number of individuals.
        num_generations: Number of generations.
        cxpb: Crossover probability.
        mutpb: Mutation probability.
        data: Optional raw data for fitness evaluation.

    Returns:
        Tuple of final population and logbook.
    """
    try:
        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)  # Track best individual
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)

        # Self-modification setup
        modification_count = 0
        max_modifications = CONFIG["self_modification"]["max_mutations"]

        for gen in range(num_generations):
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1,
                stats=stats, halloffame=hof, verbose=False
            )
            logger.info(
                "Generation %d: Avg Fitness %.2f, Max Fitness %.2f",
                gen + 1,
                stats.compile(pop)["avg"],
                stats.compile(pop)["max"])

            # Self-modification trigger
            if (random.random() < CONFIG["self_modification"] [
                "autonomous_rate"] and modification_count < max_modifications):
                modification_count += 1
                _self_modify_toolbox(toolbox, gen, num_generations)

        save_population(pop, "final_population.pkl")
        return pop, logbook

    except Exception as e:
        logger.error("Error running evolution: %s", e)
        raise


def _self_modify_toolbox(
        toolbox: base.Toolbox,
        current_gen: int,
        total_gens: int) -> None:
    """
    Autonomously modify the DEAP toolbox (dangerous AI theme).

    Args:
        toolbox: DEAP toolbox to modify.
        current_gen: Current generation number.
        total_gens: Total number of generations.
    """
    if not CONFIG["self_modification"]["enabled"]:
        return

    mod_type = random.choice(["mutation", "selection", "crossover"])
    logger.warning(
        f"Self-modifying DEAP toolbox (type: {mod_type}) at generation {current_gen}")

    if mod_type == "mutation":
        new_sigma = random.uniform(0.05, 0.5)
        toolbox.unregister("mutate")
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=new_sigma,
            indpb=0.3)
        logger.info("Mutation sigma changed to %.2f", new_sigma)

    elif mod_type == "selection":
        new_tournsize = random.randint(2, 5)
        toolbox.unregister("select")
        toolbox.register(
            "select",
            tools.selTournament,
            tournsize=new_tournsize)
        logger.info("Tournament size changed to %d", new_tournsize)

    elif mod_type == "crossover":
        toolbox.unregister("mate")
        toolbox.register("mate", tools.cxBlend, alpha=random.uniform(0.1, 0.5))
        logger.info(
            "Switched to blend crossover with alpha %.2f",
            toolbox.mate.keywords["alpha"])


def save_population(population: List, filename: str) -> bool:
    """
    Save the evolved population to a file.

    Args:
        population: List of individuals.
        filename: Name of the file to save to.

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(population, f)
        logger.info("Population saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving population to %s: %s", filepath, e)
        return False


def load_population(filename: str) -> Optional[List]:
    """
    Load a previously saved population.

    Args:
        filename: Name of the file to load from.

    Returns:
        List of individuals or None if loading fails.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            population = pickle.load(f)
        logger.info("Population loaded from %s", filepath)
        return population
    except Exception as e:
        logger.error("Error loading population from %s: %s", filepath, e)
        return None


def integrate_raw_data() -> pd.DataFrame:
    """Load and combine raw data for evolution."""
    raw_data = load_raw_data()
    if not raw_data:
        logger.warning("No raw data available, generating simulated data")
        simulate_raw_data()
        raw_data = load_raw_data()

    combined = pd.concat(raw_data.values(), ignore_index=True)
    return combined[:CONFIG["data"]["max_samples"]]


def simulate_raw_data(filename: str = "deap_simulated.csv") -> None:
    """Generate simulated raw data if none exists."""
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        data = pd.DataFrame({
            "x1": np.random.uniform(-10, 10, 1000),
            "x2": np.random.uniform(-10, 10, 1000),
            "y": np.random.normal(0, 1, 1000)
        })
        data.to_csv(filepath, index=False)
        logger.info("Simulated DEAP data generated at %s", filepath)


class DEAPRealTimeOptimizer:
    """Real-time evolutionary optimization using DEAP and data streams."""

    def __init__(self, toolbox: base.Toolbox, rt_loader: RealTimeDataLoader):
        """
        Initialize the real-time optimizer.

        Args:
            toolbox: DEAP toolbox for evolution.
            rt_loader: RealTimeDataLoader instance for streaming data.
        """
        self.toolbox = toolbox
        self.rt_loader = rt_loader
        self.population = self.toolbox.population(
            n=DEAP_CONFIG["population_size"])
        self.generation = 0
        self.data_buffer = []
        self.rt_loader.register_callback('market', self._update_fitness)
        logger.info("DEAPRealTimeOptimizer initialized")

    async def _update_fitness(self, data: float) -> None:
        """Update population fitness based on real-time data."""
        self.data_buffer.append(data)
        if len(self.data_buffer) >= 10:  # Evolve every 10 data points
            for ind in self.population:
                ind.fitness.values = (sum(ind) * np.mean(self.data_buffer),)
            self.population = self.toolbox.select(
                self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, self.population))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < DEAP_CONFIG["crossover_prob"]:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < DEAP_CONFIG["mutation_prob"]:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            self.data_buffer = []
            self.generation += 1
            logger.info(
                "Real-time evolution at generation %d",
                self.generation)

    async def optimize(self, max_generations: int = 100) -> None:
        """Run real-time optimization."""
        while self.generation < max_generations:
            await asyncio.sleep(1)  # Simulate processing delay
        best = tools.selBest(self.population, 1)[0]
        logger.info(
            "Real-time optimization complete. Best fitness: %s",
            best.fitness.values)


def main():
    """Main function to demonstrate DEAP functionality."""
    random.seed(42)

    # Static evolution with raw data
    toolbox = setup_deap()
    raw_data = integrate_raw_data()
    pop, logbook = run_evolution(toolbox, data=raw_data)
    best_ind = tools.selBest(pop, 1)[0]
    logger.info("Static evolution completed. Best individual: %s, Fitness: %s",
                best_ind, best_ind.fitness.values)

    # Real-time evolution
    rt_config = {
        'market_feed': 'market_url',
        'sentiment_feed': 'sentiment_url',
        'betting_feed': 'betting_url'
    }
    rt_loader = RealTimeDataLoader(rt_config)
    optimizer = DEAPRealTimeOptimizer(toolbox, rt_loader)

    import asyncio
    asyncio.run(optimizer.optimize())

    # Save results
    save_population(pop, "deap_final_population.pkl")


if __name__ == "__main__":
    main()

# Additional utilities


def validate_population(population: List) -> bool:
    """Validate the population for integrity."""
    return all(hasattr(ind, 'fitness')
               and ind.fitness.values for ind in population)


def export_results(
        pop: List,
        logbook: Dict,
        filename: str = "deap_results.csv") -> None:
    """Export evolution results to CSV."""
    filepath = os.path.join(MODELS_DIR, filename)
    data = {"generation": range(len(logbook)), **logbook}
    pd.DataFrame(data).to_csv(filepath, index=False)
    logger.info("Results exported to %s", filepath)


def load_previous_run(
        filename: str = "deap_final_population.pkl") -> Tuple[Optional[List], base.Toolbox]:
    """Load a previous run and set up toolbox."""
    pop = load_population(filename)
    toolbox = setup_deap()
    return pop, toolbox
