# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Callable
from datetime import datetime
import pickle
import asyncio
from deap import base, creator, gp, tools, algorithms
from config import PROJECT_CONFIG, get_project_config, DEAP_CONFIG, MODELS_DIR, RAW_DIR, export_config_to_dict
from data_loader import load_raw_data, save_processed_data, RealTimeDataLoader
# Removed: from evolutionaryoptimizer import optimize_deap  # Unresolved and unused

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(MODELS_DIR, 'gp.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load project and DEAP configuration
CONFIG = get_project_config()
DEAP_CONFIG_DICT = export_config_to_dict(DEAP_CONFIG)  # Convert string to dict

# Define the primitive set for mathematical expressions
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addEphemeralConstant("rand", lambda: random.uniform(-10, 10))

# Create fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize MSE
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Set up the DEAP GP toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

def evaluate_individual(individual: gp.PrimitiveTree, points: np.ndarray, targets: np.ndarray) -> Tuple[float]:
    """Evaluate an individual’s fitness as mean squared error."""
    try:
        func = toolbox.compile(expr=individual)
        predictions = np.array([func(x) for x in points])
        mse = np.mean((predictions - targets) ** 2)
        return (float("inf"),) if not np.isfinite(mse) else (mse,)
    except Exception as e:
        logger.error("Error evaluating individual %s: %s", str(individual), e)
        return (float("inf"),)

def evaluate_with_raw_data(individual: gp.PrimitiveTree, data: Optional[pd.DataFrame] = None) -> Tuple[float]:
    """Evaluate using raw data from data_loader."""
    if data is None or 'y' not in data.columns:
        points = np.linspace(0, 2 * np.pi, 100)
        targets = np.sin(points)
    else:
        points = data.drop(columns=['y']).values[:, 0]
        targets = data['y'].values
    return evaluate_individual(individual, points, targets)

def run_gp(
    population_size: int = int(DEAP_CONFIG_DICT["DEAP"]["population_size"]),
    generations: int = int(DEAP_CONFIG_DICT["DEAP"]["generations"]),
    cxpb: float = float(DEAP_CONFIG_DICT["Crossover"]["cx_probability"]),
    mutpb: float = float(DEAP_CONFIG_DICT["Mutation"]["mut_probability"]),
    use_raw_data: bool = True
) -> Tuple[List, Dict]:
    """Run genetic programming evolution process."""
    try:
        logger.info("Starting GP with %d individuals over %d generations", population_size, generations)
        pop = toolbox.population(n=population_size)

        raw_data = load_raw_data() if use_raw_data else {}
        combined_data = pd.concat(raw_data.values(), ignore_index=True)[:CONFIG["data"]["max_samples"]] if raw_data else None
        toolbox.register("evaluate", evaluate_with_raw_data, data=combined_data)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        modification_count = 0
        max_modifications = CONFIG["self_modification"]["max_mutations"]

        for gen in range(generations):
            pop, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, stats=stats, verbose=False
            )
            logger.info("GP Gen %d: Avg %.2f, Min %.2f", gen + 1, stats.compile(pop)["avg"], stats.compile(pop)["min"])

            if random.random() < CONFIG["self_modification"]["autonomous_rate"] and modification_count < max_modifications:
                modification_count += 1
                _self_modify_gp(gen, generations)

        best_ind = tools.selBest(pop, 1)[0]
        logger.info("Best GP individual: %s, Fitness: %.2f", str(best_ind), best_ind.fitness.values[0])
        save_gp_results(pop, logbook, "gp_results.pkl")
        return pop, logbook
    except Exception as e:
        logger.error("Error running GP: %s", e)
        raise

def _self_modify_gp(current_gen: int, total_gens: int) -> None:
    """Autonomously modify GP toolbox."""
    if not CONFIG["self_modification"]["enabled"]:
        return
    mod_type = random.choice(["primitives", "mutation"])
    logger.warning(f"Self-modifying GP (type: {mod_type}) at generation {current_gen}")
    if mod_type == "primitives":
        pset.addPrimitive(np.tanh, 1, name="tanh")
        logger.info("Added tanh primitive to pset")
    elif mod_type == "mutation":
        toolbox.unregister("mutate")
        toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
        logger.info("Switched to node replacement mutation")

def save_gp_results(population: List, logbook: Dict, filename: str) -> bool:
    """Save GP results to a file."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({"population": population, "logbook": logbook}, f)
        logger.info("GP results saved to %s", filepath)
        return True
    except (OSError, pickle.PickleError) as e:
        logger.error("Error saving GP results to %s: %s", filepath, e)
        return False

def load_gp_results(filename: str) -> Optional[Dict]:
    """Load previous GP results."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        logger.error("Error loading GP results from %s: %s", filepath, e)
        return None

class RealTimeGP:
    """Real-time genetic programming with data streams."""
    def __init__(self, rt_loader: RealTimeDataLoader):
        self.rt_loader = rt_loader
        self.population = toolbox.population(n=int(DEAP_CONFIG_DICT["DEAP"]["population_size"]))
        self.generation = 0
        self.data_buffer = []
        self.rt_loader.register_callback('market', self._update_evolution)
        logger.info("RealTimeGP initialized")

    async def _update_evolution(self, data: float) -> None:
        """Update GP evolution with real-time data."""
        self.data_buffer.append(data)
        if len(self.data_buffer) >= 5:
            points = np.array(self.data_buffer)
            targets = np.sin(points)  # Placeholder target
            toolbox.register("evaluate", evaluate_individual, points=points, targets=targets)

            offspring = toolbox.select(self.population, len(self.population))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < float(DEAP_CONFIG_DICT["Crossover"]["cx_probability"]):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < float(DEAP_CONFIG_DICT["Mutation"]["mut_probability"]):
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            self.population = offspring

            self.generation += 1
            best = tools.selBest(self.population, 1)[0]
            logger.info("Real-time GP Gen %d: Best Fitness %.2f", self.generation, best.fitness.values[0])
            self.data_buffer = []

    async def run(self, max_generations: int = 50) -> Dict:
        """Run real-time GP."""
        try:
            while self.generation < max_generations:
                await asyncio.sleep(1)
            results = {"population": self.population, "generations": self.generation}
            save_gp_results(self.population, {}, "rt_gp_results.pkl")
            return results
        except Exception as e:
            logger.error("Error in RealTimeGP run: %s", e)
            return {"population": [], "generations": self.generation}

def main():
    """Demonstrate genetic programming functionality."""
    random.seed(42)
    np.random.seed(42)

    # Static GP with raw data
    pop, log = run_gp(population_size=50, generations=40)
    best_ind = tools.selBest(pop, 1)[0]
    logger.info("Static GP completed. Best: %s, Fitness: %.2f", str(best_ind), best_ind.fitness.values[0])

    # Real-time GP
    rt_config = {
        'market_feed': 'market_url',
        'sentiment_feed': 'sentiment_url',
        'betting_feed': 'betting_url'
    }
    rt_loader = RealTimeDataLoader(rt_config)
    rt_gp = RealTimeGP(rt_loader)
    asyncio.run(rt_gp.run(max_generations=10))

    logger.info("Genetic programming demo completed.")

if __name__ == "__main__":
    main()

# Utilities
def validate_population(population: List) -> bool:
    """Validate GP population."""
    return all(hasattr(ind, 'fitness') and np.isfinite(ind.fitness.values[0]) for ind in population)

def export_results(pop: List, logbook: Dict, filename: str) -> None:
    """Export GP results to CSV."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        data = {"generation": range(len(logbook)), **logbook}
        pd.DataFrame(data).to_csv(filepath, index=False)
        logger.info("GP results exported to %s", filepath)
    except Exception as e:
        logger.error("Error exporting results to %s: %s", filepath, e)

def simulate_raw_data(filename: str = "gp_simulated.csv") -> None:
    """Generate simulated raw data if none exists."""
    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        data = pd.DataFrame({
            "x": np.linspace(0, 2 * np.pi, 100),
            "y": np.sin(np.linspace(0, 2 * np.pi, 100)) + np.random.normal(0, 0.1, 100)
        })
        data.to_csv(filepath, index=False)
        logger.info("Simulated GP data generated at %s", filepath)