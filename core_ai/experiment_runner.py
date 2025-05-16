# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/experimentrunner.py
import os
import time
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime
import pickle
import asyncio
from config import PROJECT_CONFIG, get_project_config, NEAT_CONFIG, DEAP_CONFIG
from data_loader import load_raw_data, save_processed_data, RealTimeDataLoader
# Replaces selfmodificationengine
from src.evolutionaryoptimizer import optimize_neat, optimize_deap
from codemodifier import load_and_apply_code_modification  # Assumes this exists
from deap import base, tools  # For DEAP toolbox type hint and selection tools
from metalearner import MetaLearner  # Assumes this exists

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


class MathemagicianModel:
    """Simple model class to simulate architecture changes."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.architecture_stats = {"layers": 1, "nodes": 10, "changes": 0}

    def set_target_function(self, func_name: str, **params) -> None:
        """Set the target function for optimization."""
        self.target_func = func_name
        self.func_params = params
        logger.info(
            "Set target function: %s with params %s",
            func_name,
            params)

    def modify_architecture(self) -> None:
        """Simulate architecture modification."""
        if random.random() < CONFIG["self_modification"]["autonomous_rate"]:
            self.architecture_stats["layers"] += random.randint(0, 2)
            self.architecture_stats["nodes"] += random.randint(0, 5)
            self.architecture_stats["changes"] += 1
            logger.warning(
                "Model architecture modified: %s",
                self.architecture_stats)


def run_experiment(
    config_path: str = os.path.join(SRC_DIR, "config", "config.txt"),
    generations: int = 300,
    experiment_name: str = "default",
    function_suite: Optional[Dict] = None,
    use_deap: bool = False,
    toolbox: Optional[base.Toolbox] = None
) -> Dict:
    """
    Enhanced experiment runner with function suite testing and dual NEAT/DEAP support.

    Args:
        config_path: Path to configuration file (NEAT or DEAP context).
        generations: Number of generations to run.
        experiment_name: Name of the experiment.
        function_suite: Dictionary of functions to test (e.g., {'sin': {}}).
        use_deap: If True, use DEAP instead of NEAT.
        toolbox: DEAP toolbox if using DEAP (optional).

    Returns:
        Dictionary of results for each function or default run.
    """
    try:
        logger.info(
            "Starting experiment: %s (DEAP: %s)",
            experiment_name,
            use_deap)
        start_time = time.time()

        # Initialize meta-learner and model
        meta_learner = MetaLearner()
        model = MathemagicianModel(config_path)

        # Load raw data for evaluation
        raw_data = load_raw_data()
        combined_data = pd.concat(raw_data.values(), ignore_index=True)[
            :CONFIG["data"]["max_samples"]] if raw_data else None

        results = {}
        if function_suite:
            for func_name, params in function_suite.items():
                logger.info("Testing function: %s", func_name)
                model.set_target_function(func_name, **params)

                if use_deap:
                    pop, logbook = optimize_deap(
                        toolbox=toolbox, population_size=DEAP_CONFIG["population_size"],
                        generations=generations
                    )
                    winner = tools.selBest(pop, 1)[0]
                    stats = logbook
                else:
                    winner, stats = optimize_neat(
                        config_path, generations=generations)

                model.modify_architecture()  # Self-modification
                results[func_name] = {
                    'winner': winner,
                    'stats': stats,
                    'architecture_changes': model.architecture_stats,
                    'runtime': time.time() - start_time
                }
        else:
            if use_deap:
                pop, logbook = optimize_deap(
                    toolbox=toolbox, population_size=DEAP_CONFIG["population_size"],
                    generations=generations
                )
                winner = tools.selBest(pop, 1)[0]
                stats = logbook
            else:
                winner, stats = optimize_neat(
                    config_path, generations=generations)

            model.modify_architecture()
            results['default'] = {
                'winner': winner,
                'stats': stats,
                'architecture_changes': model.architecture_stats,
                'runtime': time.time() - start_time
            }

        save_experiment_results(results, f"{experiment_name}_results.pkl")
        logger.info("Experiment %s completed in %.2f seconds",
                    experiment_name, time.time() - start_time)
        return results

    except Exception as e:
        logger.error("Error in experiment %s: %s", experiment_name, e)
        raise


def run_multiple_experiments(
    config_paths: List[str],
    generations_list: List[int],
    use_deap: bool = False,
    toolbox: Optional[base.Toolbox] = None
) -> List[Dict]:
    """
    Run multiple experiments with different configurations.

    Args:
        config_paths: List of configuration file paths.
        generations_list: List of generation counts.
        use_deap: If True, use DEAP for all experiments.
        toolbox: DEAP toolbox if using DEAP.

    Returns:
        List of results from each experiment.
    """
    results = []
    for config, generations in zip(config_paths, generations_list):
        exp_name = os.path.basename(config)
        result = run_experiment(
            config_path=config, generations=generations,
            experiment_name=exp_name, use_deap=use_deap, toolbox=toolbox
        )
        results.append(result)
    logger.info("All %d experiments completed", len(results))
    return results


def save_experiment_results(results: Dict, filename: str) -> bool:
    """Save experiment results to a file."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        logger.info("Experiment results saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving results to %s: %s", filepath, e)
        return False


def load_experiment_results(filename: str) -> Optional[Dict]:
    """Load previous experiment results."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        logger.info("Loaded experiment results from %s", filepath)
        return results
    except Exception as e:
        logger.error("Error loading results from %s: %s", filepath, e)
        return None


class RealTimeExperimentRunner:
    """Real-time experiment runner with NEAT or DEAP and data streams."""

    def __init__(self,
                 config_path: str,
                 rt_loader: RealTimeDataLoader,
                 use_deap: bool = False,
                 toolbox: Optional[base.Toolbox] = None):
        """
        Initialize the real-time experiment runner.

        Args:
            config_path: Configuration file path.
            rt_loader: RealTimeDataLoader instance.
            use_deap: If True, use DEAP instead of NEAT.
            toolbox: DEAP toolbox if using DEAP.
        """
        self.config_path = config_path
        self.rt_loader = rt_loader
        self.use_deap = use_deap
        self.toolbox = toolbox if use_deap else None
        self.model = MathemagicianModel(config_path)
        self.generation = 0
        self.results = {}
        self.rt_loader.register_callback('market', self._real_time_update)
        logger.info(
            "RealTimeExperimentRunner initialized (DEAP: %s)",
            use_deap)

    async def _real_time_update(self, data: float) -> None:
        """Update experiment based on real-time data."""
        self.generation += 1
        logger.info(
            "Real-time experiment update at generation %d with data %.2f",
            self.generation,
            data)

        if self.use_deap:
            pop, logbook = optimize_deap(self.toolbox, generations=1)
            winner = tools.selBest(pop, 1)[0]
            stats = logbook
        else:
            winner, stats = optimize_neat(self.config_path, generations=1)

        self.model.modify_architecture()
        self.results[f"gen_{self.generation}"] = {
            'winner': winner,
            'stats': stats,
            'architecture_changes': self.model.architecture_stats,
            'data_point': data
        }

    async def run(self, max_generations: int = 50) -> Dict:
        """Run the real-time experiment."""
        start_time = time.time()
        while self.generation < max_generations:
            await asyncio.sleep(1)
        self.results['runtime'] = time.time() - start_time
        save_experiment_results(
            self.results, f"rt_experiment_{
                self.generation}.pkl")
        logger.info(
            "Real-time experiment completed after %d generations",
            self.generation)
        return self.results


# Example function suite
FUNCTION_SUITE = {
    'sin': {},
    'cos': {},
    'exp': {},
    'polynomial': {'coeffs': [1, 2, 1]}  # x^2 + 2x + 1
}


def main():
    """Demonstrate experiment runner functionality."""
    from deap import setup_deap  # Import here for demo

    # Static experiments with NEAT
    configs = [
        os.path.join(SRC_DIR, "config", "config1.txt"),
        os.path.join(SRC_DIR, "config", "config2.txt")
    ]
    generations = [200, 300]
    results = run_multiple_experiments(configs, generations)
    for idx, result in enumerate(results):
        best_fit = result['default']['winner'].fitness if not hasattr(
            result['default']['winner'],
            'fitness') else result['default']['winner'].fitness.values[0]
        logger.info(
            "NEAT Result from %s: Best fitness: %.4f",
            configs[idx],
            best_fit)

    # DEAP experiment
    toolbox = setup_deap()
    deap_result = run_experiment(
        config_path=configs[0], generations=50, experiment_name="deap_test",
        function_suite=FUNCTION_SUITE, use_deap=True, toolbox=toolbox
    )
    logger.info("DEAP experiment completed: %s", deap_result.keys())

    # Real-time experiment
    rt_config = {
        'market_feed': 'market_url',
        'sentiment_feed': 'sentiment_url',
        'betting_feed': 'betting_url'
    }
    rt_loader = RealTimeDataLoader(rt_config)
    rt_runner = RealTimeExperimentRunner(
        configs[0], rt_loader, use_deap=True, toolbox=toolbox)
    asyncio.run(rt_runner.run(max_generations=10))

    logger.info("Experiment runner demo completed.")


if __name__ == "__main__":
    main()

# Utilities


def validate_results(results: Dict) -> bool:
    """Validate experiment results."""
    return all('winner' in r and 'stats' in r for r in results.values())


def export_results_to_csv(results: Dict, filename: str) -> None:
    """Export results to CSV."""
    filepath = os.path.join(MODELS_DIR, filename)
    data = {}
    for key, res in results.items():
        if 'winner' in res:
            fitness = res['winner'].fitness if not hasattr(
                res['winner'], 'fitness') else res['winner'].fitness.values[0]
            data[key] = {'fitness': fitness, 'runtime': res.get('runtime', 0)}
    pd.DataFrame.from_dict(data, orient='index').to_csv(filepath)
    logger.info("Results exported to %s", filepath)


def simulate_config_files() -> List[str]:
    """Simulate config files if missing."""
    configs = []
    for i in [1, 2]:
        path = os.path.join(SRC_DIR, "config", f"config{i}.txt")
        if not os.path.exists(path):
            with open(path, 'w') as f:
                f.write("[NEAT]\nfitness_criterion = max\npop_size = 150\n")
            logger.info("Simulated config file created: %s", path)
        configs.append(path)
    return configs
