# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/hyperparameters.py
import os
import random
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Union, Callable
from datetime import datetime
import pickle
from config import PROJECT_CONFIG, get_project_config, NEAT_CONFIG, DEAP_CONFIG
from data_loader import load_raw_data, save_processed_data
from helpers import normalize_data, split_data

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


class HyperparameterBase:
    """Base class for hyperparameters with self-tuning capabilities."""

    def __init__(self):
        self.tuning_history = []
        self.modification_count = 0
        self.max_modifications = CONFIG["self_modification"]["max_mutations"]

    def log_tuning(self, performance: float, context: str) -> None:
        """Log tuning event."""
        self.tuning_history.append({
            "timestamp": datetime.now(),
            "performance": performance,
            "context": context
        })
        logger.info(
            "Hyperparameter tuning logged for %s: %.4f",
            context,
            performance)

    def _self_modify(self,
                     param_name: str,
                     current_value: Union[int,
                                          float]) -> Union[int,
                                                           float]:
        """Autonomously modify a hyperparameter (dangerous AI theme)."""
        if (random.random() < CONFIG["self_modification"]["autonomous_rate"]
                and self.modification_count < self.max_modifications):
            self.modification_count += 1
            if isinstance(current_value, int):
                new_value = int(current_value * random.uniform(0.8, 1.2))
                new_value = max(1, new_value)  # Ensure positive
            else:
                new_value = current_value * random.uniform(0.9, 1.1)
                new_value = min(max(new_value, 0.0),
                                1.0) if "prob" in param_name else new_value
            logger.warning(
                "Self-modifying %s from %.4f to %.4f (count: %d)",
                param_name,
                current_value,
                new_value,
                self.modification_count)
            return new_value
        return current_value

    def save_state(self, filename: str) -> bool:
        """Save hyperparameter state."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.__dict__, f)
            logger.info("Hyperparameter state saved to %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving hyperparameter state: %s", e)
            return False

    def load_state(self, filename: str) -> bool:
        """Load hyperparameter state."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                self.__dict__.update(pickle.load(f))
            logger.info("Hyperparameter state loaded from %s", filepath)
            return True
        except Exception as e:
            logger.error("Error loading hyperparameter state: %s", e)
            return False


class GP_Hyperparameters(HyperparameterBase):
    """Hyperparameters for genetic programming."""

    def __init__(self):
        super().__init__()
        self.population_size = DEAP_CONFIG.get("population_size", 300)
        self.num_generations = DEAP_CONFIG.get("num_generations", 40)
        self.crossover_prob = DEAP_CONFIG.get("crossover_prob", 0.5)
        self.mutation_prob = DEAP_CONFIG.get("mutation_prob", 0.1)
        self.tournament_size = DEAP_CONFIG.get(
            "selection", {}).get(
            "tournament_size", 3)
        self.elitism = 2
        self.max_depth = 5  # Max tree depth for GP
        self.init_min_depth = 1
        self.init_max_depth = 3

    def tune(self, performance: float) -> None:
        """Tune GP hyperparameters based on performance."""
        self.population_size = self._self_modify(
            "population_size", self.population_size)
        self.num_generations = self._self_modify(
            "num_generations", self.num_generations)
        self.crossover_prob = self._self_modify(
            "crossover_prob", self.crossover_prob)
        self.mutation_prob = self._self_modify(
            "mutation_prob", self.mutation_prob)
        self.tournament_size = self._self_modify(
            "tournament_size", self.tournament_size)
        self.elitism = self._self_modify("elitism", self.elitism)
        self.log_tuning(performance, "GP")

    def to_dict(self) -> Dict:
        """Convert hyperparameters to dictionary."""
        return {
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "crossover_prob": self.crossover_prob,
            "mutation_prob": self.mutation_prob,
            "tournament_size": self.tournament_size,
            "elitism": self.elitism,
            "max_depth": self.max_depth,
            "init_min_depth": self.init_min_depth,
            "init_max_depth": self.init_max_depth
        }


class NN_Hyperparameters(HyperparameterBase):
    """Hyperparameters for neural networks."""

    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.hidden_size = 64
        self.dropout_rate = 0.2
        self.weight_decay = 1e-5
        self.num_layers = 2
        self.activation = "relu"
        self.optimizer = "adam"

    def tune(self, performance: float) -> None:
        """Tune NN hyperparameters based on performance."""
        self.learning_rate = self._self_modify(
            "learning_rate", self.learning_rate)
        self.batch_size = self._self_modify("batch_size", self.batch_size)
        self.num_epochs = self._self_modify("num_epochs", self.num_epochs)
        self.hidden_size = self._self_modify("hidden_size", self.hidden_size)
        self.dropout_rate = self._self_modify(
            "dropout_rate", self.dropout_rate)
        self.weight_decay = self._self_modify(
            "weight_decay", self.weight_decay)
        self.num_layers = self._self_modify("num_layers", self.num_layers)
        self.log_tuning(performance, "NN")

    def to_dict(self) -> Dict:
        """Convert hyperparameters to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "weight_decay": self.weight_decay,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "optimizer": self.optimizer
        }


class NEAT_Hyperparameters(HyperparameterBase):
    """Hyperparameters for NEAT (assumed usage in project)."""

    def __init__(self):
        super().__init__()
        self.population_size = NEAT_CONFIG.get("pop_size", 150)
        self.num_generations = NEAT_CONFIG.get("num_generations", 300)
        self.mutation_rate = NEAT_CONFIG.get("mutation_rate", 0.1)
        self.species_elitism = 2
        self.stagnation_limit = 15

    def tune(self, performance: float) -> None:
        """Tune NEAT hyperparameters based on performance."""
        self.population_size = self._self_modify(
            "population_size", self.population_size)
        self.num_generations = self._self_modify(
            "num_generations", self.num_generations)
        self.mutation_rate = self._self_modify(
            "mutation_rate", self.mutation_rate)
        self.species_elitism = self._self_modify(
            "species_elitism", self.species_elitism)
        self.stagnation_limit = self._self_modify(
            "stagnation_limit", self.stagnation_limit)
        self.log_tuning(performance, "NEAT")

    def to_dict(self) -> Dict:
        """Convert hyperparameters to dictionary."""
        return {
            "population_size": self.population_size,
            "num_generations": self.num_generations,
            "mutation_rate": self.mutation_rate,
            "species_elitism": self.species_elitism,
            "stagnation_limit": self.stagnation_limit
        }


def get_gp_hyperparams() -> GP_Hyperparameters:
    """Return GP hyperparameters instance."""
    return GP_Hyperparameters()


def get_nn_hyperparams() -> NN_Hyperparameters:
    """Return NN hyperparameters instance."""
    return NN_Hyperparameters()


def get_neat_hyperparams() -> NEAT_Hyperparameters:
    """Return NEAT hyperparameters instance."""
    return NEAT_Hyperparameters()


def tune_hyperparameters(
    hyperparams: HyperparameterBase,
    performance: float,
    data: Optional[pd.DataFrame] = None
) -> None:
    """
    Tune hyperparameters based on performance and optionally data statistics.

    Args:
        hyperparams: Hyperparameter instance to tune.
        performance: Performance metric (e.g., fitness, loss).
        data: Optional data to inform tuning.
    """
    try:
        if data is not None:
            data_size = len(data)
            if hasattr(hyperparams, "population_size"):
                hyperparams.population_size = max(50, int(data_size * 0.1))
                logger.info(
                    "Adjusted population_size to %d based on data size",
                    hyperparams.population_size)
            if hasattr(hyperparams, "batch_size"):
                hyperparams.batch_size = min(
                    128, max(16, int(data_size * 0.05)))
                logger.info(
                    "Adjusted batch_size to %d based on data size",
                    hyperparams.batch_size)

        hyperparams.tune(performance)
    except Exception as e:
        logger.error("Error tuning hyperparameters: %s", e)
        raise


def save_all_hyperparams(
        hyperparams_dict: Dict[str, HyperparameterBase], filename: str) -> bool:
    """Save all hyperparameter instances."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump({k: v.to_dict()
                        for k, v in hyperparams_dict.items()}, f)
        logger.info("All hyperparameters saved to %s", filepath)
        return True
    except Exception as e:
        logger.error("Error saving all hyperparameters: %s", e)
        return False


def load_all_hyperparams(filename: str) -> Optional[Dict[str, Dict]]:
    """Load all hyperparameter dictionaries."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(
            "Error loading all hyperparameters from %s: %s",
            filepath,
            e)
        return None


class HyperparameterTuner:
    """Class to manage and tune multiple hyperparameter sets."""

    def __init__(self):
        self.hyperparams = {
            "GP": get_gp_hyperparams(),
            "NN": get_nn_hyperparams(),
            "NEAT": get_neat_hyperparams()
        }
        self.performance_history = {}

    def tune_all(self,
                 performances: Dict[str,
                                    float],
                 data: Optional[pd.DataFrame] = None) -> None:
        """Tune all hyperparameter sets."""
        for key, perf in performances.items():
            if key in self.hyperparams:
                tune_hyperparameters(self.hyperparams[key], perf, data)
                self.performance_history[key] = self.performance_history.get(
                    key, []) + [perf]

        if CONFIG["self_modification"]["enabled"] and random.random(
        ) < 0.1:  # Rare global tweak
            self._global_self_modify()

    def _global_self_modify(self) -> None:
        """Globally modify hyperparameters across types (dangerous AI theme)."""
        logger.warning("Performing global hyperparameter self-modification")
        for hp_type, hp in self.hyperparams.items():
            if hasattr(hp, "population_size"):
                hp.population_size = int(
                    hp.population_size *
                    random.uniform(
                        0.9,
                        1.5))
            if hasattr(hp, "num_generations"):
                hp.num_generations = int(
                    hp.num_generations *
                    random.uniform(
                        0.8,
                        1.2))
            logger.info("Globally modified %s hyperparameters", hp_type)

    def get_best_params(self, metric: str = "last") -> Dict[str, Dict]:
        """Get best hyperparameters based on performance history."""
        best = {}
        for key, history in self.performance_history.items():
            if history:
                if metric == "last":
                    best[key] = self.hyperparams[key].to_dict()
                elif metric == "max":
                    idx = np.argmax(history)
                    # Simplified; assumes prior save
                    best[key] = self.performance_history[key][idx]
        return best

    def save(self, filename: str = "hyperparams_tuner.pkl") -> bool:
        """Save tuner state."""
        return save_all_hyperparams(self.hyperparams, filename)


def main():
    """Demonstrate hyperparameter functionality."""
    random.seed(42)
    np.random.seed(42)

    # Initialize tuner
    tuner = HyperparameterTuner()

    # Simulate performance data
    performances = {
        "GP": 0.85,
        "NN": 0.92,
        "NEAT": 0.78
    }

    # Load raw data for tuning
    raw_data = load_raw_data()
    data = pd.concat(raw_data.values(), ignore_index=True)[
        :CONFIG["data"]["max_samples"]] if raw_data else None

    # Tune hyperparameters
    tuner.tune_all(performances, data)

    # Log results
    for key, hp in tuner.hyperparams.items():
        logger.info("%s Hyperparameters: %s", key, hp.to_dict())

    # Save and load
    tuner.save()
    loaded_params = load_all_hyperparams("hyperparams_tuner.pkl")
    logger.info("Loaded hyperparameters: %s", loaded_params)

    logger.info("Hyperparameters demo completed.")


if __name__ == "__main__":
    main()
