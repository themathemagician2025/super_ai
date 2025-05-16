# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/metalearning.py
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import pickle
from config import PROJECT_CONFIG, get_project_config
from data_loader import load_raw_data, save_processed_data, RealTimeDataLoader
from helpers import normalize_data
from hyperparameters import GP_Hyperparameters, NN_Hyperparameters, NEAT_Hyperparameters, tune_hyperparameters

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


class MetaLearner:
    """
    MetaLearner dynamically adjusts hyperparameters across multiple domains (numeric, symbolic,
    market, trading) based on performance metrics, market states, and real-time data. It supports
    self-modification, strategy selection via bandit-like learning, and integration with other
    project modules, embodying a potentially autonomous and unpredictable AI.
    """

    def __init__(self):
        self.hyperparameters = {
            'numeric': GP_Hyperparameters(),
            'symbolic': GP_Hyperparameters(),  # Reused for symbolic tasks
            'market': NN_Hyperparameters(),    # Adapted for market prediction
            'trading': NEAT_Hyperparameters()  # Adapted for trading strategies
        }
        self.performance_history: Dict[str, List[float]] = {
            mode: [] for mode in self.hyperparameters}
        self.adaptation_count = 0
        self.mode = 'numeric'  # Default mode
        self.last_selected_strategies: Dict[str, Dict[str, int]] = {}
        self.strategy_scores: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        self.parameter_bounds: Dict[str,
                                    Tuple[Union[int, float], Union[int, float]]] = {}
        self.adaptation_strategies: Dict[str, List[Callable]] = {}
        self._initialize_strategies_and_bounds()
        self.rt_loader = None  # For real-time integration

    def _initialize_strategies_and_bounds(self) -> None:
        """Initialize adaptation strategies and parameter bounds."""
        # Numeric (GP-like)
        self.parameter_bounds.update(
            {
                'population_size': (
                    50, 1000), 'num_generations': (
                    10, 500), 'crossover_prob': (
                    0.0, 1.0), 'mutation_prob': (
                        0.0, 1.0), 'tournament_size': (
                            2, 10), 'elitism': (
                                0, 10), 'max_depth': (
                                    1, 10), 'init_min_depth': (
                                        1, 5), 'init_max_depth': (
                                            2, 7)})
        self.adaptation_strategies.update({
            'population_size': [
                lambda perf, curr: int(curr * (1.1 if perf < 0.5 else 0.9)),
                lambda perf, curr: int(200 + 300 * perf),
                lambda perf, curr: np.random.randint(50, 1000)  # Risky
            ],
            'mutation_prob': [
                lambda perf, curr: curr * (1.1 if perf < 0.5 else 0.9),
                lambda perf, curr: 0.1 + 0.2 * perf,
                lambda perf, curr: np.random.uniform(0.0, 1.0)
            ]
        })

        # Symbolic (GP-like)
        self.parameter_bounds.update({'tree_depth': (1, 10), 'primitives_weight': (
            0.0, 1.0), 'complexity_penalty': (0.0, 1.0)})
        self.adaptation_strategies.update({
            'max_depth': [
                lambda perf, curr: min(curr + 1, 10) if perf < 0.5 else max(curr - 1, 1),
                lambda perf, curr: int(3 + 4 * perf),
                lambda perf, curr: np.random.randint(1, 11)
            ]
        })

        # Market (NN-like)
        self.parameter_bounds.update({
            'learning_rate': (1e-5, 0.1), 'batch_size': (16, 256), 'num_epochs': (10, 200),
            'hidden_size': (16, 256), 'dropout_rate': (0.0, 0.5), 'weight_decay': (1e-6, 1e-2),
            'num_layers': (1, 5)
        })
        self.adaptation_strategies.update({
            'learning_rate': [
                lambda perf, curr: curr * (1.1 if perf < 0.5 else 0.9),
                lambda perf, curr: 0.001 * (1 + perf),
                lambda perf, curr: np.random.uniform(1e-5, 0.1)
            ]
        })

        # Trading (NEAT-like)
        self.parameter_bounds.update(
            {
                'position_size': (
                    0.0, 1.0), 'stop_loss': (
                    0.0, 0.5), 'take_profit': (
                    0.0, 0.5), 'risk_tolerance': (
                        0.0, 1.0), 'sentiment_weight': (
                            0.0, 1.0), 'technical_weight': (
                                0.0, 1.0)})
        self.adaptation_strategies.update({
            'position_size': [
                lambda perf, curr: curr * (1.1 if perf < 0.5 else 0.9),
                lambda perf, curr: 0.1 + 0.2 * perf,
                lambda perf, curr: np.random.uniform(0.0, 1.0)
            ]
        })

        # Initialize strategy scores
        for mode, hp in self.hyperparameters.items():
            self.strategy_scores[mode] = {
                key: [{'sum': 0, 'count': 0} for _ in self.adaptation_strategies.get(key, [lambda p, c: c])]
                for key in hp.to_dict()
            }

    def update_hyperparameters(
            self,
            performance: float,
            market_state: Dict = None) -> Dict:
        """
        Update hyperparameters based on performance and market state.

        Args:
            performance: Performance metric (0-1 scale).
            market_state: Market conditions (volatility, sentiment, etc.).

        Returns:
            Current hyperparameters.
        """
        try:
            if not isinstance(performance, (int, float)):
                raise ValueError("Performance must be numeric")
            performance = np.clip(performance, 0.0, 1.0)

            market_state = market_state or {
                'volatility': 0.5, 'sentiment': 0.0}
            volatility = np.clip(market_state.get('volatility', 0.5), 0.0, 1.0)
            sentiment = np.clip(market_state.get('sentiment', 0.0), -1.0, 1.0)

            self.performance_history[self.mode].append(performance)
            self._update_strategy_scores(performance)
            self._select_and_apply_strategies(
                performance, volatility, sentiment)
            self._adjust_risk_parameters(volatility, sentiment)
            self._validate_parameters()

            if len(self.performance_history[self.mode]) >= 3 and self._analyze_performance_trend(
            ) == 'stagnating':
                self._switch_optimization_mode()

            return self.get_current_parameters()
        except Exception as e:
            logger.error("Error updating hyperparameters: %s", e)
            self.reset_hyperparameters()
            return self.get_current_parameters()

    def _update_strategy_scores(self, performance: float) -> None:
        """Update strategy scores based on performance delta."""
        if self.last_selected_strategies.get(self.mode) and len(
                self.performance_history[self.mode]) >= 2:
            delta = performance - self.performance_history[self.mode][-2]
            for hp, idx in self.last_selected_strategies[self.mode].items():
                self.strategy_scores[self.mode][hp][idx]['sum'] += delta
                self.strategy_scores[self.mode][hp][idx]['count'] += 1

    def _select_and_apply_strategies(
            self,
            performance: float,
            volatility: float,
            sentiment: float) -> None:
        """Select and apply adaptation strategies using a bandit approach."""
        selected_strategies = {}
        hp_dict = self.hyperparameters[self.mode].to_dict()

        for hp in hp_dict:
            if hp in self.adaptation_strategies:
                scores = self.strategy_scores[self.mode][hp]
                averages = [s['sum'] / s['count']
                            if s['count'] > 0 else 0 for s in scores]
                exp_averages = [np.exp(a) for a in averages]
                total = sum(exp_averages) or 1
                probs = [e / total for e in exp_averages]
                idx = np.random.choice(len(probs), p=probs)
                selected_strategies[hp] = idx
                strategy = self.adaptation_strategies[hp][idx]
                current_value = hp_dict[hp]
                new_value = strategy(performance, current_value)
                setattr(self.hyperparameters[self.mode], hp, new_value)

        self.last_selected_strategies[self.mode] = selected_strategies
        self.adaptation_count += 1
        logger.info("Applied adaptation strategies for %s mode", self.mode)

        # Dangerous AI: Random full reset
        if CONFIG["self_modification"]["enabled"] and random.random() < 0.05:
            self._radical_self_modification()

    def _radical_self_modification(self) -> None:
        """Radically modify hyperparameters (dangerous AI theme)."""
        hp_dict = self.hyperparameters[self.mode].to_dict()
        for hp in hp_dict:
            if hp in self.parameter_bounds:
                low, high = self.parameter_bounds[hp]
                new_value = random.uniform(
                    low,
                    high) if isinstance(
                    low,
                    float) else random.randint(
                    low,
                    high)
                setattr(self.hyperparameters[self.mode], hp, new_value)
        logger.warning(
            "Performed radical self-modification on %s hyperparameters",
            self.mode)

    def _analyze_performance_trend(self) -> str:
        """Analyze recent performance trend."""
        recent = self.performance_history[self.mode][-3:]
        if len(recent) < 3:
            return 'improving'
        improvements = [b - a for a, b in zip(recent, recent[1:])]
        avg_improvement = sum(improvements) / len(improvements)
        return 'stagnating' if avg_improvement < 0.001 else 'improving'

    def _switch_optimization_mode(self) -> None:
        """Switch optimization mode based on performance stagnation."""
        modes = list(self.hyperparameters.keys())
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        logger.info("Switched to %s optimization mode", self.mode)

    def _adjust_risk_parameters(
            self,
            volatility: float,
            sentiment: float) -> None:
        """Adjust risk-related parameters based on market conditions."""
        if self.mode in ['market', 'trading']:
            hp = self.hyperparameters[self.mode]
            if volatility > 0.8:
                if hasattr(hp, 'position_size'):
                    hp.position_size *= 0.8
                if hasattr(hp, 'stop_loss'):
                    hp.stop_loss *= 0.7
            elif volatility < 0.2:
                if hasattr(hp, 'position_size'):
                    hp.position_size *= 1.2
                if hasattr(hp, 'stop_loss'):
                    hp.stop_loss *= 1.3
            if abs(sentiment) > 0.7:
                if hasattr(hp, 'sentiment_weight'):
                    hp.sentiment_weight *= 1.2
                if hasattr(hp, 'technical_weight'):
                    hp.technical_weight *= 0.8

    def _validate_parameters(self) -> None:
        """Validate and enforce parameter bounds."""
        hp_dict = self.hyperparameters[self.mode].to_dict()
        for key, value in hp_dict.items():
            if key in self.parameter_bounds:
                low, high = self.parameter_bounds[key]
                new_value = np.clip(value, low, high)
                if isinstance(low, int):
                    new_value = int(new_value)
                setattr(self.hyperparameters[self.mode], key, new_value)

    def get_hyperparameters(self) -> Dict:
        """Retrieve current hyperparameters as dictionary."""
        return self.hyperparameters[self.mode].to_dict()

    def reset_hyperparameters(self) -> Dict:
        """Reset hyperparameters to defaults."""
        self.hyperparameters[self.mode] = {
            'numeric': GP_Hyperparameters(),
            'symbolic': GP_Hyperparameters(),
            'market': NN_Hyperparameters(),
            'trading': NEAT_Hyperparameters()
        }[self.mode]
        self.last_selected_strategies[self.mode] = {}
        logger.info("Reset %s hyperparameters to defaults", self.mode)
        return self.get_current_parameters()

    def get_performance_history(self) -> List[float]:
        """Retrieve performance history for current mode."""
        return self.performance_history[self.mode]

    def get_current_parameters(self) -> Dict:
        """Get current parameters with validation."""
        self._validate_parameters()
        return self.hyperparameters[self.mode].to_dict()

    def integrate_real_time(self, rt_config: Dict) -> None:
        """Integrate with RealTimeDataLoader for market/trading modes."""
        self.rt_loader = RealTimeDataLoader(rt_config)
        self.rt_loader.register_callback('market', self._real_time_update)
        logger.info("Integrated real-time data loader for MetaLearner")

    async def _real_time_update(self, data: float) -> None:
        """Update hyperparameters based on real-time market data."""
        market_state = {'volatility': abs(data - 0.5), 'sentiment': data - 0.5}
        # Placeholder; replace with real metric
        performance = np.random.uniform(0, 1)
        self.update_hyperparameters(performance, market_state)
        logger.info(
            "Real-time update with data %.2f, new params: %s",
            data,
            self.get_current_parameters())

    def save_state(self, filename: str) -> bool:
        """Save MetaLearner state."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            state = {
                'hyperparameters': {
                    k: v.to_dict() for k,
                    v in self.hyperparameters.items()},
                'performance_history': self.performance_history,
                'mode': self.mode,
                'strategy_scores': self.strategy_scores,
                'last_selected_strategies': self.last_selected_strategies}
            with open(filepath, 'wb') as f:
                pickle.dump(state, f)
            logger.info("MetaLearner state saved to %s", filepath)
            return True
        except Exception as e:
            logger.error("Error saving MetaLearner state: %s", e)
            return False

    def load_state(self, filename: str) -> bool:
        """Load MetaLearner state."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            for mode, hp_dict in state['hyperparameters'].items():
                hp_class = {
                    'numeric': GP_Hyperparameters,
                    'symbolic': GP_Hyperparameters,
                    'market': NN_Hyperparameters,
                    'trading': NEAT_Hyperparameters}[mode]()
                for k, v in hp_dict.items():
                    setattr(hp_class, k, v)
                self.hyperparameters[mode] = hp_class
            self.performance_history = state['performance_history']
            self.mode = state['mode']
            self.strategy_scores = state['strategy_scores']
            self.last_selected_strategies = state['last_selected_strategies']
            logger.info("MetaLearner state loaded from %s", filepath)
            return True
        except Exception as e:
            logger.error("Error loading MetaLearner state: %s", e)
            return False


def main():
    """Demonstrate MetaLearner functionality."""
    learner = MetaLearner()

    # Initial state
    logger.info(
        "Initial %s Hyperparameters: %s",
        learner.mode,
        learner.get_hyperparameters())

    # Simulate updates
    for perf in [0.45, 0.6, 0.3, 0.7]:
        market_state = {'volatility': random.uniform(
            0, 1), 'sentiment': random.uniform(-1, 1)}
        params = learner.update_hyperparameters(perf, market_state)
        logger.info(
            "Updated %s params after perf %.2f: %s",
            learner.mode,
            perf,
            params)

    # Real-time integration (simulated)
    rt_config = {'market_feed': 'market_url'}
    learner.integrate_real_time(rt_config)
    import asyncio
    asyncio.run(asyncio.sleep(2))  # Simulate real-time run

    # Save and load
    learner.save_state("metalearner_state.pkl")
    new_learner = MetaLearner()
    new_learner.load_state("metalearner_state.pkl")
    logger.info("Loaded params: %s", new_learner.get_current_parameters())

    logger.info("MetaLearner demo completed.")


if __name__ == "__main__":
    main()
