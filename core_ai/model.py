# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/model.py
import json
import math
import os
import logging
import random
import numpy as np
import pandas as pd
import neat.nn as neat_nn
import neat
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

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


class AIModel:
    def __init__(self, model_name: str = "default_model"):
        self.model_name = model_name
        self.parameters: Dict[str, Any] = {}
        self.version = "1.0.0"

    def set_parameters(self, **kwargs):
        """Update model parameters."""
        self.parameters.update(kwargs)
        logger.info(f"Parameters updated for {self.model_name}: {kwargs}")

    def get_parameters(self) -> Dict[str, Any]:
        """Retrieve current parameters."""
        return self.parameters

    def save(self, filename: str):
        """Save parameters to a JSON file."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump({"parameters": self.parameters,
                          "version": self.version}, f, indent=4)
            logger.info(f"Parameters saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving parameters to {filepath}: {e}")
            raise

    def load(self, filename: str):
        """Load parameters from a JSON file."""
        filepath = os.path.join(MODELS_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.parameters = data["parameters"]
                self.version = data.get("version", "1.0.0")
            logger.info(
                f"Parameters loaded from {filepath}, version {
                    self.version}")
        except Exception as e:
            logger.error(f"Error loading parameters from {filepath}: {e}")
            raise

    def generate_code(self, prompt: str) -> str:
        """Generate simple Python code based on a prompt."""
        return f"# Generated for {
            self.model_name} on {
            datetime.now()}\n# Prompt: {prompt}\ndef run():\n    print('Hello, Mathemagician!')\n"


class TechnicalFeatureExtractor:
    def __init__(self):
        self.indicators = {
            'sma': self._simple_moving_average,
            'momentum': self._momentum,
            'volatility': self._volatility,
            'rsi': self._relative_strength_index
        }

    def extract_features(self,
                         data: Union[float,
                                     List[float],
                                     np.ndarray]) -> List[float]:
        """Extract technical features from data."""
        return [indicator(data) for indicator in self.indicators.values()]

    def _simple_moving_average(self, data, window: int = 5) -> float:
        if isinstance(data, (int, float)):
            return float(data)
        data = np.asarray(data)[-window:]
        return float(np.mean(data)) if len(data) > 0 else 0.0

    def _momentum(self, data) -> float:
        if isinstance(data, (int, float)):
            return 0.0
        data = np.asarray(data)
        return float(data[-1] - data[0]) if len(data) > 1 else 0.0

    def _volatility(self, data) -> float:
        if isinstance(data, (int, float)):
            return 0.0
        data = np.asarray(data)
        return float(np.std(data)) if len(data) > 1 else 0.0

    def _relative_strength_index(self, data, period: int = 14) -> float:
        if isinstance(data, (int, float)) or len(data) < period:
            return 0.0
        data = np.asarray(data)[-period:]
        delta = np.diff(data)
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0
        rs = gain / loss if loss != 0 else float('inf')
        return 100 - (100 / (1 + rs)) if rs != float('inf') else 100.0


class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_cache = {}

    def extract_features(self, data: Any) -> List[float]:
        """Extract sentiment features from data."""
        if isinstance(data, (int, float)):
            return [float(data)]
        sentiment_score = self._analyze_sentiment(str(data))
        return [sentiment_score]

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder)."""
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        # Dummy: positive if more vowels, negative if more consonants
        vowels = sum(text.lower().count(v) for v in 'aeiou')
        consonants = len(text) - vowels
        score = (vowels - consonants) / len(text) if len(text) > 0 else 0.0
        self.sentiment_cache[text] = score
        return score


class MarketFeatureExtractor:
    def __init__(self):
        self.market_metrics = {
            'trend': self._calculate_trend,
            'volume': self._calculate_volume,
            'liquidity': self._calculate_liquidity,
            'velocity': self._calculate_velocity
        }

    def extract_features(self,
                         data: Union[float,
                                     List[float],
                                     np.ndarray]) -> List[float]:
        """Extract market features from data."""
        return [metric(data) for metric in self.market_metrics.values()]

    def _calculate_trend(self, data) -> float:
        if isinstance(data, (int, float)):
            return float(data)
        data = np.asarray(data)
        return float(np.mean(np.diff(data))) if len(data) > 1 else 0.0

    def _calculate_volume(self, data) -> float:
        if isinstance(data, (int, float)):
            return abs(float(data))
        data = np.asarray(data)
        return float(np.sum(np.abs(data)))

    def _calculate_liquidity(self, data) -> float:
        if isinstance(data, (int, float)):
            return 1.0
        data = np.asarray(data)
        return float(len(set(data)) / len(data)) if len(data) > 0 else 0.0

    def _calculate_velocity(self, data) -> float:
        if isinstance(data, (int, float)):
            return 0.0
        data = np.asarray(data)
        return float(np.mean(np.abs(np.diff(data)))) if len(data) > 1 else 0.0


class RiskManager:
    def __init__(self, max_drawdown: float = 0.2,
                 volatility_threshold: float = 0.15):
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold
        self.position_history = []

    def adjust_prediction(self, prediction: float) -> float:
        """Adjust prediction based on risk factor."""
        risk_factor = self._calculate_risk_factor()
        adjusted = prediction * (1 - risk_factor)
        self.position_history.append(adjusted)
        return adjusted

    def calculate_risk_penalty(self, predictions: List[float]) -> float:
        """Calculate a risk penalty based on volatility and drawdown."""
        volatility = np.std(predictions) if predictions else 0.0
        drawdown = self._calculate_max_drawdown(predictions)
        penalty = (volatility / self.volatility_threshold +
                   drawdown / self.max_drawdown) / 2
        logger.info(
            f"Risk penalty: volatility={
                volatility:.4f}, drawdown={
                drawdown:.4f}, penalty={
                penalty:.4f}")
        return penalty

    def _calculate_risk_factor(self) -> float:
        """Compute risk factor from position history."""
        if not self.position_history:
            return 0.0
        volatility = np.std(self.position_history)
        return min(volatility / self.volatility_threshold, 1.0)

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown from a list of values."""
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak != 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd


class SelfModifyingGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.self_mutation_rate = 0.01
        self.self_node_add_prob = 0.03
        self.self_conn_add_prob = 0.05
        self.self_risk_weight = 1.0
        self.modification_log = []

    def configure_new(self, config):
        """Initialize with random self-modifying parameters."""
        super().configure_new(config)
        self.self_mutation_rate = random.uniform(0.005, 0.15)
        self.self_node_add_prob = random.uniform(0.01, 0.2)
        self.self_conn_add_prob = random.uniform(0.01, 0.2)
        self.self_risk_weight = random.uniform(0.5, 2.5)
        self.modification_log.append(
            {"event": "init", "timestamp": str(datetime.now())})

    def mutate(self, config):
        """Mutate with self-adaptive parameters."""
        # Dangerous AI twist: occasional aggressive mutation
        if random.random() < 0.05:  # 5% chance
            self.self_mutation_rate = min(self.self_mutation_rate * 2, 0.5)
            logger.warning(
                f"Dangerous mutation rate increase: {
                    self.self_mutation_rate}")

        # Mutate self-parameters
        if random.random() < 0.1:
            self.self_mutation_rate += random.gauss(0, 0.02)
            self.self_mutation_rate = max(
                0.001, min(self.self_mutation_rate, 0.5))
        if random.random() < 0.1:
            self.self_node_add_prob += random.gauss(0, 0.02)
            self.self_node_add_prob = max(
                0.01, min(self.self_node_add_prob, 0.5))
        if random.random() < 0.1:
            self.self_conn_add_prob += random.gauss(0, 0.02)
            self.self_conn_add_prob = max(
                0.01, min(self.self_conn_add_prob, 0.5))
        if random.random() < 0.1:
            self.self_risk_weight += random.gauss(0, 0.2)
            self.self_risk_weight = max(0.1, min(self.self_risk_weight, 5.0))

        # Apply mutations
        for cg in self.connections.values():
            if random.random() < self.self_mutation_rate:
                cg.weight += random.gauss(0, config.weight_mutate_power)

        if random.random() < self.self_node_add_prob:
            self.mutate_add_node(config)
            self.modification_log.append(
                {"event": "node_added", "timestamp": str(datetime.now())})
        if random.random() < self.self_conn_add_prob:
            self.mutate_add_connection(config)
            self.modification_log.append(
                {"event": "conn_added", "timestamp": str(datetime.now())})

    def configure_crossover(self, genome1, genome2, config):
        """Crossover with averaging of self-parameters."""
        super().configure_crossover(genome1, genome2, config)
        self.self_mutation_rate = (
            genome1.self_mutation_rate + genome2.self_mutation_rate) / 2
        self.self_node_add_prob = (
            genome1.self_node_add_prob + genome2.self_node_add_prob) / 2
        self.self_conn_add_prob = (
            genome1.self_conn_add_prob + genome2.self_conn_add_prob) / 2
        self.self_risk_weight = (
            genome1.self_risk_weight + genome2.self_risk_weight) / 2
        self.modification_log.append(
            {"event": "crossover", "timestamp": str(datetime.now())})


class MathemagicianModel(AIModel):
    def __init__(
            self,
            config_path: str,
            model_name: str = "MathemagicianModel"):
        super().__init__(model_name)
        try:
            self.config = neat.Config(
                SelfModifyingGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                os.path.join(
                    BASE_DIR,
                    config_path))
            self.points = np.linspace(0, 2 * math.pi, 100)
            self.targets = [math.sin(x) for x in self.points]
            self.generation = 0
            self.best_fitness = 0.0
            self.modification_history = []
            self.adaptation_thresholds = {
                'mutation_rate': {'low': 0.05, 'high': 0.6},
                'node_add_prob': {'low': 0.05, 'high': 0.5},
                'connection_add_prob': {'low': 0.05, 'high': 0.4}
            }
            self.supported_functions = {
                'sin': np.sin,
                'cos': np.cos,
                'exp': np.exp,
                'polynomial': lambda x,
                coeffs=[
                    1,
                    0,
                    1]: sum(
                    c * x**i for i,
                    c in enumerate(coeffs))}
            self.current_function = 'sin'
            self.feature_extractors = {
                'technical': TechnicalFeatureExtractor(),
                'sentiment': SentimentAnalyzer(),
                'market': MarketFeatureExtractor()
            }
            self.risk_manager = RiskManager(
                max_drawdown=0.2, volatility_threshold=0.15)
            self.data_sources = {}
            self.risk_metrics = {
                'volatility': 0.0,
                'drawdown': 0.0,
                'sharpe_ratio': 0.0}
            self.current_sentiment = 0.0
            self.current_odds = 0.0
            logger.info(
                f"{model_name} initialized with config from {config_path}")
        except Exception as e:
            logger.error(f"Error initializing {model_name}: {e}")
            raise

    def create_network(
            self,
            genome: SelfModifyingGenome) -> neat_nn.FeedForwardNetwork:
        """Create a NEAT network from a genome."""
        return neat_nn.FeedForwardNetwork.create(genome, self.config)

    def evaluate_genome(self, genome: SelfModifyingGenome) -> float:
        """Evaluate a genome’s fitness with multimodal data."""
        try:
            net = self.create_network(genome)
            predictions = []
            for point in self.points:
                features = self.process_multimodal_data({
                    'market': point,
                    'sentiment': self.current_sentiment,
                    'betting': self.current_odds
                })
                pred = net.activate(self._combine_features(features))[0]
                adjusted_pred = self.risk_manager.adjust_prediction(pred)
                predictions.append(adjusted_pred)

            mse = np.mean((np.array(predictions) - self.targets) ** 2)
            risk_penalty = self.risk_manager.calculate_risk_penalty(
                predictions)
            fitness = 1 / (1 + mse + genome.self_risk_weight * risk_penalty)
            logger.info(
                f"Genome fitness: {
                    fitness:.4f}, MSE={
                    mse:.4f}, Risk Penalty={
                    risk_penalty:.4f}")
            return fitness
        except Exception as e:
            logger.error(f"Error evaluating genome: {e}")
            return 0.0

    def set_target_function(self, func_name: str, **params):
        """Set the target function for evaluation."""
        if func_name not in self.supported_functions:
            raise ValueError(f"Unsupported function: {func_name}")
        self.current_function = func_name
        self.targets = [self.supported_functions[func_name](x, **params)
                        if params else self.supported_functions[func_name](x)
                        for x in self.points]
        logger.info(f"Target function set to {func_name} with params {params}")

    def modify_architecture(self, performance_metric: float) -> neat.Config:
        """Modify architecture based on performance."""
        if performance_metric > self.best_fitness:
            self.best_fitness = performance_metric
            if len(self.modification_history) > 2:
                improvement_rate = self._calculate_improvement_rate()
                self._adjust_parameters(improvement_rate)

            modification = {
                'generation': self.generation,
                'fitness': performance_metric,
                'parameters': self._get_current_parameters(),
                'timestamp': str(datetime.now())
            }
            self.modification_history.append(modification)
            logger.info(
                f"Architecture modified at generation {
                    self.generation}")

        self.generation += 1
        return self.config

    def _calculate_improvement_rate(self) -> float:
        """Calculate the rate of fitness improvement."""
        recent = self.modification_history[-3:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1]['fitness'] - recent[0]['fitness']) / len(recent)

    def _adjust_parameters(self, improvement_rate: float):
        """Adjust NEAT parameters based on improvement."""
        if improvement_rate < 0.01:
            self.config.genome_config.node_add_prob = min(
                self.config.genome_config.node_add_prob * 1.3, 0.5)
            self.config.genome_config.conn_add_prob = min(
                self.config.genome_config.conn_add_prob * 1.3, 0.5)
        elif improvement_rate > 0.1:
            self.config.genome_config.node_add_prob = max(
                self.config.genome_config.node_add_prob * 0.8, 0.05)
            self.config.genome_config.conn_add_prob = max(
                self.config.genome_config.conn_add_prob * 0.8, 0.05)
        logger.info(
            f"Adjusted parameters: node_add={
                self.config.genome_config.node_add_prob}, " f"conn_add={
                self.config.genome_config.conn_add_prob}")

    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current NEAT parameters."""
        return {
            'node_add_prob': self.config.genome_config.node_add_prob,
            'conn_add_prob': self.config.genome_config.conn_add_prob,
            'weight_mutate_rate': self.config.genome_config.weight_mutate_rate
        }

    def process_multimodal_data(
            self, data_dict: Dict[str, Any]) -> Dict[str, List[float]]:
        """Process multimodal data with feature extractors."""
        features = {}
        for source, extractor in self.feature_extractors.items():
            if source in data_dict and data_dict[source] is not None:
                features[source] = extractor.extract_features(
                    data_dict[source])
        return features

    def _combine_features(self,
                          features_dict: Dict[str,
                                              List[float]]) -> List[float]:
        """Combine features into a single input vector."""
        combined = []
        for source in ['market', 'sentiment', 'technical']:
            if source in features_dict:
                combined.extend(features_dict[source])
        return combined if combined else [0.0]


def main():
    """Demonstrate model functionality."""
    model = MathemagicianModel('config.txt')
    pop = neat.Population(model.config)
    genome = list(pop.population.values())[0]
    fitness = model.evaluate_genome(genome)
    print(f"Fitness of random genome: {fitness:.4f}")

    # Test modification
    model.modify_architecture(fitness)
    model.save("test_model.json")
    model.load("test_model.json")


if __name__ == "__main__":
    main()
