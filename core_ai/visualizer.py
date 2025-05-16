# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import neat
import logging
import random
import os
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from config import PROJECT_CONFIG  # For directory structure and settings

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, RAW_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, temperature: float = 1.0):
        """
        Initialize the Visualizer with multiple strategies and scores for self-modification.

        Args:
            temperature: Controls exploration vs exploitation in strategy selection.
        """
        self.strategies = {
            'function_approximation': [
                self._plot_strategy_1,
                self._plot_strategy_2,
                self._plot_strategy_3
            ],
            'evolution_stats': [
                self._plot_evolution_strategy_1,
                self._plot_evolution_strategy_2,
                self._plot_evolution_strategy_3
            ],
            'multi_model_approximation': [
                self._plot_multi_strategy_1,
                self._plot_multi_strategy_2,
                self._plot_multi_strategy_3
            ],
            'architecture_evolution': [
                self._plot_architecture_strategy_1,
                self._plot_architecture_strategy_2,
                self._plot_architecture_strategy_3
            ]
        }
        self.scores = {key: [0] * len(strats)
                       for key, strats in self.strategies.items()}
        self.temperature = temperature
        self.plot_history = []  # Track plots for self-modification feedback

    def _select_strategy(self, strategy_type: str) -> Callable:
        """Select a plotting strategy based on softmax probabilities."""
        scores = self.scores[strategy_type]
        exp_scores = [np.exp(score / self.temperature) for score in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        strategy = np.random.choice(self.strategies[strategy_type], p=probs)
        logger.info(f"Selected {strategy_type} strategy with probs: {probs}")
        return strategy

    def _update_scores(
            self,
            strategy_type: str,
            strategy_idx: int,
            reward: float) -> None:
        """Update the score of the selected strategy."""
        self.scores[strategy_type][strategy_idx] += reward
        logger.info(
            f"Updated {strategy_type} strategy {strategy_idx} score by {reward}")

    def plot_function_approximation(
            self,
            model: Any,
            points: np.ndarray,
            targets: np.ndarray,
            filename: str = "approximation.png") -> None:
        """Plot the model's predictions vs. actual function."""
        try:
            filename = os.path.join(OUTPUT_DIR, filename)
            strategy = self._select_strategy('function_approximation')
            strategy(model, points, targets, filename)
            reward = self._evaluate_plot_quality(
                filename)  # Simulated feedback
            strategy_idx = self.strategies['function_approximation'].index(
                strategy)
            self._update_scores('function_approximation', strategy_idx, reward)
            self.plot_history.append(
                {'type': 'function_approximation', 'filename': filename})
        except Exception as e:
            logger.error(f"Error plotting function approximation: {e}")
            raise

    def _plot_strategy_1(
            self,
            model: Any,
            points: np.ndarray,
            targets: np.ndarray,
            filename: str) -> None:
        """Standard plot with blue and red lines."""
        predictions = [model.activate([x])[0] for x in points]
        plt.figure(figsize=(8, 6))
        plt.plot(
            points,
            targets,
            label="True Function",
            color="blue",
            linewidth=2)
        plt.plot(
            points,
            predictions,
            label="Approximation",
            color="red",
            linestyle="--")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Function Approximation")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved function approximation plot to {filename} using strategy 1")

    def _plot_strategy_2(
            self,
            model: Any,
            points: np.ndarray,
            targets: np.ndarray,
            filename: str) -> None:
        """Alternative plot with green lines and markers."""
        predictions = [model.activate([x])[0] for x in points]
        plt.figure(figsize=(8, 6))
        plt.plot(
            points,
            targets,
            label="True Function",
            color="green",
            marker="o")
        plt.plot(
            points,
            predictions,
            label="Approximation",
            color="orange",
            linestyle=":",
            marker="x")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Function Approximation (Strategy 2)")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved function approximation plot to {filename} using strategy 2")

    def _plot_strategy_3(
            self,
            model: Any,
            points: np.ndarray,
            targets: np.ndarray,
            filename: str) -> None:
        """Scatter plot with error shading."""
        predictions = [model.activate([x])[0] for x in points]
        errors = np.abs(predictions - targets)
        plt.figure(figsize=(8, 6))
        plt.scatter(points, targets, label="True Function", color="purple")
        plt.scatter(points, predictions, label="Approximation", color="yellow")
        plt.fill_between(
            points,
            predictions -
            errors,
            predictions +
            errors,
            color="gray",
            alpha=0.3)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Function Approximation with Error (Strategy 3)")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved function approximation plot to {filename} using strategy 3")

    def plot_evolution_stats(
            self,
            stats: neat.StatisticsReporter,
            filename: str = "evolution_stats.png") -> None:
        """Plot evolution statistics."""
        try:
            filename = os.path.join(OUTPUT_DIR, filename)
            strategy = self._select_strategy('evolution_stats')
            strategy(stats, filename)
            reward = self._evaluate_plot_quality(filename)
            strategy_idx = self.strategies['evolution_stats'].index(strategy)
            self._update_scores('evolution_stats', strategy_idx, reward)
            self.plot_history.append(
                {'type': 'evolution_stats', 'filename': filename})
        except Exception as e:
            logger.error(f"Error plotting evolution stats: {e}")
            raise

    def _plot_evolution_strategy_1(
            self,
            stats: neat.StatisticsReporter,
            filename: str) -> None:
        """Standard evolution stats plot."""
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = stats.get_fitness_mean()
        plt.figure(figsize=(10, 6))
        plt.plot(generation, best_fitness, label="Best Fitness", marker="o")
        plt.plot(generation, avg_fitness, label="Average Fitness", marker="x")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution Statistics")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved evolution stats plot to {filename} using strategy 1")

    def _plot_evolution_strategy_2(
            self,
            stats: neat.StatisticsReporter,
            filename: str) -> None:
        """Evolution plot with median fitness."""
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = stats.get_fitness_mean()
        median_fitness = stats.get_fitness_median()
        plt.figure(figsize=(10, 6))
        plt.plot(generation, best_fitness, label="Best Fitness", marker="o")
        plt.plot(generation, avg_fitness, label="Average Fitness", marker="x")
        plt.plot(
            generation,
            median_fitness,
            label="Median Fitness",
            marker="^",
            linestyle="--")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution Statistics with Median")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved evolution stats plot to {filename} using strategy 2")

    def _plot_evolution_strategy_3(
            self,
            stats: neat.StatisticsReporter,
            filename: str) -> None:
        """Evolution plot with standard deviation."""
        generation = range(len(stats.most_fit_genomes))
        best_fitness = [c.fitness for c in stats.most_fit_genomes]
        avg_fitness = stats.get_fitness_mean()
        std_fitness = stats.get_fitness_stdev()
        plt.figure(figsize=(10, 6))
        plt.plot(generation, best_fitness, label="Best Fitness", marker="o")
        plt.plot(generation, avg_fitness, label="Average Fitness")
        plt.fill_between(
            generation,
            np.array(avg_fitness) -
            std_fitness,
            np.array(avg_fitness) +
            std_fitness,
            color="blue",
            alpha=0.2)
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Evolution Statistics with Std Dev (Strategy 3)")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved evolution stats plot to {filename} using strategy 3")

    def plot_multi_model_approximation(
            self,
            models: List[Any],
            points: np.ndarray,
            targets: np.ndarray,
            labels: List[str],
            filename: str = "multi_approximation.png") -> None:
        """Plot multi-model approximations."""
        try:
            filename = os.path.join(OUTPUT_DIR, filename)
            strategy = self._select_strategy('multi_model_approximation')
            strategy(models, points, targets, labels, filename)
            reward = self._evaluate_plot_quality(filename)
            strategy_idx = self.strategies['multi_model_approximation'].index(
                strategy)
            self._update_scores(
                'multi_model_approximation', 
                strategy_idx,
                reward)
            self.plot_history.append(
                {'type': 'multi_model_approximation', 'filename': filename})
        except Exception as e:
            logger.error(f"Error plotting multi-model approximation: {e}")
            raise

    def _plot_multi_strategy_1(
            self,
            models: List[Any],
            points: np.ndarray,
            targets: np.ndarray,
            labels: List[str],
            filename: str) -> None:
        """Standard multi-model plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(
            points,
            targets,
            label="True Function",
            linewidth=2,
            color="black")
        for model, label in zip(models, labels):
            predictions = [model.activate([x])[0] for x in points]
            plt.plot(points, predictions, linestyle="--", label=label)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Multi-Model Function Approximation")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved multi-model approximation plot to {filename} using strategy 1")

    def _plot_multi_strategy_2(
            self,
            models: List[Any],
            points: np.ndarray,
            targets: np.ndarray,
            labels: List[str],
            filename: str) -> None:
        """Multi-model plot with subplots."""
        fig, axes = plt.subplots(len(models), 1, figsize=(8, 6 * len(models)))
        if len(models) == 1:
            axes = [axes]
        for ax, model, label in zip(axes, models, labels):
            predictions = [model.activate([x])[0] for x in points]
            ax.plot(points, targets, label="True Function", color="black")
            ax.plot(points, predictions, label=label, linestyle="--")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.set_title(f"Approximation: {label}")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved multi-model approximation plot to {filename} using strategy 2")

    def _plot_multi_strategy_3(
            self,
            models: List[Any],
            points: np.ndarray,
            targets: np.ndarray,
            labels: List[str],
            filename: str) -> None:
        """Multi-model plot with error bars."""
        plt.figure(figsize=(10, 6))
        plt.plot(points, targets, label="True Function", color="black")
        for model, label in zip(models, labels):
            predictions = np.array([model.activate([x])[0] for x in points])
            errors = np.abs(predictions - targets)
            plt.errorbar(
                points,
                predictions,
                yerr=errors,
                label=label,
                linestyle="--",
                capsize=3)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Multi-Model Approximation with Errors (Strategy 3)")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved multi-model approximation plot to {filename} using strategy 3")

    def plot_architecture_evolution(
            self,
            model: Any,
            filename: str = "architecture_evolution.png") -> None:
        """Plot architecture evolution."""
        try:
            filename = os.path.join(OUTPUT_DIR, filename)
            strategy = self._select_strategy('architecture_evolution')
            strategy(model, filename)
            reward = self._evaluate_plot_quality(filename)
            strategy_idx = self.strategies['architecture_evolution'].index(
                strategy)
            self._update_scores('architecture_evolution', strategy_idx, reward)
            self.plot_history.append(
                {'type': 'architecture_evolution', 'filename': filename})
        except Exception as e:
            logger.error(f"Error plotting architecture evolution: {e}")
            raise

    def _plot_architecture_strategy_1(self, model: Any, filename: str) -> None:
        """Standard architecture evolution plot."""
        changes = model.architecture_stats.get('architecture_changes', [])
        generations = [c['generation'] for c in changes]
        node_counts = np.cumsum(
            [1 if c['type'] == 'node_addition' else 0 for c in changes])
        conn_counts = np.cumsum(
            [1 if c['type'] == 'connection_modification' else 0 for c in changes])
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(generations, node_counts, 'b-', label='Nodes Added')
        plt.ylabel('Cumulative Node Count')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(generations, conn_counts, 'r-', label='Connections Modified')
        plt.xlabel('Generation')
        plt.ylabel('Cumulative Connection Changes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved architecture evolution plot to {filename} using strategy 1")

    def _plot_architecture_strategy_2(self, model: Any, filename: str) -> None:
        """Architecture plot with fitness trend."""
        changes = model.architecture_stats.get('architecture_changes', [])
        generations = [c['generation'] for c in changes]
        fitness = [c.get('fitness', 0) for c in changes]
        plt.figure(figsize=(10, 4))
        plt.plot(generations, fitness, 'g-', label='Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved architecture evolution plot to {filename} using strategy 2")

    def _plot_architecture_strategy_3(self, model: Any, filename: str) -> None:
        """Architecture plot with node and connection bar chart."""
        changes = model.architecture_stats.get('architecture_changes', [])
        node_adds = sum(1 for c in changes if c['type'] == 'node_addition')
        conn_mods = sum(
            1 for c in changes if c['type'] == 'connection_modification')
        plt.figure(figsize=(8, 6))
        plt.bar(['Nodes Added', 'Connections Modified'], [
                node_adds, conn_mods], color=['blue', 'red'])
        plt.ylabel('Count')
        plt.title('Architecture Changes (Strategy 3)')
        plt.grid(axis='y')
        plt.savefig(filename)
        plt.close()
        logger.info(
            f"Saved architecture evolution plot to {filename} using strategy 3")

    def _evaluate_plot_quality(self, filename: str) -> float:
        """Simulate plot quality evaluation (e.g., based on file size or user feedback)."""
        try:
            size = os.path.getsize(filename) / 1024  # Size in KB
            reward = min(size / 100, 1.0) if size > 0 else - \
                1.0  # Simple heuristic
            logger.info(
                f"Evaluated plot {filename} quality with reward {
                    reward:.2f}")
            return reward
        except Exception as e:
            logger.error(f"Error evaluating plot quality for {filename}: {e}")
            return -1.0


def draw_net(
        config: neat.Config,
        genome: neat.DefaultGenome,
        filename: str = "network.png") -> None:
    """Draw a simple representation of the neural network."""
    try:
        filename = os.path.join(OUTPUT_DIR, filename)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        plt.figure(figsize=(8, 6))
        plt.text(0.5,
                 0.5,
                 f"Nodes: {len(net.node_evals)}\nConnections: {len(net.connections)}",
                 ha="center",
                 va="center",
                 fontsize=12)
        plt.title("Network Structure (Simplified)")
        plt.axis("off")
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved network structure plot to {filename}")
    except Exception as e:
        logger.error(f"Error drawing network: {e}")
        raise


def plot_hyperparameter_evolution(
        meta_learner: Any,
        filename: str = "hyperparameter_evolution.png") -> None:
    """Plot the evolution of hyperparameters."""
    try:
        filename = os.path.join(OUTPUT_DIR, filename)
        history = getattr(meta_learner, 'performance_history', [])
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(history)), history, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Performance")
        plt.title("Hyperparameter Evolution")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(f"Saved hyperparameter evolution plot to {filename}")
    except Exception as e:
        logger.error(f"Error plotting hyperparameter evolution: {e}")
        raise


def dangerous_plot_data(
        data: np.ndarray,
        filename: str = "dangerous_data.png") -> None:
    """Plot data with risky amplification (dangerous AI theme)."""
    try:
        filename = os.path.join(OUTPUT_DIR, filename)
        amplified = data * random.uniform(10, 100)  # Risky amplification
        plt.figure(figsize=(8, 6))
        plt.plot(amplified, label="Amplified Data", color="red")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Dangerous Data Visualization")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.warning(
            f"Saved dangerous data plot to {filename} with amplification")
    except Exception as e:
        logger.error(f"Error in dangerous data plot: {e}")
        raise


def main():
    """Demonstrate visualization functionality."""
    visualizer = Visualizer()

    class DummyModel:
        def __init__(self, factor): self.factor = factor
        def activate(self, inputs): return [self.factor * np.sin(inputs[0])]

    model1, model2 = DummyModel(1.0), DummyModel(0.8)
    points = np.linspace(0, 2 * np.pi, 100)
    targets = np.sin(points)

    visualizer.plot_function_approximation(
        model1, points, targets, "approximation_model1.png")

    models = [model1, model2]
    labels = ["Model 1 (factor=1.0)", "Model 2 (factor=0.8)"]
    visualizer.plot_multi_model_approximation(
        models, points, targets, labels, "multi_approximation.png")

    class DummyStats:
        def __init__(self):
            self.most_fit_genomes = [
                type(
                    "Genome", (), {
                        "fitness": np.random.uniform(
                            0, 1)})() for _ in range(10)]

        def get_fitness_mean(self): return [
            np.random.uniform(
                0.2, 0.8) for _ in range(10)]

        def get_fitness_median(self): return [
            np.random.uniform(
                0.3, 0.7) for _ in range(10)]

        def get_fitness_stdev(self): return [
            np.random.uniform(
                0.1, 0.2) for _ in range(10)]

    dummy_stats = DummyStats()
    visualizer.plot_evolution_stats(dummy_stats, "evolution_stats.png")

    class DummyModelWithStats:
        def __init__(self):
            self.architecture_stats = {
                'architecture_changes': [
                    {'generation': i, 'type': random.choice(['node_addition', 'connection_modification']),
                     'fitness': np.random.uniform(0, 1)} for i in range(10)
                ]
            }

    dummy_model = DummyModelWithStats()
    visualizer.plot_architecture_evolution(
        dummy_model, "architecture_evolution.png")

    # Dangerous mode demo
    dangerous_data = np.random.rand(100)
    dangerous_plot_data(dangerous_data)

    logger.info("Visualizer module test completed")


if __name__ == "__main__":
    main()

# Additional utilities


def plot_raw_data(filename: str = "raw_data.png") -> None:
    """Plot data from raw CSV files."""
    try:
        filename = os.path.join(OUTPUT_DIR, filename)
        for fname in os.listdir(RAW_DIR):
            if fname.endswith('.csv'):
                df = pd.read_csv(os.path.join(RAW_DIR, fname))
                if 'x' in df and 'y' in df:
                    plt.figure(figsize=(8, 6))
                    plt.plot(df['x'], df['y'], label=fname)
                    plt.xlabel("X")
                    plt.ylabel("Y")
                    plt.title(f"Raw Data: {fname}")
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(filename)
                    plt.close()
                    logger.info(f"Saved raw data plot to {filename}")
                    break  # Plot only first valid file for simplicity
    except Exception as e:
        logger.error(f"Error plotting raw data: {e}")
        raise


def batch_plot(
        models: List[Any],
        points: np.ndarray,
        targets: np.ndarray,
        labels: List[str],
        prefix: str = "batch_plot_") -> None:
    """Batch plot multiple models."""
    visualizer = Visualizer()
    for i, (model, label) in enumerate(zip(models, labels)):
        visualizer.plot_function_approximation(
            model, points, targets, f"{prefix}{i}.png")
