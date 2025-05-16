# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Evolutionary Optimizer Module

This module provides evolutionary optimization algorithms for AI model training and tuning.
It implements various evolutionary algorithms and strategies for optimizing model parameters.
"""

import os
import random
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
import multiprocessing
import pickle
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(SRC_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class EvolutionaryOptimizer:
    """
    Evolutionary optimization algorithms for AI model training.
    Uses a population-based approach with selection, crossover, and mutation.
    """

    def __init__(self,
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5,
                 max_generations: int = 100,
                 parallel: bool = False,
                 selection_method: str = "tournament",
                 crossover_method: str = "uniform",
                 mutation_method: str = "gaussian"):
        """
        Initialize the evolutionary optimizer.

        Args:
            population_size: Size of the population
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover between individuals
            elite_size: Number of top individuals to preserve unchanged
            max_generations: Maximum number of generations to run
            parallel: Whether to use parallel processing
            selection_method: Method for selecting parents ("tournament", "roulette", "rank")
            crossover_method: Method for crossover ("uniform", "onepoint", "twopoint")
            mutation_method: Method for mutation ("gaussian", "uniform", "adaptive")
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.parallel = parallel

        # Set selection method
        self.selection_method = selection_method.lower()
        if self.selection_method not in ["tournament", "roulette", "rank"]:
            logger.warning(f"Unknown selection method: {selection_method}. Using tournament selection.")
            self.selection_method = "tournament"

        # Set crossover method
        self.crossover_method = crossover_method.lower()
        if self.crossover_method not in ["uniform", "onepoint", "twopoint"]:
            logger.warning(f"Unknown crossover method: {crossover_method}. Using uniform crossover.")
            self.crossover_method = "uniform"

        # Set mutation method
        self.mutation_method = mutation_method.lower()
        if self.mutation_method not in ["gaussian", "uniform", "adaptive"]:
            logger.warning(f"Unknown mutation method: {mutation_method}. Using gaussian mutation.")
            self.mutation_method = "gaussian"

        # Runtime variables
        self.population = []
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0

        # For adaptive mutation
        self.adaptive_mutation_rate = self.mutation_rate

        logger.info(f"EvolutionaryOptimizer initialized with population size {population_size}, "
                   f"mutation rate {mutation_rate}, crossover rate {crossover_rate}")

    def initialize_population(self,
                             individual_size: int,
                             bounds: List[Tuple[float, float]],
                             initial_population: Optional[List[np.ndarray]] = None) -> None:
        """
        Initialize the population with random individuals.

        Args:
            individual_size: Size of each individual (number of parameters)
            bounds: List of (min, max) bounds for each parameter
            initial_population: Optional initial population instead of random generation
        """
        if initial_population is not None:
            self.population = initial_population[:self.population_size]
            remaining = self.population_size - len(initial_population)

            if remaining > 0:
                logger.info(f"Using {len(initial_population)} provided individuals and generating {remaining} random ones")
                # Generate remaining individuals
                for _ in range(remaining):
                    individual = np.array([random.uniform(low, high) for low, high in bounds])
                    self.population.append(individual)
        else:
            logger.info(f"Generating random population of {self.population_size} individuals")
            self.population = []

            for _ in range(self.population_size):
                individual = np.array([random.uniform(low, high) for low, high in bounds])
                self.population.append(individual)

        # Initialize histories
        self.fitness_history = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.generation = 0

    def evolve(self,
              fitness_function: Callable[[np.ndarray], float],
              max_generations: Optional[int] = None,
              bounds: Optional[List[Tuple[float, float]]] = None,
              early_stopping_generations: int = 0,
              early_stopping_threshold: float = 0.0) -> np.ndarray:
        """
        Evolve the population to find the best individual.

        Args:
            fitness_function: Function to evaluate fitness of an individual
            max_generations: Maximum number of generations to run, or None for self.max_generations
            bounds: Optional bounds for mutation (min, max) for each parameter
            early_stopping_generations: Stop if no improvement after this many generations (0 to disable)
            early_stopping_threshold: Minimum improvement to consider (relative to best fitness)

        Returns:
            The best individual found
        """
        if not self.population:
            raise ValueError("Population not initialized. Call initialize_population first.")

        max_gen = max_generations if max_generations is not None else self.max_generations
        logger.info(f"Starting evolution for {max_gen} generations")

        generations_without_improvement = 0

        # Main evolution loop
        for generation in range(max_gen):
            self.generation = generation + 1

            # Evaluate fitness of each individual
            if self.parallel:
                fitness_scores = self._parallel_evaluate_fitness(fitness_function)
            else:
                fitness_scores = [fitness_function(ind) for ind in self.population]

            # Record fitness statistics
            self.fitness_history.append(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.avg_fitness_history.append(avg_fitness)

            # Find best individual
            best_idx = fitness_scores.index(max(fitness_scores))
            current_best_fitness = fitness_scores[best_idx]
            current_best_individual = self.population[best_idx]

            self.best_fitness_history.append(current_best_fitness)

            # Update all-time best
            improvement = 0
            if current_best_fitness > self.best_fitness:
                improvement = (current_best_fitness - self.best_fitness) / max(abs(self.best_fitness), 1e-10)
                self.best_fitness = current_best_fitness
                self.best_individual = current_best_individual.copy()

                if improvement > early_stopping_threshold:
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
            else:
                generations_without_improvement += 1

            # Log progress
            logger.info(f"Generation {self.generation}/{max_gen}: "
                       f"Best fitness = {self.best_fitness:.6f}, "
                       f"Avg fitness = {avg_fitness:.6f}")

            # Check early stopping
            if early_stopping_generations > 0 and generations_without_improvement >= early_stopping_generations:
                logger.info(f"Early stopping after {generations_without_improvement} generations without improvement")
                break

            # Create next generation
            self._create_next_generation(fitness_scores, bounds)

            # Update adaptive mutation rate if using adaptive mutation
            if self.mutation_method == "adaptive":
                self._update_adaptive_mutation_rate(generation, max_gen)

        logger.info(f"Evolution completed after {self.generation} generations")
        logger.info(f"Best fitness: {self.best_fitness:.6f}")

        return self.best_individual

    def _parallel_evaluate_fitness(self, fitness_function: Callable[[np.ndarray], float]) -> List[float]:
        """Evaluate fitness in parallel."""
        with multiprocessing.Pool() as pool:
            fitness_scores = pool.map(fitness_function, self.population)
        return fitness_scores

    def _create_next_generation(self,
                              fitness_scores: List[float],
                              bounds: Optional[List[Tuple[float, float]]] = None) -> None:
        """Create the next generation through selection, crossover, and mutation."""
        # Handle elitism: keep the best individuals
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
        new_population = [self.population[i].copy() for i in elite_indices]

        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Selection
            if random.random() < self.crossover_rate:
                # Select two parents and perform crossover
                parent1_idx = self._selection(fitness_scores)
                parent2_idx = self._selection(fitness_scores)

                # Ensure we don't select the same parent twice
                while parent2_idx == parent1_idx:
                    parent2_idx = self._selection(fitness_scores)

                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]

                # Crossover
                child = self._crossover(parent1, parent2)
            else:
                # Just select one parent
                parent_idx = self._selection(fitness_scores)
                child = self.population[parent_idx].copy()

            # Mutation
            child = self._mutation(child, bounds)

            # Add to new population
            new_population.append(child)

        # Update population
        self.population = new_population

    def _selection(self, fitness_scores: List[float]) -> int:
        """
        Select an individual based on the configured selection method.

        Returns:
            Index of the selected individual
        """
        if self.selection_method == "tournament":
            return self._tournament_selection(fitness_scores)
        elif self.selection_method == "roulette":
            return self._roulette_selection(fitness_scores)
        elif self.selection_method == "rank":
            return self._rank_selection(fitness_scores)
        else:
            # Default to tournament
            return self._tournament_selection(fitness_scores)

    def _tournament_selection(self, fitness_scores: List[float], tournament_size: int = 3) -> int:
        """Tournament selection method."""
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(len(fitness_scores)), min(tournament_size, len(fitness_scores)))

        # Return the index of the best individual in the tournament
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        return tournament_indices[tournament_fitness.index(max(tournament_fitness))]

    def _roulette_selection(self, fitness_scores: List[float]) -> int:
        """Roulette wheel selection method."""
        # Handle negative fitness scores by shifting all values to be positive
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [score - min_fitness + 1e-10 for score in fitness_scores]
        else:
            adjusted_scores = [score + 1e-10 for score in fitness_scores]  # Avoid division by zero

        # Calculate selection probabilities
        total_fitness = sum(adjusted_scores)
        probabilities = [score / total_fitness for score in adjusted_scores]

        # Select based on probabilities
        return np.random.choice(len(fitness_scores), p=probabilities)

    def _rank_selection(self, fitness_scores: List[float]) -> int:
        """Rank-based selection method."""
        # Rank individuals by fitness (highest rank for highest fitness)
        ranks = [sorted(fitness_scores, reverse=True).index(score) + 1 for score in fitness_scores]

        # Calculate selection probabilities based on ranks
        total_rank = sum(ranks)
        probabilities = [(len(fitness_scores) - rank + 1) / total_rank for rank in ranks]

        # Select based on probabilities
        return np.random.choice(len(fitness_scores), p=probabilities)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents.

        Returns:
            The child created from the parents
        """
        if self.crossover_method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif self.crossover_method == "onepoint":
            return self._onepoint_crossover(parent1, parent2)
        elif self.crossover_method == "twopoint":
            return self._twopoint_crossover(parent1, parent2)
        else:
            # Default to uniform
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover method."""
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def _onepoint_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """One-point crossover method."""
        point = random.randint(1, len(parent1) - 1)
        return np.concatenate((parent1[:point], parent2[point:]))

    def _twopoint_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Two-point crossover method."""
        # Ensure points are in order and not at the extremes
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1) - 1)

        return np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))

    def _mutation(self,
                individual: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Perform mutation on an individual.

        Args:
            individual: The individual to mutate
            bounds: Optional bounds for mutation (min, max) for each parameter

        Returns:
            The mutated individual
        """
        mutation_rate = self.adaptive_mutation_rate if self.mutation_method == "adaptive" else self.mutation_rate

        # Make a copy to avoid modifying the original
        mutated = individual.copy()

        if self.mutation_method == "gaussian":
            return self._gaussian_mutation(mutated, mutation_rate, bounds)
        elif self.mutation_method == "uniform":
            return self._uniform_mutation(mutated, mutation_rate, bounds)
        elif self.mutation_method == "adaptive":
            return self._gaussian_mutation(mutated, mutation_rate, bounds)
        else:
            # Default to gaussian
            return self._gaussian_mutation(mutated, mutation_rate, bounds)

    def _gaussian_mutation(self,
                         individual: np.ndarray,
                         mutation_rate: float,
                         bounds: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Gaussian mutation method."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Calculate mutation magnitude based on bounds or using a default scale
                if bounds is not None:
                    scale = 0.1 * (bounds[i][1] - bounds[i][0])
                else:
                    scale = 0.1 * abs(individual[i]) + 0.1

                # Apply gaussian mutation
                individual[i] += np.random.normal(0, scale)

                # Apply bounds if provided
                if bounds is not None:
                    individual[i] = max(bounds[i][0], min(bounds[i][1], individual[i]))

        return individual

    def _uniform_mutation(self,
                        individual: np.ndarray,
                        mutation_rate: float,
                        bounds: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """Uniform mutation method."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                # Apply uniform mutation within bounds
                if bounds is not None:
                    individual[i] = random.uniform(bounds[i][0], bounds[i][1])
                else:
                    # Without bounds, apply a random perturbation of up to 20%
                    scale = 0.2 * abs(individual[i]) + 0.1
                    individual[i] += random.uniform(-scale, scale)

        return individual

    def _update_adaptive_mutation_rate(self, generation: int, max_generations: int) -> None:
        """Update adaptive mutation rate based on generation progress."""
        # Linear decay from initial rate to final rate (1/5 of initial)
        progress = generation / max_generations
        final_rate = self.mutation_rate / 5.0
        self.adaptive_mutation_rate = self.mutation_rate - progress * (self.mutation_rate - final_rate)

        logger.debug(f"Adaptive mutation rate updated to {self.adaptive_mutation_rate:.6f}")

    def plot_fitness_history(self,
                           filename: Optional[str] = None,
                           title: str = "Evolution Progress",
                           show: bool = True) -> None:
        """
        Plot the fitness history.

        Args:
            filename: Optional filename to save the plot
            title: Plot title
            show: Whether to display the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'r-', label='Best Fitness')
        plt.plot(self.avg_fitness_history, 'b-', label='Average Fitness')
        plt.title(title)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)

        if filename:
            plt.savefig(filename)
            logger.info(f"Fitness history plot saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_state(self, filename: str) -> str:
        """
        Save the optimizer state.

        Args:
            filename: Filename to save the state

        Returns:
            Full path to the saved file
        """
        filepath = os.path.join(MODELS_DIR, filename)

        state = {
            'population': self.population,
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generation': self.generation,
            'params': {
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size,
                'selection_method': self.selection_method,
                'crossover_method': self.crossover_method,
                'mutation_method': self.mutation_method
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Optimizer state saved to {filepath}")
        return filepath

    @classmethod
    def load_state(cls, filename: str) -> 'EvolutionaryOptimizer':
        """
        Load optimizer state from file.

        Args:
            filename: Filename to load state from

        Returns:
            Initialized EvolutionaryOptimizer
        """
        filepath = os.path.join(MODELS_DIR, filename)

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create optimizer with saved parameters
        params = state['params']
        optimizer = cls(
            population_size=params['population_size'],
            mutation_rate=params['mutation_rate'],
            crossover_rate=params['crossover_rate'],
            elite_size=params['elite_size'],
            selection_method=params['selection_method'],
            crossover_method=params['crossover_method'],
            mutation_method=params['mutation_method']
        )

        # Restore state
        optimizer.population = state['population']
        optimizer.best_individual = state['best_individual']
        optimizer.best_fitness = state['best_fitness']
        optimizer.fitness_history = state['fitness_history']
        optimizer.best_fitness_history = state['best_fitness_history']
        optimizer.avg_fitness_history = state['avg_fitness_history']
        optimizer.generation = state['generation']

        logger.info(f"Optimizer state loaded from {filepath}")
        return optimizer

# Utility functions for common optimization tasks
def optimize_function(
    fitness_function: Callable[[np.ndarray], float],
    individual_size: int,
    bounds: List[Tuple[float, float]],
    population_size: int = 100,
    max_generations: int = 100,
    verbose: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Optimize a function using the evolutionary optimizer.

    Args:
        fitness_function: Function to optimize
        individual_size: Number of parameters
        bounds: List of (min, max) bounds for each parameter
        population_size: Size of the population
        max_generations: Maximum number of generations to run
        verbose: Whether to print progress

    Returns:
        Tuple of (best individual, best fitness)
    """
    # Set up logging based on verbosity
    if not verbose:
        logging.getLogger(__name__).setLevel(logging.WARNING)

    # Create and initialize optimizer
    optimizer = EvolutionaryOptimizer(
        population_size=population_size,
        max_generations=max_generations
    )

    optimizer.initialize_population(individual_size, bounds)

    # Run evolution
    best_individual = optimizer.evolve(fitness_function, bounds=bounds)
    best_fitness = optimizer.best_fitness

    # Reset logging level
    if not verbose:
        logging.getLogger(__name__).setLevel(logging.INFO)

    return best_individual, best_fitness

def main():
    """Example usage of the EvolutionaryOptimizer."""
    # Define a simple fitness function (minimize sphere function)
    def sphere_function(x):
        return -sum(xi**2 for xi in x)  # Negative because we maximize fitness

    # Set problem parameters
    dimensions = 10
    bounds = [(-5.0, 5.0)] * dimensions

    # Run optimization
    print("Optimizing sphere function...")
    best_individual, best_fitness = optimize_function(
        sphere_function,
        dimensions,
        bounds,
        population_size=50,
        max_generations=100
    )

    print(f"Best solution found: {best_individual}")
    print(f"Best fitness: {best_fitness}")

if __name__ == "__main__":
    main()
