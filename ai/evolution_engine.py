# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import numpy as np
from deap import base, creator, tools
from typing import List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """Handles evolutionary optimization with NEAT/DEAP"""
    
    def __init__(self, population_size: int = 100):
        self.population_size = population_size
        self._initialize_deap()
        logger.info("Evolution engine initialized")

    def _initialize_deap(self):
        """Initialize DEAP framework components"""
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, 
                            creator.Individual, self.toolbox.attr_float, n=50)
        self.toolbox.register("population", tools.initRepeat, list, 
                            self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evolve(self, fitness_func: callable, generations: int = 50) -> List[Any]:
        """Run evolutionary optimization"""
        try:
            # Initialization
            pop = self.toolbox.population(n=self.population_size)
            
            # Evaluate initial population
            fitnesses = list(map(fitness_func, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit,
            
            for gen in range(generations):
                # Selection
                offspring = self.toolbox.select(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))
                
                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.random() < 0.7:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Mutation
                for mutant in offspring:
                    if np.random.random() < 0.2:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                # Evaluate offspring
                invalid_ind = [ind for ind in offspring if not ind.fitness.values]
                fitnesses = map(fitness_func, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit,
                
                # Replace population
                pop[:] = offspring
                
                # Log progress
                fits = [ind.fitness.values[0] for ind in pop]
                logger.info(f"Gen {gen}: Max={max(fits)}, Avg={sum(fits)/len(fits)}")
            
            return pop
            
        except Exception as e:
            logger.error(f"Evolution failed: {str(e)}")
            raise