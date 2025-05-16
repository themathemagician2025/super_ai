# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Genome:
    """Represents a genome in the NEAT algorithm"""

    def __init__(self, input_size: int, output_size: int, innovation_counter: int = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}  # node_id -> node_type (input, hidden, output)
        self.connections = {}  # connection_id -> (in_node, out_node, weight, enabled)
        self.fitness = 0.0

        # Create input and output nodes
        for i in range(input_size):
            self.nodes[i] = "input"

        for i in range(output_size):
            self.nodes[input_size + i] = "output"

        # Initialize with minimal structure (all inputs connected to all outputs)
        for i in range(input_size):
            for j in range(output_size):
                conn_id = innovation_counter
                innovation_counter += 1
                self.connections[conn_id] = (i, input_size + j, random.uniform(-1.0, 1.0), True)

    def mutate_weights(self, mutation_rate: float = 0.1, perturbation_scale: float = 0.2):
        """Mutate the weights of connections"""
        for conn_id in self.connections:
            if random.random() < mutation_rate:
                in_node, out_node, weight, enabled = self.connections[conn_id]
                # Either perturb the weight or assign a new random weight
                if random.random() < 0.8:  # 80% chance to perturb
                    new_weight = weight + random.uniform(-1, 1) * perturbation_scale
                else:  # 20% chance for new random weight
                    new_weight = random.uniform(-1.0, 1.0)

                self.connections[conn_id] = (in_node, out_node, new_weight, enabled)

    def mutate_add_node(self, innovation_counter: int) -> int:
        """Add a new node by splitting an existing connection"""
        if not self.connections:
            return innovation_counter

        # Choose a random enabled connection
        enabled_connections = [c_id for c_id, (_, _, _, enabled) in self.connections.items() if enabled]
        if not enabled_connections:
            return innovation_counter

        conn_id = random.choice(enabled_connections)
        in_node, out_node, weight, _ = self.connections[conn_id]

        # Disable the chosen connection
        self.connections[conn_id] = (in_node, out_node, weight, False)

        # Add a new node
        new_node_id = max(self.nodes.keys()) + 1
        self.nodes[new_node_id] = "hidden"

        # Add two new connections
        # 1. From the input node to the new node with weight 1.0
        self.connections[innovation_counter] = (in_node, new_node_id, 1.0, True)
        innovation_counter += 1

        # 2. From the new node to the output node with the original weight
        self.connections[innovation_counter] = (new_node_id, out_node, weight, True)
        innovation_counter += 1

        return innovation_counter

    def mutate_add_connection(self, innovation_counter: int) -> int:
        """Add a new connection between existing nodes"""
        # Find all possible connections that don't already exist
        possible_connections = []
        for from_node in self.nodes:
            for to_node in self.nodes:
                # Skip if trying to connect: output -> any or any -> input or existing connection
                if (self.nodes[from_node] == "output" or
                    self.nodes[to_node] == "input" or
                    any(in_node == from_node and out_node == to_node
                        for in_node, out_node, _, _ in self.connections.values())):
                    continue

                # Also skip if it would create a cycle (for simplicity, just prevent hidden->hidden)
                if self.nodes[from_node] == "hidden" and self.nodes[to_node] == "hidden":
                    # More sophisticated cycle detection would be needed here
                    continue

                possible_connections.append((from_node, to_node))

        if not possible_connections:
            return innovation_counter

        # Choose a random new connection
        from_node, to_node = random.choice(possible_connections)
        self.connections[innovation_counter] = (from_node, to_node, random.uniform(-1.0, 1.0), True)
        innovation_counter += 1

        return innovation_counter

    def crossover(self, other: 'Genome') -> 'Genome':
        """Perform crossover with another genome"""
        # The more fit parent should be self
        if other.fitness > self.fitness:
            return other.crossover(self)

        # Create a child with the same structure as self
        child = Genome(self.input_size, self.output_size, 0)
        child.nodes = self.nodes.copy()

        # Crossover the connections
        for conn_id in set(self.connections.keys()) | set(other.connections.keys()):
            if conn_id in self.connections and conn_id in other.connections:
                # Matching gene - randomly choose from either parent
                if random.random() < 0.5:
                    child.connections[conn_id] = self.connections[conn_id]
                else:
                    child.connections[conn_id] = other.connections[conn_id]
            elif conn_id in self.connections:
                # Disjoint or excess gene from the fitter parent
                child.connections[conn_id] = self.connections[conn_id]

        return child

    def activate(self, inputs: List[float]) -> List[float]:
        """Activate the network with given inputs"""
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        # Node activation values
        activations = {}

        # Set input activations
        for i in range(self.input_size):
            activations[i] = inputs[i]

        # Simple topological sort (this is simplified and doesn't handle all possible network structures)
        # For a complete implementation, we would need proper topological sorting

        # Process hidden and output nodes
        output_ids = [node_id for node_id, node_type in self.nodes.items() if node_type == "output"]
        hidden_ids = [node_id for node_id, node_type in self.nodes.items() if node_type == "hidden"]

        # Process hidden nodes first, then output nodes
        for node_id in hidden_ids + output_ids:
            # Find all connections that feed into this node
            incoming = [(in_node, weight) for conn_id, (in_node, out_node, weight, enabled)
                       in self.connections.items()
                       if out_node == node_id and enabled and in_node in activations]

            if not incoming:
                activations[node_id] = 0.0
                continue

            # Sum weighted inputs
            weighted_sum = sum(activations[in_node] * weight for in_node, weight in incoming)

            # Apply sigmoid activation
            activations[node_id] = 1.0 / (1.0 + np.exp(-weighted_sum))

        # Return output activations
        return [activations.get(self.input_size + i, 0.0) for i in range(self.output_size)]


class Species:
    """Represents a species in the NEAT algorithm"""

    def __init__(self, representative: Genome):
        self.representative = representative
        self.members = [representative]
        self.fitness = 0.0
        self.staleness = 0  # number of generations without improvement

    def calculate_adjusted_fitness(self):
        """Calculate adjusted fitness for all members"""
        total_fitness = sum(genome.fitness for genome in self.members)
        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / len(self.members)

        self.fitness = total_fitness / len(self.members) if self.members else 0


class NEATAlgorithm:
    """Implementation of the NEAT algorithm for neuroevolution"""

    def __init__(self, input_size: int = 10, output_size: int = 5,
                 population_size: int = 50, config_path: Path = None):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.innovation_counter = 0
        self.population = []
        self.species = []
        self.generation = 0
        self.best_genome = None
        self.best_fitness = 0.0

        # Configuration parameters
        self.compatibility_threshold = 3.0
        self.compatibility_disjoint_coefficient = 1.0
        self.compatibility_weight_coefficient = 0.4
        self.weight_mutation_rate = 0.8
        self.add_node_mutation_rate = 0.03
        self.add_connection_mutation_rate = 0.05
        self.crossover_rate = 0.75
        self.survival_threshold = 0.2
        self.stale_species_threshold = 15

        # Load configuration if provided
        if config_path:
            self._load_config(config_path)

        # Initialize population
        self._initialize_population()
        logger.info(f"NEAT Algorithm initialized with population size {population_size}")

    def _load_config(self, config_path: Path):
        """Load configuration from file"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

            # Apply configuration settings
            if config:
                self.compatibility_threshold = config.get('compatibility_threshold', self.compatibility_threshold)
                self.compatibility_disjoint_coefficient = config.get('compatibility_disjoint_coefficient', self.compatibility_disjoint_coefficient)
                self.compatibility_weight_coefficient = config.get('compatibility_weight_coefficient', self.compatibility_weight_coefficient)
                self.weight_mutation_rate = config.get('weight_mutation_rate', self.weight_mutation_rate)
                self.add_node_mutation_rate = config.get('add_node_mutation_rate', self.add_node_mutation_rate)
                self.add_connection_mutation_rate = config.get('add_connection_mutation_rate', self.add_connection_mutation_rate)
                self.crossover_rate = config.get('crossover_rate', self.crossover_rate)
                self.survival_threshold = config.get('survival_threshold', self.survival_threshold)
                self.stale_species_threshold = config.get('stale_species_threshold', self.stale_species_threshold)
                logger.info(f"Loaded NEAT configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading NEAT configuration: {str(e)}")

    def _initialize_population(self):
        """Initialize the population with basic genomes"""
        self.population = []
        for _ in range(self.population_size):
            genome = Genome(self.input_size, self.output_size, self.innovation_counter)
            self.innovation_counter += self.input_size * self.output_size
            self.population.append(genome)

        # Speciate the initial population
        self._speciate()

    def _speciate(self):
        """Divide the population into species based on genetic similarity"""
        # Clear current species
        for species in self.species:
            species.members = []

        # Find species for each genome
        for genome in self.population:
            found_species = False
            for species in self.species:
                # Calculate genetic distance to the representative
                distance = self._genetic_distance(genome, species.representative)
                if distance < self.compatibility_threshold:
                    species.members.append(genome)
                    found_species = True
                    break

            if not found_species:
                # Create a new species with this genome as the representative
                self.species.append(Species(genome))

        # Remove empty species
        self.species = [species for species in self.species if species.members]

        # Update representatives for next generation
        for species in self.species:
            species.representative = random.choice(species.members)

    def _genetic_distance(self, genome1: Genome, genome2: Genome) -> float:
        """Calculate genetic distance between two genomes"""
        # Count matching, disjoint, and excess genes
        if not genome1.connections or not genome2.connections:
            return self.compatibility_threshold + 1.0  # Ensure they're not compatible

        max_id1 = max(genome1.connections.keys())
        max_id2 = max(genome2.connections.keys())
        max_id = max(max_id1, max_id2)

        matching_weights = []
        disjoint_count = 0

        for i in range(max_id + 1):
            in_genome1 = i in genome1.connections
            in_genome2 = i in genome2.connections

            if in_genome1 and in_genome2:
                # Matching gene - calculate weight difference
                weight1 = genome1.connections[i][2]
                weight2 = genome2.connections[i][2]
                matching_weights.append(abs(weight1 - weight2))
            elif (in_genome1 and i <= max_id2) or (in_genome2 and i <= max_id1):
                # Disjoint gene
                disjoint_count += 1

        # Calculate excess count
        excess_count = len(genome1.connections) + len(genome2.connections) - disjoint_count - len(matching_weights)

        # Calculate average weight difference
        avg_weight_diff = sum(matching_weights) / len(matching_weights) if matching_weights else 0

        # Normalize by size
        n = max(len(genome1.connections), len(genome2.connections))
        n = 1 if n < 20 else n  # For small genomes, don't normalize

        # Calculate distance
        distance = (
            self.compatibility_disjoint_coefficient * (disjoint_count + excess_count) / n +
            self.compatibility_weight_coefficient * avg_weight_diff
        )

        return distance

    def evolve(self):
        """Evolve the population for one generation"""
        # Calculate adjusted fitness for each species
        for species in self.species:
            species.calculate_adjusted_fitness()

        # Remove stale species
        self._remove_stale_species()

        # Reproduce to form the next generation
        offspring = self._reproduce()

        # Update the population
        self.population = offspring

        # Speciate the new population
        self._speciate()

        # Increment generation counter
        self.generation += 1

        # Find and update the best genome
        self._update_best_genome()

        logger.info(f"Generation {self.generation} completed with {len(self.species)} species")
        logger.info(f"Best fitness: {self.best_fitness}")

    def _remove_stale_species(self):
        """Remove species that haven't improved in a while"""
        for species in self.species:
            species.staleness += 1

            # Check if any member is better than the best previously
            best_fitness = max(genome.fitness for genome in species.members)
            if best_fitness > species.fitness:
                species.fitness = best_fitness
                species.staleness = 0

        # Remove stale species (except the top species)
        sorted_species = sorted(self.species, key=lambda s: s.fitness, reverse=True)
        self.species = [sorted_species[0]] + [
            species for species in sorted_species[1:]
            if species.staleness < self.stale_species_threshold
        ]

    def _reproduce(self) -> List[Genome]:
        """Reproduce to form the next generation"""
        offspring = []

        # Calculate total adjusted fitness of the population
        total_fitness = sum(species.fitness for species in self.species)

        # Calculate how many offspring each species should produce
        for species in self.species:
            # Species with higher fitness get to produce more offspring
            if total_fitness > 0:
                offspring_count = round((species.fitness / total_fitness) * self.population_size)
            else:
                offspring_count = 1  # Default if total fitness is 0

            if offspring_count > 0:
                # Sort members by fitness
                sorted_members = sorted(species.members, key=lambda g: g.fitness, reverse=True)

                # Calculate survival threshold
                survivors_count = max(1, int(len(sorted_members) * self.survival_threshold))

                # Directly copy the best genome from the species
                if len(sorted_members) > 0:
                    offspring.append(sorted_members[0])
                    offspring_count -= 1

                # Generate the rest of the offspring
                for _ in range(offspring_count):
                    if random.random() < self.crossover_rate and len(sorted_members) >= 2:
                        # Crossover - select two parents from the survivors
                        parent1 = random.choice(sorted_members[:survivors_count])
                        parent2 = random.choice(sorted_members[:survivors_count])
                        child = parent1.crossover(parent2)
                    else:
                        # Mutation only - select one parent
                        parent = random.choice(sorted_members[:survivors_count])
                        child = Genome(self.input_size, self.output_size, 0)
                        child.nodes = parent.nodes.copy()
                        child.connections = parent.connections.copy()

                    # Apply mutations
                    if random.random() < self.weight_mutation_rate:
                        child.mutate_weights()

                    if random.random() < self.add_node_mutation_rate:
                        self.innovation_counter = child.mutate_add_node(self.innovation_counter)

                    if random.random() < self.add_connection_mutation_rate:
                        self.innovation_counter = child.mutate_add_connection(self.innovation_counter)

                    offspring.append(child)

        # If we didn't produce enough offspring, add more
        while len(offspring) < self.population_size:
            # Create a basic genome
            genome = Genome(self.input_size, self.output_size, self.innovation_counter)
            self.innovation_counter += self.input_size * self.output_size
            offspring.append(genome)

        # If we produced too many offspring, truncate
        if len(offspring) > self.population_size:
            offspring = offspring[:self.population_size]

        return offspring

    def _update_best_genome(self):
        """Update the best genome found so far"""
        current_best = max(self.population, key=lambda g: g.fitness)
        if self.best_genome is None or current_best.fitness > self.best_fitness:
            self.best_genome = current_best
            self.best_fitness = current_best.fitness

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the best genome"""
        if self.best_genome is None:
            logger.warning("No best genome available for prediction")
            return {}

        # Convert input data to the appropriate format
        try:
            # Extract features from input data (this would be domain-specific)
            features = self._extract_features(input_data)

            # Use the best genome to make a prediction
            outputs = self.best_genome.activate(features)

            # Convert the outputs to a prediction (this would be domain-specific)
            prediction = self._interpret_outputs(outputs)

            return prediction
        except Exception as e:
            logger.error(f"Error making prediction with NEAT: {str(e)}")
            return {}

    def _extract_features(self, input_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from input data"""
        # This is a simplified implementation - in a real system,
        # feature extraction would be more sophisticated
        features = []

        # Extract numerical data from historical data
        if "historical" in input_data and input_data["historical"]:
            # For example, extract the last 5 values
            historical = input_data["historical"]
            if isinstance(historical, list) and len(historical) > 0:
                # If it's a list of numbers, take the last n values
                if all(isinstance(x, (int, float)) for x in historical):
                    features.extend(historical[-min(5, len(historical)):])
                # If it's a list of dictionaries, extract values from a specific key
                elif all(isinstance(x, dict) for x in historical):
                    if "value" in historical[0]:
                        features.extend([x.get("value", 0.0) for x in historical[-min(5, len(historical)):]])

        # Extract numerical data from online data
        if "online" in input_data and input_data["online"]:
            online = input_data["online"]
            if isinstance(online, dict):
                # Extract numerical values from the dictionary
                numerical_values = [v for k, v in online.items() if isinstance(v, (int, float))]
                features.extend(numerical_values[:5])  # Take up to 5 values

        # Sentiment data
        if "sentiment" in input_data and isinstance(input_data["sentiment"], dict):
            sentiment = input_data["sentiment"]
            if "score" in sentiment:
                features.append(sentiment["score"])

        # Ensure we have the right number of features
        if len(features) < self.input_size:
            # Pad with zeros
            features.extend([0.0] * (self.input_size - len(features)))
        elif len(features) > self.input_size:
            # Truncate
            features = features[:self.input_size]

        return features

    def _interpret_outputs(self, outputs: List[float]) -> Dict[str, Any]:
        """Interpret the neural network outputs as a prediction"""
        # This is a simplified implementation - in a real system,
        # output interpretation would be domain-specific
        prediction = {}

        # Convert the raw outputs to a prediction structure
        # For example, if the first output is a binary classification
        if outputs and len(outputs) >= 1:
            prediction["probability"] = outputs[0]
            prediction["classification"] = 1 if outputs[0] > 0.5 else 0

        # If we have more outputs, they might represent different aspects of the prediction
        if len(outputs) >= 3:
            prediction["confidence"] = outputs[1]
            prediction["magnitude"] = outputs[2]

        # For a more specific prediction model, e.g., for forex
        if len(outputs) >= 5:
            prediction["trend"] = {
                "direction": "up" if outputs[0] > 0.5 else "down",
                "strength": outputs[1] * 10  # Scale to 0-10
            }
            prediction["targets"] = {
                "take_profit": outputs[2],
                "stop_loss": outputs[3]
            }
            prediction["timeframe"] = outputs[4] * 24  # Convert to hours

        return prediction

    def update(self):
        """Update the NEAT algorithm (e.g., evolve for a few generations)"""
        try:
            # Evolve for a few generations
            for _ in range(5):  # Arbitrary number of generations
                self.evolve()

            logger.info("NEAT algorithm updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating NEAT algorithm: {str(e)}")
            return False

    def save(self, file_path: Path):
        """Save the NEAT model to a file"""
        try:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'best_genome': self.best_genome,
                    'input_size': self.input_size,
                    'output_size': self.output_size,
                    'generation': self.generation,
                    'best_fitness': self.best_fitness
                }, f)
            logger.info(f"NEAT model saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving NEAT model: {str(e)}")
            return False

    def load(self, file_path: Path):
        """Load a NEAT model from a file"""
        try:
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.best_genome = data['best_genome']
                self.input_size = data['input_size']
                self.output_size = data['output_size']
                self.generation = data['generation']
                self.best_fitness = data['best_fitness']
            logger.info(f"NEAT model loaded from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading NEAT model: {str(e)}")
            return False
