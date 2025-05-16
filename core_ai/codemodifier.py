# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/codemodifier.py
from typing import Union
from deap import gp, creator, base, tools
import operator
import random
import os
import logging
import pickle
import operator
import numpy as np
import pandas as pd
from deap import gp, creator, base, tools
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import codegenerator  # Import for code generation and file operations

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

# Enhanced primitive set


def create_pset() -> gp.PrimitiveSetTyped:
    """Create a strongly-typed primitive set for GP."""
    pset = gp.PrimitiveSetTyped("MAIN", [float], float)
    pset.addPrimitive(operator.add, [float, float], float, name="add")
    pset.addPrimitive(operator.sub, [float, float], float, name="sub")
    pset.addPrimitive(operator.mul, [float, float], float, name="mul")
    pset.addPrimitive(operator.truediv, [float, float], float, name="div")
    pset.addPrimitive(np.sin, [float], float, name="sin")
    pset.addPrimitive(np.cos, [float], float, name="cos")
    pset.addPrimitive(lambda x: x ** 2, [float], float, name="square")
    pset.addTerminal(0.0, float)
    pset.addTerminal(1.0, float)
    pset.addTerminal(np.pi, float, name="pi")
    pset.renameArguments(ARG0='x')
    return pset


def modify_expression(individual: gp.PrimitiveTree,
                      modification_type: str = "mutate",
                      pset: Optional[gp.PrimitiveSetTyped] = None) -> Tuple[gp.PrimitiveTree,
                                                                            bool]:
    """
    Apply a specific modification to the expression tree (individual).

    Args:
        individual: A DEAP individual (PrimitiveTree) to modify.
        modification_type: Type of modification ("mutate", "crossover", "autonomous").
        pset: Primitive set (required for autonomous modification).

    Returns:
        Tuple: (modified_individual, success_flag).

    Raises:
        ValueError: If modification_type is unsupported or pset is missing when needed.
    """
    if not pset and modification_type == "autonomous":
        raise ValueError(
            "Primitive set (pset) required for autonomous modification.")

    success = True
    if modification_type == "mutate":
        modified_ind, = gp.mutUniform(individual, expr=gp.genFull(
            pset=pset, min_=1, max_=3), pset=pset)  # Fix mutation call
        logger.info(f"Applied uniform mutation to individual: {str(modified_ind)[:50]}...")
    elif modification_type == "crossover":
        if len(population) < 2:  # Ensure population is passed correctly
            raise ValueError(
                "Crossover requires at least two individuals in the population.")
        parent1, parent2 = random.sample(population, 2)
        child1, child2 = gp.cxOnePoint(parent1, parent2)
        modified_ind = child1  # Return only one for simplicity
        logger.info(
            f"Applied crossover, resulting child: {
                str(modified_ind)[
                    :50]}...")
    elif modification_type == "autonomous":
        # Dangerous autonomous modification (book theme)
        if random.random() < 0.1:  # 10% chance of aggressive change
            expr = str(individual) + " * square(x)"
            try:
                modified_ind = gp.PrimitiveTree.from_string(expr, pset)
                logger.warning(
                    f"Dangerous autonomous modification applied: {expr}")
            except Exception as e:
                logger.error(f"Autonomous modification failed: {e}")
                modified_ind = individual
                success = False
        else:
            modified_ind, = gp.mutNodeReplacement(individual, pset=pset)
            logger.info(
                f"Applied node replacement mutation: {
                    str(modified_ind)[
                        :50]}...")
    else:
        raise ValueError(
            "Unsupported modification type: choose 'mutate', 'crossover', or 'autonomous'")

    return modified_ind, success


def apply_modification(population: List[gp.PrimitiveTree],
                       rate: float = 0.1,
                       modification_type: str = "mutate",
                       pset: Optional[gp.PrimitiveSetTyped] = None) -> List[gp.PrimitiveTree]:
    """
    Apply modifications to a population of individuals with a given probability.

    Args:
        population: List of DEAP individuals (PrimitiveTree objects).
        rate: Probability of modifying each individual.
        modification_type: Type of modification to apply.
        pset: Primitive set (required for autonomous modification).

    Returns:
        List: New population with modifications applied.
    """
    modified_population = []
    for ind in population:
        if random.random() < rate:
            try:
                modified_ind, success = modify_expression(
                    ind, modification_type, pset)
                if success:
                    modified_population.append(modified_ind)
                else:
                    modified_population.append(ind)
            except Exception as e:
                logger.error(f"Modification failed for individual: {e}")
                modified_population.append(ind)
        else:
            modified_population.append(ind)
    logger.info(
        f"Applied {modification_type} to {
            len(population)} individuals with rate {rate}")
    return modified_population


def load_and_apply_code_modification(individual: gp.PrimitiveTree,
                                     code_file: str = "modified_function.py",
                                     pset: Optional[gp.PrimitiveSetTyped] = None) -> gp.PrimitiveTree:
    """
    Apply code-level modifications from a file with safety checks and versioning.

    Args:
        individual: The DEAP individual to modify.
        code_file: Path to the modified code file (relative to src).
        pset: Primitive set for parsing the code.

    Returns:
        gp.PrimitiveTree: Modified individual or original if failed.
    """
    try:
        modified_code = codegenerator.load_code_from_file(code_file)
        if not modified_code:
            logger.warning(f"No code loaded from {code_file}")
            return individual

        code_structure = analyze_code_structure(modified_code)
        if is_safe_modification(code_structure) and pset:
            new_expr = extract_expression_from_code(modified_code)
            if new_expr:
                new_individual = gp.PrimitiveTree.from_string(new_expr, pset)
                if validate_modified_individual(new_individual, pset):
                    logger.info(
                        f"Successfully applied code modification from {code_file}")
                    return new_individual
        logger.warning("Code modification failed validation or parsing")
        return individual
    except Exception as e:
        logger.error(f"Error in code modification: {e}")
        return individual


def analyze_code_structure(
        code: str) -> Dict[str, Union[List[str], int, float]]:
    """Analyze the structure of the modified code."""
    structure = {
        'operations': [],
        'complexity': 0,
        'safety_score': 1.0,
        'imports': []
    }

    lines = code.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if 'import' in line or 'from' in line:
            structure['imports'].append(line)
            # Slightly reduce safety for imports
            structure['safety_score'] *= 0.9
        elif 'return' in line:
            structure['operations'].append(line)
        structure['complexity'] += calculate_line_complexity(line)

    logger.info(
        f"Code structure analyzed: complexity={
            structure['complexity']}, safety={
            structure['safety_score']}")
    return structure


def calculate_line_complexity(line: str) -> int:
    """Calculate complexity of a code line (simplified)."""
    complexity = len([c for c in line if c in '+-*/()'])
    return complexity if complexity > 0 else 1


def is_safe_modification(
        structure: Dict[str, Union[List[str], int, float]]) -> bool:
    """Check if the code modification is safe to apply."""
    unsafe_ops = ['os.', 'sys.', 'exec(', 'eval(', 'pickle.']
    for op in structure['operations'] + structure['imports']:
        if any(unsafe in op for unsafe in unsafe_ops):
            return False
    return structure['safety_score'] > 0.7 and structure['complexity'] < 50


def extract_expression_from_code(code: str) -> Optional[str]:
    """Extract the mathematical expression from the return statement."""
    for line in code.split('\n'):
        if 'return' in line:
            expr = line.split('return', 1)[-1].strip()
            return expr if expr else None
    return None


def validate_modified_individual(
        individual: gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped) -> bool:
    """Validate the modified individual against raw data."""
    try:
        func = gp.compile(individual, pset)
        raw_data = load_raw_data()
        if not raw_data:
            return True  # No data to validate against
        for df in raw_data.values():
            for x in df['x'].values[:10]:  # Sample test
                func(x)  # Ensure it runs without crashing
        return True
    except Exception as e:
        logger.error(f"Validation failed for modified individual: {e}")
        return False


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Load CSVs from data/raw for validation."""
    raw_data = {}
    for filename in os.listdir(RAW_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(RAW_DIR, filename)
            try:
                df = pd.read_csv(filepath)
                if 'x' in df.columns and 'y' in df.columns:
                    raw_data[filename] = df
                    logger.info(f"Loaded raw data: {filepath}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    return raw_data


def optimize_population(population: List[gp.PrimitiveTree],
                        pset: gp.PrimitiveSetTyped,
                        generations: int = 5,
                        rate: float = 0.2) -> List[gp.PrimitiveTree]:
    """Optimize the population through iterative modification."""
    for gen in range(generations):
        population = apply_modification(
            population,
            rate=rate,
            modification_type="autonomous",
            pset=pset)
        for ind in population:
            mse = evaluate_individual(ind, pset)
            ind.fitness.values = (-mse,)  # Minimize MSE
        population = sorted(
            population,
            key=lambda x: x.fitness.values[0])[
            :len(population) // 2 + 1]
        logger.info(
            f"Generation {gen + 1}: Best MSE = {-population[0].fitness.values[0]}")
    return population


def evaluate_individual(
        individual: gp.PrimitiveTree,
        pset: gp.PrimitiveSetTyped) -> float:
    """Evaluate individual fitness against raw data (MSE)."""
    raw_data = load_raw_data()
    if not raw_data:
        return float('inf')
    try:
        func = gp.compile(individual, pset)
        total_mse = 0
        count = 0
        for df in raw_data.values():
            y_pred = [func(x) for x in df['x'].values]
            mse = np.mean((df['y'].values - y_pred) ** 2)
            total_mse += mse
            count += 1
        return total_mse / count if count > 0 else float('inf')
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return float('inf')


def save_population(
        population: List[gp.PrimitiveTree], filename: str = "population.pkl") -> bool:
    """Save the entire population to a file."""
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(population, f)
        logger.info(f"Population saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save population to {filepath}: {e}")
        return False


population = []  # Global population for crossover demonstration


def main():
    """Demonstrate code modification with enhanced functionality."""
    global population
    random.seed(42)
    pset = create_pset()

    # Setup DEAP types
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create(
            "Individual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMin)

    # Toolbox setup
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        toolbox.expr)

    # Generate initial population
    population = [toolbox.individual() for _ in range(10)]
    print("Original Population:")
    for ind in population[:3]:  # Show first 3 for brevity
        print(ind)

    # Apply various modifications
    mutated_pop = apply_modification(
        population,
        rate=0.3,
        modification_type="mutate",
        pset=pset)
    print("\nMutated Population (sample):")
    for ind in mutated_pop[:3]:
        print(ind)

    # Apply autonomous modification (dangerous)
    autonomous_pop = apply_modification(
        population,
        rate=0.2,
        modification_type="autonomous",
        pset=pset)
    print("\nAutonomously Modified Population (sample):")
    for ind in autonomous_pop[:3]:
        print(ind)

    # Optimize population
    optimized_pop = optimize_population(population, pset, generations=3)
    print("\nOptimized Population (sample):")
    for ind in optimized_pop[:3]:
        print(ind)

    # Save and generate code for best individual
    best_ind = optimized_pop[0]
    codegenerator.generate_and_save(
        best_ind, pset, "best_function", "best_function.py")
    save_population(optimized_pop)


if __name__ == "__main__":
    main()

# Additional utilities


def batch_modify_and_save(population: List[gp.PrimitiveTree],
                          pset: gp.PrimitiveSetTyped,
                          prefix: str = "mod") -> List[str]:
    """Batch modify and save population as code files."""
    modified_pop = apply_modification(
        population,
        rate=0.5,
        modification_type="autonomous",
        pset=pset)
    return codegenerator.batch_generate(modified_pop, pset, prefix)


def stress_test_modification(
        population: List[gp.PrimitiveTree], pset: gp.PrimitiveSetTyped, iterations: int = 100) -> None:
    """Stress test modification process."""
    for i in range(iterations):
        population = apply_modification(
            population,
            rate=0.1,
            modification_type="autonomous",
            pset=pset)
        logger.info(
            f"Stress test iteration {
                i +
                1}: Population size = {
                len(population)}")
