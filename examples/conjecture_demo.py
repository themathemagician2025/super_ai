#!/usr/bin/env python
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Mathematical Conjectures Demonstration

This script demonstrates how to use the mathematical conjectures AI engine
for advanced analysis of numerical data.
"""

import os
import sys
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from algorithms.conjecture_ai_engine import conjecture_ai_engine
from algorithms.conjecture_implementations import (
    SchanuelConjecture, RiemannHypothesis, CollatzConjecture, PvsNP
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conjecture_demo")

def generate_test_data():
    """Generate different types of test data for demonstration"""

    # Fibonacci sequence
    def fibonacci(n):
        sequence = [1, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        return sequence

    # Prime numbers
    def primes(n):
        sieve = [True] * (n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(n+1) if sieve[i]]

    # Random data with some structure
    def structured_random(n, seed=42):
        np.random.seed(seed)
        base = np.linspace(0, 10, n)
        noise = np.random.normal(0, 1, n)
        return base + noise

    return {
        "fibonacci": fibonacci(20),
        "primes": primes(100),
        "structured_random": structured_random(30),
        "complex_roots": [(0.5 + 14.1j), (0.5 + 21.0j), (0.5 + 25.0j), (0.5 + 30.4j)]
    }

def analyze_data(data_dict):
    """Analyze different datasets with the conjecture engine"""
    results = {}

    for name, data in data_dict.items():
        if name == "complex_roots":
            # Special processing for complex numbers
            flat_data = []
            for z in data:
                flat_data.extend([z.real, z.imag])
            analysis_data = np.array(flat_data)
        else:
            analysis_data = np.array(data)

        logger.info(f"Analyzing {name} data with {len(analysis_data)} values")

        # Full analysis
        results[name] = conjecture_ai_engine.analyze(analysis_data)

        # Print key results
        print(f"\n=== {name.upper()} ANALYSIS ===")
        print(f"Overall score: {results[name]['weighted_score']:.3f}")
        print("Top conjectures:")
        for conj_name, score in results[name]['top_conjectures']:
            print(f"  - {conj_name}: {score:.3f}")
        print("Insights:")
        for insight in results[name]['insights']:
            print(f"  - {insight}")

    return results

def create_custom_conjecture():
    """Demonstrate creating and using a custom conjecture"""

    from algorithms.mathematical_conjectures import MathematicalConjecture

    class FibonacciPatternConjecture(MathematicalConjecture):
        """
        Detects if data follows a Fibonacci-like pattern where each element
        is approximately the sum of the two preceding elements.
        """

        def __init__(self, confidence=0.75):
            super().__init__(name="Fibonacci Pattern Conjecture", confidence=confidence)

        def evaluate(self, input_data):
            """
            Evaluates how closely the sequence follows a Fibonacci pattern.

            Args:
                input_data: Input sequence

            Returns:
                Score between 0 and 1, where 1 indicates a perfect Fibonacci pattern
            """
            if len(input_data) < 3:
                return 0.0

            # Check if each element is approximately the sum of the previous two
            errors = []
            for i in range(2, len(input_data)):
                expected = input_data[i-1] + input_data[i-2]
                actual = input_data[i]
                # Calculate relative error
                if expected != 0:
                    error = abs(actual - expected) / abs(expected)
                else:
                    error = abs(actual - expected)
                errors.append(min(error, 1.0))  # Cap at 1.0

            # Convert to a score where 0 error = 1.0 score
            avg_error = np.mean(errors)
            score = 1.0 - avg_error

            return max(0.0, min(1.0, score))

    # Create and add custom conjecture
    fibonacci_conjecture = FibonacciPatternConjecture()
    conjecture_ai_engine.add_custom_conjecture(fibonacci_conjecture)

    logger.info(f"Added custom conjecture: {fibonacci_conjecture.name}")
    return fibonacci_conjecture

def specific_conjecture_analysis(data_dict):
    """Analyze data with specific conjectures only"""

    # Select specific conjectures for analysis
    specific_conjectures = [
        "Fibonacci Pattern Conjecture",
        "Riemann Hypothesis",
        "Collatz Conjecture"
    ]

    results = {}
    for name, data in data_dict.items():
        logger.info(f"Analyzing {name} with specific conjectures: {specific_conjectures}")
        results[name] = conjecture_ai_engine.analyze_specific(np.array(data), specific_conjectures)

        print(f"\n=== {name.upper()} SPECIFIC ANALYSIS ===")
        print(f"Score: {results[name]['weighted_score']:.3f}")
        print("Evaluations:")
        for conj_name, score in results[name]['evaluations'].items():
            print(f"  - {conj_name}: {score:.3f}")

    return results

def visualize_results(data_dict, results, output_dir=None):
    """Visualize analysis results"""

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = Path(".")

    # Create a bar chart of scores for each dataset
    plt.figure(figsize=(12, 8))

    datasets = list(data_dict.keys())
    conjectures = []
    scores = []

    # Extract top 3 conjectures for each dataset
    for dataset in datasets:
        for name, score in results[dataset]['top_conjectures'][:3]:
            if name not in conjectures:
                conjectures.append(name)

    # Prepare data for plotting
    plot_data = {}
    for conjecture in conjectures:
        plot_data[conjecture] = []
        for dataset in datasets:
            # Find score for this conjecture
            score = 0.0
            for name, val in results[dataset]['evaluations'].items():
                if name == conjecture:
                    score = val
                    break
            plot_data[conjecture].append(score)

    # Create grouped bar chart
    x = np.arange(len(datasets))
    width = 0.8 / len(conjectures)

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, conjecture in enumerate(conjectures):
        offset = (i - len(conjectures)/2 + 0.5) * width
        ax.bar(x + offset, plot_data[conjecture], width, label=conjecture)

    ax.set_xlabel('Datasets')
    ax.set_ylabel('Conjecture Score')
    ax.set_title('Mathematical Conjecture Scores by Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path / "conjecture_scores.png")
    plt.close()

    logger.info(f"Visualization saved to {output_path / 'conjecture_scores.png'}")

    # Save raw results as JSON
    with open(output_path / "conjecture_results.json", "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        clean_results = {}
        for dataset, result in results.items():
            clean_results[dataset] = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    clean_results[dataset][key] = value.tolist()
                elif isinstance(value, dict):
                    clean_results[dataset][key] = {k: float(v) if isinstance(v, np.number) else v
                                                  for k, v in value.items()}
                else:
                    clean_results[dataset][key] = value

        json.dump(clean_results, f, indent=2)

    logger.info(f"Results saved to {output_path / 'conjecture_results.json'}")

def main():
    """Main entry point"""
    logger.info("=== Mathematical Conjectures Demonstration ===")

    # List available conjectures
    available_conjectures = conjecture_ai_engine.list_available_conjectures()
    logger.info(f"Available conjectures ({len(available_conjectures)}): {available_conjectures}")

    # Generate test data
    data_dict = generate_test_data()

    # Create custom conjecture
    custom_conjecture = create_custom_conjecture()

    # Analyze with all conjectures
    logger.info("Running full analysis with all conjectures...")
    results = analyze_data(data_dict)

    # Analyze with specific conjectures
    logger.info("Running analysis with specific conjectures...")
    specific_results = specific_conjecture_analysis(data_dict)

    # Visualize results
    visualize_results(data_dict, results, output_dir="output")

    logger.info("Demonstration completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())