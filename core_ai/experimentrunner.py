# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Experiment Runner Module

A streamlined interface for running experiments with the AI system.
This module builds on experiment_runner.py, providing a more user-friendly
interface for setting up and running experiments.
"""

import os
import time
import logging
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(SRC_DIR, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Import from our own modules, with fallbacks for testing
try:
    from .experiment_runner import run_experiment, run_multiple_experiments
    from .experiment_runner import RealTimeExperimentRunner, save_experiment_results
except ImportError:
    logger.warning("Unable to import directly from experiment_runner, "
                 "using alternative import method")
    try:
        from experiment_runner import run_experiment, run_multiple_experiments
        from experiment_runner import RealTimeExperimentRunner, save_experiment_results
    except ImportError:
        logger.error("Failed to import experiment_runner module")
        # Define minimal versions for standalone testing
        def run_experiment(**kwargs):
            logger.error("Placeholder run_experiment called")
            return {"error": "experiment_runner module not available"}

        def run_multiple_experiments(**kwargs):
            logger.error("Placeholder run_multiple_experiments called")
            return [{"error": "experiment_runner module not available"}]

        class RealTimeExperimentRunner:
            def __init__(self, *args, **kwargs):
                logger.error("Placeholder RealTimeExperimentRunner initialized")

            async def run(self, *args, **kwargs):
                logger.error("Placeholder run method called")
                return {"error": "experiment_runner module not available"}

        def save_experiment_results(results, filename):
            logger.error("Placeholder save_experiment_results called")
            return False

class ExperimentRunner:
    """
    A class to manage and run AI experiments with various configurations.
    Extends the functionality in experiment_runner.py with a cleaner interface.
    """

    def __init__(self,
                base_config_path: Optional[str] = None,
                output_dir: Optional[str] = None,
                experiment_name: str = "default_experiment"):
        """
        Initialize the experiment runner.

        Args:
            base_config_path: Path to base configuration file
            output_dir: Directory to save experiment results
            experiment_name: Name of the experiment
        """
        self.base_config_path = base_config_path or os.path.join(SRC_DIR, "config", "config.txt")
        self.output_dir = output_dir or EXPERIMENTS_DIR
        self.experiment_name = experiment_name
        self.experiment_id = f"{experiment_name}_{int(time.time())}"

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize experiment configurations
        self.configurations = []
        self.function_suite = {}
        self.use_deap = False
        self.parallel = False
        self.max_workers = os.cpu_count() or 4

        logger.info(f"ExperimentRunner initialized: {self.experiment_id}")

    def add_configuration(self,
                        config_path: Optional[str] = None,
                        generations: int = 50,
                        name: Optional[str] = None,
                        use_deap: Optional[bool] = None) -> None:
        """
        Add a configuration to the experiment.

        Args:
            config_path: Path to configuration file, or use base_config_path if None
            generations: Number of generations to run
            name: Name of this configuration
            use_deap: Whether to use DEAP instead of NEAT for this configuration
        """
        config = {
            "config_path": config_path or self.base_config_path,
            "generations": generations,
            "name": name or f"config_{len(self.configurations) + 1}",
            "use_deap": self.use_deap if use_deap is None else use_deap
        }

        self.configurations.append(config)
        logger.info(f"Added configuration: {config['name']} with {generations} generations")

    def add_function_target(self,
                          function_name: str,
                          params: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a function to the target function suite.

        Args:
            function_name: Name of the function ('sin', 'cos', 'polynomial', etc.)
            params: Parameters for the function (e.g., {'coeffs': [1, 2, 1]})
        """
        self.function_suite[function_name] = params or {}
        logger.info(f"Added function target: {function_name} with params: {params}")

    def set_parallel(self, parallel: bool = True, max_workers: Optional[int] = None) -> None:
        """
        Set whether to run experiments in parallel.

        Args:
            parallel: Whether to run in parallel
            max_workers: Maximum number of worker processes to use
        """
        self.parallel = parallel
        if max_workers is not None:
            self.max_workers = max_workers

        logger.info(f"Parallel execution: {parallel}, max workers: {self.max_workers}")

    def set_deap(self, use_deap: bool = True) -> None:
        """
        Set whether to use DEAP instead of NEAT for all configurations.

        Args:
            use_deap: Whether to use DEAP
        """
        self.use_deap = use_deap

        # Update existing configurations
        for config in self.configurations:
            config["use_deap"] = use_deap

        logger.info(f"Using DEAP: {use_deap}")

    def run(self) -> Dict[str, Any]:
        """
        Run all configured experiments.

        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        logger.info(f"Starting experiment run: {self.experiment_id}")

        # Ensure we have at least one configuration
        if not self.configurations:
            logger.warning("No configurations added. Adding default configuration.")
            self.add_configuration()

        # Setup for either parallel or sequential execution
        if self.parallel and len(self.configurations) > 1:
            results = self._run_parallel()
        else:
            results = self._run_sequential()

        # Calculate and log summary
        runtime = time.time() - start_time
        results["summary"] = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_runtime": runtime,
            "configurations": len(self.configurations),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save results
        self._save_results(results)

        logger.info(f"Experiment run completed in {runtime:.2f} seconds")
        return results

    def _run_sequential(self) -> Dict[str, Any]:
        """Run experiments sequentially."""
        results = {}

        for idx, config in enumerate(self.configurations):
            logger.info(f"Running configuration {idx+1}/{len(self.configurations)}: {config['name']}")

            # Prepare arguments
            exp_name = f"{self.experiment_id}_{config['name']}"

            # Run experiment
            try:
                result = run_experiment(
                    config_path=config["config_path"],
                    generations=config["generations"],
                    experiment_name=exp_name,
                    function_suite=self.function_suite if self.function_suite else None,
                    use_deap=config["use_deap"]
                )
                results[config["name"]] = result
                logger.info(f"Configuration {config['name']} completed")

            except Exception as e:
                logger.error(f"Error running configuration {config['name']}: {e}")
                results[config["name"]] = {"error": str(e)}

        return results

    def _run_parallel(self) -> Dict[str, Any]:
        """Run experiments in parallel using process pool."""
        results = {}
        configs = self.configurations.copy()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {}
            for config in configs:
                exp_name = f"{self.experiment_id}_{config['name']}"

                future = executor.submit(
                    run_experiment,
                    config_path=config["config_path"],
                    generations=config["generations"],
                    experiment_name=exp_name,
                    function_suite=self.function_suite if self.function_suite else None,
                    use_deap=config["use_deap"]
                )
                future_to_config[future] = config

            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results[config["name"]] = result
                    logger.info(f"Configuration {config['name']} completed")

                except Exception as e:
                    logger.error(f"Error running configuration {config['name']}: {e}")
                    results[config["name"]] = {"error": str(e)}

        return results

    def _save_results(self, results: Dict[str, Any]) -> str:
        """
        Save experiment results to file.

        Args:
            results: Results dictionary

        Returns:
            Path to saved results file
        """
        # Create a timestamped results directory
        results_dir = os.path.join(self.output_dir, self.experiment_id)
        os.makedirs(results_dir, exist_ok=True)

        # Save full results as pickle
        pickle_path = os.path.join(results_dir, "results.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        # Save summary as JSON
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results.get("summary", {}), f, indent=2)

        logger.info(f"Results saved to {results_dir}")
        return results_dir

    def analyze_results(self, results_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze experiment results.

        Args:
            results_path: Path to results pickle file, or use latest if None

        Returns:
            Dictionary with analysis results
        """
        # Load results
        if results_path is None:
            # Find most recent results directory
            experiment_dirs = [os.path.join(self.output_dir, d) for d in os.listdir(self.output_dir)
                            if os.path.isdir(os.path.join(self.output_dir, d)) and
                            d.startswith(self.experiment_name)]
            if not experiment_dirs:
                logger.error("No experiment results found")
                return {"error": "No experiment results found"}

            # Get most recent directory
            results_dir = max(experiment_dirs, key=os.path.getmtime)
            results_path = os.path.join(results_dir, "results.pkl")

        # Load results
        try:
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {results_path}: {e}")
            return {"error": f"Error loading results: {e}"}

        # Perform analysis
        analysis = {"summary": results.get("summary", {})}

        # Extract fitness metrics across configurations
        fitness_data = {}
        for config_name, config_results in results.items():
            if config_name == "summary":
                continue

            if isinstance(config_results, dict) and "error" not in config_results:
                # Extract best fitness from each run
                best_fitness = self._extract_best_fitness(config_results)
                fitness_data[config_name] = best_fitness

        analysis["fitness_comparison"] = fitness_data

        # Save analysis results
        if results_path:
            results_dir = os.path.dirname(results_path)
            analysis_path = os.path.join(results_dir, "analysis.json")
            with open(analysis_path, 'w') as f:
                # Convert numpy values to Python native types for JSON serialization
                analysis_json = self._make_json_serializable(analysis)
                json.dump(analysis_json, f, indent=2)

            logger.info(f"Analysis saved to {analysis_path}")

        return analysis

    def _extract_best_fitness(self, results: Dict[str, Any]) -> float:
        """Extract best fitness value from results."""
        best_fitness = 0.0

        # Check different result formats
        if "default" in results:
            # Single run with default function
            if "winner" in results["default"]:
                winner = results["default"]["winner"]
                if hasattr(winner, "fitness"):
                    if isinstance(winner.fitness, float):
                        best_fitness = winner.fitness
                    elif hasattr(winner.fitness, "values"):
                        best_fitness = winner.fitness.values[0]

        # Check function suite results
        for func_name, func_results in results.items():
            if isinstance(func_results, dict) and "winner" in func_results:
                winner = func_results["winner"]
                if hasattr(winner, "fitness"):
                    if isinstance(winner.fitness, float):
                        fitness = winner.fitness
                    elif hasattr(winner.fitness, "values"):
                        fitness = winner.fitness.values[0]
                    best_fitness = max(best_fitness, fitness)

        return best_fitness

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        else:
            return str(obj)

class ExperimentPreset:
    """
    Predefined experiment configurations for common use cases.
    """

    @staticmethod
    def function_approximation(
        base_config_path: Optional[str] = None,
        functions: List[str] = ["sin", "cos", "polynomial"],
        generations: int = 200,
        use_deap: bool = False
    ) -> ExperimentRunner:
        """
        Create a function approximation experiment.

        Args:
            base_config_path: Path to base configuration file
            functions: List of target functions
            generations: Number of generations to run
            use_deap: Whether to use DEAP instead of NEAT

        Returns:
            Configured ExperimentRunner
        """
        runner = ExperimentRunner(
            base_config_path=base_config_path,
            experiment_name="function_approximation"
        )

        # Configure runner
        runner.set_deap(use_deap)
        runner.add_configuration(generations=generations)

        # Add function targets
        for func in functions:
            if func == "polynomial":
                runner.add_function_target(func, {"coeffs": [1, 2, 1]})  # x^2 + 2x + 1
            else:
                runner.add_function_target(func)

        return runner

    @staticmethod
    def hyperparameter_tuning(
        base_config_path: Optional[str] = None,
        param_variations: Dict[str, List[Any]] = None,
        generations: int = 100
    ) -> ExperimentRunner:
        """
        Create a hyperparameter tuning experiment with different configurations.

        Args:
            base_config_path: Path to base configuration file
            param_variations: Dictionary of parameters to tune and their values
            generations: Number of generations for each configuration

        Returns:
            Configured ExperimentRunner
        """
        if param_variations is None:
            param_variations = {
                "population_size": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5]
            }

        runner = ExperimentRunner(
            base_config_path=base_config_path,
            experiment_name="hyperparameter_tuning"
        )

        # Make a default configuration
        runner.add_configuration(
            generations=generations,
            name="default"
        )

        # Generate configurations for all parameter combinations
        import itertools
        param_names = list(param_variations.keys())
        param_values = list(param_variations.values())

        for i, combination in enumerate(itertools.product(*param_values)):
            param_dict = {name: value for name, value in zip(param_names, combination)}
            config_name = "_".join(f"{name}_{value}" for name, value in param_dict.items())

            # TODO: In a real implementation, we would create a new config file with these parameters
            # For now, we just log what we would do
            logger.info(f"Would create configuration with params: {param_dict}")

            runner.add_configuration(
                generations=generations,
                name=f"config_{i+1}_{config_name}"
            )

        # Set to parallel for efficiency
        runner.set_parallel(True)

        return runner

    @staticmethod
    def model_comparison(
        base_config_path: Optional[str] = None,
        generations: int = 150
    ) -> ExperimentRunner:
        """
        Create an experiment to compare NEAT vs DEAP performance.

        Args:
            base_config_path: Path to base configuration file
            generations: Number of generations for each model

        Returns:
            Configured ExperimentRunner
        """
        runner = ExperimentRunner(
            base_config_path=base_config_path,
            experiment_name="model_comparison"
        )

        # Add NEAT configuration
        runner.add_configuration(
            generations=generations,
            name="neat",
            use_deap=False
        )

        # Add DEAP configuration
        runner.add_configuration(
            generations=generations,
            name="deap",
            use_deap=True
        )

        # Add some functions to test
        runner.add_function_target("sin")
        runner.add_function_target("polynomial", {"coeffs": [1, 0, 0, 1]})  # x^3 + 1

        return runner

def main():
    """Example usage of the ExperimentRunner."""
    # Example 1: Basic experiment with default settings
    basic_runner = ExperimentRunner(experiment_name="basic_test")
    basic_runner.add_configuration(generations=5)  # Small number for testing
    basic_results = basic_runner.run()
    print(f"Basic experiment completed with {len(basic_results)} result sets")

    # Example 2: Using a preset
    function_runner = ExperimentPreset.function_approximation(
        functions=["sin", "cos"],
        generations=5  # Small number for testing
    )
    function_results = function_runner.run()
    print(f"Function approximation experiment completed with {len(function_results)} result sets")

    # Example 3: Parallel hyperparameter tuning
    hyperparam_runner = ExperimentPreset.hyperparameter_tuning(
        param_variations={"population_size": [50, 100]},
        generations=5  # Small number for testing
    )
    hyperparam_results = hyperparam_runner.run()
    print(f"Hyperparameter tuning experiment completed with {len(hyperparam_results)} result sets")

    # Example 4: Model comparison (NEAT vs DEAP)
    comparison_runner = ExperimentPreset.model_comparison(generations=5)  # Small number for testing
    comparison_results = comparison_runner.run()

    # Analyze results
    analysis = comparison_runner.analyze_results()
    print("Analysis completed. Fitness comparison:")
    for config, fitness in analysis.get("fitness_comparison", {}).items():
        print(f"  {config}: {fitness}")

if __name__ == "__main__":
    main()
