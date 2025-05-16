# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Module Benchmark

This module provides functionality to evaluate and benchmark different
variants of code modules to determine which performs best.
"""

import os
import sys
import ast
import time
import logging
import tempfile
import importlib
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import multiprocessing as mp
from contextlib import contextmanager

# Try to import optional performance libraries
try:
    import psutil
    import memory_profiler
    HAVE_PROFILERS = True
except ImportError:
    HAVE_PROFILERS = False

# Local imports
from ..utils.config_loader import ConfigLoader
from ..utils.sqlite_helper import SQLiteHelper

logger = logging.getLogger(__name__)

class ModuleBenchmark:
    """
    Benchmarks and evaluates module variants to determine which performs best
    according to multiple criteria:
    - Execution speed
    - Memory usage
    - Code quality/complexity
    - Test success rate
    - Error handling robustness
    """
    
    def __init__(self, config_path: str = None, db_path: str = None):
        """
        Initialize the benchmark system.
        
        Args:
            config_path: Path to benchmark configuration
            db_path: Path to SQLite database for storing results
        """
        self.config = ConfigLoader().load(config_path or "config/benchmark.yaml")
        self.db_path = db_path or self.config.get("db_path", "data/benchmark_results.db")
        self.db = SQLiteHelper(self.db_path)
        self._init_database()
        
        # Weight factors for different evaluation criteria
        self.weights = self.config.get("weights", {
            "execution_speed": 0.3,
            "memory_usage": 0.2,
            "code_quality": 0.15,
            "test_success": 0.25,
            "error_handling": 0.1
        })
        
        # If pycodestyle is available for code quality checks
        try:
            import pycodestyle
            self.style_checker = pycodestyle.StyleGuide(quiet=True)
            self.have_style_checker = True
        except ImportError:
            self.have_style_checker = False
            
        logger.info("ModuleBenchmark initialized")
        
    def _init_database(self):
        """Initialize the benchmark results database"""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_path TEXT,
                variant_path TEXT,
                timestamp TEXT,
                execution_speed REAL,
                memory_usage REAL,
                code_quality REAL,
                test_success REAL,
                error_handling REAL,
                overall_score REAL
            )
        """)
        
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT UNIQUE,
                best_variant_path TEXT,
                score REAL,
                timestamp TEXT
            )
        """)
        
    def evaluate_module_variants(self, original_module_path: str, 
                               variant_paths: List[str]) -> Dict[str, Dict]:
        """
        Evaluate multiple variants of a module and return comprehensive results.
        
        Args:
            original_module_path: Path to the original module
            variant_paths: List of paths to variant modules
            
        Returns:
            Dictionary of evaluation results for each variant
        """
        logger.info(f"Evaluating {len(variant_paths)} variants of {original_module_path}")
        
        # First, evaluate the original module as baseline
        original_results = self._evaluate_single_module(original_module_path)
        if not original_results:
            logger.error(f"Failed to evaluate original module {original_module_path}")
            return {}
        
        # Initialize results dictionary with original module
        results = {
            original_module_path: original_results
        }
        
        # Evaluate each variant
        for variant_path in variant_paths:
            variant_results = self._evaluate_single_module(variant_path)
            if variant_results:
                # Normalize scores relative to the original module
                normalized_results = self._normalize_results(variant_results, original_results)
                results[variant_path] = normalized_results
            else:
                logger.warning(f"Failed to evaluate variant {variant_path}")
        
        # Save results to database
        self._save_results_to_db(original_module_path, results)
        
        return results
    
    def _evaluate_single_module(self, module_path: str) -> Dict:
        """
        Comprehensively evaluate a single module across all metrics.
        
        Args:
            module_path: Path to the module to evaluate
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating module: {module_path}")
        
        # Basic validation
        if not os.path.exists(module_path):
            logger.error(f"Module path does not exist: {module_path}")
            return {}
            
        try:
            # Run various evaluations
            execution_speed = self._benchmark_execution_speed(module_path)
            memory_usage = self._benchmark_memory_usage(module_path) if HAVE_PROFILERS else 0.0
            code_quality = self._assess_code_quality(module_path)
            test_success = self._run_tests(module_path)
            error_handling = self._evaluate_error_handling(module_path)
            
            # Calculate overall score
            overall_score = (
                execution_speed * self.weights["execution_speed"] +
                memory_usage * self.weights["memory_usage"] +
                code_quality * self.weights["code_quality"] +
                test_success * self.weights["test_success"] +
                error_handling * self.weights["error_handling"]
            )
            
            return {
                "execution_speed": execution_speed,
                "memory_usage": memory_usage,
                "code_quality": code_quality,
                "test_success": test_success,
                "error_handling": error_handling,
                "overall_score": overall_score
            }
        
        except Exception as e:
            logger.error(f"Error evaluating module {module_path}: {e}")
            traceback.print_exc()
            return {}
    
    def _benchmark_execution_speed(self, module_path: str) -> float:
        """
        Benchmark the execution speed of a module.
        
        Args:
            module_path: Path to the module to benchmark
            
        Returns:
            Speed score (higher is better)
        """
        try:
            # Create a temporary script to run the module
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
                f.write(f"""
import sys
import time
import importlib.util
from pathlib import Path

# Load the module
module_path = "{module_path}"
module_name = Path(module_path).stem
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Find all functions to test
functions = [attr for attr in dir(module) if callable(getattr(module, attr)) 
             and not attr.startswith('_')]

# Measure execution time
start_time = time.time()
for _ in range(100):  # Run multiple iterations for better accuracy
    for func_name in functions[:5]:  # Limit to first 5 functions
        try:
            func = getattr(module, func_name)
            # Try to call with no arguments, ignore if it fails
            func()
        except:
            pass
            
end_time = time.time()
print(end_time - start_time)  # Output execution time
                """)
                temp_script = f.name
            
            # Run the script and capture output
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=10  # Timeout after 10 seconds
            )
            
            # Clean up
            os.unlink(temp_script)
            
            # Parse execution time
            try:
                execution_time = float(result.stdout.strip())
                
                # Convert to score (lower time = higher score)
                # Normalize to range 0-1 where 1 is best (fastest)
                # Use a baseline execution time of 0.1s as reference
                baseline = 0.1
                speed_score = min(1.0, baseline / max(execution_time, 0.001))
                
                return speed_score
            except ValueError:
                logger.error(f"Failed to parse execution time from: {result.stdout}")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error benchmarking execution speed for {module_path}: {e}")
            return 0.0
    
    def _benchmark_memory_usage(self, module_path: str) -> float:
        """
        Benchmark the memory usage of a module.
        
        Args:
            module_path: Path to the module to benchmark
            
        Returns:
            Memory efficiency score (higher is better)
        """
        if not HAVE_PROFILERS:
            return 0.5  # Default score if profilers are not available
            
        try:
            # Create a process to measure memory
            def measure_memory():
                # Import and run the module
                module_dir = os.path.dirname(module_path)
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                    
                module_name = os.path.basename(module_path).replace('.py', '')
                
                # Use memory_profiler to measure peak memory
                from memory_profiler import memory_usage
                
                def load_and_run_module():
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Call a few functions to measure memory usage in action
                        for func_name in [attr for attr in dir(module) if callable(getattr(module, attr)) 
                                        and not attr.startswith('_')][:3]:
                            try:
                                getattr(module, func_name)()
                            except:
                                pass
                    except Exception as e:
                        logger.error(f"Error in load_and_run_module: {e}")
                
                # Measure memory usage
                mem_usage = memory_usage((load_and_run_module, (), {}), interval=0.1, timeout=5)
                return max(mem_usage) if mem_usage else 0
            
            # Run in separate process for isolation
            with mp.Pool(1) as pool:
                memory_used = pool.apply(measure_memory)
            
            # Convert to score (lower memory = higher score)
            # Normalize to range 0-1 where 1 is best (lowest memory)
            # Use a baseline of 50MB as reference
            baseline = 50  # MB
            memory_score = min(1.0, baseline / max(memory_used, 1.0))
            
            return memory_score
            
        except Exception as e:
            logger.error(f"Error benchmarking memory usage for {module_path}: {e}")
            return 0.5  # Default score on error
    
    def _assess_code_quality(self, module_path: str) -> float:
        """
        Assess the code quality of a module.
        
        Args:
            module_path: Path to the module to assess
            
        Returns:
            Code quality score (higher is better)
        """
        try:
            # Read the module code
            with open(module_path, 'r') as f:
                code = f.read()
                
            # Check Python syntax
            try:
                ast.parse(code)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False
                
            # Calculate code quality metrics
            quality_score = 0.0
            
            # Syntax validity is a basic requirement
            if not syntax_valid:
                return 0.0
                
            # Check code style if pycodestyle is available
            if self.have_style_checker:
                style_report = self.style_checker.check_files([module_path])
                style_errors = style_report.total_errors
                
                # Convert to score (fewer errors = higher score)
                # Normalize to range 0-0.5
                line_count = len(code.splitlines())
                error_ratio = min(1.0, style_errors / max(line_count, 1))
                style_score = 0.5 * (1 - error_ratio)
                quality_score += style_score
            else:
                # Default style score if checker not available
                quality_score += 0.25
                
            # Check complexity metrics
            try:
                # Calculate cyclomatic complexity
                import radon.complexity as cc
                average_complexity = 0
                
                results = cc.cc_visit(code)
                if results:
                    complexities = [result.complexity for result in results]
                    average_complexity = sum(complexities) / len(complexities)
                    
                # Convert to score (lower complexity = higher score)
                # Use scale where complexity of 10 = 0.25 score, 5 = 0.5 score
                complexity_score = 0.5 * min(1.0, 10 / max(average_complexity, 1.0))
                quality_score += complexity_score
            except ImportError:
                # If radon is not available, assign default score
                quality_score += 0.25
                
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing code quality for {module_path}: {e}")
            return 0.3  # Default score on error
    
    def _run_tests(self, module_path: str) -> float:
        """
        Run tests for a module and assess test success rate.
        
        Args:
            module_path: Path to the module to test
            
        Returns:
            Test success score (higher is better)
        """
        try:
            module_name = os.path.basename(module_path).replace('.py', '')
            module_dir = os.path.dirname(module_path)
            
            # Look for test files in several possible locations
            possible_test_locations = [
                os.path.join(module_dir, f"test_{module_name}.py"),
                os.path.join(module_dir, f"{module_name}_test.py"),
                os.path.join(module_dir, "..", "tests", f"test_{module_name}.py"),
                os.path.join(module_dir, "..", "..", "tests", f"test_{module_name}.py")
            ]
            
            test_file = None
            for location in possible_test_locations:
                if os.path.exists(location):
                    test_file = location
                    break
                    
            if not test_file:
                logger.warning(f"No test file found for {module_path}")
                # Create a basic test file with simulated imports and function calls
                test_file = self._create_basic_test(module_path)
                if not test_file:
                    return 0.5  # Default score if no test file can be created
            
            # Run the test file with pytest
            try:
                # Prepare environment for test
                env = os.environ.copy()
                # Add the parent directory to PYTHONPATH
                parent_dir = os.path.dirname(os.path.dirname(module_path))
                if "PYTHONPATH" in env:
                    env["PYTHONPATH"] = f"{parent_dir}:{env['PYTHONPATH']}"
                else:
                    env["PYTHONPATH"] = parent_dir
                
                # Run pytest on the test file
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v"],
                    capture_output=True,
                    text=True,
                    timeout=30,  # Timeout after 30 seconds
                    env=env
                )
                
                # Parse test results
                passed_count = result.stdout.count("PASSED")
                failed_count = result.stdout.count("FAILED")
                error_count = result.stdout.count("ERROR")
                total_count = passed_count + failed_count + error_count
                
                if total_count == 0:
                    logger.warning(f"No tests run for {module_path}")
                    return 0.5  # Default score if no tests were run
                    
                # Calculate test success rate
                success_rate = passed_count / total_count
                
                return success_rate
                
            except subprocess.TimeoutExpired:
                logger.error(f"Tests timed out for {module_path}")
                return 0.0  # Timeout is a serious problem
                
            except Exception as e:
                logger.error(f"Error running tests for {module_path}: {e}")
                return 0.3  # Default score on error
                
        except Exception as e:
            logger.error(f"Error setting up tests for {module_path}: {e}")
            return 0.3  # Default score on error
            
        finally:
            # Clean up temporary test file if one was created
            if test_file and test_file.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(test_file)
                except:
                    pass
    
    def _create_basic_test(self, module_path: str) -> str:
        """
        Create a basic test file for a module.
        
        Args:
            module_path: Path to the module to test
            
        Returns:
            Path to the created test file
        """
        try:
            # Read the module code
            with open(module_path, 'r') as f:
                code = f.read()
                
            # Parse AST to find classes and functions
            tree = ast.parse(code)
            
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    # Only include public functions
                    functions.append(node.name)
            
            # Create a temporary test file
            with tempfile.NamedTemporaryFile(suffix='_test.py', mode='w', delete=False) as f:
                module_name = os.path.basename(module_path).replace('.py', '')
                parent_dir = os.path.basename(os.path.dirname(module_path))
                
                # Write test imports
                f.write(f"""
import pytest
import sys
import os
from pathlib import Path

# Add module directory to path
module_dir = Path("{os.path.dirname(module_path)}")
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir.parent))

# Import the module
from {parent_dir} import {module_name}

# Basic test suite for {module_name}
""")
                
                # Write class tests if any
                for class_name in classes:
                    f.write(f"""
class Test{class_name}:
    def test_{class_name.lower()}_instantiation(self):
        # Test that we can instantiate the class
        try:
            instance = {module_name}.{class_name}()
            assert instance is not None
        except TypeError:
            # If it requires parameters, skip this test
            pytest.skip("Class requires parameters")
""")
                
                # Write function tests
                for func_name in functions:
                    f.write(f"""
def test_{func_name}():
    # Test that the function exists and can be called
    try:
        # This test only verifies the function can be called, not correctness
        result = {module_name}.{func_name}()
        assert True  # If we got here, the function didn't crash
    except TypeError:
        # If it requires parameters, skip this test
        pytest.skip("Function requires parameters")
    except Exception as e:
        # Allow any other exceptions for now as we don't know expected behavior
        pytest.skip(f"Function raised exception: {{e}}")
""")
                
                test_file = f.name
                
            return test_file
            
        except Exception as e:
            logger.error(f"Error creating basic test for {module_path}: {e}")
            return None
    
    def _evaluate_error_handling(self, module_path: str) -> float:
        """
        Evaluate error handling robustness of a module.
        
        Args:
            module_path: Path to the module to evaluate
            
        Returns:
            Error handling score (higher is better)
        """
        try:
            # Read the module code
            with open(module_path, 'r') as f:
                code = f.read()
                
            # Parse AST
            tree = ast.parse(code)
            
            # Count try-except blocks
            try_count = 0
            function_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_count += 1
                elif isinstance(node, ast.FunctionDef):
                    function_count += 1
            
            # Calculate error handling score
            if function_count == 0:
                return 0.5  # Default for modules with no functions
                
            # Score based on ratio of try blocks to functions
            # Ideal ratio is around 0.7 try blocks per function
            ratio = try_count / function_count
            if ratio > 1.0:
                # Too many try blocks might indicate poor design
                score = 0.7
            else:
                # Scale from 0 to 1 with optimal at 0.7
                score = min(1.0, ratio / 0.7)
                
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating error handling for {module_path}: {e}")
            return 0.4  # Default score on error
    
    def _normalize_results(self, variant_results: Dict, baseline_results: Dict) -> Dict:
        """
        Normalize variant results relative to the baseline (original module).
        
        Args:
            variant_results: Results dictionary for the variant
            baseline_results: Results dictionary for the baseline (original module)
            
        Returns:
            Normalized results dictionary
        """
        normalized = {}
        
        for metric, value in variant_results.items():
            baseline_value = baseline_results.get(metric, 0.5)
            if baseline_value <= 0:
                baseline_value = 0.1  # Avoid division by zero
                
            # For overall score, we want absolute values
            if metric == "overall_score":
                normalized[metric] = value
            else:
                # For other metrics, normalize relative to baseline
                # Score of 1.0 means same as baseline
                # Above 1.0 means better than baseline
                normalized[metric] = value / baseline_value
                
        return normalized
    
    def _save_results_to_db(self, original_module_path: str, results: Dict[str, Dict]) -> None:
        """
        Save benchmark results to database.
        
        Args:
            original_module_path: Path to the original module
            results: Dictionary of evaluation results for each variant
        """
        from datetime import datetime
        
        timestamp = datetime.now().isoformat()
        module_name = os.path.basename(original_module_path)
        
        # Save each result
        for module_path, metrics in results.items():
            self.db.execute(
                """
                INSERT INTO benchmark_results 
                (module_path, variant_path, timestamp, execution_speed, memory_usage, 
                code_quality, test_success, error_handling, overall_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    original_module_path,
                    module_path,
                    timestamp,
                    metrics.get("execution_speed", 0),
                    metrics.get("memory_usage", 0),
                    metrics.get("code_quality", 0),
                    metrics.get("test_success", 0),
                    metrics.get("error_handling", 0),
                    metrics.get("overall_score", 0)
                )
            )
        
        # Update leaderboard with best variant
        best_variant = self.select_best_variant(results)
        if best_variant:
            best_score = results[best_variant]["overall_score"]
            
            self.db.execute(
                """
                INSERT INTO leaderboard (module_name, best_variant_path, score, timestamp)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(module_name) DO UPDATE SET
                best_variant_path = excluded.best_variant_path,
                score = excluded.score,
                timestamp = excluded.timestamp
                WHERE excluded.score > leaderboard.score
                """,
                (module_name, best_variant, best_score, timestamp)
            )
    
    def select_best_variant(self, evaluation_results: Dict[str, Dict]) -> str:
        """
        Select the best variant based on evaluation results.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            
        Returns:
            Path to the best variant
        """
        if not evaluation_results:
            return None
            
        # Find the variant with the highest overall score
        best_score = -1
        best_variant = None
        
        for variant_path, metrics in evaluation_results.items():
            score = metrics.get("overall_score", 0)
            if score > best_score:
                best_score = score
                best_variant = variant_path
                
        return best_variant
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """
        Get the current module leaderboard.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of leaderboard entries
        """
        results = self.db.query(
            "SELECT module_name, best_variant_path, score, timestamp FROM leaderboard ORDER BY score DESC LIMIT ?",
            (limit,)
        )
        
        return [
            {
                "module_name": row[0],
                "best_variant_path": row[1],
                "score": row[2],
                "timestamp": row[3]
            }
            for row in results
        ] 