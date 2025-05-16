# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Module Executor

Provides functionality for dynamically loading, executing, and monitoring
key modules within the Super AI system in an asynchronous manner.
"""

import os
import sys
import time
import logging
import asyncio
import importlib.util
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import inspect

# Set up logger
logger = logging.getLogger(__name__)

class ModuleExecutor:
    """
    Manages the execution of key modules within the Super AI system.

    Provides functionality for:
    - Dynamically loading modules
    - Executing module functions in separate threads
    - Monitoring execution and tracking results
    - Handling dependencies and execution order
    """

    def __init__(self, base_dir: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ModuleExecutor with configuration settings.

        Args:
            base_dir: Base directory for relative paths (default: current directory)
            config: Configuration settings for the executor
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config = config or {}

        # Track loaded modules
        self.loaded_modules: Dict[str, Any] = {}

        # Track running tasks
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Track execution results
        self.execution_results: Dict[str, Any] = {}

        # Threading pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4)
        )

        # Initialize event loop for async operations
        self.loop = asyncio.new_event_loop()

        # Configure module execution options
        self.dangerous_ai_enabled = self.config.get('ENABLE_DANGEROUS_AI', False)

        logger.info(f"ModuleExecutor initialized with base_dir: {self.base_dir}")
        if not self.dangerous_ai_enabled:
            logger.warning("DANGEROUS_AI is disabled. Some modules may be skipped.")

    async def load_module(self, module_path: Union[str, Path]) -> Optional[Any]:
        """
        Dynamically load a Python module.

        Args:
            module_path: Path to the module

        Returns:
            The loaded module or None if loading fails
        """
        try:
            module_path = Path(module_path)
            if not module_path.is_absolute():
                module_path = self.base_dir / module_path

            if not module_path.exists():
                logger.error(f"Module not found: {module_path}")
                return None

            module_name = module_path.stem

            # Skip dangerous modules if disabled
            if module_name == 'dangerousai' and not self.dangerous_ai_enabled:
                logger.warning(f"Skipping module {module_name} because DANGEROUS_AI is disabled")
                return None

            logger.info(f"Loading module: {module_name} from {module_path}")

            # Check if already loaded
            if module_name in self.loaded_modules:
                logger.info(f"Module {module_name} already loaded, returning cached instance")
                return self.loaded_modules[module_name]

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for module: {module_name}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Add to sys.modules for imports
            spec.loader.exec_module(module)

            # Cache the loaded module
            self.loaded_modules[module_name] = module

            logger.info(f"Successfully loaded module: {module_name}")
            return module

        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {str(e)}")
            return None

    async def execute_module_function(self, module_path: Union[str, Path],
                                     function_name: str = 'main',
                                     args: Optional[List] = None,
                                     kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a function within a module asynchronously.

        Args:
            module_path: Path to the module
            function_name: Name of the function to execute (default: 'main')
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary containing execution results
        """
        args = args or []
        kwargs = kwargs or {}
        module_name = Path(module_path).stem

        # Track execution in results
        execution_id = f"{module_name}.{function_name}_{int(time.time())}"
        self.execution_results[execution_id] = {"status": "started", "start_time": time.time()}

        try:
            # Load the module
            module = await self.load_module(module_path)
            if module is None:
                result = {"status": "error", "message": f"Failed to load module: {module_name}"}
                self.execution_results[execution_id].update(result)
                return result

            # Get the function to execute
            if not hasattr(module, function_name):
                result = {"status": "error", "message": f"Function {function_name} not found in module {module_name}"}
                self.execution_results[execution_id].update(result)
                return result

            function = getattr(module, function_name)

            # Check if function is async or not
            is_async = inspect.iscoroutinefunction(function)

            logger.info(f"Executing {'async ' if is_async else ''}function {module_name}.{function_name}")

            # Execute based on type
            if is_async:
                # Run directly if async
                result = await function(*args, **kwargs)
            else:
                # Run in thread pool if synchronous
                result = await asyncio.to_thread(function, *args, **kwargs)

            # Handle result
            if result is None:
                result = {"status": "completed", "message": f"{module_name}.{function_name} executed without return value"}
            elif not isinstance(result, dict):
                result = {"status": "completed", "result": result}

            end_time = time.time()
            execution_time = end_time - self.execution_results[execution_id]["start_time"]

            # Update execution results
            result.update({
                "execution_time": execution_time,
                "end_time": end_time
            })

            self.execution_results[execution_id].update(result)

            logger.info(f"Completed {module_name}.{function_name} in {execution_time:.2f}s")
            return result

        except Exception as e:
            error_message = f"Error executing {module_name}.{function_name}: {str(e)}"
            logger.error(error_message, exc_info=True)

            result = {
                "status": "error",
                "message": error_message,
                "exception": str(e)
            }

            self.execution_results[execution_id].update(result)
            return result

    async def execute_modules(self, modules: List[Tuple[Union[str, Path], str, Optional[List], Optional[Dict]]]) -> Dict[str, Any]:
        """
        Execute multiple module functions concurrently.

        Args:
            modules: List of (module_path, function_name, args, kwargs) tuples

        Returns:
            Dictionary mapping module names to execution results
        """
        tasks = []
        for module_args in modules:
            # Unpack the module arguments
            if len(module_args) == 4:
                module_path, function_name, args, kwargs = module_args
            elif len(module_args) == 3:
                module_path, function_name, args = module_args
                kwargs = {}
            elif len(module_args) == 2:
                module_path, function_name = module_args
                args, kwargs = [], {}
            else:
                module_path = module_args[0]
                function_name, args, kwargs = 'main', [], {}

            # Create task
            task = asyncio.create_task(
                self.execute_module_function(module_path, function_name, args, kwargs)
            )

            module_name = Path(module_path).stem
            task_id = f"{module_name}.{function_name}"
            self.running_tasks[task_id] = task
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        output = {}
        for i, module_args in enumerate(modules):
            module_path = module_args[0]
            function_name = module_args[1] if len(module_args) > 1 else 'main'

            module_name = Path(module_path).stem
            task_id = f"{module_name}.{function_name}"

            result = results[i]
            if isinstance(result, Exception):
                output[task_id] = {
                    "status": "error",
                    "message": f"Exception during execution: {str(result)}",
                    "exception": str(result)
                }
            else:
                output[task_id] = result

            # Clean up task
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

        return output

    def run_modules(self, modules: List[Tuple[Union[str, Path], str, Optional[List], Optional[Dict]]]) -> Dict[str, Any]:
        """
        Synchronous wrapper to execute modules.

        Args:
            modules: List of (module_path, function_name, args, kwargs) tuples

        Returns:
            Dictionary mapping module names to execution results
        """
        return asyncio.run(self.execute_modules(modules))

    async def run_key_modules(self):
        """
        Run the key modules for the Super AI system.

        This executes the main prediction and processing modules in the appropriate order.
        """
        # Define key modules to run
        key_modules = [
            ('prediction/sports_predictor.py', 'main', [], {}),
            ('betting/betting_prediction.py', 'main', [], {}),
            ('web_interface/app.py', 'main', [], {})
        ]

        # Check for dangerous AI
        if self.dangerous_ai_enabled:
            key_modules.append(('core_ai/dangerousai.py', 'main', [], {}))

        logger.info(f"Running {len(key_modules)} key modules")

        # Execute modules
        results = await self.execute_modules(key_modules)

        # Log summary
        successful = sum(1 for result in results.values() if result.get('status') != 'error')
        logger.info(f"Completed key module execution: {successful}/{len(key_modules)} successful")

        return results

    def get_execution_results(self) -> Dict[str, Any]:
        """Get all execution results."""
        return self.execution_results

    def get_loaded_modules(self) -> Dict[str, Any]:
        """Get all loaded modules."""
        return self.loaded_modules

    def shutdown(self):
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ModuleExecutor")
        self.thread_pool.shutdown(wait=True)
        for task_id, task in list(self.running_tasks.items()):
            if not task.done():
                logger.warning(f"Cancelling task {task_id}")
                task.cancel()
        self.running_tasks.clear()


# Helper function to execute modules
async def execute_modules(modules: List[Tuple[Union[str, Path], str, Optional[List], Optional[Dict]]],
                        base_dir: Optional[Path] = None,
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute multiple modules concurrently.

    Args:
        modules: List of (module_path, function_name, args, kwargs) tuples
        base_dir: Base directory for module paths
        config: Configuration settings

    Returns:
        Dictionary mapping module names to execution results
    """
    executor = ModuleExecutor(base_dir, config)
    return await executor.execute_modules(modules)


# Run this module directly for testing
if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Sample execution
    async def main():
        base_dir = Path(__file__).parent.parent
        config = {"ENABLE_DANGEROUS_AI": False}

        executor = ModuleExecutor(base_dir, config)

        # Define test modules
        test_modules = [
            ('run.py', 'check_dependencies', [], {}),
            ('utils/file_processor.py', 'process_files', [str(base_dir)], {})
        ]

        # Execute modules
        results = await executor.execute_modules(test_modules)

        # Print results
        for module_name, result in results.items():
            print(f"\n--- {module_name} ---")
            for key, value in result.items():
                print(f"{key}: {value}")

        return results

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    results = asyncio.run(main())

    print("\nExecution complete!")
