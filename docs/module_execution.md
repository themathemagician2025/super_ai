# Module Execution System

## Overview

The Super AI Module Execution System provides a flexible and robust way to dynamically load and execute key modules within the application. It features asynchronous execution, parallel module processing, and comprehensive error handling.

## Key Components

### 1. ModuleExecutor

The `ModuleExecutor` class is the core component responsible for:
- Dynamically loading Python modules at runtime
- Executing module functions in separate threads
- Tracking execution status and results
- Managing dependencies between modules
- Handling errors and timeouts

### 2. Module Execution Script

The `module_execution.py` script serves as the entry point for executing key modules. It:
- Parses command-line arguments
- Loads configuration settings
- Determines which modules to execute
- Creates and configures the ModuleExecutor
- Processes and reports execution results

## Execution Workflow

1. **Initialization**: The ModuleExecutor is initialized with configuration settings.
2. **Module Selection**: Key modules for execution are identified (e.g., sports_predictor, betting_prediction).
3. **Safety Check**: If ENABLE_DANGEROUS_AI is False, certain modules (like dangerousai.py) are skipped.
4. **Asynchronous Execution**: Selected modules are executed in parallel using asyncio and threading.
5. **Results Processing**: Execution results are collected and processed.

## Supported Modules

The system currently supports the following key modules:

| Module Name | Path | Description |
|-------------|------|-------------|
| sports_predictor | prediction/sports_predictor.py | Sports prediction model with PyTorch |
| betting_prediction | betting/betting_prediction.py | Betting outcome prediction model |
| web_interface | web_interface/app.py | Flask web API for accessing predictions |

## Usage

### Command-Line Options

```bash
# Execute all modules
python module_execution.py

# Execute specific modules
python module_execution.py --modules sports_predictor,betting_prediction

# Skip web interface
python module_execution.py --no-web

# Specify custom configuration
python module_execution.py --config custom.env

# Enable debug logging
python module_execution.py --debug

# Specify model paths
python module_execution.py --sports-model models/custom_sports.pth --betting-model models/custom_betting.pth
```

### Programmatic Usage

```python
from utils.module_executor import ModuleExecutor
import asyncio

async def run_modules():
    # Create executor with configuration
    executor = ModuleExecutor(base_dir='/path/to/project', config={
        'ENABLE_DANGEROUS_AI': False
    })

    # Define modules to execute
    modules = [
        ('prediction/sports_predictor.py', 'main', [], {}),
        ('betting/betting_prediction.py', 'main', [], {})
    ]

    # Execute modules and get results
    results = await executor.execute_modules(modules)

    # Process results
    for module_id, result in results.items():
        print(f"{module_id}: {result.get('status')}")

    # Shutdown executor
    executor.shutdown()

# Run the async function
asyncio.run(run_modules())
```

## Module Implementation Requirements

For a module to be compatible with the execution system, it should:

1. Provide a `main()` function that serves as the entry point
2. Return a dictionary with a `status` key set to "success" or "error"
3. Include a `message` key with details about execution
4. Handle its own exceptions internally

### Example Module Structure

```python
def main(*args, **kwargs):
    try:
        # Module implementation
        # ...

        return {
            "status": "success",
            "message": "Module executed successfully",
            # Additional result data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Module execution failed: {str(e)}"
        }
```

## Safety and Security

- The execution system respects the `ENABLE_DANGEROUS_AI` configuration setting
- Modules are executed in separate threads to prevent blocking the main thread
- Exceptions in one module do not affect the execution of other modules
- Resource cleanup is ensured through the shutdown process
