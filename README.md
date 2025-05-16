# Super AI Source Code

This directory contains the source code for the Super AI prediction system. The system is a sophisticated Artificial Narrow Intelligence (ANI) designed for financial and sports market predictions.

## Directory Structure

- `core/`: Core AI functionality and algorithms
  - `brain/`: Neural network architecture and deep learning modules
  - `reasoning/`: Logical reasoning and decision-making components
  - `learning/`: Learning algorithms including reinforcement learning
  - `self_modification/`: Self-improvement and code generation capabilities

- `prediction/`: Prediction engines and models
  - `models/`: Various prediction models
  - `forecasting/`: Time-series forecasting components
  - `trading/`: Trading strategy implementations
  - `betting/`: Sports betting algorithms

- `api/`: API interfaces and controllers
  - `api_interface/`: Integration points for external systems

- `data_pipeline/`: Data processing and ETL workflows
  - `scrapers/`: Web scraping utilities
  - `etl/`: Extract, transform, load processes

- `ui/`: User interfaces
  - `web_interface/`: Web application for user interaction
  - `visualization/`: Data visualization components

- `utils/`: Utility functions and common tools
  - `config/`: Configuration management
  - `logging/`: Logging utilities

## Running the System

There are multiple entry points depending on which functionality you need:

1. **Main AI System**:
   ```
   python run.py
   ```
   or
   ```
   python super_ai_runner.py
   ```

2. **Web Interface**:
   ```
   python run_server.py
   ```

3. **Forex Trading**:
   ```
   python run_forex.py
   ```
   or
   ```
   python demo_forex.py
   ```

## Technologies Used

The system employs multiple AI methodologies:

- Neural networks (PyTorch/TensorFlow)
- Reinforcement learning
- Evolutionary algorithms
- Natural language processing
- Time-series forecasting
- Decision intelligence

## Development

Before contributing to the codebase, please familiarize yourself with the overall architecture and component interactions. The system is designed with modularity in mind, allowing for components to be improved or replaced individually.

For testing new features, use the `development/examples/` directory to create demonstration scripts that test specific functionality in isolation.
