#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Super AI Controller

The central orchestrator for the autonomous Super AI system, managing the daily
cycle of data collection, model fine-tuning, self-modification, browser automation,
and logging/checkpointing.
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

# Set up logger
logger = logging.getLogger(__name__)

class SuperAI:
    """
    Main controller class for the Super AI system.
    
    Orchestrates the five phases of the daily cycle and manages continuous operation.
    """
    
    def __init__(self, config_path: str = None, config: Dict[str, Any] = None, debug: bool = False):
        """
        Initialize the Super AI system.
        
        Args:
            config_path: Path to the YAML configuration file
            config: Configuration dictionary (overrides config_path if provided)
        """
        self.start_time = datetime.now()
        self.debug = debug
        logger.info(f"=== SUPER AI INITIALIZATION STARTED {'(DEBUG MODE)' if debug else ''} ===")
        
        # Load configuration
        if config:
            # Ensure debug mode is set in config
            config['debug'] = debug
            self.config = config
        else:
            self.config = self._load_config(config_path or "config/main.yaml")
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Super AI initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}")
                logger.info("Creating default configuration")
                default_config = self._create_default_config(config_file)
            default_config['debug'] = debug
            return default_config
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            logger.info(f"Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            logger.info("Creating default configuration")
            return self._create_default_config()
    
    def _create_default_config(self, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Create a default configuration.
        
        Args:
            config_file: Path to save the default configuration
            
        Returns:
            Default configuration dictionary
        """
        default_config = {
            "system": {
                "name": "Super AI",
                "version": "1.0.0"
            },
            "data_collection": {
                "domains": ["forex", "sports", "stocks"],
                "frequency": "daily"
            },
            "model_fine_tuning": {
                "models": ["forex", "betting", "stock"],
                "epochs": 10,
                "batch_size": 64
            },
            "self_modification": {
                "modules": ["super_ai/prediction_engines/forex_predictor.py"],
                "max_attempts": 5
            },
            "browser_automation": {
                "tasks": ["data_collection", "test_solving"],
                "headless": True
            },
            "error_handling": {
                "auto_rollback": True,
                "max_retries": 3
            }
        }
        
        # Save default config if path provided
        if config_file:
            os.makedirs(config_file.parent, exist_ok=True)
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f)
            logger.info(f"Default configuration saved to {config_file}")
            
        return default_config
    
    def _init_components(self):
        """Initialize the required components for each phase."""
        try:
            # Import components lazily to avoid circular imports
            from super_ai.web_scraper.data_collector import DataCollector
            from super_ai.nlp_processor.llm_orchestrator import LLMOrchestrator
            from super_ai.self_modification.code_generator import CodeGenerator
            from super_ai.self_modification.module_benchmark import ModuleBenchmark
            from super_ai.self_modification.rollback_manager import RollbackManager
            from super_ai.self_modification.changelog_tracker import ChangelogTracker
            from super_ai.automation.browser_operator import BrowserOperator
            
            # Initialize components
            self.data_collector = DataCollector()
            self.llm_orchestrator = LLMOrchestrator()
            self.code_generator = CodeGenerator()
            self.module_benchmark = ModuleBenchmark()
            self.rollback_manager = RollbackManager()
            self.changelog_tracker = ChangelogTracker()
            self.browser_operator = BrowserOperator()
            
            logger.info("All components initialized successfully")
            
        except ImportError as e:
            # Handle missing modules gracefully for easier debugging
            logger.warning(f"Component initialization error: {str(e)}")
            logger.info("The system will try to operate with available components only")
    
    def run_daily_cycle(self):
        """Run a complete daily cycle with all five phases."""
        logger.info("Starting daily operational cycle")
        
        # Morning: Data Collection
        self._run_data_collection()
        
        # Midday: Model Fine-Tuning
        self._run_model_fine_tuning()
        
        # Afternoon: Self-Modification
        self._run_self_modification()
        
        # Evening: Browser Automation
        self._run_browser_automation()
        
        # Night: Logging and Checkpointing
        self._run_logging_and_checkpointing()
        
        logger.info("Daily operational cycle completed")
        
    def run_single_phase(self, phase: str):
        """
        Run a single phase of the daily cycle.
        
        Args:
            phase: Phase to run (data_collection, model_fine_tuning, self_modification, 
                   browser_automation, logging_checkpointing)
        """
        logger.info(f"Running single phase: {phase}")
        
        try:
            if phase == "data_collection":
                self._run_data_collection()
            elif phase == "model_fine_tuning":
                self._run_model_fine_tuning()
            elif phase == "self_modification":
                self._run_self_modification()
            elif phase == "browser_automation":
                self._run_browser_automation()
            elif phase == "logging_checkpointing":
                self._run_logging_and_checkpointing()
            else:
                logger.error(f"Unknown phase: {phase}")
                
        except Exception as e:
            logger.error(f"Error in phase {phase}: {str(e)}")
            self._handle_cycle_error(phase, e)
            
        logger.info(f"Phase {phase} completed")
        
    def run_continuous(self, interval_hours: int = 24):
        """
        Run the system continuously with the specified interval.
        
        Args:
            interval_hours: Interval between cycles in hours
        """
        logger.info(f"Starting continuous operation with {interval_hours}-hour interval")
        
        try:
            while True:
                cycle_start = datetime.now()
                logger.info(f"Starting cycle at {cycle_start}")
                
                # Run the complete daily cycle
                self.run_daily_cycle()
                
                # Calculate the wait time until the next cycle
                cycle_end = datetime.now()
                cycle_duration = (cycle_end - cycle_start).total_seconds()
                wait_seconds = max(0, interval_hours * 3600 - cycle_duration)
                
                if wait_seconds > 0:
                    logger.info(f"Waiting {wait_seconds / 3600:.2f} hours until next cycle")
                    time.sleep(wait_seconds)
                    
        except KeyboardInterrupt:
            logger.info("Continuous operation stopped by user")
            self._handle_shutdown()
            
    def is_debug_mode(self) -> bool:
        """Check if the system is running in debug mode."""
        return self.debug or self.config.get('debug', False)
        
    def set_debug_mode(self, enabled: bool = False) -> None:
        """Enable or disable debug mode.
        
        Args:
            enabled: If True, enables debug mode. If False, disables it.
        """
        self.debug = enabled
        self.config['debug'] = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")


    def _run_data_collection(self):
        """Morning phase: Collect data from various web sources."""
        logger.info(f"Starting data collection phase {'(DEBUG MODE)' if self.is_debug_mode() else ''}")
        
        try:
            domains = self.config.get("data_collection", {}).get("domains", [])
            
            if self.is_debug_mode():
                logger.info(f"[DEBUG MODE] Skipping actual data collection")
                for domain in domains:
                    logger.info(f"[DEBUG MODE] Simulating data collection for domain: {domain}")
                logger.info(f"[DEBUG MODE] Data collection phase completed (simulated)")
                return

            if not domains:
                logger.warning("No domains specified for data collection")
                return
                
            for domain in domains:
                logger.info(f"Collecting data for domain: {domain}")
                
                try:
                    # Use the data collector to fetch data
                    if hasattr(self, 'data_collector'):
                        result = self.data_collector.collect_data(domain)
                        logger.info(f"Data collection for {domain} completed: {result}")
                    else:
                        logger.warning("DataCollector not initialized, skipping actual collection")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {domain}: {str(e)}")
                    
            logger.info("Data collection phase completed")
            
        except Exception as e:
            logger.error(f"Error in data collection phase: {str(e)}")
            self._handle_cycle_error("data_collection", e)
            
    def _run_model_fine_tuning(self):
        """Midday phase: Fine-tune models with collected data."""
        logger.info(f"Starting model fine-tuning phase {'(DEBUG MODE)' if self.is_debug_mode() else ''}")
        
        if self.is_debug_mode():
            logger.info(f"[DEBUG MODE] Skipping actual model fine-tuning")
            models = self.config.get("model_fine_tuning", {}).get("models", [])
            for model in models:
                logger.info(f"[DEBUG MODE] Simulating fine-tuning for model: {model}")
            logger.info(f"[DEBUG MODE] Model fine-tuning phase completed (simulated)")
            return
            
        try:
            logger.info("Starting model fine-tuning process")
            models = self.config.get("model_fine_tuning", {}).get("models", [])
            if not models:
                logger.warning("No models specified for fine-tuning")
                return
                
            for model in models:
                logger.info(f"Fine-tuning model: {model}")
                
                try:
                    # Use the LLM orchestrator to fine-tune models
                    if hasattr(self, 'llm_orchestrator'):
                        result = self.llm_orchestrator.fine_tune(model)
                        logger.info(f"Fine-tuning for {model} completed: {result}")
                    else:
                        logger.warning("LLMOrchestrator not initialized, skipping actual fine-tuning")
                        
                except Exception as e:
                    logger.error(f"Error fine-tuning model {model}: {str(e)}")
                    
            logger.info("Model fine-tuning phase completed")
            
        except Exception as e:
            logger.error(f"Error in model fine-tuning phase: {str(e)}")
            self._handle_cycle_error("model_fine_tuning", e)
            
    def _run_self_modification(self):
        """Afternoon phase: Modify and improve the system's code."""
        logger.info(f"Starting self-modification phase {'(DEBUG MODE)' if self.is_debug_mode() else ''}")
        
        if self.is_debug_mode():
            logger.info(f"[DEBUG MODE] Skipping actual self-modification")
            modules = self.config.get("self_modification", {}).get("modules", [])
            for module_path in modules:
                logger.info(f"[DEBUG MODE] Simulating code improvement for module: {module_path}")
            logger.info(f"[DEBUG MODE] Self-modification phase completed (simulated)")
            return
            
        try:
            logger.info("Starting self-modification process")
            modules = self.config.get("self_modification", {}).get("modules", [])
            if not modules:
                logger.warning("No modules specified for self-modification")
                return
                
            for module_path in modules:
                logger.info(f"Analyzing module for improvements: {module_path}")
                
                try:
                    if not hasattr(self, 'code_generator') or not hasattr(self, 'module_benchmark') or not hasattr(self, 'rollback_manager'):
                        logger.warning("Required components not initialized, skipping actual self-modification")
                        continue
                        
                    # Create a checkpoint before modification
                    checkpoint_path = self.rollback_manager.create_checkpoint(module_path)
                    
                    # Generate improved code
                    improved_code = self.code_generator.generate_improvement(module_path)
                    
                    # Apply improvements temporarily for testing
                    temp_path = f"{module_path}.new"
                    with open(temp_path, 'w') as f:
                        f.write(improved_code)
                        
                    # Benchmark the improvements
                    is_improvement, result = self.module_benchmark.test_module(temp_path)
                    
                    if is_improvement:
                        # Apply improvements permanently
                        os.replace(temp_path, module_path)
                        logger.info(f"Module {module_path} successfully improved: {result}")
                        
                        # Log the changes
                        self.changelog_tracker.add_entry(module_path, result)
                    else:
                        # Discard improvements
                        os.remove(temp_path)
                        logger.info(f"Generated code for {module_path} did not improve performance, discarded")
                        
                except Exception as e:
                    logger.error(f"Error modifying module {module_path}: {str(e)}")
                    
                    # Rollback if auto-rollback is enabled
                    if self.config.get("error_handling", {}).get("auto_rollback", True):
                        if hasattr(self, 'rollback_manager'):
                            self.rollback_manager.restore_checkpoint(module_path, checkpoint_path)
                            logger.info(f"Rolled back changes to {module_path}")
                            
            logger.info("Self-modification phase completed")
            
        except Exception as e:
            logger.error(f"Error in self-modification phase: {str(e)}")
            self._handle_cycle_error("self_modification", e)
            
    def _run_browser_automation(self):
        """Evening phase: Automate browser tasks."""
        logger.info(f"Starting browser automation phase {'(DEBUG MODE)' if self.is_debug_mode() else ''}")
        
        if self.is_debug_mode():
            logger.info(f"[DEBUG MODE] Skipping actual browser automation")
            tasks = self.config.get("browser_automation", {}).get("tasks", [])
            for task in tasks:
                logger.info(f"[DEBUG MODE] Simulating browser task: {task}")
            logger.info(f"[DEBUG MODE] Browser automation phase completed (simulated)")
            return
            
        try:
            logger.info("Starting browser automation process")
            tasks = self.config.get("browser_automation", {}).get("tasks", [])
            if not tasks:
                logger.warning("No tasks specified for browser automation")
                return
                
            for task in tasks:
                logger.info(f"Running browser task: {task}")
                
                try:
                    # Use the browser operator to run tasks
                    if hasattr(self, 'browser_operator'):
                        result = self.browser_operator.run_task(task)
                        logger.info(f"Browser task {task} completed: {result}")
                    else:
                        logger.warning("BrowserOperator not initialized, skipping actual task execution")
                        
                except Exception as e:
                    logger.error(f"Error running browser task {task}: {str(e)}")
                    
            logger.info("Browser automation phase completed")
            
        except Exception as e:
            logger.error(f"Error in browser automation phase: {str(e)}")
            self._handle_cycle_error("browser_automation", e)
            
    def _run_logging_and_checkpointing(self):
        """Night phase: Log system state and create checkpoints."""
        logger.info(f"Starting logging and checkpointing phase {'(DEBUG MODE)' if self.is_debug_mode() else ''}")
        
        if self.is_debug_mode():
            logger.info(f"[DEBUG MODE] Simulating checkpointing and logging")
            modules = self.config.get("self_modification", {}).get("modules", [])
            for module_path in modules:
                logger.info(f"[DEBUG MODE] Simulating checkpoint creation for: {module_path}")
            logger.info(f"[DEBUG MODE] Logging and checkpointing phase completed (simulated)")
            return
            
        try:
            logger.info("Starting logging and checkpointing process")
            # Create checkpoints for key modules
            if hasattr(self, 'rollback_manager'):
                modules = self.config.get("self_modification", {}).get("modules", [])
                for module_path in modules:
                    self.rollback_manager.create_checkpoint(module_path)
                logger.info(f"Created checkpoints for {len(modules)} modules")
                
            # Generate and save changelog
            if hasattr(self, 'changelog_tracker'):
                changelog_path = self.changelog_tracker.generate_changelog()
                logger.info(f"Generated changelog at {changelog_path}")
                
            # Log module performance leaderboard
            if hasattr(self, 'module_benchmark'):
                leaderboard_path = self.module_benchmark.save_leaderboard()
                logger.info(f"Updated performance leaderboard at {leaderboard_path}")
                
            logger.info("Logging and checkpointing phase completed")
            
        except Exception as e:
            logger.error(f"Error in logging and checkpointing phase: {str(e)}")
            self._handle_cycle_error("logging_checkpointing", e)
            
    def _handle_cycle_error(self, phase: str, error: Exception):
        """
        Handle errors that occur during the daily cycle.
        
        Args:
            phase: The phase where the error occurred
            error: The exception that was raised
        """
        logger.error(f"Cycle error in phase {phase}: {str(error)}")
        
        # Add additional error handling logic here
        # For example, sending notifications, attempting recovery, etc.
        
    def _handle_shutdown(self):
        """Handle graceful shutdown of the system."""
        logger.info("Shutting down Super AI system")
        
        # Perform final checkpointing
        self._run_logging_and_checkpointing()
        
        # Add additional shutdown logic here
        # For example, closing connections, saving state, etc.
        
        logger.info("Shutdown complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run the system
    system = SuperAI()
    system.run_daily_cycle()
