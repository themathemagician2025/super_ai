# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Phase Scheduler

This module provides time-based scheduling of different processing phases
for the Super AI system's daily operations cycle.
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Callable, List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PhaseScheduler:
    """Manages scheduled execution of system phases based on time."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the phase scheduler with configuration.
        
        Args:
            config: Configuration dictionary with schedule settings
        """
        self.config = config or {}
        self.default_schedule = [
            ("data_collection", "08:00"),    # Morning
            ("model_fine_tuning", "12:00"),  # Midday
            ("self_modification", "15:00"),  # Afternoon
            ("browser_automation", "18:00"), # Evening
            ("logging_checkpointing", "22:00") # Night
        ]
        
        # Get schedule from config or use default
        self.schedule = self.config.get("schedule", self.default_schedule)
        logger.info(f"Phase scheduler initialized with {len(self.schedule)} phases")
        
    def get_next_phase_time(self) -> Tuple[str, str]:
        """
        Get the next phase and its scheduled time based on current time.
        
        Returns:
            Tuple containing (phase_name, scheduled_time)
        """
        current_time = datetime.now().strftime("%H:%M")
        
        # Find the next phase that hasn't run yet today
        for phase_name, scheduled_time in self.schedule:
            if current_time < scheduled_time:
                return phase_name, scheduled_time
                
        # If all phases for today have run, return the first phase for tomorrow
        return self.schedule[0]
    
    async def wait_until(self, target_time: str) -> None:
        """
        Wait until the specified target time.
        
        Args:
            target_time: Time in format "HH:MM"
        """
        while datetime.now().strftime("%H:%M") < target_time:
            # Log every hour while waiting
            if datetime.now().strftime("%M") == "00":
                current = datetime.now().strftime("%H:%M")
                logger.info(f"Waiting for scheduled time {target_time}, current time: {current}")
            await asyncio.sleep(60)  # Check every minute
            
    def get_phase_info(self, phase_name: str) -> Dict[str, Any]:
        """
        Get configuration information for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            Dictionary with phase configuration
        """
        phase_config = self.config.get(phase_name, {})
        return {
            "name": phase_name,
            "enabled": phase_config.get("enabled", True),
            "max_runtime": phase_config.get("max_runtime", 3600),  # Default 1 hour
            "retry_count": phase_config.get("retry_count", 3),
            "params": phase_config.get("params", {})
        }
        
    async def run_scheduled_phases(self, phase_handlers: Dict[str, Callable]) -> None:
        """
        Run all phases according to their schedule.
        
        Args:
            phase_handlers: Dictionary mapping phase names to handler functions
        """
        logger.info("Starting scheduled phase execution")
        
        for phase_name, scheduled_time in self.schedule:
            # Get handler for this phase
            handler = phase_handlers.get(phase_name)
            if not handler:
                logger.warning(f"No handler defined for phase '{phase_name}', skipping")
                continue
                
            # Get phase configuration
            phase_info = self.get_phase_info(phase_name)
            if not phase_info.get("enabled", True):
                logger.info(f"Phase '{phase_name}' is disabled, skipping")
                continue
                
            # Wait until scheduled time
            logger.info(f"Waiting for phase '{phase_name}' scheduled at {scheduled_time}")
            await self.wait_until(scheduled_time)
            
            # Execute the phase
            logger.info(f"Starting phase: {phase_name}")
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**phase_info.get("params", {}))
                else:
                    result = handler(**phase_info.get("params", {}))
                    
                duration = time.time() - start_time
                logger.info(f"Completed phase '{phase_name}' in {duration:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error in phase '{phase_name}': {str(e)}", exc_info=True)

# Sample usage
async def run_phases_example():
    scheduler = PhaseScheduler()
    
    # Define phase handlers
    async def handle_data_collection(**kwargs):
        logger.info(f"Collecting data with params: {kwargs}")
        await asyncio.sleep(5)  # Simulate work
        return {"records_collected": 1250}
        
    # Map phase names to handlers
    phase_handlers = {
        "data_collection": handle_data_collection,
        # Add other phase handlers here
    }
    
    # Run the scheduled phases
    await scheduler.run_scheduled_phases(phase_handlers)
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(run_phases_example())
