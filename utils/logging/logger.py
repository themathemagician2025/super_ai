# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Logging Module

This module configures logging for the Super AI system.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import traceback

# Import configuration
from super_ai.utils.config import settings

# Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "logs")

class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the log record."""

    def __init__(self, **kwargs):
        """Initialize the formatter with the specified JSON attributes."""
        self.json_attributes = kwargs
        super().__init__()

    def format(self, record):
        """Format the record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add custom fields from the record attributes
        for key, value in self.json_attributes.items():
            if key in record.__dict__:
                log_data[value] = record.__dict__[key]

        return json.dumps(log_data)

def setup_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    json_format: bool = False,
    rotate_size: Optional[int] = None,
    rotate_time: Optional[str] = None,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.

    Args:
        name: Logger name (default: super_ai)
        level: Logging level (default from config or INFO)
        log_file: Path to log file (default from config or logs/super_ai.log)
        log_format: Log format string (default from config or standard format)
        json_format: Whether to use JSON formatting (default: False)
        rotate_size: Size in bytes to rotate log file (default: None)
        rotate_time: When to rotate logs ('midnight', 'h' for hourly, etc.) (default: None)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured logger
    """
    # Get configuration instance
    config = settings.get_config()

    # Set up parameters with defaults
    name = name or "super_ai"
    level_str = level or config.get("logging", "level") or DEFAULT_LOG_LEVEL
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Get log file path
    log_file = log_file or config.get("logging", "file")
    if not log_file:
        log_dir = config.get("logging", "directory") or DEFAULT_LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")

    # Get log format
    log_format = log_format or config.get("logging", "format") or DEFAULT_LOG_FORMAT
    date_format = config.get("logging", "date_format") or DEFAULT_DATE_FORMAT

    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Choose handler type based on rotation settings
        if rotate_size:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=rotate_size,
                backupCount=backup_count
            )
        elif rotate_time:
            file_handler = TimedRotatingFileHandler(
                log_file,
                when=rotate_time,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)

        # Set formatter
        if json_format:
            file_handler.setFormatter(JsonFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(log_format, date_format))

        logger.addHandler(file_handler)

    return logger

# Create a default logger for the package
default_logger = setup_logger()

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (if None, returns default logger)

    Returns:
        Logger instance
    """
    if name is None:
        return default_logger

    # Create child logger of default logger
    return logging.getLogger(f"super_ai.{name}")
