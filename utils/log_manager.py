# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import os
import sys
import logging
import threading
from typing import Dict, Optional
from datetime import datetime
from functools import wraps
import time

class LogManager:
    """
    Centralized logging manager for the application.
    Sets up various loggers for different purposes (main, error, performance, access)
    and provides methods for logging errors, performance metrics, and access information.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the LogManager with the specified log directory.

        Args:
            log_dir: Directory to store log files. If None, uses 'logs' in the current directory.
        """
        self.log_dir = log_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup loggers
        self.main_logger = self._setup_logger('main', os.path.join(self.log_dir, 'main.log'))
        self.error_logger = self._setup_logger('error', os.path.join(self.log_dir, 'error.log'))
        self.performance_logger = self._setup_logger('performance', os.path.join(self.log_dir, 'performance.log'))
        self.access_logger = self._setup_logger('access', os.path.join(self.log_dir, 'access.log'))

    def setup_logging(self, level: int = logging.INFO, log_to_console: bool = True, log_file: Optional[str] = None) -> None:
        """
        Set up the root logger with the specified level, console output, and log file.

        Args:
            level: Logging level (default: INFO)
            log_to_console: Whether to log to console (default: True)
            log_file: Path to the log file (default: None, which uses main.log)
        """
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # Add file handler if log_file is specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # Log initial message
        root_logger.info(f"Logging initialized at level {logging.getLevelName(level)}")
        if log_file:
            root_logger.info(f"Log file: {log_file}")

    def _setup_logger(self, name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
        """
        Set up a logger with the specified name, log file, and level.

        Args:
            name: Name of the logger
            log_file: Path to the log file
            level: Logging level (default: INFO)

        Returns:
            Configured logger instance
        """
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicate logging
        if logger.handlers:
            logger.handlers = []

        logger.addHandler(handler)

        # Add console handler for main logger
        if name == 'main':
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        self.main_logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.main_logger.warning(message)

    def log_error(self, message: str, exc_info=None) -> None:
        """
        Log an error message and optionally the exception info.

        Args:
            message: Error message
            exc_info: Exception information (default: None)
        """
        self.error_logger.error(message, exc_info=exc_info)
        self.main_logger.error(message)

    def log_performance(self, component: str, operation: str, duration: float) -> None:
        """
        Log performance metrics.

        Args:
            component: Component being measured
            operation: Operation being performed
            duration: Duration in seconds
        """
        self.performance_logger.info(f"{component} - {operation}: {duration:.4f}s")

    def log_access(self, endpoint: str, method: str, ip: str, status_code: int, duration: float) -> None:
        """
        Log API access information.

        Args:
            endpoint: API endpoint accessed
            method: HTTP method used
            ip: IP address of the client
            status_code: HTTP status code returned
            duration: Request duration in seconds
        """
        self.access_logger.info(f"{method} {endpoint} from {ip} - {status_code} - {duration:.4f}s")

def log_execution_time(logger=None):
    """
    Decorator to log the execution time of a function.

    Args:
        logger: Logger to use (default: None, which uses the default LogManager)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            duration = end_time - start_time
            log_manager = logger or ThreadSafeLogManager()
            log_manager.log_performance(
                func.__module__,
                func.__name__,
                duration
            )

            return result
        return wrapper
    return decorator

class ThreadSafeLogManager(LogManager):
    """
    Thread-safe singleton implementation of LogManager.
    Ensures that only one instance of LogManager is created and used across threads.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ThreadSafeLogManager, cls).__new__(cls)
                cls._instance.__initialized = False
            return cls._instance

    def __init__(self, log_dir: Optional[str] = None):
        with self._lock:
            if not getattr(self, "__initialized", False):
                super(ThreadSafeLogManager, self).__init__(log_dir)
                self.__initialized = True
