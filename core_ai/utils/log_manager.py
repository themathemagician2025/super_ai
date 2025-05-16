# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import traceback
import threading
from functools import wraps
import time

class LogManager:
    def __init__(self, log_dir: Path, app_name: str = "super_ai"):
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create different log files for different purposes
        self.main_logger = self._setup_logger('main', 'main.log')
        self.error_logger = self._setup_logger('error', 'error.log', level=logging.ERROR)
        self.performance_logger = self._setup_logger('performance', 'performance.log')
        self.access_logger = self._setup_logger('access', 'access.log')

    def _setup_logger(self, name: str, filename: str,
                     level: int = logging.INFO) -> logging.Logger:
        """Set up a logger with file and console handlers."""
        logger = logging.getLogger(f"{self.app_name}.{name}")
        logger.setLevel(level)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with full traceback and context."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }

        self.error_logger.error(json.dumps(error_info, indent=2))

    def log_performance(self, operation: str, duration: float, details: Dict[str, Any] = None):
        """Log performance metrics."""
        perf_info = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_ms': duration * 1000,
            'details': details or {}
        }

        self.performance_logger.info(json.dumps(perf_info))

    def log_access(self, endpoint: str, method: str, status_code: int,
                  response_time: float, user_info: Dict[str, Any] = None):
        """Log API access information."""
        access_info = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': response_time * 1000,
            'user_info': user_info or {}
        }

        self.access_logger.info(json.dumps(access_info))

    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """Get a specific logger by name."""
        return logging.getLogger(f"{self.app_name}.{name}")

def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if logger:
                    logger.info(f"{func.__name__} executed in {duration:.2f} seconds")
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger:
                    logger.error(f"{func.__name__} failed after {duration:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator

class ThreadSafeLogManager:
    """Thread-safe singleton log manager."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: Optional[Path] = None, app_name: str = "super_ai"):
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not hasattr(self, '_initialized'):
                    if log_dir is None:
                        log_dir = Path(__file__).parent.parent / 'logs'
                    self.log_manager = LogManager(log_dir, app_name)
                    self._initialized = True

    def get_logger(self, name: str) -> Optional[logging.Logger]:
        return self.log_manager.get_logger(name)

    def log_error(self, *args, **kwargs):
        return self.log_manager.log_error(*args, **kwargs)

    def log_performance(self, *args, **kwargs):
        return self.log_manager.log_performance(*args, **kwargs)

    def log_access(self, *args, **kwargs):
        return self.log_manager.log_access(*args, **kwargs)
