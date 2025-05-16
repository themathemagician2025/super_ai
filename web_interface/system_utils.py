# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
System utilities for error handling, logging, and recovery mechanisms
"""

import os
import sys
import time
import json
import signal
import logging
import traceback
import functools
import threading
from datetime import datetime
import pickle
from typing import Any, Dict, Callable, TypeVar, Optional, Union

# Setup logging directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, f"super_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a special logger for tool and decision tracking
decision_logger = logging.getLogger("decision_tracker")
decision_logger.setLevel(logging.INFO)
decision_handler = logging.FileHandler(os.path.join(LOG_DIR, f"decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
decision_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
decision_logger.addHandler(decision_handler)

# For recovery - store recent user queries and state
RECOVERY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recovery")
os.makedirs(RECOVERY_DIR, exist_ok=True)
LAST_QUERY_FILE = os.path.join(RECOVERY_DIR, "last_query.pickle")

# Type var for function decorators
F = TypeVar('F', bound=Callable[..., Any])

class TimeoutError(Exception):
    """Exception raised when a function times out"""
    pass

def timeout(seconds: int) -> Callable[[F], F]:
    """
    Decorator to add timeout to a function

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorated function with timeout capability
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True

            thread.start()
            thread.join(seconds)

            if isinstance(result[0], Exception):
                raise result[0]

            return result[0]

        return wrapper  # type: ignore

    return decorator

def fallback(default_return: Any = None, log_exception: bool = True) -> Callable[[F], F]:
    """
    Decorator to add fallback to a function

    Args:
        default_return: Return value if function fails
        log_exception: Whether to log the exception

    Returns:
        Decorated function with fallback capability
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exception:
                    logging.exception(f"Function {func.__name__} failed: {str(e)}")
                return default_return

        return wrapper  # type: ignore

    return decorator

def log_decision(decision: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log a decision to the decision tracker

    Args:
        decision: Decision description
        metadata: Additional metadata
    """
    msg = f"DECISION: {decision}"
    if metadata:
        msg += f" - META: {json.dumps(metadata)}"
    decision_logger.info(msg)

def log_tool_call(tool_name: str, args: Dict[str, Any], result: Any) -> None:
    """
    Log a tool call to the decision tracker

    Args:
        tool_name: Name of the tool being called
        args: Arguments to the tool
        result: Result of the tool call (or exception)
    """
    # Truncate result if too large
    result_str = str(result)
    if len(result_str) > 500:
        result_str = result_str[:500] + "..."

    decision_logger.info(
        f"TOOL CALL: {tool_name} - ARGS: {json.dumps(args)} - RESULT: {result_str}"
    )

def save_user_query(user_query: str, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Save the latest user query for recovery purposes

    Args:
        user_query: User's query string
        context: Additional context to restore
    """
    data = {
        "query": user_query,
        "timestamp": datetime.now().isoformat(),
        "context": context or {},
    }

    with open(LAST_QUERY_FILE, 'wb') as f:
        pickle.dump(data, f)

    logging.info(f"Saved user query for recovery: {user_query[:50]}...")

def get_last_user_query() -> Optional[Dict[str, Any]]:
    """
    Get the last user query for recovery

    Returns:
        Dict with query and context or None if no saved query
    """
    if not os.path.exists(LAST_QUERY_FILE):
        return None

    try:
        with open(LAST_QUERY_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load last query: {str(e)}")
        return None

class RequestTracker:
    """Track request processing for monitoring and auto-recovery"""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.current_request: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.pid = os.getpid()

    def start_request(self, request_id: str, request_data: Any) -> None:
        """Start tracking a request"""
        self.current_request = {
            "id": request_id,
            "data": request_data,
            "start_time": datetime.now().isoformat()
        }
        self.start_time = time.time()
        logging.info(f"Started request {request_id} in {self.app_name}")

    def end_request(self, request_id: str, status: str = "success") -> None:
        """End tracking a request"""
        if self.start_time:
            duration = time.time() - self.start_time
            logging.info(f"Completed request {request_id} in {self.app_name} ({status}) in {duration:.2f}s")
        self.current_request = {}
        self.start_time = None

    def get_current_request(self) -> Dict[str, Any]:
        """Get information about the current request"""
        return self.current_request

def setup_crash_recovery(app_name: str, restart_cmd: str) -> None:
    """
    Set up crash recovery handlers

    Args:
        app_name: Name of the application
        restart_cmd: Command to run to restart the application
    """
    def handle_crash(sig, frame):
        logging.critical(f"{app_name} crashed with signal {sig}!")

        # Save crash info
        crash_file = os.path.join(RECOVERY_DIR, f"crash_{app_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        with open(crash_file, 'w') as f:
            f.write(f"Crash at {datetime.now().isoformat()}\n")
            f.write(f"Signal: {sig}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write("\nTraceback:\n")
            traceback.print_stack(frame, file=f)

        # Attempt to restart
        try:
            logging.info(f"Attempting to restart {app_name} with: {restart_cmd}")
            os.system(restart_cmd)
        except Exception as e:
            logging.error(f"Failed to restart {app_name}: {str(e)}")

        # Exit
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_crash)
    signal.signal(signal.SIGSEGV, handle_crash)
    signal.signal(signal.SIGABRT, handle_crash)
