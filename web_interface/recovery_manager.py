# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Recovery Manager for Super AI System

This script handles automatic recovery after crashes by:
1. Checking for crashes
2. Restarting services as needed
3. Reprocessing the last user query if available
"""

import os
import sys
import time
import logging
import subprocess
import argparse
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Optional, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our utilities
from web_interface.system_utils import (
    get_last_user_query,
    LOG_DIR,
    RECOVERY_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

# Services to monitor and recover
SERVICES = {
    "main_web": {
        "name": "Main Web Interface",
        "url": "http://localhost:5000/api/health",
        "start_cmd": "python simple_server.py",
        "working_dir": os.path.join(current_dir),
    },
    "forex_api": {
        "name": "Forex API Service",
        "url": "http://localhost:5001/api/health",
        "start_cmd": "python simple_forex_server.py",
        "working_dir": os.path.join(current_dir),
    },
}

def check_service(service_id: str) -> bool:
    """
    Check if a service is running by querying its health endpoint

    Args:
        service_id: ID of the service to check

    Returns:
        bool: True if service is running, False otherwise
    """
    service = SERVICES.get(service_id)
    if not service:
        logging.error(f"Unknown service ID: {service_id}")
        return False

    try:
        response = requests.get(service["url"], timeout=2)
        if response.status_code == 200:
            logging.info(f"✅ {service['name']} is running at {service['url']}")
            return True
        else:
            logging.warning(f"❌ {service['name']} returned status code {response.status_code}")
            return False
    except Exception as e:
        logging.warning(f"❌ {service['name']} is not running: {str(e)}")
        return False

def restart_service(service_id: str) -> bool:
    """
    Restart a service

    Args:
        service_id: ID of the service to restart

    Returns:
        bool: True if service restarted successfully, False otherwise
    """
    service = SERVICES.get(service_id)
    if not service:
        logging.error(f"Unknown service ID: {service_id}")
        return False

    logging.info(f"Restarting {service['name']}...")

    try:
        # Start the process in a new terminal/command window
        if sys.platform == "win32":
            cmd = f'start cmd /k "cd {service["working_dir"]} && {service["start_cmd"]}"'
            subprocess.Popen(cmd, shell=True)
        else:
            # Linux/Mac approach
            cmd = f'cd {service["working_dir"]} && {service["start_cmd"]}'
            subprocess.Popen(cmd, shell=True, start_new_session=True)

        # Wait for service to come online
        for i in range(10):
            time.sleep(2)
            if check_service(service_id):
                logging.info(f"Service {service['name']} successfully restarted")
                return True

        logging.error(f"Service {service['name']} failed to restart within timeout")
        return False

    except Exception as e:
        logging.error(f"Failed to restart {service['name']}: {str(e)}")
        return False

def reprocess_last_query() -> None:
    """
    Reprocess the last user query if available
    """
    query_data = get_last_user_query()
    if not query_data:
        logging.warning("No last query found for recovery")
        return

    logging.info(f"Found last query from {query_data.get('timestamp')}")
    logging.info(f"Query: {query_data.get('query')[:100]}...")

    # In a real implementation, this would send the query to the processing engine
    # For now, we just log that we would process it
    logging.info("Would reprocess the last query (integration to be implemented)")

def check_and_recover() -> None:
    """
    Check all services and recover as needed
    """
    logging.info("Starting service check and recovery process")

    for service_id, service in SERVICES.items():
        if not check_service(service_id):
            logging.warning(f"Service {service['name']} is down, attempting to restart")
            if restart_service(service_id):
                logging.info(f"Successfully recovered {service['name']}")
            else:
                logging.error(f"Failed to recover {service['name']}")

    # After all services are up, reprocess the last query
    reprocess_last_query()

def run_recovery_monitor(interval: int = 60) -> None:
    """
    Run continuous monitoring and recovery

    Args:
        interval: Check interval in seconds
    """
    logging.info(f"Starting recovery monitor with {interval}s interval")

    try:
        while True:
            check_and_recover()
            logging.info(f"Sleeping for {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Recovery monitor stopped by user")
    except Exception as e:
        logging.error(f"Recovery monitor error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super AI System Recovery Manager")
    parser.add_argument(
        "--check", action="store_true",
        help="Check services and exit"
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Start continuous monitoring"
    )
    parser.add_argument(
        "--interval", type=int, default=60,
        help="Monitoring interval in seconds"
    )
    parser.add_argument(
        "--restart", type=str,
        help="Restart a specific service (main_web, forex_api)"
    )

    args = parser.parse_args()

    if args.restart:
        if restart_service(args.restart):
            print(f"Service {args.restart} restarted successfully")
        else:
            print(f"Failed to restart service {args.restart}")
            sys.exit(1)
    elif args.check:
        check_and_recover()
    elif args.monitor:
        run_recovery_monitor(args.interval)
    else:
        parser.print_help()
