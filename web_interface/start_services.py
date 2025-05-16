# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Start all Super AI System services with recovery monitoring
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def start_service(service_id, service_name, command):
    """Start a service in a new process window"""
    print(f"Starting {service_name}...")

    try:
        if sys.platform == "win32":
            # Windows approach - open in new window
            cmd = f'start cmd /k "cd {current_dir} && {command}"'
            subprocess.Popen(cmd, shell=True)
        else:
            # Linux/Mac approach
            cmd = f'cd {current_dir} && {command}'
            subprocess.Popen(cmd, shell=True, start_new_session=True)

        print(f"✅ {service_name} startup initiated")
        return True

    except Exception as e:
        print(f"❌ Failed to start {service_name}: {str(e)}")
        return False

def main():
    """Start all services and optionally the recovery monitor"""
    parser = argparse.ArgumentParser(description="Start Super AI System Services")
    parser.add_argument(
        "--no-monitor", action="store_true",
        help="Do not start the recovery monitor"
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=60,
        help="Monitoring interval in seconds"
    )

    args = parser.parse_args()

    print(f"Starting Super AI System services at {datetime.now().isoformat()}\n")

    # Define services
    services = [
        {
            "id": "main_web",
            "name": "Main Web Interface",
            "command": "python simple_server.py"
        },
        {
            "id": "forex_api",
            "name": "Forex API Service",
            "command": "python simple_forex_server.py"
        }
    ]

    # Start each service
    started = []
    for service in services:
        if start_service(service["id"], service["name"], service["command"]):
            started.append(service["id"])

    # Wait for services to initialize
    if started:
        print(f"\nWaiting for services to initialize...")
        time.sleep(5)

        # Check services
        check_cmd = f"python {os.path.join(current_dir, 'check_services.py')}"
        subprocess.call(check_cmd, shell=True)

    # Start recovery monitor unless disabled
    if not args.no_monitor and started:
        print(f"\nStarting recovery monitor (interval: {args.monitor_interval}s)...")

        monitor_cmd = f"python {os.path.join(current_dir, 'recovery_manager.py')} --monitor --interval {args.monitor_interval}"

        if sys.platform == "win32":
            # Windows approach - open in new window
            cmd = f'start cmd /k "{monitor_cmd}"'
            subprocess.Popen(cmd, shell=True)
        else:
            # Linux/Mac approach
            subprocess.Popen(monitor_cmd, shell=True, start_new_session=True)

        print(f"✅ Recovery monitor started")

    print(f"\nSystem startup complete at {datetime.now().isoformat()}")
    print("Use Ctrl+C in each window to stop individual services")

if __name__ == "__main__":
    main()
