# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Check if Super AI services are running with robust error handling and logging
"""

import os
import sys
import requests
import json
import argparse
from datetime import datetime
import traceback

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our utilities
from web_interface.system_utils import (
    timeout,
    fallback,
    log_decision,
    log_tool_call
)

@timeout(5)  # 5 second timeout for service checks
@fallback(default_return=(False, "Check timed out or failed"))
def check_service(url, name):
    """
    Check if a service is running with timeout and fallback

    Args:
        url: Service URL to check
        name: Name of the service

    Returns:
        tuple: (success, message)
    """
    try:
        # Log this tool call
        log_decision(f"Checking service: {name}", {"url": url})

        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            message = f"✅ {name} is running at {url}\n   Response: {json.dumps(response.json(), indent=2)[:80]}..."
            print(message)

            # Log the successful check
            log_tool_call(
                "service_check",
                {"service": name, "url": url},
                f"Status: {response.status_code}, Response: {str(response.json())[:100]}..."
            )

            return True, message
        else:
            message = f"❌ {name} returned status code {response.status_code}"
            print(message)

            # Log the failed check
            log_tool_call(
                "service_check",
                {"service": name, "url": url},
                f"Failed with status: {response.status_code}"
            )

            return False, message
    except Exception as e:
        message = f"❌ {name} is not running: {str(e)}"
        print(message)

        # Log the exception
        log_tool_call(
            "service_check",
            {"service": name, "url": url},
            f"Exception: {str(e)}"
        )

        return False, message

def check_all_services():
    """
    Check all services and return status

    Returns:
        dict: Service status information
    """
    services = {
        "main_web": {
            "name": "Main Web Interface",
            "url": "http://localhost:5000/api/health",
        },
        "forex_api": {
            "name": "Forex API Service",
            "url": "http://localhost:5001/api/health",
        },
    }

    results = {}
    all_okay = True

    # Track the overall operation
    log_decision("Starting comprehensive service check", {
        "timestamp": datetime.now().isoformat(),
        "services": list(services.keys())
    })

    for service_id, service in services.items():
        success, message = check_service(service["url"], service["name"])
        results[service_id] = {
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if not success:
            all_okay = False

    # Log the final result
    log_decision("Service check completed", {
        "all_okay": all_okay,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "all_okay": all_okay,
        "services": results,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Super AI System Services")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    print(f"Checking Super AI services at {datetime.now().isoformat()}...\n")

    try:
        results = check_all_services()

        if args.json:
            print(json.dumps(results, indent=2))
        else:
    print("\nService check complete.")
            if results["all_okay"]:
                print("✅ All services are running correctly.")
            else:
                print("❌ Some services have issues. See details above.")

        # Exit with appropriate code
        sys.exit(0 if results["all_okay"] else 1)

    except Exception as e:
        print(f"\n❌ Error during service check: {str(e)}")
        traceback.print_exc()
        sys.exit(2)
