#!/bin/bash
echo "Starting Super AI System..."
cd "$(dirname "$0")"
python start_services.py "$@"
echo ""
echo "To check services status: python check_services.py"
echo "To stop services: Use 'pkill -f simple_server.py' and 'pkill -f simple_forex_server.py'"
