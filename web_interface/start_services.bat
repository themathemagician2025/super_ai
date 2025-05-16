@echo off
echo Starting Super AI System...
cd %~dp0
python start_services.py %*
echo.
echo To check services status: python check_services.py
echo To stop services: Close each service window or press Ctrl+C in each window
