# Super AI System

## Overview
This system provides APIs for sports predictions, betting analysis, forex trading, and athlete performance metrics.

## Running the Application

### Main Web Interface
The main web interface serves the frontend and API endpoints for sports predictions, betting analysis, and athlete metrics.

```
cd Development/src/web_interface
python simple_server.py
```

This starts the server on http://localhost:5000

### Forex API Server
The Forex API server provides endpoints for forex market predictions, analysis, and backtesting.

```
cd Development/src/web_interface
python simple_forex_server.py
```

This starts the server on http://localhost:5001

## API Endpoints

### Main Web Interface (Port 5000)
- `GET /api/health` - Health check endpoint
- `GET /` - Main page
- `GET /api/test-timeout` - Test endpoint for timeout demonstration
- `GET /api/test-fallback` - Test endpoint for fallback demonstration

### Forex API (Port 5001)
- `GET /api/health` - Health check endpoint
- `POST /api/forex/predict` - Generate forex price predictions and trading signals
- `GET /api/forex/available_models` - Get a list of available trained forex models
- `GET /api/test-timeout` - Test endpoint for timeout demonstration
- `GET /api/test-crash` - Test endpoint for crash recovery demonstration

## Testing the APIs

### Check All Services
```
cd Development/src/web_interface
python check_services.py
```

For JSON output:
```
python check_services.py --json
```

### Test Forex Prediction API
```powershell
# PowerShell
$body = @{currency_pair='EUR/USD'; timeframe='1h'} | ConvertTo-Json
Invoke-WebRequest -Uri 'http://localhost:5001/api/forex/predict' -Method Post -Body $body -ContentType 'application/json' | Select-Object -ExpandProperty Content
```

### View Available Forex Models
```powershell
# PowerShell
Invoke-WebRequest -Uri 'http://localhost:5001/api/forex/available_models' -Method Get | Select-Object -ExpandProperty Content
```

## Error Handling & Recovery Features

### Robust Error Handling

The system now includes comprehensive error handling:

1. **Timeout Protection**: All API endpoints have configurable timeouts, preventing long-running operations from blocking the system.
2. **Fallback Responses**: If a service fails, it will return a sensible fallback response rather than crashing.
3. **Graceful Exception Handling**: All exceptions are caught, logged, and returned as structured error responses.

### Comprehensive Logging

A detailed logging system is now in place:

1. **Decision Logging**: All significant decisions are logged with timestamps and contextual metadata.
2. **Tool Call Tracking**: External service and tool calls are tracked with inputs and outputs.
3. **Request Tracking**: Every API request is tracked from start to finish with unique IDs.

Logs can be found in the `logs` directory with the following files:
- `super_ai_YYYYMMDD_HHMMSS.log` - General application logs
- `decisions_YYYYMMDD_HHMMSS.log` - Decision and tool call logs

### Auto-Restart & Recovery

The system now includes automatic crash recovery:

1. **Crash Detection**: Signal handlers detect crashes and trigger the recovery process.
2. **Service Recovery**: The recovery manager automatically restarts crashed services.
3. **Query Recovery**: The last user query is saved and can be replayed after a crash.

To manually trigger recovery:
```
python recovery_manager.py --restart main_web
python recovery_manager.py --restart forex_api
```

To run continuous monitoring:
```
python recovery_manager.py --monitor --interval 60
```

## Development Notes

If you need to modify app.py or another file with complex dependencies, make sure to:
1. Fix import paths (use relative imports instead of absolute ones)
2. Handle missing dependencies gracefully
3. Update any deprecated Flask decorators (`@app.before_first_request` → use `with app.app_context()` instead)

### Using the Error Handling Utilities

To add timeout protection to a function:
```python
from web_interface.system_utils import timeout

@timeout(5)  # 5 second timeout
def my_function():
    # Potentially slow operation
    pass
```

To add fallback to a function:
```python
from web_interface.system_utils import fallback

@fallback(default_return={"status": "error"})
def my_function():
    # Operation that might fail
    pass
```

## Next Steps
- Train real models for sports prediction and forex trading
- Set up a database for storing historical data
- Connect to real data sources for live predictions
- Extend recovery system to include state persistence
- Add health metrics dashboard to monitor system stability
