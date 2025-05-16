# Web Interface and Server Setup

## Overview

The Super AI Web Interface provides a web-based frontend and API for accessing the prediction models and system functionality. It includes:

- A Flask-based web server
- API endpoints for sports and betting predictions
- System status monitoring
- Debug interface
- Module execution capabilities

## Server Components

### 1. Web Server (`server.py`)

The core server component handles:
- Loading prediction models
- Managing API endpoints
- Routing requests
- Module execution tracking

### 2. Templates

- `index.html`: Main landing page for users
- `debug.html`: Debug interface for developers and system administrators

### 3. Run Server (`run_server.py`)

The main entry point for starting the server:
- Handles command-line arguments
- Loads configuration
- Initializes components
- Starts the web server

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the main index page |
| `/debug` | GET | Serves the debug interface |
| `/api/health` | GET | Returns health status of the system |
| `/api/status` | GET | Returns detailed system status |
| `/api/predictions/sports` | POST | Make a sports prediction |
| `/api/predictions/betting` | POST | Make a betting prediction |
| `/api/modules/execute` | POST | Execute a specific module |

## Usage

### Starting the Server

```bash
# Basic usage
python run_server.py

# Specify host and port
python run_server.py --host 0.0.0.0 --port 8000

# Enable debug mode
python run_server.py --debug

# Specify model paths
python run_server.py --sports-model models/custom_sports.pth --betting-model models/custom_betting.pth

# Load configuration from .env file
python run_server.py --config custom.env
```

### Configuration Options

The server can be configured through:
1. Command-line arguments
2. Environment variables
3. `.env` file

#### Available Configuration Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| Host | `API_HOST` | 127.0.0.1 | Host address to bind to |
| Port | `API_PORT` | 5000 | Port to listen on |
| Debug Mode | `API_DEBUG` | False | Enable debug mode |
| Log Level | `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| Sports Model | `SPORTS_MODEL_PATH` | models/sports_predictor.pth | Path to sports model |
| Betting Model | `BETTING_MODEL_PATH` | models/betting_predictor.pth | Path to betting model |
| Enable Dangerous AI | `ENABLE_DANGEROUS_AI` | False | Enable dangerous AI operations |

### API Usage Examples

#### Sports Prediction

```bash
curl -X POST http://localhost:5000/api/predictions/sports \
  -H "Content-Type: application/json" \
  -d '{"features": [0.2, 0.5, 0.3, 0.8, 0.7, 0.9, 0.1, 0.4]}'
```

Response:
```json
{
  "status": "success",
  "prediction": 0.75,
  "home_win_probability": 0.75,
  "away_win_probability": 0.25
}
```

#### Betting Prediction

```bash
curl -X POST http://localhost:5000/api/predictions/betting \
  -H "Content-Type: application/json" \
  -d '{"features": [0.2, 0.5, 0.3, 0.8, 0.7, 0.9, 0.1, 0.4, 0.6, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.2, 0.1, 0.9], "stake": 100, "odds": 2.5}'
```

Response:
```json
{
  "status": "success",
  "prediction": 0.65,
  "expected_return": 162.5,
  "expected_value": 1.625,
  "recommended_stake": 100
}
```

#### Execute Module

```bash
curl -X POST http://localhost:5000/api/modules/execute \
  -H "Content-Type: application/json" \
  -d '{"module_path": "prediction/sports_predictor.py", "function_name": "main"}'
```

Response:
```json
{
  "status": "started",
  "message": "Module execution started for prediction/sports_predictor.py.main"
}
```

## Debug Interface

The debug interface provides a web-based UI for:

1. Monitoring system status
2. Testing prediction API endpoints
3. Executing modules
4. Viewing system logs

Access the debug interface at: `http://localhost:5000/debug`

## Security Notes

- The web interface binds to localhost (127.0.0.1) by default for security
- To expose the server externally, specify `--host 0.0.0.0` (use with caution)
- The server does not implement authentication by default
- The `ENABLE_DANGEROUS_AI` setting restricts certain operations

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Port already in use**: Change the port with `--port` or check for other running processes
3. **Model not found**: Verify model paths and run training scripts if needed
4. **Permission denied**: Ensure the user has permission to read model files and write logs

### Checking Logs

Server logs are output to the console by default. For more detailed logging, set the log level:

```bash
python run_server.py --debug
```

Or set in configuration:
```
LOG_LEVEL=DEBUG
```
