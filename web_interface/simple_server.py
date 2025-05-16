# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Simple Flask web server with robust error handling, logging, and auto-recovery
"""

import os
import sys
import uuid
import traceback
from flask import Flask, jsonify, request, Response
from datetime import datetime
import signal
import time

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our utilities
from web_interface.system_utils import (
    timeout,
    fallback,
    log_decision,
    log_tool_call,
    save_user_query,
    RequestTracker,
    setup_crash_recovery
)

app = Flask(__name__)

# Configure request tracking
request_tracker = RequestTracker("main_web")

# Setup crash recovery
setup_crash_recovery(
    "main_web",
    f"python {os.path.join(current_dir, 'recovery_manager.py')} --restart main_web"
)

@app.before_request
def before_request():
    """Log and track every request"""
    request_id = str(uuid.uuid4())
    request.request_id = request_id

    # Start tracking this request
    request_tracker.start_request(
        request_id,
        {
            "path": request.path,
            "method": request.method,
            "args": dict(request.args),
            "headers": {k: v for k, v in request.headers.items()},
            "remote_addr": request.remote_addr
        }
    )

    # Log the decision to process this request
    log_decision(
        f"Processing request to {request.path}",
        {
            "request_id": request_id,
            "method": request.method,
            "path": request.path
        }
    )

@app.after_request
def after_request(response):
    """Log after every request"""
    if hasattr(request, 'request_id'):
        request_tracker.end_request(request.request_id, status=str(response.status_code))
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle any unhandled exceptions"""
    if hasattr(request, 'request_id'):
        request_tracker.end_request(request.request_id, status="error")

    # Log the error
    error_id = str(uuid.uuid4())
    app.logger.error(f"Error {error_id}: {str(e)}\n{traceback.format_exc()}")

    # Return an error response
    return jsonify({
        "error": str(e),
        "error_id": error_id,
        "timestamp": datetime.now().isoformat()
    }), 500

@app.route('/api/health')
@fallback(default_return=jsonify({"status": "degraded", "error": "Health check failed"}))
def health_check():
    """Health check endpoint with fallback for robustness"""
    return jsonify({
        'status': 'online',
        'message': 'Simple server is running',
        'timestamp': datetime.now().isoformat(),
        'process_id': os.getpid()
    })

@app.route('/')
@timeout(5)  # 5 second timeout
def index():
    """Index page with timeout protection"""
    # Log this tool call
    log_tool_call(
        "index_page_render",
        {},
        "Rendering index page"
    )

    # Save this as the last query for recovery demonstration
    save_user_query(
        "index_page_access",
        {"path": "/", "timestamp": datetime.now().isoformat()}
    )

    return "<h1>Super AI Simple Server</h1>" + \
           "<p>API is available at /api/health</p>" + \
           f"<p>Server time: {datetime.now().isoformat()}</p>"

@app.route('/api/test-timeout')
@timeout(2)  # 2 second timeout
def test_timeout():
    """Test endpoint to demonstrate timeout"""
    # Simulate a long-running operation
    time.sleep(5)  # This will trigger a timeout
    return jsonify({"status": "This should never be returned"})

@app.route('/api/test-fallback')
@fallback(default_return=jsonify({"status": "fallback", "message": "Using fallback response"}))
def test_fallback():
    """Test endpoint to demonstrate fallback"""
    # Simulate a failure
    if True:  # Always fail for demo purposes
        raise Exception("Simulated failure to demonstrate fallback")
    return jsonify({"status": "This should never be returned"})

if __name__ == '__main__':
    print("Starting robust Flask server with error handling, logging and recovery...")
    app.run(host='0.0.0.0', port=5000, debug=True)
