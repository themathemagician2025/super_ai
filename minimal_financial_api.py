#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Minimal Financial API Server

A minimal Flask application that only registers the financial API blueprint.
"""

import os
import sys
import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_caching import Cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minimal_financial_api")

# Initialize cache
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',  # Use Redis in production
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes default
})

def create_app():
    """Create a minimal Flask application"""
    app = Flask(__name__)

    # Configure security settings
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', os.environ.get('SECRET_KEY'))

    # Configure performance settings
    app.config['THREAD_POOL_WORKERS'] = int(os.environ.get('THREAD_POOL_WORKERS', 8))
    app.config['COMPRESS_MIMETYPES'] = [
        'text/html', 'text/css', 'text/xml',
        'application/json', 'application/javascript'
    ]
    app.config['COMPRESS_LEVEL'] = 6  # GZip compression level (1-9)
    app.config['COMPRESS_MIN_SIZE'] = 500  # Minimum size in bytes to compress

    # Configure caching
    app.config['CACHE_TYPE'] = os.environ.get('CACHE_TYPE', 'SimpleCache')
    app.config['CACHE_REDIS_URL'] = os.environ.get('REDIS_URL')

    # Initialize extensions
    jwt = JWTManager(app)
    cache.init_app(app)

    # Configure CORS with restricted origins
    CORS(app, origins=["http://localhost:3000", "https://super-ai-frontend.example.com"])

    # Add basic route
    @app.route('/')
    def index():
        return jsonify({
            "status": "ok",
            "message": "Minimal Financial API server running"
        })

    # Serve static files with caching headers
    @app.route('/js/<path:filename>')
    def serve_js(filename):
        response = send_from_directory('static/dist/js', filename)
        # Set cache headers: 1 week for production, no cache for development
        if app.config['ENV'] == 'production':
            response.headers['Cache-Control'] = 'public, max-age=604800'
        else:
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/css/<path:filename>')
    def serve_css(filename):
        response = send_from_directory('static/dist/css', filename)
        # Set cache headers: 1 week for production, no cache for development
        if app.config['ENV'] == 'production':
            response.headers['Cache-Control'] = 'public, max-age=604800'
        else:
            response.headers['Cache-Control'] = 'no-store'
        return response

    # Cached API endpoints
    @app.route('/api/model/info')
    @cache.cached(timeout=3600)  # Cache for 1 hour
    def model_info():
        # This would normally fetch model info from a database
        return jsonify({
            "version": "1.0.0",
            "last_trained": "2023-09-01T12:00:00Z",
            "accuracy": 0.92,
            "precision": 0.89,
            "models": [
                {"name": "forex_predictor", "version": "1.2.0"},
                {"name": "sports_predictor", "version": "1.1.0"}
            ]
        })

    @app.route('/api/dashboard_metrics')
    @cache.cached(timeout=300)  # Cache for 5 minutes
    def dashboard_metrics():
        # This would normally compute dashboard metrics
        return jsonify({
            "predictions_today": 1250,
            "accuracy_7d": 0.91,
            "active_users": 328,
            "avg_response_time_ms": 125
        })

    # Import and register celery tasks
    try:
        from celery_tasks import register_celery_routes
        register_celery_routes(app)
        logger.info("Celery routes registered successfully")
    except Exception as e:
        logger.error(f"Failed to register celery routes: {str(e)}")

    # Import and register financial API
    try:
        from api_interface.financial_api import register_financial_blueprint
        register_financial_blueprint(app)
        logger.info("Financial API blueprint registered successfully")
    except Exception as e:
        logger.error(f"Failed to register financial API blueprint: {str(e)}", exc_info=True)

    # Register performance optimizations if available
    try:
        from performance_optimization import register_performance_optimizations
        register_performance_optimizations(app)
        logger.info("Performance optimizations registered successfully")
    except Exception as e:
        logger.error(f"Failed to register performance optimizations: {str(e)}")

    return app

# Example of input validation pattern (can be used in route handlers)
def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    # Create app
    app = create_app()

    # Check if SECRET_KEY is set
    if not os.environ.get('SECRET_KEY'):
        logger.warning("SECRET_KEY not set! Using an insecure default. Set SECRET_KEY environment variable for production.")
        # In development only - NEVER use this in production
        app.config['SECRET_KEY'] = 'dev_key_not_secure'

    # List routes
    logger.info("Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"{rule.endpoint}: {rule}")

    # Run server
    port = 8888
    logger.info(f"Starting minimal server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port)
