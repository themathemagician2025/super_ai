# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from aiohttp import web
import logging
import json
from typing import Dict, Any
import asyncio
from datetime import datetime
import traceback

from utils.log_manager import ThreadSafeLogManager

# Initialize logging
log_manager = ThreadSafeLogManager()
logger = log_manager.get_logger('api')

routes = web.RouteTableDef()

@routes.get('/')
async def home(request: web.Request) -> web.Response:
    """Home endpoint."""
    start_time = datetime.now()
    try:
        response_data = {"message": "Welcome to Super AI API"}
        response = web.json_response(response_data)

        # Log access
        duration = (datetime.now() - start_time).total_seconds()
        log_manager.log_access(
            endpoint='/',
            method='GET',
            status_code=200,
            response_time=duration
        )

        return response
    except Exception as e:
        # Log error
        log_manager.log_error(e, {'endpoint': '/', 'method': 'GET'})
        return web.json_response(
            {'error': str(e)},
            status=500
        )

@routes.get('/health')
async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    start_time = datetime.now()
    try:
        response_data = {"status": "healthy"}
        response = web.json_response(response_data)

        # Log access
        duration = (datetime.now() - start_time).total_seconds()
        log_manager.log_access(
            endpoint='/health',
            method='GET',
            status_code=200,
            response_time=duration
        )

        return response
    except Exception as e:
        # Log error
        log_manager.log_error(e, {'endpoint': '/health', 'method': 'GET'})
        return web.json_response(
            {'error': str(e)},
            status=500
        )

@routes.get('/metrics')
async def metrics(request: web.Request) -> web.Response:
    """System metrics endpoint."""
    start_time = datetime.now()
    try:
        # Collect basic metrics
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'uptime': 0,  # TODO: Implement uptime tracking
            'requests_processed': 0,  # TODO: Implement request counting
            'system_status': 'operational'
        }

        response = web.json_response(metrics_data)

        # Log access
        duration = (datetime.now() - start_time).total_seconds()
        log_manager.log_access(
            endpoint='/metrics',
            method='GET',
            status_code=200,
            response_time=duration
        )

        return response
    except Exception as e:
        # Log error
        log_manager.log_error(e, {'endpoint': '/metrics', 'method': 'GET'})
        return web.json_response(
            {'error': str(e)},
            status=500
        )

async def start_api(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """Start the API server."""
    app = web.Application()
    app.add_routes(routes)

    # Configure CORS if needed
    # cors = aiohttp_cors.setup(app, defaults={
    #     "*": aiohttp_cors.ResourceOptions(
    #         allow_credentials=True,
    #         expose_headers="*",
    #         allow_headers="*"
    #     )
    # })

    # Add middleware for request timing
    @web.middleware
    async def timing_middleware(request: web.Request, handler):
        start_time = datetime.now()
        try:
            response = await handler(request)
            duration = (datetime.now() - start_time).total_seconds()
            log_manager.log_performance(
                operation=f"{request.method} {request.path}",
                duration=duration
            )
            return response
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            log_manager.log_error(e, {
                'request_path': request.path,
                'request_method': request.method,
                'duration': duration
            })
            raise

    app.middlewares.append(timing_middleware)

    # Start the server
    logger.info(f"Starting API server on http://{host}:{port}")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Keep the server running
    while True:
        await asyncio.sleep(3600)  # Sleep for an hour
