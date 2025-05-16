#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Performance Optimization Module

This module provides performance optimization features for the Super AI API
including caching, compression, and asynchronous task processing.
"""

import os
import io
import gzip
import functools
import time
import logging
from typing import Dict, Any, Callable, Optional
from flask import Flask, request, Response, current_app
from PIL import Image
import base64
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logger = logging.getLogger("performance")

# Global thread pool executor with optimized worker count
# Default to number of CPUs + 4 for I/O bound tasks
CPU_COUNT = os.cpu_count() or 1
DEFAULT_MAX_WORKERS = CPU_COUNT + 4
thread_pool = ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS)

# Memory cache for API responses
# In production, this should be replaced with Redis or another distributed cache
CACHE = {}
CACHE_TIMEOUTS = {}

def configure_executor(max_workers: Optional[int] = None) -> None:
    """
    Configure the global thread pool executor

    Args:
        max_workers: Maximum number of worker threads
    """
    global thread_pool

    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    # Shutdown existing executor if it exists
    if thread_pool:
        thread_pool.shutdown(wait=False)

    # Create new executor
    thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ThreadPoolExecutor configured with {max_workers} workers")

def gzip_response(response: Response) -> Response:
    """
    Compress response data with gzip

    Args:
        response: Flask response object

    Returns:
        Compressed response
    """
    # Skip compression for small responses (less than 500 bytes)
    if len(response.data) < 500:
        return response

    # Skip if already compressed
    if response.headers.get('Content-Encoding') == 'gzip':
        return response

    # Skip compression for certain content types
    if response.mimetype in ['image/jpeg', 'image/png', 'image/gif', 'application/pdf']:
        return response

    # Compress content
    gzip_buffer = io.BytesIO()
    gzip_file = gzip.GzipFile(mode='wb', fileobj=gzip_buffer)
    gzip_file.write(response.data)
    gzip_file.close()

    # Update response
    response.data = gzip_buffer.getvalue()
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Length'] = len(response.data)
    response.headers['Vary'] = 'Accept-Encoding'

    return response

def cache_response(timeout: int = 300):
    """
    Cache API responses in memory

    Args:
        timeout: Cache timeout in seconds (default: 5 minutes)
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Create cache key from request path and query parameters
            cache_key = f"{request.path}?{request.query_string.decode('utf-8')}"

            # Check if response is in cache and not expired
            current_time = time.time()
            if cache_key in CACHE and CACHE_TIMEOUTS.get(cache_key, 0) > current_time:
                logger.debug(f"Cache hit for {cache_key}")
                return CACHE[cache_key]

            # Get fresh response
            logger.debug(f"Cache miss for {cache_key}")
            response = f(*args, **kwargs)

            # Cache response
            CACHE[cache_key] = response
            CACHE_TIMEOUTS[cache_key] = current_time + timeout

            return response
        return wrapper
    return decorator

def optimize_base64_image(base64_string: str, quality: int = 85, max_size: int = 800) -> str:
    """
    Optimize a base64-encoded image by resizing and compressing

    Args:
        base64_string: Base64-encoded image string
        quality: JPEG compression quality (0-100)
        max_size: Maximum dimension (width or height) in pixels

    Returns:
        Optimized base64-encoded image string
    """
    try:
        # Extract the base64 data part (remove header if exists)
        if ',' in base64_string:
            header, data = base64_string.split(',', 1)
        else:
            header = "data:image/jpeg;base64"
            data = base64_string

        # Decode base64 data
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data))

        # Resize if needed
        width, height = image.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))

            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Compress image
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=True)
        compressed_data = output.getvalue()

        # Encode back to base64
        compressed_base64 = base64.b64encode(compressed_data).decode('utf-8')

        # Return with header if original had one
        if ',' in base64_string:
            return f"{header},{compressed_base64}"
        return compressed_base64

    except Exception as e:
        logger.error(f"Error optimizing base64 image: {str(e)}")
        return base64_string  # Return original on error

def run_async(func: Callable, *args, **kwargs) -> threading.Thread:
    """
    Run a function asynchronously in a thread

    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Thread object
    """
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()
    return thread

def submit_to_thread_pool(func: Callable, *args, **kwargs) -> Any:
    """
    Submit a task to the thread pool

    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Future object
    """
    return thread_pool.submit(func, *args, **kwargs)

def register_performance_optimizations(app: Flask) -> None:
    """
    Register performance optimizations with Flask app

    Args:
        app: Flask application instance
    """
    # Configure thread pool based on environment variables or config
    max_workers = app.config.get('THREAD_POOL_WORKERS', DEFAULT_MAX_WORKERS)
    configure_executor(max_workers)

    # Add after_request handler for compression
    @app.after_request
    def compress_response(response: Response) -> Response:
        # Only compress if client accepts gzip encoding
        if 'gzip' not in request.headers.get('Accept-Encoding', ''):
            return response

        return gzip_response(response)

    # Add route to clear cache
    @app.route('/api/admin/clear_cache', methods=['POST'])
    def clear_cache():
        global CACHE, CACHE_TIMEOUTS
        CACHE.clear()
        CACHE_TIMEOUTS.clear()
        return {'status': 'success', 'message': 'Cache cleared'}, 200

    logger.info("Performance optimizations registered")