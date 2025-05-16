from prometheus_client import Counter, Histogram, Gauge
from typing import Dict

# Model performance metrics
MODEL_PREDICTION_TIME = Histogram(
    'model_prediction_seconds',
    'Time spent on model prediction',
    ['model_name', 'prediction_type']
)

MODEL_PREDICTION_ACCURACY = Counter(
    'model_prediction_accuracy_total',
    'Total number of accurate predictions',
    ['model_name', 'prediction_type']
)

MODEL_PREDICTION_ERROR = Counter(
    'model_prediction_error_total',
    'Total number of prediction errors',
    ['model_name', 'error_type']
)

# Resource utilization metrics
GPU_MEMORY_USAGE = Gauge(
    'gpu_memory_usage_bytes',
    'Current GPU memory usage in bytes',
    ['device_id']
)

MODEL_MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Current model memory usage in bytes',
    ['model_name']
)

# Request metrics
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['endpoint', 'method']
)

REQUEST_COUNT = Counter(
    'request_total',
    'Total number of requests',
    ['endpoint', 'method', 'status']
)

# Cache metrics
CACHE_HIT = Counter(
    'cache_hit_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISS = Counter(
    'cache_miss_total',
    'Total number of cache misses',
    ['cache_type']
)

# Batch processing metrics
BATCH_PROCESSING_TIME = Histogram(
    'batch_processing_seconds',
    'Time spent processing batches',
    ['model_name', 'batch_size']
)

BATCH_SIZE = Histogram(
    'batch_size_total',
    'Distribution of batch sizes',
    ['model_name']
)

def record_prediction_time(model_name: str, prediction_type: str, duration: float):
    """Record the time taken for a model prediction."""
    MODEL_PREDICTION_TIME.labels(model_name=model_name, prediction_type=prediction_type).observe(duration)

def record_prediction_accuracy(model_name: str, prediction_type: str, is_accurate: bool):
    """Record whether a prediction was accurate."""
    if is_accurate:
        MODEL_PREDICTION_ACCURACY.labels(model_name=model_name, prediction_type=prediction_type).inc()

def record_prediction_error(model_name: str, error_type: str):
    """Record a prediction error."""
    MODEL_PREDICTION_ERROR.labels(model_name=model_name, error_type=error_type).inc()

def update_gpu_memory_usage(device_id: str, memory_bytes: float):
    """Update GPU memory usage metric."""
    GPU_MEMORY_USAGE.labels(device_id=device_id).set(memory_bytes)

def update_model_memory_usage(model_name: str, memory_bytes: float):
    """Update model memory usage metric."""
    MODEL_MEMORY_USAGE.labels(model_name=model_name).set(memory_bytes)

def record_request_latency(endpoint: str, method: str, duration: float):
    """Record API request latency."""
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration)

def record_request(endpoint: str, method: str, status: str):
    """Record an API request."""
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status=status).inc()

def record_cache_operation(cache_type: str, hit: bool):
    """Record cache hit/miss."""
    if hit:
        CACHE_HIT.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISS.labels(cache_type=cache_type).inc()

def record_batch_metrics(model_name: str, batch_size: int, duration: float):
    """Record batch processing metrics."""
    BATCH_PROCESSING_TIME.labels(model_name=model_name, batch_size=batch_size).observe(duration)
    BATCH_SIZE.labels(model_name=model_name).observe(batch_size) 