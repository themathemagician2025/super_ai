import time
import logging
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsManager:
    """Centralized metrics management for the Super AI system"""
    
    def __init__(self, push_gateway: str = "localhost:9091", job_name: str = "super_ai"):
        self.registry = CollectorRegistry()
        self.push_gateway = push_gateway
        self.job_name = job_name
        
        # File processing metrics
        self.files_processed = Counter(
            'files_processed_total',
            'Total number of files processed',
            ['file_type', 'status'],
            registry=self.registry
        )
        
        self.processing_time = Histogram(
            'file_processing_duration_seconds',
            'Time spent processing files',
            ['file_type'],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600),
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total number of cache hits',
            ['file_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'cache_size_bytes',
            'Current size of cache in bytes',
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference_time = Histogram(
            'model_inference_duration_seconds',
            'Time spent on model inference',
            ['model_type'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Current memory usage by models',
            ['model_type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_memory = Gauge(
            'system_memory_usage_bytes',
            'System memory usage',
            ['type'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'gpu_memory_usage_bytes',
            'GPU memory usage',
            ['device'],
            registry=self.registry
        )
        
    def track_processing(self, file_type: str):
        """Decorator to track file processing metrics"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.files_processed.labels(
                        file_type=file_type,
                        status='success'
                    ).inc()
                    return result
                except Exception as e:
                    self.files_processed.labels(
                        file_type=file_type,
                        status='error'
                    ).inc()
                    raise e
                finally:
                    duration = time.time() - start_time
                    self.processing_time.labels(
                        file_type=file_type
                    ).observe(duration)
            return wrapper
        return decorator
    
    def track_inference(self, model_type: str):
        """Decorator to track model inference metrics"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.model_inference_time.labels(
                        model_type=model_type
                    ).observe(duration)
            return wrapper
        return decorator
    
    def record_cache_hit(self, file_type: str):
        """Record a cache hit"""
        self.cache_hits.labels(file_type=file_type).inc()
    
    def update_cache_size(self, size_bytes: int):
        """Update cache size metric"""
        self.cache_size.set(size_bytes)
    
    def update_model_memory(self, model_type: str, memory_bytes: int):
        """Update model memory usage metric"""
        self.model_memory_usage.labels(model_type=model_type).set(memory_bytes)
    
    def update_system_metrics(self, metrics: Dict[str, float]):
        """Update system metrics"""
        for metric_type, value in metrics.items():
            self.system_memory.labels(type=metric_type).set(value)
    
    def update_gpu_metrics(self, device: str, memory_bytes: int):
        """Update GPU memory metrics"""
        self.gpu_memory.labels(device=device).set(memory_bytes)
    
    def push_metrics(self):
        """Push all metrics to Prometheus gateway"""
        try:
            push_to_gateway(
                self.push_gateway,
                job=self.job_name,
                registry=self.registry
            )
        except Exception as e:
            logger.error(f"Failed to push metrics: {str(e)}")

class PerformanceMonitor:
    """Performance monitoring and profiling utility"""
    
    def __init__(self, metrics_manager: Optional[MetricsManager] = None):
        self.metrics = metrics_manager or MetricsManager()
        self.start_time = None
        self.checkpoints = {}
    
    def start(self):
        """Start monitoring session"""
        self.start_time = time.time()
        self.checkpoints = {}
    
    def checkpoint(self, name: str):
        """Record a checkpoint"""
        if self.start_time is None:
            raise RuntimeError("Monitoring session not started")
        self.checkpoints[name] = time.time() - self.start_time
    
    def end(self) -> Dict:
        """End monitoring session and return results"""
        if self.start_time is None:
            raise RuntimeError("Monitoring session not started")
            
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        results = {
            'total_duration': total_duration,
            'checkpoints': self.checkpoints,
            'timestamp': datetime.now().isoformat()
        }
        
        self.start_time = None
        self.checkpoints = {}
        
        return results 