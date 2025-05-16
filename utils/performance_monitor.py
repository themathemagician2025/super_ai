# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Performance Monitor for Super AI System

This module provides a PerformanceMonitor class that tracks system performance metrics
in real-time, including CPU usage, memory consumption, response times, and error rates.
"""

import os
import time
import json
import logging
import threading
import platform
import psutil
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Development/logs/performance_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PerformanceMonitor")

class PerformanceMonitor:
    """
    A class for monitoring system performance metrics during runtime.

    This monitor tracks:
    - CPU usage
    - Memory usage
    - Disk I/O
    - Network activity
    - Response times
    - Error rates
    - Request throughput
    """

    def __init__(self, metrics_dir="Development/logs/metrics",
                 collection_interval=5, save_interval=60):
        """
        Initialize the performance monitor.

        Args:
            metrics_dir (str): Directory to save metric data
            collection_interval (int): How often to collect metrics (seconds)
            save_interval (int): How often to save metrics to disk (seconds)
        """
        self.metrics_dir = metrics_dir
        self.collection_interval = collection_interval
        self.save_interval = save_interval

        # Create metrics directory if it doesn't exist
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Performance data
        self.metrics = {
            "system": {
                "cpu_percent": [],
                "memory_percent": [],
                "disk_usage_percent": [],
                "network_sent_bytes": [],
                "network_recv_bytes": [],
            },
            "application": {
                "response_times": [],
                "error_count": 0,
                "request_count": 0,
                "active_sessions": 0,
            },
            "timestamps": []
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_save_time = time.time()

        # Track network stats for calculating deltas
        self.last_net_io = psutil.net_io_counters()

        logger.info("Performance monitor initialized")

    def start_monitoring(self):
        """Start the monitoring thread."""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread and save final metrics."""
        if not self.is_monitoring:
            logger.warning("Monitor not running")
            return

        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        self._save_metrics()
        logger.info("Performance monitoring stopped")

    def record_response_time(self, response_time):
        """
        Record a response time for a request.

        Args:
            response_time (float): Response time in seconds
        """
        if self.is_monitoring:
            self.metrics["application"]["response_times"].append(response_time)
            self.metrics["application"]["request_count"] += 1

    def record_error(self):
        """Record an error occurrence."""
        if self.is_monitoring:
            self.metrics["application"]["error_count"] += 1

    def update_active_sessions(self, count):
        """
        Update the count of active sessions.

        Args:
            count (int): Number of active sessions
        """
        if self.is_monitoring:
            self.metrics["application"]["active_sessions"] = count

    def _monitoring_loop(self):
        """Main monitoring loop that collects metrics at regular intervals."""
        while self.is_monitoring:
            try:
                self._collect_metrics()

                # Save metrics at the specified interval
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self._save_metrics()
                    self.last_save_time = current_time

                # Sleep until next collection
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.collection_interval)

    def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent

            # Get network IO
            current_net_io = psutil.net_io_counters()
            net_sent = current_net_io.bytes_sent - self.last_net_io.bytes_sent
            net_recv = current_net_io.bytes_recv - self.last_net_io.bytes_recv
            self.last_net_io = current_net_io

            # Store metrics
            timestamp = datetime.now().isoformat()
            self.metrics["timestamps"].append(timestamp)

            self.metrics["system"]["cpu_percent"].append(cpu_percent)
            self.metrics["system"]["memory_percent"].append(memory_percent)
            self.metrics["system"]["disk_usage_percent"].append(disk_percent)
            self.metrics["system"]["network_sent_bytes"].append(net_sent)
            self.metrics["system"]["network_recv_bytes"].append(net_recv)

            # Log high resource usage
            if cpu_percent > 80:
                logger.warning(f"High CPU usage detected: {cpu_percent}%")
            if memory_percent > 80:
                logger.warning(f"High memory usage detected: {memory_percent}%")

        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")

    def _save_metrics(self):
        """Save the collected metrics to a file."""
        if not self.metrics["timestamps"]:
            logger.info("No metrics to save")
            return

        try:
            # Create a snapshot of current metrics
            now = datetime.now()
            filename = f"metrics_{now.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.metrics_dir, filename)

            # Calculate summary statistics
            summary = self._calculate_summary()

            # Combine data for saving
            save_data = {
                "timestamp": now.isoformat(),
                "duration": len(self.metrics["timestamps"]) * self.collection_interval,
                "system_info": self._get_system_info(),
                "metrics": self.metrics,
                "summary": summary
            }

            # Write to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Metrics saved to {filepath}")

            # Reset response times list but keep counters
            response_times = self.metrics["application"]["response_times"]
            self.metrics["application"]["response_times"] = []

            # Only keep the most recent system metrics (last 10 minutes)
            keep_points = min(len(self.metrics["timestamps"]),
                              int(600 / self.collection_interval))

            if keep_points < len(self.metrics["timestamps"]):
                self.metrics["timestamps"] = self.metrics["timestamps"][-keep_points:]
                for key in self.metrics["system"]:
                    self.metrics["system"][key] = self.metrics["system"][key][-keep_points:]

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    def _calculate_summary(self):
        """Calculate summary statistics for the collected metrics."""
        summary = {
            "system": {},
            "application": {}
        }

        # Calculate system metrics summaries
        for key, values in self.metrics["system"].items():
            if values:
                summary["system"][key] = {
                    "avg": sum(values) / len(values),
                    "max": max(values),
                    "min": min(values)
                }

        # Calculate application metrics summaries
        response_times = self.metrics["application"]["response_times"]
        if response_times:
            response_times.sort()
            summary["application"]["response_time"] = {
                "avg": sum(response_times) / len(response_times),
                "median": response_times[len(response_times) // 2],
                "p95": response_times[int(len(response_times) * 0.95)] if len(response_times) >= 20 else None,
                "max": max(response_times),
                "min": min(response_times)
            }

        # Error rate
        request_count = self.metrics["application"]["request_count"]
        error_count = self.metrics["application"]["error_count"]
        if request_count > 0:
            summary["application"]["error_rate"] = (error_count / request_count) * 100
        else:
            summary["application"]["error_rate"] = 0

        return summary

    def _get_system_info(self):
        """Get system information for context."""
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory": psutil.virtual_memory().total,
                "python_version": platform.python_version(),
            }
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {}

# Singleton instance
_monitor_instance = None

def get_monitor():
    """Get the singleton PerformanceMonitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = PerformanceMonitor()
    return _monitor_instance
