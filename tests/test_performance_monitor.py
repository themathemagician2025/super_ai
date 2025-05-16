# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Unit tests for the Performance Monitor module.

This test suite validates the functionality of the Performance Monitor,
ensuring it correctly tracks, stores, and retrieves system metrics.
"""

import os
import sys
import time
import json
import unittest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the performance monitor
from src.utils.performance_monitor import PerformanceMonitor, get_monitor

class TestPerformanceMonitor(unittest.TestCase):
    """Test suite for the PerformanceMonitor class."""

    def setUp(self):
        """Set up a temporary directory for test metrics."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metrics_dir = Path(self.temp_dir.name) / "metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Create a test-specific monitor instance
        self.monitor = PerformanceMonitor(metrics_dir=str(self.metrics_dir))

    def tearDown(self):
        """Clean up temporary files."""
        # Ensure monitoring is stopped
        if hasattr(self, 'monitor') and self.monitor._monitoring:
            self.monitor.stop_monitoring()

        # Clean up temp directory
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the monitor initializes correctly."""
        # Check that directories are created
        self.assertTrue(os.path.exists(self.metrics_dir))

        # Check that the metrics dictionary is initialized
        self.assertIsInstance(self.monitor.metrics, dict)
        self.assertEqual(self.monitor.active_sessions, 0)
        self.assertEqual(self.monitor.errors, 0)
        self.assertIsInstance(self.monitor.response_times, list)

    def test_start_stop_monitoring(self):
        """Test starting and stopping the monitoring thread."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Check that monitoring is running
        self.assertTrue(self.monitor._monitoring)
        self.assertIsInstance(self.monitor._monitor_thread, threading.Thread)
        self.assertTrue(self.monitor._monitor_thread.is_alive())

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Check that monitoring has stopped
        self.assertFalse(self.monitor._monitoring)

        # Allow thread to fully terminate
        time.sleep(0.5)
        self.assertFalse(self.monitor._monitor_thread.is_alive())

    def test_collect_metrics(self):
        """Test that metrics collection works."""
        # Mock psutil to return predictable values
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory', return_value=MagicMock(percent=60.0)), \
             patch('psutil.disk_usage', return_value=MagicMock(percent=70.0)), \
             patch('psutil.net_io_counters', return_value=MagicMock(
                 bytes_sent=1000, bytes_recv=2000)):

            # Collect metrics
            metrics = self.monitor._collect_metrics()

            # Verify collected metrics
            self.assertEqual(metrics['cpu_percent'], 50.0)
            self.assertEqual(metrics['memory_percent'], 60.0)
            self.assertEqual(metrics['disk_percent'], 70.0)
            self.assertEqual(metrics['network_sent'], 1000)
            self.assertEqual(metrics['network_received'], 2000)

    def test_update_active_sessions(self):
        """Test updating active session count."""
        # Initial value should be 0
        self.assertEqual(self.monitor.active_sessions, 0)

        # Update to 5
        self.monitor.update_active_sessions(5)
        self.assertEqual(self.monitor.active_sessions, 5)

        # Update to 0
        self.monitor.update_active_sessions(0)
        self.assertEqual(self.monitor.active_sessions, 0)

    def test_record_response_time(self):
        """Test recording response times."""
        # Initial list should be empty
        self.assertEqual(len(self.monitor.response_times), 0)

        # Record a response time
        self.monitor.record_response_time(0.5)
        self.assertEqual(len(self.monitor.response_times), 1)
        self.assertEqual(self.monitor.response_times[0], 0.5)

        # Record multiple response times
        self.monitor.record_response_time(1.0)
        self.monitor.record_response_time(1.5)
        self.assertEqual(len(self.monitor.response_times), 3)
        self.assertEqual(sum(self.monitor.response_times), 3.0)

    def test_record_error(self):
        """Test recording errors."""
        # Initial count should be 0
        self.assertEqual(self.monitor.errors, 0)

        # Record an error
        self.monitor.record_error()
        self.assertEqual(self.monitor.errors, 1)

        # Record multiple errors
        self.monitor.record_error()
        self.monitor.record_error()
        self.assertEqual(self.monitor.errors, 3)

    def test_calculate_summary(self):
        """Test summary calculation."""
        # Set up test data
        test_metrics = [
            {'cpu_percent': 10.0, 'memory_percent': 20.0},
            {'cpu_percent': 20.0, 'memory_percent': 30.0},
            {'cpu_percent': 30.0, 'memory_percent': 40.0},
        ]

        # Calculate summary
        summary = self.monitor._calculate_summary(test_metrics)

        # Verify summary calculations
        self.assertEqual(summary['cpu_percent']['avg'], 20.0)
        self.assertEqual(summary['cpu_percent']['max'], 30.0)
        self.assertEqual(summary['cpu_percent']['min'], 10.0)

        self.assertEqual(summary['memory_percent']['avg'], 30.0)
        self.assertEqual(summary['memory_percent']['max'], 40.0)
        self.assertEqual(summary['memory_percent']['min'], 20.0)

    def test_save_metrics(self):
        """Test saving metrics to a file."""
        # Create test metrics
        test_metrics = [
            {'timestamp': '2023-05-01T10:00:00', 'cpu_percent': 50.0, 'memory_percent': 60.0},
            {'timestamp': '2023-05-01T10:00:05', 'cpu_percent': 55.0, 'memory_percent': 65.0},
        ]

        # Save metrics
        filename = self.monitor._save_metrics(test_metrics)

        # Verify file was created
        self.assertTrue(os.path.exists(filename))

        # Verify file contents
        with open(filename, 'r') as f:
            saved_data = json.load(f)

            # Check metrics
            self.assertEqual(len(saved_data['metrics']), 2)
            self.assertEqual(saved_data['metrics'][0]['cpu_percent'], 50.0)
            self.assertEqual(saved_data['metrics'][1]['memory_percent'], 65.0)

            # Check summary
            self.assertEqual(saved_data['summary']['cpu_percent']['avg'], 52.5)
            self.assertEqual(saved_data['summary']['memory_percent']['min'], 60.0)

    def test_singleton_pattern(self):
        """Test that get_monitor returns the same instance."""
        # Get two monitor instances
        monitor1 = get_monitor()
        monitor2 = get_monitor()

        # They should be the same instance
        self.assertIs(monitor1, monitor2)

        # Modify one, change should be reflected in the other
        monitor1.update_active_sessions(10)
        self.assertEqual(monitor2.active_sessions, 10)

    def test_high_cpu_usage_logging(self):
        """Test that high CPU usage is logged."""
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('logging.Logger.warning') as mock_log:

            # Run _collect_metrics which should log high CPU
            self.monitor._collect_metrics()

            # Verify that a warning was logged
            mock_log.assert_called_once()
            self.assertIn('CPU usage', mock_log.call_args[0][0])

    def test_metrics_under_load(self):
        """Test metrics collection under simulated load."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Simulate activity
        for i in range(5):
            # Update sessions
            self.monitor.update_active_sessions(i + 1)

            # Record response times
            self.monitor.record_response_time(0.1 * (i + 1))

            # Record errors (every other iteration)
            if i % 2 == 0:
                self.monitor.record_error()

            # Small delay
            time.sleep(0.2)

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Check that metrics were collected
        metrics_files = list(self.metrics_dir.glob('*.json'))
        self.assertGreaterEqual(len(metrics_files), 1)

        # Verify response times and errors were recorded
        self.assertEqual(len(self.monitor.response_times), 5)
        self.assertEqual(self.monitor.errors, 3)
        self.assertEqual(self.monitor.active_sessions, 5)

    def test_monitor_threading_safety(self):
        """Test that monitor is thread-safe."""
        # Start monitoring
        self.monitor.start_monitoring()

        # Function to simulate concurrent operations
        def concurrent_operations(iterations):
            for _ in range(iterations):
                self.monitor.record_response_time(0.1)
                self.monitor.record_error()
                self.monitor.update_active_sessions(1)
                time.sleep(0.01)
                self.monitor.update_active_sessions(0)

        # Create and start threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=concurrent_operations, args=(10,))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Verify data
        self.assertEqual(len(self.monitor.response_times), 50)  # 5 threads * 10 iterations
        self.assertEqual(self.monitor.errors, 50)

        # Sessions could be 0 or 1 depending on timing
        self.assertIn(self.monitor.active_sessions, [0, 1])

if __name__ == '__main__':
    unittest.main()
