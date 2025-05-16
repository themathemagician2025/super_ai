# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
System Integration Test Suite

This script performs comprehensive testing of all Super AI features:
1. Error handling (timeouts, fallbacks)
2. Logging system
3. Recovery mechanisms
4. Self-learning capabilities
5. Performance monitoring

Usage:
    python system_integration_test.py [--full] [--stress] [--no-cleanup]
"""

import os
import sys
import time
import json
import random
import argparse
import threading
import subprocess
import statistics
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import utility modules
try:
    from web_interface.system_utils import (
        log_decision,
        log_tool_call,
        timeout,
        fallback,
        RequestTracker
    )
except ImportError:
    print("ERROR: Could not import system_utils. Make sure it exists and is accessible.")
    sys.exit(1)

# Import learning modules
try:
    from core_ai.learning_integration import (
        get_best_model,
        search_before_web,
        rate_response
    )
except ImportError:
    print("ERROR: Could not import learning_integration. Make sure it exists and is accessible.")
    sys.exit(1)

# Configure logging for tests
import logging
LOG_DIR = os.path.join(parent_dir, "logs", "tests")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("integration_test")

# Test configuration
STRESS_TEST_THREADS = 10
STRESS_TEST_QUERIES_PER_THREAD = 20
DEFAULT_TEST_QUERIES = 20

# Sample data
SAMPLE_QUERIES = [
    "How do I implement a binary search tree?",
    "What is the time complexity of quicksort?",
    "Explain deep learning in simple terms",
    "How can I optimize Python code?",
    "What are design patterns in software engineering?",
    "Explain RESTful API design principles",
    "What is the difference between TCP and UDP?",
    "How to handle concurrent requests in a web server?",
    "Explain the CAP theorem",
    "What are microservices?",
    # Add complex edge cases
    "Can you write a very long and detailed explanation of quantum computing that covers all major aspects including quantum bits, superposition, entanglement, quantum gates, quantum algorithms, quantum error correction, physical implementations, challenges, and the current state of quantum computing research as of 2023?",
    "",  # Empty query
    "?",  # Single character query
    "Please respond extremely quickly, I'm in a hurry!",  # Speed priority indicator
    "I need the most accurate and thorough analysis possible.",  # Accuracy priority indicator
]

class PerformanceMetrics:
    """Tracks performance metrics for the test runs"""

    def __init__(self):
        self.response_times = []
        self.memory_retrieval_times = []
        self.model_selection_times = []
        self.feedback_times = []
        self.errors = []
        self.timeouts = []

    def add_response_time(self, time_ms: float):
        self.response_times.append(time_ms)

    def add_memory_retrieval_time(self, time_ms: float):
        self.memory_retrieval_times.append(time_ms)

    def add_model_selection_time(self, time_ms: float):
        self.model_selection_times.append(time_ms)

    def add_feedback_time(self, time_ms: float):
        self.feedback_times.append(time_ms)

    def add_error(self, error_info: Dict):
        self.errors.append(error_info)

    def add_timeout(self, timeout_info: Dict):
        self.timeouts.append(timeout_info)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics"""

        def safe_stats(data_list):
            if not data_list:
                return {"min": 0, "max": 0, "avg": 0, "median": 0, "p95": 0, "count": 0}

            sorted_data = sorted(data_list)
            p95_idx = int(len(sorted_data) * 0.95)
            return {
                "min": min(data_list),
                "max": max(data_list),
                "avg": sum(data_list) / len(data_list),
                "median": statistics.median(data_list) if data_list else 0,
                "p95": sorted_data[p95_idx] if len(sorted_data) > p95_idx else max(data_list) if data_list else 0,
                "count": len(data_list)
            }

        return {
            "response_times": safe_stats(self.response_times),
            "memory_retrieval_times": safe_stats(self.memory_retrieval_times),
            "model_selection_times": safe_stats(self.model_selection_times),
            "feedback_times": safe_stats(self.feedback_times),
            "errors": {
                "count": len(self.errors),
                "details": self.errors[:10]  # Just first 10 for brevity
            },
            "timeouts": {
                "count": len(self.timeouts),
                "details": self.timeouts[:10]  # Just first 10 for brevity
            }
        }

class SystemTester:
    """
    Main test orchestrator for the system integration test
    """

    def __init__(self,
                 server_ports: Tuple[int, int] = (5000, 5001),
                 cleanup_after: bool = True):
        self.main_port, self.forex_port = server_ports
        self.cleanup_after = cleanup_after
        self.metrics = PerformanceMetrics()
        self.servers_started = False
        self.server_processes = []
        self.request_tracker = RequestTracker("test_system")

    def start_servers(self) -> bool:
        """Start the web and forex servers for testing"""
        logger.info("Starting web and forex servers...")

        try:
            # Start the main web server
            main_web_cmd = [
                sys.executable,
                os.path.join(parent_dir, "web_interface", "simple_server.py")
            ]
            main_process = subprocess.Popen(
                main_web_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.server_processes.append(main_process)

            # Start the forex server
            forex_cmd = [
                sys.executable,
                os.path.join(parent_dir, "web_interface", "simple_forex_server.py")
            ]
            forex_process = subprocess.Popen(
                forex_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.server_processes.append(forex_process)

            # Wait for servers to start up
            logger.info("Waiting for servers to initialize...")
            time.sleep(5)

            # Verify they're running
            import requests
            try:
                main_response = requests.get(f"http://localhost:{self.main_port}/api/health", timeout=2)
                forex_response = requests.get(f"http://localhost:{self.forex_port}/api/health", timeout=2)

                if main_response.status_code == 200 and forex_response.status_code == 200:
                    logger.info("Both servers started successfully")
                    self.servers_started = True
                    return True
                else:
                    logger.error(f"Server health checks failed: Main={main_response.status_code}, Forex={forex_response.status_code}")
                    return False

            except requests.RequestException as e:
                logger.error(f"Error connecting to servers: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Failed to start servers: {str(e)}")
            return False

    def stop_servers(self):
        """Stop the web and forex servers"""
        if not self.servers_started:
            return

        logger.info("Stopping servers...")
        for process in self.server_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating server process: {str(e)}")
                # Force kill if terminate didn't work
                try:
                    process.kill()
                except:
                    pass

        self.servers_started = False
        logger.info("All servers stopped")

    def cleanup(self):
        """Clean up temporary test data"""
        if not self.cleanup_after:
            logger.info("Skipping cleanup as requested")
            return

        logger.info("Cleaning up test data...")

        # Clean up any test-specific data
        # Note: We're being careful not to delete important user data
        try:
            test_memory_dir = os.path.join(parent_dir, "memory", "test")
            test_feedback_dir = os.path.join(parent_dir, "feedback", "test")

            for directory in [test_memory_dir, test_feedback_dir]:
                if os.path.exists(directory):
                    import shutil
                    shutil.rmtree(directory)
                    logger.info(f"Removed test directory: {directory}")

        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def run_single_query(self,
                        query: str,
                        expected_response: Optional[str] = None,
                        simulate_timeout: bool = False,
                        simulate_error: bool = False) -> Dict[str, Any]:
        """
        Run a single query through the system

        Args:
            query: The query text
            expected_response: Optional expected response for validation
            simulate_timeout: Intentionally cause a timeout
            simulate_error: Intentionally cause an error

        Returns:
            Dict with test results
        """
        request_id = f"test_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
        self.request_tracker.start_request(request_id, {"query": query})

        start_time = time.time()
        result = {"query": query, "success": False, "errors": []}

        try:
            # 1. Select model
            model_selection_start = time.time()
            model_info = get_best_model(query)
            model_selection_time = time.time() - model_selection_start
            self.metrics.add_model_selection_time(model_selection_time * 1000)  # Convert to ms

            result["model_info"] = model_info
            result["model_selection_time_ms"] = model_selection_time * 1000

            # 2. Check memory (optional timeout simulation)
            memory_start = time.time()
            if simulate_timeout:
                try:
                    # Create a timeout by using a very short timeout
                    @timeout(0.001)
                    def simulate_timeout_search():
                        time.sleep(1)  # This will timeout
                        return {}

                    memory_results = simulate_timeout_search()
                    result["memory_results"] = memory_results
                except Exception as e:
                    self.metrics.add_timeout({"operation": "memory_retrieval", "query": query, "error": str(e)})
                    result["errors"].append({"type": "timeout", "phase": "memory_retrieval", "error": str(e)})
                    # Fall back to a simple response
                    memory_results = {"source": "fallback", "results": []}
            elif simulate_error:
                # Simulate an error during memory retrieval
                raise ValueError("Simulated error in memory retrieval")
            else:
                # Normal path
                memory_results = search_before_web(query)

            memory_retrieval_time = time.time() - memory_start
            self.metrics.add_memory_retrieval_time(memory_retrieval_time * 1000)

            result["memory_results"] = memory_results
            result["memory_retrieval_time_ms"] = memory_retrieval_time * 1000

            # 3. Generate "response" (in a real system this would call the actual AI)
            # For testing, we'll just use a placeholder response
            if expected_response:
                response = expected_response
            else:
                response = f"This is a test response for: {query}"

            # 4. Rate the response
            feedback_start = time.time()
            metadata = {
                "model_name": model_info.get("model", "unknown"),
                "runtime_seconds": random.uniform(0.1, 2.0),
                "speed_score": random.uniform(0.5, 0.9),
                "test_run": True
            }

            feedback = rate_response(query, response, metadata)
            feedback_time = time.time() - feedback_start
            self.metrics.add_feedback_time(feedback_time * 1000)

            result["feedback"] = feedback
            result["feedback_time_ms"] = feedback_time * 1000
            result["success"] = True

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            result["errors"].append({"type": "exception", "error": str(e)})
            self.metrics.add_error({"query": query, "error": str(e)})

        total_time = time.time() - start_time
        result["total_time_ms"] = total_time * 1000
        self.metrics.add_response_time(total_time * 1000)

        self.request_tracker.end_request(
            request_id,
            "error" if result["errors"] else "success"
        )

        return result

    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling mechanisms"""
        logger.info("Testing error handling...")
        results = []

        # Test timeout handling
        logger.info("Testing timeout handling...")
        timeout_result = self.run_single_query(
            "This query will timeout",
            simulate_timeout=True
        )
        results.append({"test": "timeout_handling", "result": timeout_result})

        # Test error fallbacks
        logger.info("Testing error fallbacks...")
        error_result = self.run_single_query(
            "This query will cause an error",
            simulate_error=True
        )
        results.append({"test": "error_fallback", "result": error_result})

        # Test empty query
        logger.info("Testing empty query...")
        empty_result = self.run_single_query("")
        results.append({"test": "empty_query", "result": empty_result})

        # Test very large query
        logger.info("Testing very large query...")
        large_query = "A" * 10000  # 10KB query
        large_result = self.run_single_query(large_query)
        results.append({"test": "large_query", "result": large_result})

        return {"name": "error_handling_tests", "results": results}

    def test_learning_capabilities(self) -> Dict[str, Any]:
        """Test the self-learning capabilities"""
        logger.info("Testing learning capabilities...")
        results = []

        # Run 12 queries to trigger a retraining (after 10)
        for i in range(12):
            query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
            logger.info(f"Running learning test query {i+1}: {query[:30]}...")

            result = self.run_single_query(query)
            results.append(result)

            # Check if retraining was triggered (every 10 queries)
            if i > 0 and i % 10 == 0:
                logger.info(f"Query {i+1} should have triggered retraining")
                # In a more sophisticated test, we could verify this by checking logs
                # or by examining the strategy file timestamps

            # Add a short delay to avoid overwhelming the system
            time.sleep(0.5)

        return {"name": "learning_capabilities_test", "results": results}

    def test_self_retrieval(self) -> Dict[str, Any]:
        """Test the self-retrieval capabilities"""
        logger.info("Testing self-retrieval capabilities...")
        results = []

        # Step 1: Store some information
        initial_queries = SAMPLE_QUERIES[:5]
        for i, query in enumerate(initial_queries):
            response = f"Test response {i} for: {query}"
            logger.info(f"Storing knowledge for: {query[:30]}...")

            # Store this in memory by rating it
            metadata = {"test_run": True, "model_name": "test_model"}
            rate_response(query, response, metadata)

            # Add to results
            results.append({
                "phase": "storage",
                "query": query,
                "response": response
            })

            time.sleep(0.5)  # Brief delay

        # Step 2: Try to retrieve the same information
        logger.info("Testing retrieval of stored knowledge...")
        for i, query in enumerate(initial_queries):
            # Modify query slightly to test fuzzy matching
            retrieval_query = query.replace("?", "").replace("How", "How exactly")

            logger.info(f"Retrieving knowledge for: {retrieval_query[:30]}...")
            memory_results = search_before_web(retrieval_query)

            # Check if we got results
            success = memory_results.get("source") == "memory" and len(memory_results.get("results", [])) > 0

            # Add to results
            results.append({
                "phase": "retrieval",
                "query": retrieval_query,
                "success": success,
                "results_count": len(memory_results.get("results", [])),
                "source": memory_results.get("source")
            })

            time.sleep(0.5)  # Brief delay

        return {"name": "self_retrieval_test", "results": results}

    def test_self_feedback(self) -> Dict[str, Any]:
        """Test the self-feedback scoring capabilities"""
        logger.info("Testing self-feedback capabilities...")
        results = []

        test_cases = [
            # Good responses
            {
                "query": "What is a binary search tree?",
                "response": "A binary search tree (BST) is a data structure where each node has at most two children, with values less than the node on the left and values greater than the node on the right. This property makes search, insertion, and deletion operations efficient, typically O(log n) in balanced trees.\n\nHere's a simple implementation in Python:\n\n```python\nclass Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\nclass BinarySearchTree:\n    def __init__(self):\n        self.root = None\n        \n    def insert(self, value):\n        if not self.root:\n            self.root = Node(value)\n        else:\n            self._insert_recursive(self.root, value)\n            \n    def _insert_recursive(self, node, value):\n        if value < node.value:\n            if node.left is None:\n                node.left = Node(value)\n            else:\n                self._insert_recursive(node.left, value)\n        else:\n            if node.right is None:\n                node.right = Node(value)\n            else:\n                self._insert_recursive(node.right, value)\n```\n\nThis implementation handles insertion. Search and deletion would follow similar recursive patterns.",
                "expected_rating": "GOOD"
            },
            # Bad responses (too short, irrelevant)
            {
                "query": "Explain how neural networks work in detail",
                "response": "Neural networks have neurons.",
                "expected_rating": "BAD"
            },
            # Neutral responses (partial)
            {
                "query": "How does quicksort work?",
                "response": "Quicksort is a sorting algorithm that uses a divide-and-conquer approach. It picks a pivot element and partitions the array around it.",
                "expected_rating": "NEUTRAL"
            }
        ]

        for i, test_case in enumerate(test_cases):
            query = test_case["query"]
            response = test_case["response"]
            expected_rating = test_case["expected_rating"]

            logger.info(f"Testing feedback for case {i+1}: expected {expected_rating}...")

            # Get feedback rating
            metadata = {"test_run": True, "model_name": "test_model"}
            feedback = rate_response(query, response, metadata)

            # Check if rating matches expectation
            actual_rating = feedback.get("rating")
            success = actual_rating == expected_rating

            if not success:
                logger.warning(f"Rating mismatch: expected {expected_rating}, got {actual_rating}")

            # Add to results
            results.append({
                "case": i+1,
                "query": query,
                "response_snippet": response[:50] + "...",
                "expected_rating": expected_rating,
                "actual_rating": actual_rating,
                "score": feedback.get("score"),
                "success": success,
                "suggestions": feedback.get("suggestions", [])
            })

        return {"name": "self_feedback_test", "results": results}

    def test_recovery_mechanism(self) -> Dict[str, Any]:
        """Test the recovery mechanisms"""
        logger.info("Testing recovery mechanisms...")
        results = []

        # This is a simulated recovery test
        # In a real test, we would force-kill a server and verify it restarts

        logger.info("Setting up recovery test...")

        # 1. Store a query that we'll try to recover
        recovery_query = "This is a query that should be recovered after a crash"
        recovery_response = "This is the response that should be recovered"

        metadata = {"test_run": True, "model_name": "test_model", "recovery_test": True}
        pre_result = rate_response(recovery_query, recovery_response, metadata)

        results.append({
            "phase": "pre_crash",
            "query": recovery_query,
            "response": recovery_response,
            "metadata": metadata,
            "result": pre_result
        })

        # 2. Simulate crash and recovery
        # In a production test, we would actually kill the server process here
        # and verify the recovery manager restarts it

        logger.info("Simulating crash and recovery...")

        # 3. After "recovery", check if we can retrieve the previous query
        time.sleep(2)  # Pretend recovery took some time

        post_memory = search_before_web(recovery_query)

        # Check if we got our recovery query back
        success = False
        for result in post_memory.get("results", []):
            if recovery_query in result.get("query", ""):
                success = True
                break

        results.append({
            "phase": "post_recovery",
            "retrieval_query": recovery_query,
            "memory_results": post_memory,
            "recovery_successful": success
        })

        return {"name": "recovery_mechanism_test", "results": results}

    def run_stress_test(self,
                       num_threads: int = STRESS_TEST_THREADS,
                       queries_per_thread: int = STRESS_TEST_QUERIES_PER_THREAD) -> Dict[str, Any]:
        """
        Run a stress test with multiple concurrent threads

        Args:
            num_threads: Number of concurrent threads to run
            queries_per_thread: Number of queries per thread

        Returns:
            Dict with stress test results
        """
        logger.info(f"Starting stress test with {num_threads} threads, {queries_per_thread} queries per thread...")
        results = []

        def thread_worker(thread_id: int) -> List[Dict[str, Any]]:
            """Worker function for each thread"""
            thread_results = []
            for i in range(queries_per_thread):
                query_idx = random.randrange(0, len(SAMPLE_QUERIES))
                query = SAMPLE_QUERIES[query_idx]

                # Randomly simulate errors or timeouts (10% chance)
                simulate_timeout = random.random() < 0.05
                simulate_error = random.random() < 0.05 and not simulate_timeout

                try:
                    result = self.run_single_query(
                        query,
                        simulate_timeout=simulate_timeout,
                        simulate_error=simulate_error
                    )

                    result["thread_id"] = thread_id
                    result["query_id"] = i
                    thread_results.append(result)

                except Exception as e:
                    logger.error(f"Thread {thread_id}, query {i} failed: {str(e)}")
                    thread_results.append({
                        "thread_id": thread_id,
                        "query_id": i,
                        "query": query,
                        "success": False,
                        "error": str(e)
                    })

            return thread_results

        # Use thread pool to run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_thread = {
                executor.submit(thread_worker, thread_id): thread_id
                for thread_id in range(num_threads)
            }

            for future in concurrent.futures.as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    thread_results = future.result()
                    results.extend(thread_results)
                    logger.info(f"Thread {thread_id} completed with {len(thread_results)} results")
                except Exception as e:
                    logger.error(f"Thread {thread_id} failed: {str(e)}")

        # Calculate success rate
        success_count = sum(1 for r in results if r.get("success", False))
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0

        logger.info(f"Stress test completed with {success_count}/{total_count} successful queries ({success_rate:.2%})")

        return {
            "name": "stress_test",
            "total_queries": total_count,
            "successful_queries": success_count,
            "success_rate": success_rate,
            "thread_count": num_threads,
            "queries_per_thread": queries_per_thread,
            "results": results[:10]  # Just include first 10 for brevity
        }

    def run_full_test_suite(self, include_stress_test: bool = False) -> Dict[str, Any]:
        """Run the complete test suite"""
        start_time = time.time()
        logger.info("Starting full test suite...")

        # Start the servers if not already running
        if not self.servers_started:
            if not self.start_servers():
                logger.error("Failed to start servers, aborting tests")
                return {"success": False, "error": "Failed to start servers"}

        # Run individual test groups
        test_results = {}

        try:
            # Test error handling
            error_results = self.test_error_handling()
            test_results["error_handling"] = error_results

            # Test learning capabilities
            learning_results = self.test_learning_capabilities()
            test_results["learning_capabilities"] = learning_results

            # Test self-retrieval
            retrieval_results = self.test_self_retrieval()
            test_results["self_retrieval"] = retrieval_results

            # Test self-feedback
            feedback_results = self.test_self_feedback()
            test_results["self_feedback"] = feedback_results

            # Test recovery mechanisms
            recovery_results = self.test_recovery_mechanism()
            test_results["recovery_mechanisms"] = recovery_results

            # Stress test (optional)
            if include_stress_test:
                stress_results = self.run_stress_test()
                test_results["stress_test"] = stress_results

        except Exception as e:
            logger.error(f"Error during test suite: {str(e)}")
            test_results["error"] = str(e)

        # Get performance metrics
        metrics_summary = self.metrics.get_summary()
        test_results["performance_metrics"] = metrics_summary

        # Calculate overall success
        success = all(
            result.get("name") in test_results
            for result in [error_results, learning_results, retrieval_results, feedback_results, recovery_results]
        )

        total_time = time.time() - start_time

        return {
            "success": success,
            "total_time_seconds": total_time,
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results
        }

    def run_basic_query_test(self, num_queries: int = DEFAULT_TEST_QUERIES) -> Dict[str, Any]:
        """Run basic query tests"""
        logger.info(f"Running basic query test with {num_queries} queries...")

        # Start the servers if not already running
        if not self.servers_started:
            if not self.start_servers():
                logger.error("Failed to start servers, aborting tests")
                return {"success": False, "error": "Failed to start servers"}

        results = []

        for i in range(num_queries):
            query_idx = i % len(SAMPLE_QUERIES)
            query = SAMPLE_QUERIES[query_idx]

            logger.info(f"Running query {i+1}/{num_queries}: {query[:30]}...")

            result = self.run_single_query(query)
            results.append(result)

            # Add a short delay to avoid overwhelming the system
            time.sleep(0.1)

        # Calculate success rate
        success_count = sum(1 for r in results if r.get("success", False))
        success_rate = success_count / num_queries

        metrics_summary = self.metrics.get_summary()

        return {
            "success": success_rate > 0.9,  # Consider successful if >90% queries succeed
            "query_count": num_queries,
            "successful_queries": success_count,
            "success_rate": success_rate,
            "performance_metrics": metrics_summary,
            "results": results
        }

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save test results to a file

        Args:
            results: Test results to save
            filename: Optional filename, will generate if not provided

        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"

        output_dir = os.path.join(parent_dir, "logs", "test_results")
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, filename)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Test results saved to {output_path}")
        return output_path

    def run(self,
           test_type: str = "basic",
           num_queries: int = DEFAULT_TEST_QUERIES,
           include_stress: bool = False) -> Dict[str, Any]:
        """
        Run tests based on specified type

        Args:
            test_type: Type of test to run ('basic', 'full')
            num_queries: Number of queries for basic test
            include_stress: Whether to include stress tests

        Returns:
            Dict with test results
        """
        try:
            if test_type == "full":
                results = self.run_full_test_suite(include_stress)
            else:
                results = self.run_basic_query_test(num_queries)

            # Save results
            results_path = self.save_results(results)
            results["results_path"] = results_path

            return results

        finally:
            # Always stop servers and clean up
            self.stop_servers()
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Super AI System Integration Test")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--stress", action="store_true", help="Include stress tests")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test data")
    parser.add_argument("--queries", type=int, default=DEFAULT_TEST_QUERIES, help="Number of queries for basic test")

    args = parser.parse_args()

    print("==================================================")
    print("  Super AI System Integration Test")
    print("==================================================")
    print(f"Test type: {'Full Test Suite' if args.full else 'Basic Query Test'}")
    print(f"Stress test: {'Included' if args.stress else 'Not included'}")
    print(f"Queries: {args.queries if not args.full else 'Based on test suite'}")
    print(f"Cleanup: {'Disabled' if args.no_cleanup else 'Enabled'}")
    print("==================================================")

    # Run the tests
    tester = SystemTester(cleanup_after=not args.no_cleanup)
    test_type = "full" if args.full else "basic"

    start_time = time.time()
    results = tester.run(test_type, args.queries, args.stress)
    total_time = time.time() - start_time

    # Print summary
    print("\n==================================================")
    print("  Test Summary")
    print("==================================================")
    print(f"Success: {'Yes' if results.get('success', False) else 'No'}")
    print(f"Total time: {total_time:.2f} seconds")

    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print("\nPerformance Metrics:")
        print(f"  Avg response time: {metrics['response_times']['avg']:.2f} ms")
        print(f"  P95 response time: {metrics['response_times']['p95']:.2f} ms")
        print(f"  Errors: {metrics['errors']['count']}")
        print(f"  Timeouts: {metrics['timeouts']['count']}")

    if test_type == "basic":
        print(f"\nQuery Results:")
        print(f"  Total queries: {results.get('query_count', 0)}")
        print(f"  Successful: {results.get('successful_queries', 0)}")
        print(f"  Success rate: {results.get('success_rate', 0):.2%}")

    print(f"\nDetailed results saved to: {results.get('results_path', 'unknown')}")
    print("==================================================")

    sys.exit(0 if results.get('success', False) else 1)

if __name__ == "__main__":
    main()
