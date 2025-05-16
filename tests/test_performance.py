# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Performance Test Script for Super AI System

This script tests the performance of the Super AI system under various load conditions,
measuring response times, resource usage, and error rates.
"""

import os
import sys
import time
import asyncio
import random
import json
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.performance_monitor import get_monitor
from utils.performance_analyzer import get_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("Development/logs/performance_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PerformanceTest")

# Sample test queries with varying complexity
TEST_QUERIES = [
    # Simple factual queries
    {"query": "What is the capital of France?", "complexity": "low"},
    {"query": "Who wrote Romeo and Juliet?", "complexity": "low"},
    {"query": "What is the boiling point of water?", "complexity": "low"},
    {"query": "What is the tallest mountain in the world?", "complexity": "low"},
    {"query": "Who is the current president of the United States?", "complexity": "low"},

    # Medium complexity queries
    {"query": "Compare and contrast renewable and non-renewable energy sources.", "complexity": "medium"},
    {"query": "Explain the theory of relativity in simple terms.", "complexity": "medium"},
    {"query": "What are the major causes and effects of climate change?", "complexity": "medium"},
    {"query": "Describe the process of photosynthesis and why it's important.", "complexity": "medium"},
    {"query": "What are the main differences between classical and quantum computing?", "complexity": "medium"},

    # Complex queries requiring reasoning
    {"query": "Design a sustainable urban transportation system for a city of 1 million people.", "complexity": "high"},
    {"query": "Analyze the economic implications of universal basic income.", "complexity": "high"},
    {"query": "Propose a solution to reduce plastic pollution in oceans.", "complexity": "high"},
    {"query": "Evaluate the ethical considerations in developing artificial general intelligence.", "complexity": "high"},
    {"query": "Develop a strategy for a small nation to achieve carbon neutrality by 2030.", "complexity": "high"},

    # Edge cases
    {"query": "", "complexity": "edge"},  # Empty query
    {"query": "?", "complexity": "edge"},  # Single character
    {"query": "Please provide a detailed analysis of the entire history of human civilization from prehistoric times to present day, including major cultural, technological, and social developments across all continents.", "complexity": "edge"},  # Extremely long query
    {"query": "¿Cómo estás hoy? Comment allez-vous? Wie geht es dir?", "complexity": "edge"},  # Multi-language
    {"query": "<script>alert('XSS')</script>", "complexity": "edge"}  # Injection attempt
]

async def simulate_user_interaction(system_func, query, session_id=None, user_id="test_user"):
    """Simulate a user interaction with the system."""
    try:
        start_time = time.time()

        # If system_func is asynchronous
        if asyncio.iscoroutinefunction(system_func):
            response = await system_func(query, session_id=session_id, user_id=user_id)
        else:
            response = system_func(query, session_id=session_id, user_id=user_id)

        end_time = time.time()
        elapsed = end_time - start_time

        return {
            "query": query,
            "response_length": len(response) if response else 0,
            "response_time": elapsed,
            "success": True,
            "session_id": session_id
        }
    except Exception as e:
        end_time = time.time()
        elapsed = end_time - start_time

        logger.error(f"Error processing query: {query}")
        logger.error(f"Exception: {str(e)}")

        return {
            "query": query,
            "response_length": 0,
            "response_time": elapsed,
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

async def run_sequential_test(system_func, num_queries=10, session_id=None):
    """Run sequential queries to test system performance."""
    results = []

    # Select a subset of queries if specified
    queries = random.sample(TEST_QUERIES, min(num_queries, len(TEST_QUERIES)))

    logger.info(f"Running sequential test with {len(queries)} queries")

    for i, query_data in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: {query_data['query'][:50]}...")
        result = await simulate_user_interaction(system_func, query_data["query"], session_id)
        result["complexity"] = query_data["complexity"]
        results.append(result)

    return results

async def run_concurrent_test(system_func, concurrency=5, num_queries=20):
    """Run concurrent queries to test system performance under load."""
    results = []

    # Select a subset of queries if specified
    query_pool = random.sample(TEST_QUERIES * (num_queries // len(TEST_QUERIES) + 1), num_queries)

    logger.info(f"Running concurrent test with {num_queries} queries at concurrency level {concurrency}")

    # Create session IDs for concurrent users
    session_ids = [f"test_session_{i}" for i in range(concurrency)]

    async def process_batch(batch_queries):
        batch_results = []
        tasks = []

        for j, query_data in enumerate(batch_queries):
            session_id = session_ids[j % len(session_ids)]
            task = asyncio.create_task(
                simulate_user_interaction(system_func, query_data["query"], session_id)
            )
            tasks.append((task, query_data["complexity"]))

        for task, complexity in tasks:
            result = await task
            result["complexity"] = complexity
            batch_results.append(result)

        return batch_results

    # Process queries in batches of concurrency
    batches = [query_pool[i:i+concurrency] for i in range(0, len(query_pool), concurrency)]

    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        batch_results = await process_batch(batch)
        results.extend(batch_results)

    return results

def analyze_results(results, test_type):
    """Analyze the results of a performance test."""
    if not results:
        logger.warning("No results to analyze")
        return {}

    # Extract result metrics
    response_times = [r["response_time"] for r in results if r["success"]]
    success_count = sum(1 for r in results if r["success"])
    failure_count = len(results) - success_count

    # Calculate stats
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    if response_times:
        response_times.sort()
        p50 = response_times[len(response_times) // 2]
        p90 = response_times[int(len(response_times) * 0.9)]
        p95 = response_times[int(len(response_times) * 0.95)]
        p99 = response_times[int(len(response_times) * 0.99)]
    else:
        p50 = p90 = p95 = p99 = 0

    # Group by complexity
    complexity_stats = {}
    for complexity in ["low", "medium", "high", "edge"]:
        complexity_times = [r["response_time"] for r in results if r["success"] and r["complexity"] == complexity]
        if complexity_times:
            complexity_stats[complexity] = {
                "count": len(complexity_times),
                "avg_time": sum(complexity_times) / len(complexity_times),
                "min_time": min(complexity_times),
                "max_time": max(complexity_times)
            }

    # Group by success/failure
    success_rate = (success_count / len(results)) * 100 if results else 0

    # Collect error types
    error_types = {}
    for r in results:
        if not r["success"] and "error" in r:
            error_type = str(type(r["error"]).__name__)
            error_types[error_type] = error_types.get(error_type, 0) + 1

    return {
        "test_type": test_type,
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "response_times": {
            "average": avg_response_time,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "min": min(response_times) if response_times else 0,
            "max": max(response_times) if response_times else 0
        },
        "complexity_stats": complexity_stats,
        "error_types": error_types
    }

def save_results(analysis, filename):
    """Save the analysis results to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Results saved to {filename}")

async def main(system_func, args):
    """Main test runner function."""
    # Start the performance monitor
    monitor = get_monitor()
    monitor.start_monitoring()

    try:
        # Configure test parameters
        tests_to_run = []

        if args.sequential:
            tests_to_run.append(("sequential", args.num_queries))

        if args.concurrent:
            concurrency_levels = [1, 2, 5, 10, 20] if args.stress else [args.concurrency]
            for concurrency in concurrency_levels:
                tests_to_run.append(("concurrent", args.num_queries, concurrency))

        # Run all configured tests
        all_results = {}

        for test_config in tests_to_run:
            if test_config[0] == "sequential":
                logger.info(f"Running sequential test with {test_config[1]} queries")
                session_id = f"test_session_{int(time.time())}"
                results = await run_sequential_test(system_func, test_config[1], session_id)
                analysis = analyze_results(results, "sequential")
                all_results["sequential"] = analysis

                # Save individual test results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results(analysis, f"Development/logs/test_results/sequential_{timestamp}.json")

            elif test_config[0] == "concurrent":
                num_queries = test_config[1]
                concurrency = test_config[2]
                logger.info(f"Running concurrent test with {num_queries} queries at concurrency level {concurrency}")
                results = await run_concurrent_test(system_func, concurrency, num_queries)
                analysis = analyze_results(results, f"concurrent_{concurrency}")
                all_results[f"concurrent_{concurrency}"] = analysis

                # Save individual test results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_results(analysis, f"Development/logs/test_results/concurrent_{concurrency}_{timestamp}.json")

        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_results(all_results, f"Development/logs/test_results/combined_{timestamp}.json")

        # Generate performance analysis report
        if args.analysis:
            logger.info("Generating performance analysis report")
            analyzer = get_analyzer()
            report = analyzer.generate_report(f"Development/logs/analysis/performance_report_{timestamp}.md")
            logger.info(f"Analysis report generated")

        return all_results

    finally:
        # Stop the performance monitor
        monitor.stop_monitoring()

def dummy_system_func(query, session_id=None, user_id=None):
    """Dummy function to simulate the Super AI system for testing purposes."""
    complexity = None
    for q in TEST_QUERIES:
        if q["query"] == query:
            complexity = q["complexity"]
            break

    # Simulate processing time based on complexity
    if complexity == "low":
        time.sleep(random.uniform(0.1, 0.5))
        return "This is a simple response."
    elif complexity == "medium":
        time.sleep(random.uniform(0.5, 1.5))
        return "This is a more detailed response with some analysis and context."
    elif complexity == "high":
        time.sleep(random.uniform(1.5, 3.0))
        return "This is a comprehensive response that includes detailed analysis, multiple perspectives, and actionable insights."
    elif complexity == "edge":
        # Edge cases have a chance to fail
        if random.random() < 0.3:
            raise Exception("Failed to process edge case query")
        time.sleep(random.uniform(0.5, 4.0))
        return "Edge case response."
    else:
        time.sleep(random.uniform(0.3, 1.0))
        return "Generic response."

if __name__ == "__main__":
    # Create results directory
    os.makedirs("Development/logs/test_results", exist_ok=True)

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Performance test for Super AI system")

    # Test type arguments
    parser.add_argument("--sequential", action="store_true", help="Run sequential test")
    parser.add_argument("--concurrent", action="store_true", help="Run concurrent test")
    parser.add_argument("--stress", action="store_true", help="Run stress test with multiple concurrency levels")

    # Test configuration
    parser.add_argument("--num-queries", type=int, default=20, help="Number of queries to run")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency level for concurrent test")
    parser.add_argument("--analysis", action="store_true", help="Generate performance analysis report")

    args = parser.parse_args()

    # Default to sequential if no test type specified
    if not (args.sequential or args.concurrent):
        args.sequential = True

    # Run the tests using asyncio
    asyncio.run(main(dummy_system_func, args))
