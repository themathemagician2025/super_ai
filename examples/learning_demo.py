# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Demonstration of Super AI Learning Engine Features

This script demonstrates the three key features:
1. Self-learning: Automatic retraining after every 10 conversations
2. Self-retrieval: Internal memory search before using web
3. Self-feedback: Rating system for answer quality

Usage:
    python learning_demo.py
"""

import os
import sys
import time
import json
import random
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the learning integration
from core_ai.learning_integration import (
    get_best_model,
    search_before_web,
    rate_response
)

# Sample questions and answers for testing
SAMPLE_QUERIES = [
    "How do I create a binary search tree in Python?",
    "What is the time complexity of quicksort?",
    "Explain how neural networks work",
    "How can I optimize TensorFlow models?",
    "Write a function to calculate Fibonacci numbers",
    "What is the difference between supervised and unsupervised learning?",
    "How to implement a hash table from scratch?",
    "What are the principles of clean code?",
    "Explain the CAP theorem in distributed systems",
    "How to design a RESTful API?",
    "What are design patterns in software engineering?",
    "How to handle concurrency in multithreaded applications?",
]

SAMPLE_RESPONSES = [
    "A binary search tree is a data structure where each node has at most two children, with left child values less than node value and right child values greater. Implementation example: [code for BST implementation]",
    "Quicksort has an average time complexity of O(n log n), but worst case can be O(n²) when poorly implemented or with bad pivot selection. It's generally faster in practice than other O(n log n) sorting algorithms.",
    "Neural networks are computational models inspired by the human brain. They consist of layers of nodes (neurons) that process input data through weighted connections, activation functions, and backpropagation for learning.",
    "TensorFlow models can be optimized through quantization (reducing precision), pruning (removing unnecessary connections), model distillation (training smaller models from larger ones), and hardware-specific optimizations.",
    "Here's a Python function to calculate Fibonacci numbers: [code for Fibonacci function]",
    "Supervised learning uses labeled data where the algorithm learns to map inputs to known outputs. Unsupervised learning works with unlabeled data to find patterns or structure without predefined output categories.",
    "A hash table implementation requires a hash function to map keys to array indices, collision resolution (like chaining or open addressing), and methods for insertion, deletion, and lookup. [implementation details]",
    "Clean code principles include meaningful names, small functions with single responsibility, consistent formatting, proper error handling, minimal comments (self-documenting code), and avoiding duplication (DRY principle).",
    "The CAP theorem states that distributed systems can provide only two of three guarantees: Consistency (all nodes see the same data), Availability (every request receives a response), and Partition tolerance (system works despite network failures).",
    "RESTful API design includes using proper HTTP methods (GET, POST, PUT, DELETE), stateless interactions, resource-based URLs, proper status codes, and supporting content negotiation.",
    "Design patterns are reusable solutions to common software problems. Categories include Creational (e.g., Singleton, Factory), Structural (e.g., Adapter, Decorator), and Behavioral (e.g., Observer, Strategy) patterns.",
    "Concurrency in multithreaded applications requires synchronization mechanisms like locks, semaphores, and atomic operations to prevent race conditions and deadlocks. Thread-safe data structures and immutability also help.",
]

def simulate_query_response_cycle(query_idx: int, use_memory: bool = True) -> Dict[str, Any]:
    """
    Simulate a complete query-response cycle with model selection,
    memory retrieval, response generation and feedback

    Args:
        query_idx: Index of query to use
        use_memory: Whether to use memory retrieval

    Returns:
        Dict with results
    """
    query = SAMPLE_QUERIES[query_idx]

    print(f"\n--- Query {query_idx + 1} ---")
    print(f"User query: {query}")

    # 1. Select the best model for this query
    start_time = time.time()
    model_info = get_best_model(query)
    model_selection_time = time.time() - start_time

    print(f"Selected model: {model_info['model']} (query type: {model_info['query_type']}, priority: {model_info['priority']})")
    print(f"Model selection took {model_selection_time:.3f} seconds")

    # 2. Try to retrieve from memory first (if enabled)
    memory_results = None
    if use_memory:
        start_time = time.time()
        memory_results = search_before_web(query)
        memory_retrieval_time = time.time() - start_time

        source = memory_results.get('source', 'unknown')
        result_count = len(memory_results.get('results', []))

        print(f"Memory search: found {result_count} results from {source}")
        print(f"Memory retrieval took {memory_retrieval_time:.3f} seconds")

        # If we got a good memory match, use that instead of generating a new response
        if source == 'memory' and result_count > 0 and memory_results['results'][0]['score'] > 0.8:
            response = memory_results['results'][0]['response']
            print("Using response from memory (high confidence match)")
        else:
            # Generate new response
            response = SAMPLE_RESPONSES[query_idx]
            print("Generated new response (no good memory match)")
    else:
        # Generate new response without memory lookup
        response = SAMPLE_RESPONSES[query_idx]
        print("Generated new response (memory lookup disabled)")

    # 3. Rate the response quality
    start_time = time.time()
    metadata = {
        "model_name": model_info["model"],
        "runtime_seconds": random.uniform(0.5, 3.0),
        "speed_score": random.uniform(0.4, 0.9)
    }

    feedback = rate_response(query, response, metadata)
    feedback_time = time.time() - start_time

    print(f"Response rating: {feedback['rating']} (score: {feedback['score']:.2f})")
    if feedback.get('suggestions'):
        print(f"Improvement suggestions: {', '.join(feedback['suggestions'])}")
    print(f"Feedback generation took {feedback_time:.3f} seconds")

    # Return all results for analysis
    return {
        "query": query,
        "response": response,
        "model_info": model_info,
        "memory_results": memory_results,
        "feedback": feedback,
        "conversation_count": feedback.get("conversation_count", 0)
    }

def run_learning_demo(num_queries: int = 15):
    """
    Run a demonstration of the learning engine with multiple queries

    Args:
        num_queries: Number of queries to simulate
    """
    print("==================================================")
    print("  Super AI Learning Engine Demonstration")
    print("==================================================")
    print(f"Running {num_queries} simulated queries to demonstrate:")
    print("1. Self-learning: Automatic retraining after every 10 conversations")
    print("2. Self-retrieval: Internal memory search before using web")
    print("3. Self-feedback: Rating system for answer quality")
    print("==================================================")

    results = []

    # Run queries in sequence
    for i in range(num_queries):
        # Select a query (cycle through samples or choose random)
        query_idx = i % len(SAMPLE_QUERIES)

        # For the first half, don't use memory retrieval to build up the memory
        use_memory = i >= len(SAMPLE_QUERIES) // 2

        # Run the query-response cycle
        result = simulate_query_response_cycle(query_idx, use_memory)
        results.append(result)

        # Check if retraining occurred
        if result["conversation_count"] % 10 == 0:
            print("\n*** RETRAINING TRIGGERED ***")
            print(f"Learning engine has processed {result['conversation_count']} conversations")
            print("Model selection strategies have been updated based on feedback")

        # Pause between queries
        if i < num_queries - 1:
            print("\nWaiting 1 second before next query...")
            time.sleep(1)

    # Print summary
    print("\n==================================================")
    print("  Learning Engine Demonstration Summary")
    print("==================================================")
    print(f"Processed {len(results)} queries")
    print(f"Final conversation count: {results[-1]['conversation_count']}")
    print(f"Retraining cycles: {results[-1]['conversation_count'] // 10}")

    # Calculate statistics
    ratings = [r["feedback"]["rating"] for r in results]
    scores = [r["feedback"]["score"] for r in results]

    rating_counts = {}
    for rating in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1

    print("\nResponse Ratings:")
    for rating, count in rating_counts.items():
        percentage = (count / len(ratings)) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")

    avg_score = sum(scores) / len(scores)
    print(f"Average score: {avg_score:.2f}")

    print("\nDemonstration complete!")

if __name__ == "__main__":
    run_learning_demo()
