# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Learning Integration Module

This module integrates the learning engine with the core AI system.
It provides hooks for other components to use the learning capabilities:
- Self-learning (auto-retraining after conversations)
- Self-retrieval (internal memory search before web)
- Self-feedback (internal rating of answers)
"""

import os
import sys
import time
import logging
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our learning engine
from core_ai.learning_engine import (
    process_query,
    retrieve_info,
    select_best_model,
    learning_engine
)

# Try to import system utilities
try:
    from web_interface.system_utils import (
        log_decision,
        log_tool_call,
        timeout,
        fallback
    )
except ImportError:
    # Fallback logging functions
    def log_decision(decision, metadata=None):
        logging.info(f"DECISION: {decision} - META: {metadata}")

    def log_tool_call(tool_name, args, result):
        logging.info(f"TOOL CALL: {tool_name} - ARGS: {args} - RESULT: {result}")

    # Decorator stubs
    def timeout(seconds):
        def decorator(func):
            return func
        return decorator

    def fallback(default_return=None):
        def decorator(func):
            return func
        return decorator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LearningIntegration:
    """
    Integrates learning capabilities with the core AI system
    """

    def __init__(self):
        """Initialize the learning integration"""
        self.last_query_time = time.time()
        self.query_count = 0
        logger.info("Learning integration initialized")

    @fallback(default_return={"model": "medium", "priority": "balanced"})
    def get_optimal_model(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Get the optimal model to handle a given query

        Args:
            query: The user query
            context: Additional context for decision making

        Returns:
            Dict with model name and priority
        """
        # Classify the query type
        query_type = self._classify_query_type(query)

        # Determine performance priority based on context
        priority = self._determine_priority(query, context)

        # Get the best model based on learned strategies
        model = select_best_model(query_type, priority)

        log_decision(
            "Selected optimal model",
            {"query_type": query_type, "priority": priority, "model": model}
        )

        return {
            "model": model,
            "priority": priority,
            "query_type": query_type
        }

    @fallback(default_return={"source": "none", "results": []})
    def retrieve_information(self, query: str, use_web_fallback: bool = True) -> Dict[str, Any]:
        """
        Retrieve information for a query, first checking internal memory

        Args:
            query: The search query
            use_web_fallback: Whether to use web search if memory fails

        Returns:
            Dict with source and results
        """
        return retrieve_info(query, use_web_fallback)

    @fallback(default_return={"rating": "NEUTRAL", "score": 0.5})
    def evaluate_response(self,
                         query: str,
                         response: str,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the quality of a response

        Args:
            query: The user query
            response: The AI's response
            metadata: Additional context about the response generation

        Returns:
            Dict with evaluation results
        """
        # Track query count for retraining
        self.query_count += 1

        # Process and store the interaction
        result = process_query(query, response, metadata)

        # Return a simplified view of the feedback
        return {
            "memory_id": result.get("memory_id", ""),
            "rating": result.get("feedback", {}).get("rating", "NEUTRAL"),
            "score": result.get("feedback", {}).get("overall_score", 0.5),
            "suggestions": result.get("feedback", {}).get("suggestions", []),
            "conversation_count": result.get("conversation_count", 0)
        }

    def _classify_query_type(self, query: str) -> str:
        """
        Classify the query type

        Args:
            query: The user query

        Returns:
            str: Query type classification
        """
        # Simple heuristic classification
        query_lower = query.lower()

        # Check for code-related indicators
        code_keywords = ["code", "function", "program", "script", "error", "bug", "debug",
                        "compile", "syntax", "api", "library", "import"]
        if any(keyword in query_lower for keyword in code_keywords):
            return "code"

        # Check for factual question indicators
        factual_starters = ["what is", "who is", "when did", "where is", "how does",
                           "explain", "describe", "define", "tell me about"]
        if any(query_lower.startswith(starter) for starter in factual_starters):
            return "factual"

        # Check for creative indicators
        creative_keywords = ["generate", "create", "write", "design", "imagine",
                            "story", "creative", "ideas", "suggest"]
        if any(keyword in query_lower for keyword in creative_keywords):
            return "creative"

        # Default to general
        return "general"

    def _determine_priority(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Determine the performance priority for a query

        Args:
            query: The user query
            context: Additional context

        Returns:
            str: Priority ('speed', 'accuracy', or 'balanced')
        """
        # Default priority
        priority = "balanced"

        # If context explicitly specifies priority, use that
        if context and "priority" in context:
            if context["priority"] in ["speed", "accuracy", "balanced"]:
                return context["priority"]

        # Check for indicators of urgency in the query
        urgency_indicators = ["quickly", "fast", "urgent", "immediately", "asap", "hurry"]
        if any(indicator in query.lower() for indicator in urgency_indicators):
            priority = "speed"

        # Check for indicators of accuracy needs
        accuracy_indicators = ["accurate", "precise", "exact", "detailed", "thorough", "comprehensive"]
        if any(indicator in query.lower() for indicator in accuracy_indicators):
            priority = "accuracy"

        return priority

    def shutdown(self):
        """Shutdown the integration, ensuring all data is saved"""
        learning_engine.shutdown()
        logger.info("Learning integration shutdown cleanly")

# Create singleton instance
learning_integration = LearningIntegration()

# Convenience functions for using the integration

def get_best_model(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Get the best model for a query (singleton accessor)"""
    return learning_integration.get_optimal_model(query, context)

def search_before_web(query: str, use_web_fallback: bool = True) -> Dict[str, Any]:
    """Search internal memory before web (singleton accessor)"""
    return learning_integration.retrieve_information(query, use_web_fallback)

def rate_response(query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rate a response (singleton accessor)"""
    return learning_integration.evaluate_response(query, response, metadata)

# Clean up on process exit
import atexit
atexit.register(learning_integration.shutdown)

if __name__ == "__main__":
    # Simple self-test
    print("Testing learning integration...")

    # Test model selection
    test_query = "Write a function to calculate Fibonacci numbers"
    model_info = get_best_model(test_query)
    print(f"Query type: {model_info.get('query_type', 'unknown')}")
    print(f"Selected model: {model_info.get('model', 'unknown')}")
    print(f"Priority: {model_info.get('priority', 'unknown')}")

    # Test memory retrieval
    test_query = "How to implement a binary search tree in Python?"
    response = "A binary search tree (BST) is a data structure where each node has at most two children, with left child values less than the node and right child values greater than the node. Here's how to implement it in Python: [code example]"

    # First, store this information
    rate_response(test_query, response, {"model_name": "medium", "runtime_seconds": 1.2})

    # Now test retrieval
    retrieved = search_before_web("binary search tree python")
    print(f"Retrieved from: {retrieved.get('source', 'unknown')}")
    if retrieved.get('results'):
        print(f"Found {len(retrieved['results'])} results")

    print("Test complete")
