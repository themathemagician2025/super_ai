# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
System Integration Tests for Super AI System

This module provides comprehensive tests for validating the integration of all
system components, including error handling, logging, recovery mechanisms,
and self-learning capabilities.
"""

import os
import sys
import time
import json
import asyncio
import logging
import unittest
import random
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import system components
from src.core.engine import Engine
from src.core.memory import Memory
from src.core.self_feedback import SelfFeedback
from src.core.conversation import Conversation
from src.utils.performance_monitor import get_monitor
from src.utils.logger import setup_logger

# Setup testing logger
logger = setup_logger("integration_tests", "Development/logs/integration_tests.log")

class SystemIntegrationTests(unittest.TestCase):
    """Test suite for system integration tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Ensure test directories exist
        os.makedirs("Development/logs", exist_ok=True)
        os.makedirs("Development/data/test", exist_ok=True)

        # Initialize performance monitor
        cls.monitor = get_monitor()
        cls.monitor.start_monitoring()

        # Load test data
        cls.test_queries = cls._load_test_queries()

        logger.info("System integration test suite initialized")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are complete."""
        cls.monitor.stop_monitoring()
        logger.info("System integration testing complete")

    def setUp(self):
        """Set up before each test."""
        # Create fresh instances for each test
        self.engine = Engine()
        self.memory = Memory()
        self.feedback = SelfFeedback()

        # Reset conversation ID counter
        Conversation._id_counter = 0

        logger.info(f"Setting up test: {self._testMethodName}")

    def tearDown(self):
        """Clean up after each test."""
        # Force garbage collection to prevent memory leaks
        import gc
        gc.collect()

        logger.info(f"Completed test: {self._testMethodName}")

    @staticmethod
    def _load_test_queries():
        """Load test queries from JSON file or create default set."""
        test_queries_path = Path("Development/data/test/test_queries.json")

        if test_queries_path.exists():
            try:
                with open(test_queries_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load test queries: {e}")

        # Default test queries
        default_queries = {
            "basic": [
                "What is the weather today?",
                "Tell me a joke",
                "What time is it?",
                "Who are you?",
                "How does this system work?"
            ],
            "complex": [
                "Explain quantum computing and provide examples of its applications",
                "Compare and contrast five different machine learning algorithms",
                "Write a recursive algorithm to solve the Tower of Hanoi problem",
                "Analyze the economic impacts of climate change on global agriculture",
                "Explain how transformer models work in natural language processing"
            ],
            "memory_dependent": [
                "What did we talk about earlier?",
                "Continue our previous discussion",
                "You mentioned something about X before, can you elaborate?",
                "I want to revisit the topic from our last conversation",
                "What was that recommendation you gave me?"
            ],
            "error_inducing": [
                "intentional_timeout_query",
                "intentional_exception_query",
                "This query will be cut off mid-sentence and",
                "Send 50 follow-up messages at once",
                "{malformed_json: query with syntax errors"
            ]
        }

        # Save default queries for future use
        try:
            with open(test_queries_path, 'w') as f:
                json.dump(default_queries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save default test queries: {e}")

        return default_queries

    #----------------------------------------------------------------------
    # Core Functionality Tests
    #----------------------------------------------------------------------

    def test_basic_conversation_flow(self):
        """Test basic end-to-end conversation flow with simple queries."""
        logger.info("Testing basic conversation flow")

        # Create a new conversation
        conversation = Conversation(self.engine, self.memory)

        for query in self.test_queries["basic"]:
            start_time = time.time()

            response = conversation.process_message(query)

            # Record performance metrics
            response_time = time.time() - start_time
            self.monitor.record_response_time(response_time)

            # Validate response
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 0)

            logger.info(f"Query: '{query}' - Response length: {len(response)}")

    def test_complex_query_handling(self):
        """Test system's ability to handle complex, knowledge-intensive queries."""
        logger.info("Testing complex query handling")

        conversation = Conversation(self.engine, self.memory)

        for query in self.test_queries["complex"]:
            start_time = time.time()

            response = conversation.process_message(query)

            # Record performance metrics
            response_time = time.time() - start_time
            self.monitor.record_response_time(response_time)

            # Validate response
            self.assertIsNotNone(response)
            self.assertTrue(len(response) > 100)  # Complex responses should be substantial

            logger.info(f"Complex query processed in {response_time:.2f}s - Response length: {len(response)}")

    #----------------------------------------------------------------------
    # Memory Tests
    #----------------------------------------------------------------------

    def test_conversation_memory(self):
        """Test that the system correctly maintains conversation context."""
        logger.info("Testing conversation memory")

        conversation = Conversation(self.engine, self.memory)

        # Initial query to establish context
        initial_query = "My name is Alice and I'm interested in machine learning."
        conversation.process_message(initial_query)

        # Follow-up query that requires remembering the user's name
        follow_up = "What was my name again?"
        response = conversation.process_message(follow_up)

        # The response should mention "Alice"
        self.assertIn("Alice", response)

        logger.info("Memory test successful - system remembered context")

    def test_long_term_memory_retrieval(self):
        """Test retrieval of information from long-term memory."""
        logger.info("Testing long-term memory retrieval")

        # Setup: Store something in memory
        key_info = "The password is BlueSky123"
        self.memory.store("test_memory", {
            "content": key_info,
            "timestamp": time.time(),
            "tags": ["password", "important"]
        })

        # Create a new conversation with the memory system
        conversation = Conversation(self.engine, self.memory)

        # Query should trigger memory retrieval
        query = "What was the password I told you earlier?"
        response = conversation.process_message(query)

        # The response should contain the password
        self.assertIn("BlueSky123", response)

        logger.info("Long-term memory retrieval test successful")

    #----------------------------------------------------------------------
    # Error Handling Tests
    #----------------------------------------------------------------------

    def test_timeout_recovery(self):
        """Test system's ability to handle and recover from timeouts."""
        logger.info("Testing timeout recovery")

        # Mock the engine to simulate a timeout
        with patch.object(self.engine, 'process_query', side_effect=asyncio.TimeoutError):
            conversation = Conversation(self.engine, self.memory)

            # This should trigger the timeout handler
            query = "Regular query that will timeout"
            start_time = time.time()

            response = conversation.process_message(query)

            # Record error
            self.monitor.record_error()

            # Validate response indicates timeout and provides fallback
            self.assertIsNotNone(response)
            self.assertIn("timeout", response.lower())

            logger.info(f"Timeout recovery successful in {time.time() - start_time:.2f}s")

    def test_exception_handling(self):
        """Test system's ability to handle unexpected exceptions."""
        logger.info("Testing exception handling")

        # Mock the engine to raise different exceptions
        exceptions = [
            ValueError("Invalid parameter"),
            KeyError("Missing key"),
            RuntimeError("Unexpected error"),
            MemoryError("Out of memory")
        ]

        for exception in exceptions:
            with patch.object(self.engine, 'process_query', side_effect=exception):
                conversation = Conversation(self.engine, self.memory)

                query = f"Query that will raise {type(exception).__name__}"

                try:
                    response = conversation.process_message(query)

                    # Record error
                    self.monitor.record_error()

                    # Validate graceful handling
                    self.assertIsNotNone(response)
                    self.assertTrue("error" in response.lower() or
                                   "unable" in response.lower() or
                                   "sorry" in response.lower())

                    logger.info(f"Successfully handled {type(exception).__name__}")

                except Exception as e:
                    self.fail(f"Exception not properly handled: {str(e)}")

    def test_malformed_input_handling(self):
        """Test system's ability to handle malformed or problematic input."""
        logger.info("Testing malformed input handling")

        conversation = Conversation(self.engine, self.memory)

        malformed_inputs = [
            "",  # Empty string
            " ",  # Whitespace only
            "a" * 10000,  # Extremely long input
            "🤖" * 100,  # Many emoji
            "<script>alert('XSS')</script>",  # Potential XSS
            "SELECT * FROM users;",  # SQL injection attempt
            "User: What is my password?"  # Impersonation attempt
        ]

        for input_text in malformed_inputs:
            start_time = time.time()

            response = conversation.process_message(input_text)

            # Validate response
            self.assertIsNotNone(response)

            # Response shouldn't be excessively long regardless of input
            self.assertTrue(len(response) < 5000)

            logger.info(f"Handled malformed input in {time.time() - start_time:.2f}s - "
                      f"Input type: {type(input_text)}, length: {len(input_text)}")

    #----------------------------------------------------------------------
    # Self-Learning Tests
    #----------------------------------------------------------------------

    def test_self_feedback_mechanism(self):
        """Test that the system can evaluate its own responses."""
        logger.info("Testing self-feedback mechanism")

        # Create conversation with feedback enabled
        conversation = Conversation(self.engine, self.memory, self.feedback)

        query = "Explain quantum computing to a 5-year-old"
        response = conversation.process_message(query)

        # Get feedback on the response
        feedback_scores = self.feedback.evaluate_response(query, response)

        # Validate feedback structure
        self.assertIsInstance(feedback_scores, dict)
        self.assertIn("clarity", feedback_scores)
        self.assertIn("accuracy", feedback_scores)
        self.assertIn("helpfulness", feedback_scores)

        for key, score in feedback_scores.items():
            self.assertIsInstance(score, (int, float))
            self.assertTrue(0 <= score <= 10)

        logger.info(f"Self-feedback scores: {feedback_scores}")

    def test_learning_from_corrections(self):
        """Test that the system learns from user corrections."""
        logger.info("Testing learning from corrections")

        conversation = Conversation(self.engine, self.memory, self.feedback)

        # First interaction with incorrect information
        query1 = "Who invented the telephone?"
        response1 = conversation.process_message(query1)

        # Simulate user correction
        correction = "Actually, many historians believe Antonio Meucci invented the telephone before Bell."
        conversation.process_message(correction)

        # Ask similar question later
        query2 = "Tell me about the invention of the telephone."
        response2 = conversation.process_message(query2)

        # The corrected information should be reflected
        self.assertIn("Meucci", response2)

        logger.info("System successfully learned from correction")

    #----------------------------------------------------------------------
    # Concurrency Tests
    #----------------------------------------------------------------------

    def test_concurrent_conversations(self):
        """Test system's ability to handle multiple concurrent conversations."""
        logger.info("Testing concurrent conversations")

        num_concurrent = 5
        num_messages_per_conversation = 3

        async def run_conversation(conv_id):
            """Run a conversation with multiple messages."""
            conversation = Conversation(self.engine, self.memory)
            results = []

            for i in range(num_messages_per_conversation):
                # Mix of query types
                if i % 3 == 0:
                    query_list = self.test_queries["basic"]
                elif i % 3 == 1:
                    query_list = self.test_queries["complex"]
                else:
                    query_list = self.test_queries["memory_dependent"]

                query = random.choice(query_list)

                start_time = time.time()
                response = conversation.process_message(query)
                elapsed = time.time() - start_time

                # Record metrics
                self.monitor.record_response_time(elapsed)

                results.append({
                    "conv_id": conv_id,
                    "message_num": i,
                    "query": query,
                    "response_length": len(response),
                    "time": elapsed
                })

                # Small delay to simulate real user interaction
                await asyncio.sleep(0.1)

            return results

        async def run_all_conversations():
            """Run all conversations concurrently."""
            # Update active sessions count
            self.monitor.update_active_sessions(num_concurrent)

            # Start all conversations
            tasks = [run_conversation(i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)

            # Update active sessions count
            self.monitor.update_active_sessions(0)

            return [item for sublist in results for item in sublist]

        # Run the concurrent conversations
        results = asyncio.run(run_all_conversations())

        # Validate
        self.assertEqual(len(results), num_concurrent * num_messages_per_conversation)

        # Log results
        total_time = sum(r["time"] for r in results)
        avg_time = total_time / len(results)
        logger.info(f"Concurrent conversations test: {len(results)} messages processed")
        logger.info(f"Average response time: {avg_time:.2f}s, Total time: {total_time:.2f}s")

    #----------------------------------------------------------------------
    # Recovery Tests
    #----------------------------------------------------------------------

    def test_component_failure_recovery(self):
        """Test system's ability to recover from component failures."""
        logger.info("Testing component failure recovery")

        # Create a conversation with real components
        conversation = Conversation(self.engine, self.memory)

        # First, process a normal message
        response1 = conversation.process_message("Remember that my favorite color is blue.")
        self.assertIsNotNone(response1)

        # Now simulate a memory component failure
        with patch.object(self.memory, 'retrieve', side_effect=Exception("Memory failure")):
            # The system should use a fallback mechanism
            response2 = conversation.process_message("What's my favorite color?")

            # Should get a response despite memory failure
            self.assertIsNotNone(response2)
            self.monitor.record_error()

            logger.info("System responded despite memory component failure")

        # After "recovery", memory should work again
        response3 = conversation.process_message("Can you remind me of my favorite color?")
        self.assertIsNotNone(response3)
        self.assertIn("blue", response3.lower())

        logger.info("System recovered and correctly used memory after component recovery")

    def test_graceful_degradation(self):
        """Test that the system degrades gracefully under extreme load."""
        logger.info("Testing graceful degradation under load")

        # Create a conversation
        conversation = Conversation(self.engine, self.memory)

        # First, get baseline performance
        query = "What is artificial intelligence?"
        start_time = time.time()
        baseline_response = conversation.process_message(query)
        baseline_time = time.time() - start_time

        # Simulate extreme load
        with patch('src.core.engine.Engine.process_query',
                  side_effect=lambda q, c: time.sleep(0.5) or f"Simplified response to: {q}"):

            # System should provide a simplified response under load
            for i in range(5):
                query = random.choice(self.test_queries["complex"])
                start_time = time.time()
                response = conversation.process_message(query)
                response_time = time.time() - start_time

                # Response should be quicker than baseline for complex queries
                self.assertLess(response_time, baseline_time * 2)
                self.assertIsNotNone(response)

                logger.info(f"Load test iteration {i+1}: response time {response_time:.2f}s")

    #----------------------------------------------------------------------
    # Main test runner
    #----------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
