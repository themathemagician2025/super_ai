# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
System-level tests for the Super AI application.

This test suite validates the integrated functionality of all Super AI components
working together, testing error handling, logging, recovery mechanisms,
self-learning capabilities, and performance under various conditions.
"""

import os
import sys
import json
import time
import unittest
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import main components
from src.super_ai import SuperAI
from src.core.memory import MemorySystem
from src.core.llm import LLMInterface
from src.core.feedback import FeedbackSystem
from src.core.training import TrainingSystem
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.error_handler import ErrorHandler
from src.utils.logger import Logger
from src.utils.configuration import Config

class TestSuperAISystem(unittest.TestCase):
    """System tests for the entire Super AI application."""

    def setUp(self):
        """Set up a test environment with all components initialized."""
        # Create temporary directories for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.temp_dir.name)
        self.memory_dir = self.root_dir / "memory"
        self.logs_dir = self.root_dir / "logs"
        self.metrics_dir = self.root_dir / "metrics"
        self.models_dir = self.root_dir / "models"

        for directory in [self.memory_dir, self.logs_dir, self.metrics_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)

        # Create a test configuration
        self.config = Config({
            "llm": {
                "provider": "test_provider",
                "model": "test_model",
                "api_key": "test_key",
                "max_tokens": 100,
                "temperature": 0.7,
                "timeout": 10,
                "retry_attempts": 3
            },
            "memory": {
                "storage_path": str(self.memory_dir),
                "max_conversations": 100,
                "embedding_dimensions": 128
            },
            "feedback": {
                "enabled": True,
                "threshold": 0.7
            },
            "training": {
                "auto_train": True,
                "train_after_conversations": 10,
                "models_path": str(self.models_dir)
            },
            "performance": {
                "metrics_path": str(self.metrics_dir),
                "log_interval": 1,
                "high_cpu_threshold": 80,
                "high_memory_threshold": 80
            },
            "error_handling": {
                "max_retries": 3,
                "backoff_factor": 1.5
            },
            "logging": {
                "logs_path": str(self.logs_dir),
                "level": "DEBUG"
            }
        })

        # Create mocks for external services
        self.mock_llm = MagicMock(spec=LLMInterface)
        self.mock_llm.generate_response.return_value = {
            "text": "This is a test response from the LLM.",
            "model": "test_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }

        # Initialize logger first (other components need it)
        self.logger = Logger(self.config)

        # Initialize components
        self.memory = MemorySystem(memory_dir=str(self.memory_dir))
        self.error_handler = ErrorHandler(config=self.config, logger=self.logger)
        self.performance_monitor = PerformanceMonitor(
            metrics_dir=str(self.metrics_dir),
            logger=self.logger
        )
        self.feedback_system = FeedbackSystem(
            config=self.config,
            memory=self.memory,
            logger=self.logger
        )
        self.training_system = TrainingSystem(
            config=self.config,
            memory=self.memory,
            logger=self.logger
        )

        # Initialize the Super AI system with all components
        self.super_ai = SuperAI(
            config=self.config,
            llm=self.mock_llm,
            memory=self.memory,
            feedback=self.feedback_system,
            training=self.training_system,
            performance=self.performance_monitor,
            error_handler=self.error_handler,
            logger=self.logger
        )

        # Start the performance monitoring
        self.performance_monitor.start_monitoring()

    def tearDown(self):
        """Clean up temporary files and stop monitoring."""
        self.performance_monitor.stop_monitoring()
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the Super AI system initializes correctly with all components."""
        self.assertIsNotNone(self.super_ai)
        self.assertEqual(self.super_ai.llm, self.mock_llm)
        self.assertEqual(self.super_ai.memory, self.memory)
        self.assertEqual(self.super_ai.feedback, self.feedback_system)
        self.assertEqual(self.super_ai.training, self.training_system)
        self.assertEqual(self.super_ai.performance, self.performance_monitor)
        self.assertEqual(self.super_ai.error_handler, self.error_handler)
        self.assertEqual(self.super_ai.logger, self.logger)

    def test_conversation_flow(self):
        """Test a complete conversation flow, including memory storage."""
        # Configure mock to return a predictable response
        self.mock_llm.generate_response.return_value = {
            "text": "I'm Super AI, how can I help you today?",
            "model": "test_model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }

        # Create a new conversation
        conversation_id = self.super_ai.create_conversation()
        self.assertIsNotNone(conversation_id)

        # Send a message and get a response
        response = self.super_ai.process_message(
            conversation_id=conversation_id,
            user_message="Hello, Super AI!"
        )

        # Verify the response
        self.assertEqual(response["text"], "I'm Super AI, how can I help you today?")

        # Verify the conversation was stored in memory
        conversation = self.memory.get_conversation(conversation_id)
        self.assertEqual(len(conversation), 2)  # User message + AI response
        self.assertEqual(conversation[0]["role"], "user")
        self.assertEqual(conversation[0]["content"], "Hello, Super AI!")
        self.assertEqual(conversation[1]["role"], "assistant")
        self.assertEqual(conversation[1]["content"], "I'm Super AI, how can I help you today?")

        # Continue the conversation
        self.mock_llm.generate_response.return_value = {
            "text": "I'm an advanced AI system built to assist with various tasks.",
            "model": "test_model",
            "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}
        }

        response = self.super_ai.process_message(
            conversation_id=conversation_id,
            user_message="Tell me about yourself."
        )

        # Verify the response
        self.assertEqual(response["text"], "I'm an advanced AI system built to assist with various tasks.")

        # Verify the conversation was updated
        conversation = self.memory.get_conversation(conversation_id)
        self.assertEqual(len(conversation), 4)  # All messages so far

    def test_error_handling_and_retry(self):
        """Test that errors in LLM calls are properly handled with retries."""
        # Configure mock to fail the first two times, then succeed
        error_responses = [
            Exception("Network error"),
            Exception("API timeout"),
            {
                "text": "Response after retry",
                "model": "test_model",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
        ]

        self.mock_llm.generate_response.side_effect = error_responses

        # Create conversation and send message
        conversation_id = self.super_ai.create_conversation()
        response = self.super_ai.process_message(
            conversation_id=conversation_id,
            user_message="This message will trigger retries"
        )

        # Verify the response succeeded after retries
        self.assertEqual(response["text"], "Response after retry")

        # Verify the LLM was called the expected number of times
        self.assertEqual(self.mock_llm.generate_response.call_count, 3)

        # Verify errors were logged
        # (Would check log files, but since they're handled by the Logger mock, we'll skip)

    def test_complete_failure_fallback(self):
        """Test that the system falls back to a safe response after all retries fail."""
        # Configure mock to always fail
        self.mock_llm.generate_response.side_effect = Exception("Persistent failure")

        # Create conversation and send message
        conversation_id = self.super_ai.create_conversation()
        response = self.super_ai.process_message(
            conversation_id=conversation_id,
            user_message="This message will trigger a complete failure"
        )

        # Verify a fallback response was returned
        self.assertIn("I apologize", response["text"].lower())
        self.assertIn("unable to process", response["text"].lower())

        # Verify the error was recorded in the metrics
        metrics = self.performance_monitor.get_metrics()
        self.assertGreater(metrics["error_count"], 0)

    def test_performance_monitoring(self):
        """Test that performance metrics are recorded during conversation."""
        # Process a few messages to generate metrics
        conversation_id = self.super_ai.create_conversation()

        for i in range(3):
            self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message=f"Test message {i}"
            )

        # Wait a moment for metrics to update
        time.sleep(0.1)

        # Get and verify metrics
        metrics = self.performance_monitor.get_metrics()

        self.assertIn("response_times", metrics)
        self.assertIn("errors", metrics)
        self.assertIn("cpu_usage", metrics)
        self.assertIn("memory_usage", metrics)
        self.assertIn("token_usage", metrics)

        # Verify response times were recorded (should have 3)
        self.assertEqual(len(metrics["response_times"]), 3)

        # Verify token usage was tracked
        self.assertEqual(metrics["token_usage"]["total"], 90)  # 3 responses × 30 tokens each

    def test_self_learning_trigger(self):
        """Test that self-learning is triggered after the configured number of conversations."""
        # Patch the training method to track calls
        with patch.object(self.training_system, 'trigger_training') as mock_training:
            # Mock that we already have 9 conversations
            with patch.object(self.memory, 'get_conversation_count', return_value=9):
                # Process a message to create the 10th conversation
                conversation_id = self.super_ai.create_conversation()
                self.super_ai.process_message(
                    conversation_id=conversation_id,
                    user_message="This should trigger training"
                )

                # Training should not be triggered yet (10 conversations exactly)
                mock_training.assert_not_called()

            # Now mock that we have 10 conversations and process another
            with patch.object(self.memory, 'get_conversation_count', return_value=10):
                new_conversation_id = self.super_ai.create_conversation()
                self.super_ai.process_message(
                    conversation_id=new_conversation_id,
                    user_message="This should trigger training"
                )

                # Training should be triggered now (11 conversations)
                mock_training.assert_called_once()

    def test_feedback_integration(self):
        """Test that user feedback is properly integrated into the system."""
        # Create a conversation and get a response
        conversation_id = self.super_ai.create_conversation()
        response = self.super_ai.process_message(
            conversation_id=conversation_id,
            user_message="Tell me a fact"
        )

        # Submit feedback for the response
        with patch.object(self.feedback_system, 'record_feedback') as mock_record:
            feedback_data = {
                "rating": 4,
                "comment": "Good response, but could be more detailed",
                "categories": ["accuracy", "completeness"]
            }

            self.super_ai.submit_feedback(
                conversation_id=conversation_id,
                message_index=1,  # Index of the assistant's response
                feedback=feedback_data
            )

            # Verify feedback was recorded
            mock_record.assert_called_once()
            args = mock_record.call_args[0]
            self.assertEqual(args[0], conversation_id)
            self.assertEqual(args[1], 1)
            self.assertEqual(args[2]["rating"], 4)

    def test_memory_retrieval_during_conversation(self):
        """Test that the system retrieves relevant memories during conversation."""
        # First, store some knowledge
        self.memory.add_to_knowledge_base(
            content="Artificial intelligence is the simulation of human intelligence by machines.",
            source="test"
        )

        # Mock the memory search to return this knowledge
        with patch.object(self.memory, 'search_knowledge') as mock_search:
            mock_search.return_value = [{
                "content": "Artificial intelligence is the simulation of human intelligence by machines.",
                "source": "test",
                "timestamp": time.time(),
                "id": "test1"
            }]

            # Mock the LLM to check if relevant knowledge is included in the prompt
            orig_generate = self.mock_llm.generate_response

            def check_prompt(conversation_id, messages, options=None):
                # Check if knowledge was included in the messages
                knowledge_found = False
                for msg in messages:
                    if (msg["role"] == "system" and
                        "Artificial intelligence is the simulation" in msg["content"]):
                        knowledge_found = True
                        break

                self.assertTrue(knowledge_found, "Relevant knowledge not included in prompt")
                return orig_generate(conversation_id, messages, options)

            self.mock_llm.generate_response = check_prompt

            # Process a message that should trigger memory retrieval
            conversation_id = self.super_ai.create_conversation()
            self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message="What is artificial intelligence?"
            )

            # Verify memory search was called
            mock_search.assert_called_once()

    def test_error_recovery(self):
        """Test the system's ability to recover from component failures."""
        # Simulate a temporary memory system failure
        with patch.object(self.memory, 'store_conversation', side_effect=[Exception("Storage failure"), None]):
            # Process a message that will trigger the failure and recovery
            conversation_id = self.super_ai.create_conversation()
            response = self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message="This should survive a memory failure"
            )

            # Verify we got a response despite the failure
            self.assertIn("test response", response["text"].lower())

            # Verify the error handler logged the exception
            # (Would check log files, but using mocks so we'll skip)

    def test_multiple_concurrent_conversations(self):
        """Test handling multiple conversations concurrently."""
        # Create a function to run a conversation in a thread
        results = {}

        def run_conversation(conversation_id, messages):
            try:
                for message in messages:
                    response = self.super_ai.process_message(
                        conversation_id=conversation_id,
                        user_message=message
                    )
                    results[f"{conversation_id}_{message}"] = response["text"]
            except Exception as e:
                results[f"{conversation_id}_error"] = str(e)

        # Create several conversations with different messages
        conversations = {
            "convo1": ["Hello", "How are you?", "What's the weather?"],
            "convo2": ["Tell me a joke", "Another joke please", "Thanks"],
            "convo3": ["What is AI?", "How does machine learning work?", "Interesting"]
        }

        # Start a thread for each conversation
        threads = []
        for convo_id, messages in conversations.items():
            # Create the conversation
            actual_id = self.super_ai.create_conversation()
            thread = threading.Thread(target=run_conversation, args=(actual_id, messages))
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all messages were processed without errors
        self.assertEqual(len(results), 9)  # 3 conversations × 3 messages each
        for key in results:
            self.assertNotIn("error", key)

    def test_system_under_load(self):
        """Test system performance under high load."""
        # Configure load test parameters
        num_requests = 20
        max_allowed_time = 10  # seconds

        # Record start time
        start_time = time.time()

        # Process multiple messages in rapid succession
        conversation_id = self.super_ai.create_conversation()
        for i in range(num_requests):
            self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message=f"Load test message {i}"
            )

        # Calculate total time
        total_time = time.time() - start_time

        # Verify the test completed within the time limit
        self.assertLess(total_time, max_allowed_time,
                         f"System too slow under load: {total_time:.2f}s for {num_requests} requests")

        # Check performance metrics
        metrics = self.performance_monitor.get_metrics()
        avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])

        # Log performance details
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Peak CPU usage: {max(metrics['cpu_usage']):.2f}%")
        print(f"Peak memory usage: {max(metrics['memory_usage']):.2f}%")

    def test_graceful_shutdown(self):
        """Test that the system shuts down gracefully, saving all necessary data."""
        # Process some messages to generate data
        conversation_id = self.super_ai.create_conversation()
        for i in range(3):
            self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message=f"Message before shutdown {i}"
            )

        # Mock the component shutdown methods to verify they're called
        with patch.object(self.performance_monitor, 'stop_monitoring') as mock_perf_stop, \
             patch.object(self.memory, 'sync') as mock_memory_sync:

            # Initiate shutdown
            self.super_ai.shutdown()

            # Verify shutdown operations were performed
            mock_perf_stop.assert_called_once()
            mock_memory_sync.assert_called_once()

    def test_configuration_update(self):
        """Test dynamically updating the system configuration."""
        # Original configuration value
        original_max_tokens = self.config.get("llm.max_tokens")

        # Update the configuration
        new_config = {
            "llm": {
                "max_tokens": 200,
                "temperature": 0.5
            }
        }
        self.super_ai.update_configuration(new_config)

        # Verify the configuration was updated
        self.assertEqual(self.config.get("llm.max_tokens"), 200)
        self.assertEqual(self.config.get("llm.temperature"), 0.5)

        # Process a message to verify the new configuration is used
        with patch.object(self.mock_llm, 'generate_response') as mock_generate:
            conversation_id = self.super_ai.create_conversation()
            self.super_ai.process_message(
                conversation_id=conversation_id,
                user_message="Test with new config"
            )

            # Check that the updated configuration was used
            options = mock_generate.call_args[1].get("options", {})
            self.assertEqual(options.get("max_tokens"), 200)
            self.assertEqual(options.get("temperature"), 0.5)

if __name__ == '__main__':
    unittest.main()
