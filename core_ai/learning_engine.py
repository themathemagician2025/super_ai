# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Learning Engine for Super AI System

This module implements continuous self-improvement capabilities:
1. Self-learning - Automatically retrain strategies after every 10 conversations
2. Self-retrieval - Internal knowledge search before using external sources
3. Self-feedback - Rate its own answers and learn from performance
"""

import os
import sys
import json
import time
import logging
import pickle
import random
import datetime
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, deque

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import system utilities
try:
    from web_interface.system_utils import (
        timeout,
        fallback,
        log_decision,
        log_tool_call
    )
except ImportError:
    # Fallback implementations if system_utils not available
    def timeout(seconds):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def fallback(default_return=None, log_exception=True):
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.exception(f"Error in {func.__name__}: {str(e)}")
                    return default_return
            return wrapper
        return decorator

    def log_decision(decision, metadata=None):
        logging.info(f"DECISION: {decision} - META: {metadata}")

    def log_tool_call(tool_name, args, result):
        logging.info(f"TOOL CALL: {tool_name} - ARGS: {args} - RESULT: {result}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs", f"learning_engine_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths for storing memory, feedback, and model data
MEMORY_DIR = os.path.join(parent_dir, "memory")
MODELS_DIR = os.path.join(parent_dir, "models")
FEEDBACK_DIR = os.path.join(parent_dir, "feedback")

for directory in [MEMORY_DIR, MODELS_DIR, FEEDBACK_DIR]:
    os.makedirs(directory, exist_ok=True)

class MemoryManager:
    """Manages the system's internal memory for self-retrieval"""

    def __init__(self, memory_file: str = "system_memory.json"):
        self.memory_file = os.path.join(MEMORY_DIR, memory_file)
        self.memory_lock = threading.Lock()
        self.memory_cache = {}
        self.memory_embeddings = {}
        self.memory_counter = 0
        self.load_memory()

    def load_memory(self):
        """Load memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memory_cache = data.get('memories', {})
                    self.memory_embeddings = data.get('embeddings', {})
                    self.memory_counter = data.get('counter', 0)
                logger.info(f"Loaded {len(self.memory_cache)} memories from {self.memory_file}")
            else:
                logger.info(f"No memory file found at {self.memory_file}, starting with empty memory")
                self.save_memory()  # Create the initial file
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
            # Initialize with empty memory
            self.memory_cache = {}
            self.memory_embeddings = {}
            self.memory_counter = 0
            self.save_memory()

    def save_memory(self):
        """Save memory to disk"""
        with self.memory_lock:
            try:
                data = {
                    'memories': self.memory_cache,
                    'embeddings': self.memory_embeddings,
                    'counter': self.memory_counter,
                    'last_updated': datetime.datetime.now().isoformat()
                }
                with open(self.memory_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved {len(self.memory_cache)} memories to {self.memory_file}")
            except Exception as e:
                logger.error(f"Error saving memory: {e}")

    def add_memory(self,
                  query: str,
                  response: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  embedding: Optional[List[float]] = None) -> str:
        """
        Add a new memory (query-response pair)

        Args:
            query: The user query
            response: The system response
            metadata: Additional metadata about this interaction
            embedding: Vector embedding for semantic search (if available)

        Returns:
            str: Memory ID
        """
        with self.memory_lock:
            # Generate memory ID
            memory_id = f"mem_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.memory_counter}"
            self.memory_counter += 1

            # Store memory
            self.memory_cache[memory_id] = {
                'query': query,
                'response': response,
                'metadata': metadata or {},
                'created_at': datetime.datetime.now().isoformat(),
                'accessed_count': 0,
                'feedback_score': None
            }

            # Store embedding if provided
            if embedding is not None:
                self.memory_embeddings[memory_id] = embedding

            # Save periodically (every 10 additions)
            if self.memory_counter % 10 == 0:
                self.save_memory()

            log_decision(
                "Added new memory",
                {"memory_id": memory_id, "query_preview": query[:50] + "..." if len(query) > 50 else query}
            )

            return memory_id

    @timeout(5)  # 5 second timeout for search
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search internal memory for relevant past interactions

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List[Dict]: List of memory entries
        """
        try:
            log_decision("Searching internal memory", {"query": query, "limit": limit})

            # Simple keyword-based search (in a real implementation, use embeddings)
            results = []
            query_terms = set(query.lower().split())

            for memory_id, memory in self.memory_cache.items():
                memory_text = f"{memory['query']} {memory['response']}".lower()
                score = sum(1 for term in query_terms if term in memory_text) / max(1, len(query_terms))

                if score > 0:
                    results.append({
                        'memory_id': memory_id,
                        'query': memory['query'],
                        'response': memory['response'],
                        'created_at': memory['created_at'],
                        'score': score,
                        'metadata': memory['metadata']
                    })

                    # Update access count
                    with self.memory_lock:
                        memory['accessed_count'] = memory.get('accessed_count', 0) + 1

            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)

            # Log search results
            log_tool_call(
                "memory_search",
                {"query": query, "limit": limit},
                f"Found {len(results[:limit])} relevant memories"
            )

            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []

    def setup_multi_gpu(self):
        """Configure distributed training across GPUs"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.distributed = self.num_gpus > 1
            if self.distributed:
                torch.distributed.init_process_group('nccl')

    def setup_task_agnostic_engine(self):
        """Initialize task-agnostic prediction components"""
        self.task_configs = {
            'forex': {'model': 'LSTM', 'features': ['price', 'volume', 'sentiment']},
            'sports': {'model': 'XGBoost', 'features': ['stats', 'history', 'weather']},
            'stocks': {'model': 'Transformer', 'features': ['price', 'fundamentals', 'news']}
        }

    def setup_reinforcement_learning(self):
        """Setup RL components for self-improvement"""
        self.rl_config = {
            'reward_scale': 1.0,
            'punishment_scale': 0.5,
            'memory_size': 10000,
            'batch_size': 64
        }

    async def train(self, data, model_type):
        """Enhanced training with multi-GPU support and auto-optimization"""
        try:
            # Auto-detect best normalization
            normalized_data = self.auto_normalize_data(data)
            
            # Handle missing data
            complete_data = self.impute_missing_values(normalized_data)
            
            # Add contextual features
            enriched_data = self.add_contextual_features(complete_data)
            
            # Distribute training across GPUs if available
            if self.distributed:
                model = self.train_distributed(enriched_data, model_type)
            else:
                model = self.train_single_device(enriched_data, model_type)
            
            # Evaluate and store in experience bank
            accuracy = self.evaluate_model(model, enriched_data)
            self.update_experience_bank(model, accuracy)
            
            return model
            
        except Exception as e:
            logger.error(f'Training error: {e}')
            self.trigger_fallback_logic()

class FeedbackScorer:
    """Self-evaluation system for rating answers and learning from performance"""

    # Scoring constants
    SCORE_GOOD = 1.0
    SCORE_NEUTRAL = 0.5
    SCORE_BAD = 0.0

    def __init__(self, feedback_file: str = "feedback_scores.json"):
        self.feedback_file = os.path.join(FEEDBACK_DIR, feedback_file)
        self.feedback_lock = threading.Lock()
        self.feedback_cache = {}
        self.scoring_models = {}
        self.load_feedback()
        self.evaluation_criteria = {
            'completeness': self._evaluate_completeness,
            'relevance': self._evaluate_relevance,
            'clarity': self._evaluate_clarity,
            'efficiency': self._evaluate_efficiency
        }

    def load_feedback(self):
        """Load feedback data from disk"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    self.feedback_cache = json.load(f)
                logger.info(f"Loaded feedback data with {len(self.feedback_cache)} entries")
            else:
                logger.info(f"No feedback file found at {self.feedback_file}, starting with empty feedback")
                self.save_feedback()  # Create the initial file
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            self.feedback_cache = {}
            self.save_feedback()

    def save_feedback(self):
        """Save feedback data to disk"""
        with self.feedback_lock:
            try:
                with open(self.feedback_file, 'w') as f:
                    json.dump(self.feedback_cache, f, indent=2)
                logger.info(f"Saved feedback data with {len(self.feedback_cache)} entries")
            except Exception as e:
                logger.error(f"Error saving feedback: {e}")

    def rate_answer(self,
                   query: str,
                   response: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Rate an answer based on internal criteria

        Args:
            query: The user query
            response: The system response
            metadata: Additional context about the interaction

        Returns:
            Dict: Evaluation results with scores and feedback
        """
        try:
            feedback_id = f"fb_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            log_decision("Rating answer quality", {"query": query[:50] + "..." if len(query) > 50 else query})

            # Calculate scores for each criterion
            scores = {}
            for criterion, eval_func in self.evaluation_criteria.items():
                scores[criterion] = eval_func(query, response, metadata)

            # Calculate overall score (weighted average)
            weights = {
                'completeness': 0.3,
                'relevance': 0.4,
                'clarity': 0.2,
                'efficiency': 0.1
            }

            overall_score = sum(scores[c] * weights[c] for c in scores) / sum(weights.values())

            # Determine rating category
            if overall_score >= 0.8:
                rating = "GOOD"
            elif overall_score >= 0.4:
                rating = "NEUTRAL"
            else:
                rating = "BAD"

            # Generate improvement suggestions
            suggestions = self._generate_suggestions(scores, query, response)

            # Store feedback
            result = {
                'feedback_id': feedback_id,
                'query': query,
                'response_snippet': response[:100] + "..." if len(response) > 100 else response,
                'scores': scores,
                'overall_score': overall_score,
                'rating': rating,
                'suggestions': suggestions,
                'timestamp': datetime.datetime.now().isoformat()
            }

            with self.feedback_lock:
                self.feedback_cache[feedback_id] = result
                # Save periodically
                if len(self.feedback_cache) % 10 == 0:
                    self.save_feedback()

            log_tool_call(
                "self_feedback",
                {"query": query[:50] + "..." if len(query) > 50 else query},
                f"Rating: {rating}, Score: {overall_score:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error rating answer: {e}")
            return {
                'rating': "NEUTRAL",
                'overall_score': 0.5,
                'error': str(e)
            }

    def _evaluate_completeness(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate if the response completely answers the query"""
        # Simple heuristic: length ratio and keyword coverage
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Check keyword coverage
        coverage = sum(1 for w in query_words if w in response_words) / max(1, len(query_words))

        # Check length adequacy (penalize too short responses)
        length_ratio = min(1.0, len(response) / max(1, len(query) * 2))

        return 0.7 * coverage + 0.3 * length_ratio

    def _evaluate_relevance(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate how relevant the response is to the query"""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())

        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(response_words))
        union = len(query_words.union(response_words))

        similarity = intersection / max(1, union)

        # Adjust based on metadata if available
        if metadata and 'similarity_score' in metadata:
            return 0.3 * similarity + 0.7 * metadata['similarity_score']

        return similarity

    def _evaluate_clarity(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate clarity and organization of the response"""
        # Simple heuristics based on structure
        has_paragraphs = '\n\n' in response
        has_lists = ('- ' in response) or ('* ' in response) or any(f"{i}." in response for i in range(1, 10))
        has_headings = any(line.strip().startswith('#') for line in response.split('\n'))

        structure_score = sum([
            0.4 if has_paragraphs else 0.0,
            0.3 if has_lists else 0.0,
            0.3 if has_headings else 0.0
        ])

        # Penalize very long sentences
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        sentence_score = 1.0 if avg_sentence_length < 20 else (30 / max(20, avg_sentence_length))

        return 0.5 * structure_score + 0.5 * sentence_score

    def _evaluate_efficiency(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate response efficiency (conciseness vs verbosity)"""
        # Consider response length relative to query complexity
        query_complexity = len(query.split())
        response_length = len(response.split())

        # Simple heuristic for expected length
        expected_length = query_complexity * 3

        if response_length <= expected_length:
            efficiency = 1.0  # Perfectly concise
        else:
            # Penalize verbosity
            verbosity_ratio = expected_length / response_length
            efficiency = max(0.3, verbosity_ratio)

        # Consider runtime from metadata if available
        if metadata and 'runtime_seconds' in metadata:
            runtime = metadata['runtime_seconds']
            runtime_score = 1.0 if runtime < 2 else (5 / max(2, runtime))
            return 0.7 * efficiency + 0.3 * runtime_score

        return efficiency

    def _generate_suggestions(self, scores: Dict[str, float], query: str, response: str) -> List[str]:
        """Generate specific improvement suggestions based on scores"""
        suggestions = []

        # Add criterion-specific suggestions
        if scores['completeness'] < 0.6:
            suggestions.append("Provide more comprehensive answers that fully address the query")

        if scores['relevance'] < 0.6:
            suggestions.append("Focus more directly on answering the specific question asked")

        if scores['clarity'] < 0.6:
            suggestions.append("Improve clarity with better structure (paragraphs, lists, examples)")

        if scores['efficiency'] < 0.6:
            suggestions.append("Be more concise; avoid unnecessary explanation")

        return suggestions

class LearningEngine:
    """Advanced Learning Engine with AGI capabilities
    Handles self-improvement, multi-model learning, and adaptive optimization
    """

    """Main learning engine that coordinates self-improvement strategies"""

    def __init__(self, config_path: str = 'config/learning_engine.json'):
        self.config = self._load_config(config_path)
        self.models = {}
        self.history = []
        self.experience_bank = {}
        self.current_accuracy = 0.0
        self.accuracy_target = 0.95
        self.setup_multi_gpu()
        self.setup_task_agnostic_engine()
        self.setup_reinforcement_learning()
        self.memory = MemoryManager()
        self.feedback = FeedbackScorer()
        self.conversation_counter = 0
        self.retraining_frequency = 10  # Retrain after every 10 conversations
        self.strategies = {}
        self.model_stats = defaultdict(dict)
        self.model_lock = threading.Lock()
        self.load_strategies()

        # Create analysis thread
        self.should_stop = False
        self.analysis_thread = threading.Thread(target=self._background_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()

        logger.info("Learning Engine initialized")

    def load_strategies(self):
        """Load routing and model selection strategies"""
        try:
            strategy_file = os.path.join(MODELS_DIR, "strategies.json")
            if os.path.exists(strategy_file):
                with open(strategy_file, 'r') as f:
                    self.strategies = json.load(f)
                logger.info(f"Loaded strategies from {strategy_file}")
            else:
                # Initialize with default strategies
                self.strategies = {
                    'routing': {
                        'default': 'balanced',
                        'options': {
                            'speed': {'weight': 0.3},
                            'accuracy': {'weight': 0.4},
                            'balanced': {'weight': 0.3}
                        }
                    },
                    'models': {
                        'default': 'medium',
                        'speed_rankings': {
                            'fast': 0.8,
                            'medium': 0.5,
                            'slow': 0.2
                        },
                        'accuracy_rankings': {
                            'fast': 0.3,
                            'medium': 0.7,
                            'slow': 0.9
                        }
                    }
                }
                self.save_strategies()
        except Exception as e:
            logger.error(f"Error loading strategies: {e}")
            # Set reasonable defaults
            self.strategies = {
                'routing': {'default': 'balanced'},
                'models': {'default': 'medium'}
            }

    def save_strategies(self):
        """Save current strategies to disk"""
        try:
            strategy_file = os.path.join(MODELS_DIR, "strategies.json")
            with open(strategy_file, 'w') as f:
                json.dump(self.strategies, f, indent=2)
            logger.info(f"Saved strategies to {strategy_file}")
        except Exception as e:
            logger.error(f"Error saving strategies: {e}")

    def process_interaction(self,
                           query: str,
                           response: str,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user interaction, save to memory, and provide feedback

        Args:
            query: The user query
            response: The system response
            metadata: Additional context about the interaction

        Returns:
            Dict: Feedback and memory information
        """
        # Record in memory
        memory_id = self.memory.add_memory(query, response, metadata)

        # Generate feedback
        feedback = self.feedback.rate_answer(query, response, metadata)

        # Update conversation counter
        self.conversation_counter += 1

        # Check if it's time to retrain
        if self.conversation_counter % self.retraining_frequency == 0:
            self._trigger_retraining()

        return {
            'memory_id': memory_id,
            'feedback': feedback,
            'conversation_count': self.conversation_counter
        }

    @timeout(10)  # 10 second timeout for retrieval
    def retrieve_from_memory(self, query: str, use_web_fallback: bool = True) -> Dict[str, Any]:
        """
        Search internal memory first, then fallback to web if needed

        Args:
            query: The search query
            use_web_fallback: Whether to use web search as fallback

        Returns:
            Dict: Search results and source information
        """
        log_decision(
            "Performing self-retrieval",
            {"query": query, "use_web_fallback": use_web_fallback}
        )

        # First search internal memory
        memory_results = self.memory.search_memory(query)

        # If we have good memory results, use them
        if memory_results and memory_results[0]['score'] > 0.7:
            return {
                'source': 'memory',
                'results': memory_results,
                'query': query
            }

        # If no good results and web fallback is enabled
        if use_web_fallback:
            try:
                # This would call an actual web search API in a real implementation
                web_results = [{"title": f"Web result for {query}", "snippet": "Example web content"}]

                log_tool_call(
                    "web_search_fallback",
                    {"query": query},
                    f"Retrieved {len(web_results)} results from web"
                )

                return {
                    'source': 'web',
                    'results': web_results,
                    'query': query,
                    'memory_results': memory_results  # Include partial memory matches
                }
            except Exception as e:
                logger.error(f"Web search fallback failed: {e}")

        # Return the memory results even if they're not perfect
        return {
            'source': 'memory',
            'results': memory_results,
            'query': query,
            'fallback': True
        }

    def select_model(self, query_type: str, performance_priority: str = 'balanced') -> str:
        """
        Select the best model based on query type and performance priority

        Args:
            query_type: Type of query (e.g., 'factual', 'creative', 'code')
            performance_priority: Priority ('speed', 'accuracy', or 'balanced')

        Returns:
            str: Selected model name
        """
        with self.model_lock:
            log_decision(
                "Selecting model based on learned strategies",
                {"query_type": query_type, "priority": performance_priority}
            )

            # Ensure valid performance priority
            if performance_priority not in ['speed', 'accuracy', 'balanced']:
                performance_priority = self.strategies['routing']['default']

            # Get model options
            model_options = self.strategies.get('models', {})
            default_model = model_options.get('default', 'medium')

            # Simple model selection logic
            if performance_priority == 'speed':
                # Sort by speed ranking
                rankings = model_options.get('speed_rankings', {})
                return max(rankings.items(), key=lambda x: x[1])[0] if rankings else default_model

            elif performance_priority == 'accuracy':
                # Sort by accuracy ranking
                rankings = model_options.get('accuracy_rankings', {})
                return max(rankings.items(), key=lambda x: x[1])[0] if rankings else default_model

            else:  # balanced
                # Use combined score
                speed_rankings = model_options.get('speed_rankings', {})
                accuracy_rankings = model_options.get('accuracy_rankings', {})

                combined_scores = {}
                for model in set(list(speed_rankings.keys()) + list(accuracy_rankings.keys())):
                    speed_score = speed_rankings.get(model, 0.5)
                    accuracy_score = accuracy_rankings.get(model, 0.5)
                    combined_scores[model] = 0.5 * speed_score + 0.5 * accuracy_score

                return max(combined_scores.items(), key=lambda x: x[1])[0] if combined_scores else default_model

    def _trigger_retraining(self):
        """Trigger strategy retraining based on accumulated data"""
        log_decision(
            "Triggering strategy retraining",
            {"conversation_count": self.conversation_counter}
        )

        try:
            # This would be async in a real implementation
            self._retrain_strategies()
        except Exception as e:
            logger.error(f"Error during retraining: {e}")

    def _retrain_strategies(self):
        """Retrain routing and model selection strategies"""
        with self.model_lock:
            # Load recent feedback
            feedback_cache = self.feedback.feedback_cache

            if not feedback_cache:
                logger.warning("No feedback data available for retraining")
                return

            logger.info(f"Retraining with {len(feedback_cache)} feedback entries")

            # Analyze model performance
            model_performance = defaultdict(lambda: {'count': 0, 'speed_score': 0, 'accuracy_score': 0})

            for fb_id, feedback in feedback_cache.items():
                # Extract model info from metadata
                metadata = feedback.get('metadata', {})
                model_name = metadata.get('model_name', 'unknown')

                # Update performance metrics
                model_performance[model_name]['count'] += 1
                model_performance[model_name]['speed_score'] += metadata.get('speed_score', 0.5)
                model_performance[model_name]['accuracy_score'] += feedback.get('overall_score', 0.5)

            # Calculate averages
            for model, stats in model_performance.items():
                if stats['count'] > 0:
                    stats['speed_score'] /= stats['count']
                    stats['accuracy_score'] /= stats['count']

            # Update strategy weights
            if 'models' not in self.strategies:
                self.strategies['models'] = {}

            # Update speed rankings
            speed_rankings = {}
            for model, stats in model_performance.items():
                if stats['count'] >= 5:  # Only update if we have enough data
                    speed_rankings[model] = stats['speed_score']

            if speed_rankings:
                self.strategies['models']['speed_rankings'] = speed_rankings

            # Update accuracy rankings
            accuracy_rankings = {}
            for model, stats in model_performance.items():
                if stats['count'] >= 5:  # Only update if we have enough data
                    accuracy_rankings[model] = stats['accuracy_score']

            if accuracy_rankings:
                self.strategies['models']['accuracy_rankings'] = accuracy_rankings

            # Save updated strategies
            self.save_strategies()

            logger.info("Retraining completed successfully")

    def _background_analysis(self):
        """Background thread for continuous analysis and optimization"""
        while not self.should_stop:
            try:
                # Sleep for a while between analysis runs
                time.sleep(300)  # 5 minutes

                # Perform maintenance tasks
                self._cleanup_old_feedback()
                self._analyze_query_patterns()

            except Exception as e:
                logger.error(f"Error in background analysis: {e}")

    def _cleanup_old_feedback(self):
        """Clean up old feedback entries"""
        with self.feedback.feedback_lock:
            # Keep only recent entries (last 1000)
            if len(self.feedback.feedback_cache) > 1000:
                sorted_keys = sorted(
                    self.feedback.feedback_cache.keys(),
                    key=lambda k: self.feedback.feedback_cache[k].get('timestamp', '')
                )

                # Remove oldest entries
                keys_to_remove = sorted_keys[:-1000]
                for key in keys_to_remove:
                    del self.feedback.feedback_cache[key]

                logger.info(f"Cleaned up {len(keys_to_remove)} old feedback entries")
                self.feedback.save_feedback()

    def _analyze_query_patterns(self):
        """Analyze query patterns to optimize retrieval"""
        # This would analyze common queries and pre-compute useful embeddings
        pass

    def shutdown(self):
        """Shutdown the learning engine gracefully"""
        logger.info("Shutting down learning engine")
        self.should_stop = True

        if hasattr(self, 'analysis_thread') and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=5)

        # Save any pending data
        self.memory.save_memory()
        self.feedback.save_feedback()
        self.save_strategies()

        logger.info("Learning engine shutdown complete")

# Create singleton instance
learning_engine = LearningEngine()

def process_query(query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a query-response pair (utility function for the singleton)

    Args:
        query: User query
        response: System response
        metadata: Additional metadata

    Returns:
        Dict: Processing results with feedback
    """
    return learning_engine.process_interaction(query, response, metadata)

def retrieve_info(query: str, use_web_fallback: bool = True) -> Dict[str, Any]:
    """
    Retrieve information for a query (utility function for the singleton)

    Args:
        query: Search query
        use_web_fallback: Whether to use web search as fallback

    Returns:
        Dict: Retrieval results
    """
    return learning_engine.retrieve_from_memory(query, use_web_fallback)

def select_best_model(query_type: str, performance_priority: str = 'balanced') -> str:
    """
    Select the best model for a query (utility function for the singleton)

    Args:
        query_type: Type of query
        performance_priority: Performance priority

    Returns:
        str: Selected model name
    """
    return learning_engine.select_model(query_type, performance_priority)

# Clean up on process exit
import atexit
atexit.register(learning_engine.shutdown)

if __name__ == "__main__":
    # Simple self-test
    print("Testing learning engine...")

    # Test memory and retrieval
    test_query = "How do I optimize TensorFlow models?"
    test_response = "To optimize TensorFlow models, you can use techniques like quantization, pruning, and model distillation. These approaches can significantly reduce model size and improve inference speed."

    result = process_query(test_query, test_response, {'model_name': 'medium', 'speed_score': 0.7})
    print(f"Feedback: {result['feedback']['rating']} ({result['feedback']['overall_score']:.2f})")

    # Test retrieval
    retrieved = retrieve_info("How to optimize TensorFlow?")
    print(f"Retrieved from: {retrieved['source']}")
    if retrieved['results']:
        print(f"Top result score: {retrieved['results'][0]['score']:.2f}")

    # Test model selection
    selected_model = select_best_model('factual', 'speed')
    print(f"Selected model: {selected_model}")

    print("Test complete")
