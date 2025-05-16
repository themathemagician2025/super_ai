#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Orchestrator for SUPER AI PROJECT

This module provides a unified interface to interact with Ollama-powered language models,
specifically Llama3.1 and Phi, for various tasks including code generation,
log analysis, and response refinement.
"""

import os
import json
import logging
import sqlite3
import time
import hashlib
import sys
import subprocess
from typing import Dict, Any, List, Optional, Union, Callable

# Ensure ollama is available
try:
    import ollama
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

# Set up logging
log_dir = os.getenv("LOG_DIR", "/app/logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "llm_orchestrator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llm_orchestrator")


class LLMCache:
    """
    Cache for LLM responses to avoid redundant model calls and improve performance.
    Uses SQLite for persistence.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the LLM response cache.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path in data directory.
        """
        if db_path is None:
            data_dir = os.getenv("DATA_BASE_PATH", "/app/data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "llm_cache.db")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
    
    def _initialize_db(self):
        """Create necessary database tables if they don't exist."""
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            prompt_hash TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp REAL NOT NULL,
            metadata TEXT
        )
        ''')
        self.conn.commit()
    
    def _hash_prompt(self, prompt: str, model: str) -> str:
        """
        Create a deterministic hash for a prompt and model combination.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            Hash string to use as cache key
        """
        # Use SHA-256 for more reliable hashing than built-in hash()
        hash_input = f"{model}:{prompt}"
        hash_obj = hashlib.sha256(hash_input.encode())
        return hash_obj.hexdigest()
    
    def cache_response(self, 
                       prompt: str, 
                       response: str, 
                       model: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The prompt text
            response: The model's response 
            model: The model name
            metadata: Optional metadata about the generation
        """
        prompt_hash = self._hash_prompt(prompt, model)
        metadata_json = json.dumps(metadata) if metadata else None
        
        self.conn.execute(
            "INSERT OR REPLACE INTO response_cache (prompt_hash, model, prompt, response, timestamp, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (prompt_hash, model, prompt, response, time.time(), metadata_json)
        )
        self.conn.commit()
        logger.debug(f"Cached response for prompt hash: {prompt_hash[:8]}...")
    
    def get_cached_response(self, 
                           prompt: str, 
                           model: str,
                           max_age: Optional[float] = None) -> Optional[str]:
        """
        Retrieve a cached response if available.
        
        Args:
            prompt: The prompt text
            model: The model name
            max_age: Maximum age in seconds for a valid cache entry
            
        Returns:
            Cached response string or None if not found or too old
        """
        prompt_hash = self._hash_prompt(prompt, model)
        
        query = "SELECT response, timestamp FROM response_cache WHERE prompt_hash = ?"
        cursor = self.conn.execute(query, (prompt_hash,))
        result = cursor.fetchone()
        
        if result:
            response, timestamp = result
            
            # Check if cache entry is too old
            if max_age and (time.time() - timestamp) > max_age:
                logger.debug(f"Cache hit for {prompt_hash[:8]}... but entry too old")
                self.misses += 1
                return None
            
            logger.debug(f"Cache hit for {prompt_hash[:8]}...")
            self.hits += 1
            return response
        
        logger.debug(f"Cache miss for {prompt_hash[:8]}...")
        self.misses += 1
        return None
    
    def clear_cache(self, older_than: Optional[float] = None) -> int:
        """
        Clear the cache, optionally only entries older than a specified time.
        
        Args:
            older_than: If provided, only clear entries older than this many seconds
            
        Returns:
            Number of entries cleared
        """
        if older_than:
            cutoff_time = time.time() - older_than
            cursor = self.conn.execute(
                "DELETE FROM response_cache WHERE timestamp < ?", 
                (cutoff_time,)
            )
        else:
            cursor = self.conn.execute("DELETE FROM response_cache")
        
        self.conn.commit()
        return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict containing cache statistics
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM response_cache")
        entry_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute(
            "SELECT model, COUNT(*) FROM response_cache GROUP BY model"
        )
        models = {model: count for model, count in cursor.fetchall()}
        
        cursor = self.conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM response_cache"
        )
        min_time, max_time = cursor.fetchone()
        
        # Calculate hit rate
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_entries": entry_count,
            "models": models,
            "oldest_entry": min_time,
            "newest_entry": max_time,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": hit_rate,
            "db_path": self.db_path
        }
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
    
    def __del__(self):
        """Ensure connection is closed when the object is garbage collected."""
        self.close()


class PromptTemplates:
    """
    Collection of prompt templates for different LLM tasks.
    Provides methods to format prompts with specific parameters.
    """
    
    # Code generation templates
    CODE_GENERATION = """
    You are an expert Python developer. Generate a Python function for {task_description} that is optimized for {optimization_goal}. Ensure the code is syntactically correct, follows PEP 8, and includes comments.

    Task: {task_details}
    Optimization Goal: {optimization_goal}
    
    Return ONLY the Python code without any additional explanation or markdown formatting.
    """
    
    # Log analysis template
    LOG_ANALYSIS = """
    Analyze the following logs to identify performance bottlenecks, errors, and warnings. Provide a JSON response with:
    - summary: Key issues and their impact.
    - recommendations: Actionable changes with code snippets.
    - priority: High, Medium, Low.

    Logs:
    {log_content}
    
    Return ONLY valid JSON without any additional explanation.
    """
    
    # Response refinement template
    RESPONSE_REFINEMENT = """
    Refine the following response to improve accuracy, clarity, and relevance for the SUPER AI PROJECT. Ensure the output is technically correct and aligned with achieving 99.9% accuracy. Return JSON with refined_response and changes_made.

    Initial Response:
    {initial_response}

    Context:
    {context}
    
    Return ONLY valid JSON without any additional explanation.
    """
    
    # Code debugging template
    CODE_DEBUGGING = """
    Debug the following Python code that has an issue. Identify the problem, explain the root cause, and provide the corrected code.

    Issue Description: {issue_description}
    
    Code:
    ```python
    {code}
    ```
    
    Return a JSON response with the following structure:
    {
        "root_cause": "Description of the root cause",
        "explanation": "Detailed explanation of the issue",
        "fixed_code": "The corrected code"
    }
    """
    
    # Docker optimization template
    DOCKER_OPTIMIZATION = """
    Optimize the following Dockerfile to improve build speed, reduce image size, and enhance security.
    
    Current Dockerfile:
    ```dockerfile
    {dockerfile}
    ```
    
    Build Context Size: {context_size}
    Build Time: {build_time}
    
    Return a JSON response with:
    {
        "optimized_dockerfile": "The optimized Dockerfile",
        "improvements": ["List of specific improvements made"],
        "expected_benefits": {
            "size_reduction": "Estimated size reduction",
            "build_time_reduction": "Estimated build time reduction",
            "security_enhancements": ["Security improvements"]
        }
    }
    """
    
    @classmethod
    def format_code_generation(cls, 
                              task_description: str, 
                              task_details: str, 
                              optimization_goal: str) -> str:
        """
        Format the code generation prompt.
        
        Args:
            task_description: Brief description of the coding task
            task_details: Detailed requirements for the code
            optimization_goal: What aspect to optimize for
            
        Returns:
            Formatted prompt string
        """
        return cls.CODE_GENERATION.format(
            task_description=task_description,
            task_details=task_details,
            optimization_goal=optimization_goal
        )
    
    @classmethod
    def format_log_analysis(cls, log_content: str) -> str:
        """
        Format the log analysis prompt.
        
        Args:
            log_content: Log content to analyze
            
        Returns:
            Formatted prompt string
        """
        return cls.LOG_ANALYSIS.format(log_content=log_content)
    
    @classmethod
    def format_response_refinement(cls, 
                                  initial_response: str, 
                                  context: str) -> str:
        """
        Format the response refinement prompt.
        
        Args:
            initial_response: Original response to refine
            context: Additional context for refinement
            
        Returns:
            Formatted prompt string
        """
        return cls.RESPONSE_REFINEMENT.format(
            initial_response=initial_response,
            context=context
        )
    
    @classmethod
    def format_code_debugging(cls, 
                             issue_description: str, 
                             code: str) -> str:
        """
        Format the code debugging prompt.
        
        Args:
            issue_description: Description of the issue
            code: Code with the issue
            
        Returns:
            Formatted prompt string
        """
        return cls.CODE_DEBUGGING.format(
            issue_description=issue_description,
            code=code
        )
    
    @classmethod
    def format_docker_optimization(cls, 
                                  dockerfile: str, 
                                  context_size: str = "Unknown", 
                                  build_time: str = "Unknown") -> str:
        """
        Format the Docker optimization prompt.
        
        Args:
            dockerfile: Current Dockerfile content
            context_size: Size of the Docker build context
            build_time: Current build time
            
        Returns:
            Formatted prompt string
        """
        return cls.DOCKER_OPTIMIZATION.format(
            dockerfile=dockerfile,
            context_size=context_size,
            build_time=build_time
        )


class LLMOrchestrator:
    """
    Main orchestrator for LLM operations. Manages model selection,
    prompt generation, and response handling.
    """
    
    def __init__(self):
        """Initialize the LLM orchestrator with default models and settings."""
        self.models = {
            "complex": "llama3.1:8b",  # For code rewriting, log analysis
            "fast": "phi:latest"       # For quick tasks, response refinement
        }
        self.cache = LLMCache()
        self.prompt_templates = PromptTemplates
        
        # Track performance metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0,
            "total_response_time": 0,
            "calls_by_model": {model: 0 for model in self.models.values()}
        }
        
        logger.info(f"LLMOrchestrator initialized with models: {self.models}")
    
    def _call_model(self, 
                   model: str, 
                   prompt: str, 
                   temperature: float = 0.7, 
                   max_tokens: int = 2000,
                   use_cache: bool = True,
                   cache_max_age: Optional[float] = None) -> Dict[str, Any]:
        """
        Call an Ollama model with the given prompt.
        
        Args:
            model: Name of the Ollama model
            prompt: Prompt text
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use cache
            cache_max_age: Maximum age for cache entries in seconds
            
        Returns:
            Dict containing the model response and metadata
        """
        self.metrics["total_calls"] += 1
        self.metrics["calls_by_model"][model] = self.metrics["calls_by_model"].get(model, 0) + 1
        
        # Check cache first if enabled
        if use_cache:
            cached_response = self.cache.get_cached_response(prompt, model, max_age=cache_max_age)
            if cached_response:
                logger.info(f"Using cached response for prompt with {model}")
                return {"text": cached_response, "from_cache": True}
        
        start_time = time.time()
        
        try:
            # Call the Ollama API
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
            result = response["text"]
            
            # Update success metrics
            self.metrics["successful_calls"] += 1
            
            # Cache the result if enabled
            if use_cache:
                self.cache.cache_response(
                    prompt=prompt,
                    response=result,
                    model=model,
                    metadata={
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timestamp": time.time()
                    }
                )
            
            logger.info(f"Successfully called {model} model")
            
        except Exception as e:
            # Update failure metrics
            self.metrics["failed_calls"] += 1
            logger.error(f"Error calling {model}: {e}")
            raise
        
        finally:
            # Update timing metrics
            response_time = time.time() - start_time
            self.metrics["total_response_time"] += response_time
            self.metrics["average_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["total_calls"]
            )
        
        return {"text": result, "response_time": response_time, "from_cache": False}
    
    def generate_code(self, 
                     task_description: str, 
                     task_details: str, 
                     optimization_goal: str, 
                     task_type: str = "complex") -> str:
        """
        Generate code for a specified task.
        
        Args:
            task_description: Brief description of the task
            task_details: Detailed requirements
            optimization_goal: What to optimize for
            task_type: Which model to use (complex or fast)
            
        Returns:
            Generated code as a string
        """
        prompt = self.prompt_templates.format_code_generation(
            task_description=task_description,
            task_details=task_details,
            optimization_goal=optimization_goal
        )
        
        model = self.models.get(task_type, self.models["complex"])
        response = self._call_model(
            model=model,
            prompt=prompt,
            temperature=0.3,  # Lower temperature for code generation
            max_tokens=3000,  # More tokens for complex code
            use_cache=True
        )
        
        result = response["text"]
        logger.info(f"Generated code with {model}: {result[:50]}...")
        return result
    
    def analyze_logs(self, log_content: str, task_type: str = "complex") -> Dict[str, Any]:
        """
        Analyze logs to identify issues and recommend solutions.
        
        Args:
            log_content: Log content to analyze
            task_type: Which model to use (complex or fast)
            
        Returns:
            Dict containing analysis results
        """
        prompt = self.prompt_templates.format_log_analysis(log_content=log_content)
        
        model = self.models.get(task_type, self.models["complex"])
        response = self._call_model(
            model=model,
            prompt=prompt,
            temperature=0.2,  # Lower for more deterministic analysis
            max_tokens=2000,
            use_cache=True,
            cache_max_age=86400  # Cache for 24 hours (logs might change)
        )
        
        try:
            result = json.loads(response["text"])
        except json.JSONDecodeError:
            # Fallback to extracting JSON from text if there's other content
            text = response["text"]
            json_str = text[text.find('{'):text.rfind('}')+1]
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from log analysis response")
                result = {
                    "summary": "Failed to parse response",
                    "recommendations": [],
                    "priority": "High"
                }
        
        logger.info(f"Log analysis completed with {model}: {result.get('summary', '')[:50]}...")
        return result
    
    def refine_response(self, 
                       initial_response: str, 
                       context: str, 
                       task_type: str = "fast") -> Dict[str, Any]:
        """
        Refine a response to improve quality.
        
        Args:
            initial_response: Original response to refine
            context: Additional context for refinement
            task_type: Which model to use (complex or fast)
            
        Returns:
            Dict containing refined response and changes
        """
        prompt = self.prompt_templates.format_response_refinement(
            initial_response=initial_response,
            context=context
        )
        
        model = self.models.get(task_type, self.models["fast"])
        response = self._call_model(
            model=model,
            prompt=prompt,
            temperature=0.5,
            max_tokens=2000,
            use_cache=True
        )
        
        try:
            result = json.loads(response["text"])
        except json.JSONDecodeError:
            # Fallback to simple result if JSON parsing fails
            logger.warning("Failed to parse JSON from response refinement")
            result = {
                "refined_response": response["text"],
                "changes_made": "Unknown (JSON parsing failed)"
            }
        
        refined_response = result.get("refined_response", "")
        logger.info(f"Refined response with {model}: {refined_response[:50]}...")
        return result
    
    def debug_code(self, 
                  issue_description: str, 
                  code: str, 
                  task_type: str = "complex") -> Dict[str, Any]:
        """
        Debug code to identify and fix issues.
        
        Args:
            issue_description: Description of the issue
            code: Code with the issue
            task_type: Which model to use (complex or fast)
            
        Returns:
            Dict containing debugging results
        """
        prompt = self.prompt_templates.format_code_debugging(
            issue_description=issue_description,
            code=code
        )
        
        model = self.models.get(task_type, self.models["complex"])
        response = self._call_model(
            model=model,
            prompt=prompt,
            temperature=0.2,
            max_tokens=3000,
            use_cache=True
        )
        
        try:
            result = json.loads(response["text"])
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse JSON from code debugging response")
            text = response["text"]
            
            # Try to extract fixed code between code blocks
            if "```python" in text and "```" in text:
                fixed_code = text.split("```python")[1].split("```")[0].strip()
            else:
                fixed_code = text
            
            result = {
                "root_cause": "Unknown (JSON parsing failed)",
                "explanation": "Failed to parse structured response",
                "fixed_code": fixed_code
            }
        
        logger.info(f"Code debugging completed with {model}")
        return result
    
    def optimize_dockerfile(self, 
                          dockerfile: str, 
                          context_size: str = "Unknown", 
                          build_time: str = "Unknown") -> Dict[str, Any]:
        """
        Optimize a Dockerfile for better performance and smaller size.
        
        Args:
            dockerfile: Current Dockerfile content
            context_size: Size of the build context
            build_time: Current build time
            
        Returns:
            Dict containing optimization results
        """
        prompt = self.prompt_templates.format_docker_optimization(
            dockerfile=dockerfile,
            context_size=context_size,
            build_time=build_time
        )
        
        model = self.models["complex"]
        response = self._call_model(
            model=model,
            prompt=prompt,
            temperature=0.3,
            max_tokens=3000,
            use_cache=True
        )
        
        try:
            result = json.loads(response["text"])
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse JSON from Docker optimization response")
            text = response["text"]
            
            # Try to extract optimized Dockerfile
            if "```dockerfile" in text and "```" in text:
                optimized_dockerfile = text.split("```dockerfile")[1].split("```")[0].strip()
            else:
                optimized_dockerfile = text
            
            result = {
                "optimized_dockerfile": optimized_dockerfile,
                "improvements": ["Parsing structured response failed"],
                "expected_benefits": {
                    "size_reduction": "Unknown",
                    "build_time_reduction": "Unknown",
                    "security_enhancements": []
                }
            }
        
        logger.info(f"Dockerfile optimization completed with {model}")
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the orchestrator.
        
        Returns:
            Dict containing performance metrics
        """
        cache_stats = self.cache.get_stats()
        return {
            "llm": self.metrics,
            "cache": cache_stats,
            "models": self.models
        }
    
    def update_model(self, role: str, model_name: str) -> bool:
        """
        Update a model for a specific role.
        
        Args:
            role: The role to update (complex or fast)
            model_name: New model name
            
        Returns:
            Boolean indicating success
        """
        if role not in self.models:
            logger.error(f"Invalid role: {role}")
            return False
        
        # Check if model is available in Ollama
        try:
            # Simple test query to check if model exists
            ollama.generate(
                model=model_name,
                prompt="test",
                options={"max_tokens": 10}
            )
            
            # Update model if test was successful
            self.models[role] = model_name
            logger.info(f"Updated {role} model to {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False


# Create a global instance for easy import
orchestrator = LLMOrchestrator()


def generate_code(task_description: str, task_details: str, optimization_goal: str) -> str:
    """
    Helper function to generate code using the global orchestrator.
    
    Args:
        task_description: Brief description of the task
        task_details: Detailed requirements
        optimization_goal: What to optimize for
        
    Returns:
        Generated code as a string
    """
    return orchestrator.generate_code(
        task_description=task_description,
        task_details=task_details,
        optimization_goal=optimization_goal
    )


def analyze_logs(log_content: str) -> Dict[str, Any]:
    """
    Helper function to analyze logs using the global orchestrator.
    
    Args:
        log_content: Log content to analyze
        
    Returns:
        Dict containing analysis results
    """
    return orchestrator.analyze_logs(log_content=log_content)


def refine_response(initial_response: str, context: str) -> Dict[str, Any]:
    """
    Helper function to refine a response using the global orchestrator.
    
    Args:
        initial_response: Original response to refine
        context: Additional context for refinement
        
    Returns:
        Dict containing refined response and changes
    """
    return orchestrator.refine_response(
        initial_response=initial_response,
        context=context
    )


if __name__ == "__main__":
    # Example usage
    example_code = generate_code(
        task_description="Fetch stock data from an API",
        task_details="The function should fetch historical stock data for a given symbol from a public API, handle errors gracefully, implement rate limiting, and return data as a pandas DataFrame.",
        optimization_goal="reliability and performance"
    )
    
    print("Generated Code Example:")
    print(example_code) 