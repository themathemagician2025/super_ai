#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Response Amplification Module for SUPER AI PROJECT

This module handles the process of enhancing and amplifying responses from the AI system
using Ollama-powered LLMs (Llama3.1 and Phi). It ensures all responses are verbose,
informative, and exceed the minimum required length.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple

# Ensure ollama is available
try:
    import ollama
except ImportError:
    raise ImportError("Ollama package is required. Install with: pip install ollama")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv("LOG_DIR", "/app/logs") + "/response_amplifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("response_amplifier")

class ResponseCache:
    """Cache for storing and retrieving amplified responses to reduce LLM calls."""
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the response cache.
        
        Args:
            cache_size: Maximum number of responses to cache
        """
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_access_times = {}
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a cached response.
        
        Args:
            key: Cache key (hash of original response and query)
            
        Returns:
            Cached amplified response or None if not found
        """
        if key in self.cache:
            self.cache_hits += 1
            self.last_access_times[key] = time.time()
            return self.cache[key]
        
        self.cache_misses += 1
        return None
    
    def put(self, key: str, value: str):
        """
        Store a response in the cache.
        
        Args:
            key: Cache key
            value: Amplified response to cache
        """
        # Evict least recently used item if cache is full
        if len(self.cache) >= self.cache_size:
            lru_key = min(self.last_access_times, key=self.last_access_times.get)
            del self.cache[lru_key]
            del self.last_access_times[lru_key]
        
        self.cache[key] = value
        self.last_access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache usage statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_ratio": hit_ratio
        }


class ResponseAmplifier:
    """
    Main class for amplifying AI responses to make them more detailed,
    comprehensive, and verbose.
    """
    
    def __init__(self, 
                 primary_model: str = "llama3.1:8b", 
                 fallback_model: str = "phi:latest",
                 min_words: int = 100,
                 max_tokens: int = 4000,
                 use_cache: bool = True):
        """
        Initialize the response amplifier.
        
        Args:
            primary_model: Primary Ollama model for response amplification
            fallback_model: Fallback model if primary fails
            min_words: Minimum word count for amplified responses
            max_tokens: Maximum tokens for the generated response
            use_cache: Whether to use response caching
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.min_words = min_words
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        
        # Initialize cache if enabled
        self.cache = ResponseCache() if use_cache else None
        
        # Statistics tracking
        self.call_stats = {
            "total_calls": 0,
            "successful_amplifications": 0,
            "fallback_used": 0,
            "cache_hits": 0,
            "average_amplification_ratio": 0,
            "average_response_time": 0,
            "total_response_time": 0
        }
    
    def _create_cache_key(self, original_response: str, user_query: str) -> str:
        """
        Create a cache key from the original response and user query.
        
        Args:
            original_response: Original AI response
            user_query: User's query
            
        Returns:
            String hash to use as cache key
        """
        # Simple hash function for the cache key
        combined = f"{user_query}:{original_response}"
        return str(hash(combined))
    
    def amplify_response(self, 
                        original_response: str, 
                        user_query: str,
                        context: Optional[str] = None,
                        domain_specific_instructions: Optional[List[str]] = None) -> str:
        """
        Amplify a response to make it more detailed and informative.
        
        Args:
            original_response: The original response to amplify
            user_query: The user query that generated the original response
            context: Additional context to guide amplification
            domain_specific_instructions: Additional domain-specific instructions
            
        Returns:
            Amplified response text
        """
        self.call_stats["total_calls"] += 1
        start_time = time.time()
        
        # Check cache first if enabled
        if self.use_cache:
            cache_key = self._create_cache_key(original_response, user_query)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                self.call_stats["cache_hits"] += 1
                logger.info("Using cached amplified response")
                return cached_response
        
        # Count original words
        original_words = len(original_response.split())
        
        # Skip amplification if original is already verbose
        if original_words > self.min_words * 2:
            logger.info(f"Original response already verbose ({original_words} words). Skipping amplification.")
            return original_response
        
        # Prepare context
        context_text = f"\nADDITIONAL CONTEXT:\n{context}" if context else ""
        
        # Prepare domain-specific instructions
        domain_instructions = ""
        if domain_specific_instructions:
            domain_instructions = "\nDOMAIN-SPECIFIC REQUIREMENTS:\n" + "\n".join(
                [f"- {instr}" for instr in domain_specific_instructions]
            )
        
        prompt = f"""
        You are an expert at enhancing AI responses to make them more detailed, comprehensive, and insightful.
        
        USER QUERY:
        {user_query}
        
        ORIGINAL RESPONSE:
        {original_response}
        {context_text}
        {domain_instructions}
        
        Please enhance and amplify this response to be more comprehensive, with these requirements:
        1. The response must be at least {self.min_words} words
        2. Add technical depth and sophistication
        3. Include practical examples where relevant
        4. Maintain a professional but approachable tone
        5. Organize information logically with clear structure
        6. Expand explanations of complex concepts
        7. Ensure the response fully addresses all aspects of the user's query
        
        Your enhanced response should build upon the original, not contradict it.
        PROVIDE ONLY THE ENHANCED RESPONSE, NO INTRODUCTION OR EXPLANATION.
        """
        
        try:
            # Try with primary model first
            response = ollama.generate(
                model=self.primary_model,
                prompt=prompt,
                options={"temperature": 0.7, "max_tokens": self.max_tokens}
            )
            amplified_text = response["text"].strip()
            
        except Exception as e:
            logger.warning(f"Primary model failed: {e}. Falling back to {self.fallback_model}")
            self.call_stats["fallback_used"] += 1
            
            try:
                # Fallback to secondary model
                response = ollama.generate(
                    model=self.fallback_model,
                    prompt=prompt,
                    options={"temperature": 0.7, "max_tokens": self.max_tokens - 1000}
                )
                amplified_text = response["text"].strip()
                
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}. Returning original response.")
                return original_response
        
        # Calculate amplification ratio
        amplified_words = len(amplified_text.split())
        
        # If amplification didn't meet minimum requirements, return original
        if amplified_words < self.min_words:
            logger.warning(f"Amplification insufficient ({amplified_words} words). Returning original.")
            return original_response
        
        # Update stats
        if amplified_words > original_words:
            self.call_stats["successful_amplifications"] += 1
            # Calculate running average of amplification ratio
            current_ratio = amplified_words / max(original_words, 1)
            prev_avg = self.call_stats["average_amplification_ratio"]
            prev_count = self.call_stats["successful_amplifications"] - 1
            if prev_count > 0:
                new_avg = (prev_avg * prev_count + current_ratio) / self.call_stats["successful_amplifications"]
                self.call_stats["average_amplification_ratio"] = new_avg
            else:
                self.call_stats["average_amplification_ratio"] = current_ratio
        
        # Update response time stats
        response_time = time.time() - start_time
        self.call_stats["total_response_time"] += response_time
        self.call_stats["average_response_time"] = (
            self.call_stats["total_response_time"] / self.call_stats["total_calls"]
        )
        
        logger.info(f"Response amplified: {original_words} → {amplified_words} words ({response_time:.2f}s)")
        
        # Store in cache if enabled
        if self.use_cache:
            cache_key = self._create_cache_key(original_response, user_query)
            self.cache.put(cache_key, amplified_text)
        
        return amplified_text
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get amplifier statistics and performance metrics.
        
        Returns:
            Dict containing performance statistics
        """
        stats = {
            "general": self.call_stats.copy(),
            "models": {
                "primary": self.primary_model,
                "fallback": self.fallback_model
            },
            "settings": {
                "min_words": self.min_words,
                "max_tokens": self.max_tokens,
                "use_cache": self.use_cache
            }
        }
        
        if self.use_cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


class AmplificationMiddleware:
    """
    Middleware that intercepts API responses and amplifies them
    before they're returned to the user.
    """
    
    def __init__(self, amplifier: ResponseAmplifier):
        """
        Initialize the middleware with a response amplifier.
        
        Args:
            amplifier: ResponseAmplifier instance
        """
        self.amplifier = amplifier
    
    async def process_response(self, response: Dict[str, Any], 
                              request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and amplify an API response.
        
        Args:
            response: Original API response
            request_data: Request data including user query
            
        Returns:
            Dict containing the amplified response
        """
        # Only process responses that have content to amplify
        if "response" in response and isinstance(response["response"], str):
            user_query = request_data.get("query", "")
            context = request_data.get("context", "")
            
            # Domain-specific instructions based on request type
            domain_instructions = []
            if "model" in request_data:
                if request_data["model"] == "forex":
                    domain_instructions = [
                        "Include specific forex terminology and concepts",
                        "Provide detailed market analysis where relevant",
                        "Explain risk management considerations"
                    ]
                elif request_data["model"] == "sports":
                    domain_instructions = [
                        "Include relevant statistics and performance metrics",
                        "Reference historical game data where applicable",
                        "Discuss strategy and tactical elements"
                    ]
            
            # Amplify the response
            original_response = response["response"]
            amplified_response = self.amplifier.amplify_response(
                original_response=original_response,
                user_query=user_query,
                context=context,
                domain_specific_instructions=domain_instructions
            )
            
            # Update the response
            response["response"] = amplified_response
            
            # Add metadata about amplification if debug mode
            if request_data.get("debug", False):
                response["_debug"] = {
                    "original_length": len(original_response.split()),
                    "amplified_length": len(amplified_response.split()),
                    "amplification_ratio": len(amplified_response.split()) / max(len(original_response.split()), 1)
                }
        
        return response


# Create singleton instance
default_amplifier = ResponseAmplifier(
    primary_model="llama3.1:8b",
    fallback_model="phi:latest",
    min_words=100,
    max_tokens=4000,
    use_cache=True
)

# Helper function to amplify a response
def amplify(response: str, query: str, context: str = None) -> str:
    """
    Amplify a response using the default amplifier.
    
    Args:
        response: Original response
        query: User query
        context: Optional context
        
    Returns:
        Amplified response
    """
    return default_amplifier.amplify_response(
        original_response=response,
        user_query=query,
        context=context
    )


if __name__ == "__main__":
    # Example usage
    original_response = "Optimized Dockerfile with multi-stage builds."
    user_query = "How can I optimize my Docker build process?"
    
    amplified = amplify(original_response, user_query)
    print(f"Original ({len(original_response.split())} words): {original_response}")
    print(f"Amplified ({len(amplified.split())} words): {amplified}")
    print(f"Stats: {json.dumps(default_amplifier.get_stats(), indent=2)}") 