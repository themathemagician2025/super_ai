# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
LLM Module

This module provides interfaces for interacting with Language Model services.
Supports various providers like Ollama, OpenAI, etc.
"""

import os
import requests
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API to access LLM models
    """
    
    def __init__(self, model_name: str = "llama3.1", url: str = "http://host.docker.internal:11434"):
        """
        Initialize the Ollama client
        
        Args:
            model_name: Name of the model to use
            url: URL of the Ollama service
        """
        self.model_name = model_name
        self.base_url = url
        logger.info(f"Initialized OllamaClient with model {model_name} at {url}")
        
        # Verify connection
        try:
            self.list_models()
            logger.info(f"Successfully connected to Ollama at {url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {e}")
    
    def query(self, prompt: str, **kwargs) -> str:
        """
        Send a query to the model and get a response
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the model
            
        Returns:
            The model's response as a string
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return f"Error: {str(e)}"
    
    def list_models(self) -> Dict:
        """
        List available models from Ollama
        
        Returns:
            Dictionary containing model information
        """
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()