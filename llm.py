"""
LLM Module

This module provides convenience imports for LLM services.
"""

# Import the OllamaClient from the core module
from .core.llm import OllamaClient

# Export the OllamaClient for easier imports
__all__ = ["OllamaClient"]
