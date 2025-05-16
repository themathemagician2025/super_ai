# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
LLM Training module for Super AI system.
Implements transformer-based language model training from scratch.
"""

from .trainer import LLMTrainer
from .model import TransformerLLM
from .data_processor import DataProcessor
from .config import TrainingConfig

__all__ = ['LLMTrainer', 'TransformerLLM', 'DataProcessor', 'TrainingConfig'] 