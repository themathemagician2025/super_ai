# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
SUPER AI - Self-Evolving Autonomous Intelligence System

A self-evolving autonomous intelligence system designed to generate, modify, test,
and deploy its own improvements across multiple domains including forex prediction, 
sports betting, stock market analysis, and automated data collection.
"""

__version__ = '0.1.0'
__author__ = 'Super AI Team'

import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import main components
from .self_modification import (
    CodeGenerator,
    ModuleBenchmark,
    RollbackManager,
    MutationStrategies,
    ChangelogTracker
)

from .utils import (
    ConfigLoader,
    FileManager,
    SQLiteHelper
)

__all__ = [
    # Self-modification
    'CodeGenerator',
    'ModuleBenchmark',
    'RollbackManager',
    'MutationStrategies',
    'ChangelogTracker',
    
    # Utilities
    'ConfigLoader',
    'FileManager',
    'SQLiteHelper'
]
