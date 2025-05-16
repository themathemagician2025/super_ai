# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Self-Modification Package

This package provides the core self-modification capabilities that allow
the Super AI system to evolve and improve its own code.
"""

from .code_generator import CodeGenerator
from .module_benchmark import ModuleBenchmark
from .rollback_manager import RollbackManager
from .mutation_strategies import MutationStrategies
from .changelog_tracker import ChangelogTracker

__all__ = [
    'CodeGenerator',
    'ModuleBenchmark',
    'RollbackManager',
    'MutationStrategies',
    'ChangelogTracker'
]
