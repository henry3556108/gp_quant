"""
Parallel Execution Module

This module provides parallel execution capabilities for GP evolution,
including parallel fitness evaluation and similarity calculation.
"""

from .executor import ParallelExecutor
from .fitness_evaluator import ParallelFitnessEvaluator

__all__ = [
    'ParallelExecutor',
    'ParallelFitnessEvaluator',
]
