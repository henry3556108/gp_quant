"""
Tree Similarity Module

This module provides tools for calculating similarity between GP trees,
which is essential for implementing niching strategies in genetic programming.

Components:
- TreeEditDistance: Calculate tree edit distance between GP trees
- SimilarityMatrix: Manage and compute similarity matrices for populations
- ParallelSimilarityCalculator: Parallel computation of tree similarities
- SimilarityCache: Cache mechanism for computed similarities
- Visualizer: Visualization tools for similarity analysis

Usage:
    from gp_quant.similarity import TreeEditDistance, SimilarityMatrix
    
    # Calculate distance between two trees
    ted = TreeEditDistance()
    distance = ted.compute(tree1, tree2)
    
    # Compute similarity matrix for population
    sim_matrix = SimilarityMatrix(population)
    matrix = sim_matrix.compute()
"""

__version__ = '0.1.0'

# Import implemented components
from .tree_edit_distance import (
    TreeNode,
    TreeEditDistance,
    deap_to_tree_node,
    tree_node_to_bracket,
    compute_ted,
    compute_similarity
)

from .similarity_matrix import SimilarityMatrix

# Will be imported as implementations are added
# from .parallel_calculator import ParallelSimilarityCalculator
# from .cache import SimilarityCache
# from .visualizer import SimilarityVisualizer

__all__ = [
    'TreeNode',
    'TreeEditDistance',
    'deap_to_tree_node',
    'tree_node_to_bracket',
    'compute_ted',
    'compute_similarity',
    'SimilarityMatrix',
    # 'ParallelSimilarityCalculator',
    # 'SimilarityCache',
    # 'SimilarityVisualizer',
]
