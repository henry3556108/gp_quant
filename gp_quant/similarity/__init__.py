"""
Tree Similarity Module

This module provides tools for calculating similarity between GP trees,
which is essential for implementing niching strategies in genetic programming.

Components:
- TreeEditDistance: Calculate tree edit distance between GP trees (O(n²m²))
- SubtreeKernel: Fast tree kernel for similarity (O(nm)) - TED alternative
- SimilarityMatrix: Manage and compute similarity matrices for populations
- ParallelSimilarityCalculator: Parallel computation of tree similarities
- SimilarityCache: Cache mechanism for computed similarities
- Visualizer: Visualization tools for similarity analysis

Usage:
    from gp_quant.similarity import TreeEditDistance, SimilarityMatrix
    
    # Calculate distance between two trees
    ted = TreeEditDistance()
    distance = ted.compute(tree1, tree2)
    
    # Fast alternative using Tree Kernel
    from gp_quant.similarity import SubtreeKernel
    kernel = SubtreeKernel()
    distance = kernel.compute_distance(tree1, tree2)
    
    # Compute similarity matrix for population
    sim_matrix = SimilarityMatrix(population)
    matrix = sim_matrix.compute()
"""

__version__ = '0.3.0'

# Import implemented components
from .tree_edit_distance import (
    TreeNode,
    TreeEditDistance,
    deap_to_tree_node,
    tree_node_to_bracket,
    compute_ted,
    compute_similarity
)

# Tree Kernel (fast alternative to TED)
from .tree_kernel import (
    SubtreeKernel,
    SubsetTreeKernel,
    compute_subtree_kernel,
    compute_kernel_similarity,
    compute_kernel_distance
)

from .similarity_matrix import SimilarityMatrix
from .parallel_calculator import ParallelSimilarityMatrix
from .sampled_calculator import SampledSimilarityMatrix
from .visualizer import (
    plot_diversity_evolution,
    plot_similarity_heatmap,
    plot_similarity_distribution,
    plot_population_tsne
)

# Will be imported as implementations are added
# from .cache import SimilarityCache

__all__ = [
    # Tree Edit Distance
    'TreeNode',
    'TreeEditDistance',
    'deap_to_tree_node',
    'tree_node_to_bracket',
    'compute_ted',
    'compute_similarity',
    # Tree Kernel (fast alternative)
    'SubtreeKernel',
    'SubsetTreeKernel',
    'compute_subtree_kernel',
    'compute_kernel_similarity',
    'compute_kernel_distance',
    # Similarity Matrix
    'SimilarityMatrix',
    'ParallelSimilarityMatrix',
    'SampledSimilarityMatrix',
    # Visualization
    'plot_diversity_evolution',
    'plot_similarity_heatmap',
    'plot_similarity_distribution',
    'plot_population_tsne',
    # 'SimilarityCache',
]

