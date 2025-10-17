"""
Niching Strategy Module

This module provides niching mechanisms for maintaining diversity in GP populations.

Components:
- NichingClusterer: Clustering-based niching
- CrossNicheSelector: Cross-niche parent selection
- DynamicKSelector: Dynamic k-value selection for niching
- FitnessSharing: Fitness sharing mechanism (future)

Usage:
    from gp_quant.niching import NichingClusterer, CrossNicheSelector, DynamicKSelector
    
    # Cluster population based on similarity
    clusterer = NichingClusterer(n_clusters=5)
    labels = clusterer.fit_predict(similarity_matrix)
    
    # Cross-niche parent selection
    selector = CrossNicheSelector(cross_niche_ratio=0.8, tournament_size=3)
    parents = selector.select(population, niche_labels, k=100)
    
    # Dynamic k-value selection
    k_selector = DynamicKSelector(mode='auto', k_min=2, k_max='auto')
    result = k_selector.select_k(similarity_matrix, population_size=5000)
"""

__version__ = '0.3.0'

# Import implemented components
from .clustering import NichingClusterer
from .selection import CrossNicheSelector
from .dynamic_k_selector import DynamicKSelector, create_k_selector

__all__ = [
    'NichingClusterer',
    'CrossNicheSelector',
    'DynamicKSelector',
    'create_k_selector',
]
