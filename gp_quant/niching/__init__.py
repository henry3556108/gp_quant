"""
Niching Strategy Module

This module provides niching mechanisms for maintaining diversity in GP populations.

Components:
- NichingClusterer: Clustering-based niching
- CrossNicheSelector: Cross-niche parent selection
- FitnessSharing: Fitness sharing mechanism (future)

Usage:
    from gp_quant.niching import NichingClusterer, CrossNicheSelector
    
    # Cluster population based on similarity
    clusterer = NichingClusterer(n_clusters=5)
    labels = clusterer.fit_predict(similarity_matrix)
    
    # Cross-niche parent selection
    selector = CrossNicheSelector(cross_niche_ratio=0.8, tournament_size=3)
    parents = selector.select(population, niche_labels, k=100)
"""

__version__ = '0.2.0'

# Import implemented components
from .clustering import NichingClusterer
from .selection import CrossNicheSelector

__all__ = [
    'NichingClusterer',
    'CrossNicheSelector',
]
