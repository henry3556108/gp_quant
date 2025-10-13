"""
Niching Strategy Module

This module provides niching mechanisms for maintaining diversity in GP populations.

Components:
- NichingClusterer: Clustering-based niching
- CrossNicheSelector: Cross-niche parent selection
- FitnessSharing: Fitness sharing mechanism (future)

Usage:
    from gp_quant.niching import NichingClusterer
    
    # Cluster population based on similarity
    clusterer = NichingClusterer(n_clusters=5)
    labels = clusterer.fit_predict(similarity_matrix)
"""

__version__ = '0.1.0'

# Import implemented components
from .clustering import NichingClusterer

__all__ = [
    'NichingClusterer',
]
