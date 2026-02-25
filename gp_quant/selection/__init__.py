"""
Selection module for GP Quant.

Provides various selection strategies including diversity-aware selection.
"""

from gp_quant.selection.diverse_selection import (
    select_topk_diverse,
    compute_signal_similarity,
    compute_pnl_correlation,
    compute_topk_diversity_stats,
)

__all__ = [
    "select_topk_diverse",
    "compute_signal_similarity",
    "compute_pnl_correlation",
    "compute_topk_diversity_stats",
]
