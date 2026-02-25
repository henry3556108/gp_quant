"""
Diverse Selection Module

Provides Greedy Diverse Selection for Top-K individuals,
ensuring selected individuals have low similarity to each other.
"""

from typing import List, Literal, Any, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_signal_similarity(signals_a: np.ndarray, signals_b: np.ndarray) -> float:
    """
    Compute Signal Overlap (Jaccard Index / IoU) between two signal arrays.
    
    Args:
        signals_a: Boolean array of trading signals for individual A
        signals_b: Boolean array of trading signals for individual B
    
    Returns:
        Jaccard similarity score (0 = no overlap, 1 = identical)
    """
    if len(signals_a) != len(signals_b):
        raise ValueError(f"Signal arrays must have same length. Got {len(signals_a)} and {len(signals_b)}")
    
    intersection = np.sum(signals_a & signals_b)
    union = np.sum(signals_a | signals_b)
    
    if union == 0:
        return 0.0  # No signals at all
    
    return float(intersection / union)


def compute_pnl_correlation(pnl_a: pd.Series, pnl_b: pd.Series) -> float:
    """
    Compute Pearson correlation between two PnL curves.
    
    Args:
        pnl_a: PnL curve (cumulative) for individual A
        pnl_b: PnL curve (cumulative) for individual B
    
    Returns:
        Correlation coefficient (-1 to 1, higher = more similar)
    """
    # Align indices
    common_idx = pnl_a.index.intersection(pnl_b.index)
    if len(common_idx) < 2:
        return 0.0
    
    pnl_a_aligned = pnl_a.loc[common_idx]
    pnl_b_aligned = pnl_b.loc[common_idx]
    
    # Calculate daily returns
    returns_a = pnl_a_aligned.pct_change().fillna(0)
    returns_b = pnl_b_aligned.pct_change().fillna(0)
    
    # Correlation
    corr = returns_a.corr(returns_b)
    
    return float(corr) if not np.isnan(corr) else 0.0


def select_topk_diverse(
    population: List[Any],
    k: int,
    engine: Any,
    similarity_threshold: float = 0.9,
    mode: Literal["signal", "pnl"] = "signal",
    fallback_to_naive: bool = True,
    verbose: bool = True
) -> Tuple[List[Any], dict]:
    """
    Select top-k individuals with diversity constraint.
    
    Algorithm:
    1. Sort population by fitness (descending)
    2. Select the highest fitness individual
    3. For each remaining candidate:
       a. Compute max similarity with all selected individuals
       b. If max_similarity < threshold, select this candidate
    4. Repeat until k individuals are selected
    5. If not enough candidates pass threshold, optionally fall back to naive selection
    
    Args:
        population: List of DEAP individuals with fitness values
        k: Number of individuals to select
        engine: BacktestingEngine with get_signals() and get_pnl_curve() methods
        similarity_threshold: Maximum allowed similarity (default 0.9)
        mode: "signal" for Signal IoU, "pnl" for PnL correlation
        fallback_to_naive: If True, fill remaining slots with next best fitness
        verbose: Print progress
    
    Returns:
        Tuple of:
        - List of selected individuals
        - Dict with selection statistics
    """
    if k <= 0:
        return [], {"selected": 0, "rejected": 0}
    
    # Sort by fitness
    sorted_pop = sorted(
        population,
        key=lambda ind: ind.fitness.values[0] if (hasattr(ind, 'fitness') and ind.fitness.values) else -float('inf'),
        reverse=True
    )
    
    # Remove duplicates by tree structure
    unique_pop = []
    seen_trees = set()
    for ind in sorted_pop:
        tree_str = str(ind)
        if tree_str not in seen_trees:
            seen_trees.add(tree_str)
            unique_pop.append(ind)
    
    if verbose:
        print(f"  Unique individuals: {len(unique_pop)} / {len(population)}")
    
    # Pre-compute signals/pnl for efficiency
    cache = {}
    
    def get_representation(ind):
        """Get cached signal or pnl representation."""
        ind_id = id(ind)
        if ind_id not in cache:
            try:
                if mode == "signal":
                    cache[ind_id] = engine.get_signals(ind)
                else:  # pnl
                    cache[ind_id] = engine.get_pnl_curve(ind)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to get {mode} for individual: {e}")
                cache[ind_id] = None
        return cache[ind_id]
    
    def compute_similarity(ind_a, ind_b) -> float:
        """Compute similarity between two individuals."""
        rep_a = get_representation(ind_a)
        rep_b = get_representation(ind_b)
        
        if rep_a is None or rep_b is None:
            return 0.0
        
        if mode == "signal":
            return compute_signal_similarity(rep_a, rep_b)
        else:  # pnl
            return compute_pnl_correlation(rep_a, rep_b)
    
    # Selection process
    selected = []
    rejected_by_similarity = 0
    
    iterator = tqdm(unique_pop, desc="Diverse Selection") if verbose else unique_pop
    
    for candidate in iterator:
        if len(selected) >= k:
            break
        
        if len(selected) == 0:
            # First one: always select the best
            selected.append(candidate)
            continue
        
        # Compute max similarity with all selected
        max_sim = 0.0
        for sel in selected:
            sim = compute_similarity(candidate, sel)
            max_sim = max(max_sim, sim)
            
            # Early exit if already too similar
            if max_sim >= similarity_threshold:
                break
        
        if max_sim < similarity_threshold:
            selected.append(candidate)
        else:
            rejected_by_similarity += 1
    
    # Fallback if not enough
    if len(selected) < k and fallback_to_naive:
        if verbose:
            print(f"  Fallback: filling {k - len(selected)} remaining slots with next best fitness")
        
        selected_ids = {id(s) for s in selected}
        for ind in unique_pop:
            if id(ind) not in selected_ids:
                selected.append(ind)
                if len(selected) >= k:
                    break
    
    stats = {
        "selected": len(selected),
        "rejected_by_similarity": rejected_by_similarity,
        "threshold": similarity_threshold,
        "mode": mode
    }
    
    if verbose:
        print(f"  Selected: {len(selected)}, Rejected by similarity: {rejected_by_similarity}")
    
    return selected, stats


def compute_topk_diversity_stats(
    individuals: List[Any],
    engine: Any,
    mode: Literal["signal", "pnl"] = "signal"
) -> dict:
    """
    Compute diversity statistics for a set of individuals.
    
    Args:
        individuals: List of DEAP individuals
        engine: BacktestingEngine
        mode: "signal" or "pnl"
    
    Returns:
        Dict with mean, max, min similarity
    """
    n = len(individuals)
    if n < 2:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "pairs": 0}
    
    # Get representations
    representations = []
    for ind in individuals:
        try:
            if mode == "signal":
                rep = engine.get_signals(ind)
            else:
                rep = engine.get_pnl_curve(ind)
            representations.append(rep)
        except Exception:
            representations.append(None)
    
    # Pairwise similarities
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            rep_a, rep_b = representations[i], representations[j]
            if rep_a is None or rep_b is None:
                continue
            
            if mode == "signal":
                sim = compute_signal_similarity(rep_a, rep_b)
            else:
                sim = compute_pnl_correlation(rep_a, rep_b)
            
            similarities.append(sim)
    
    if not similarities:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "pairs": 0}
    
    return {
        "mean": float(np.mean(similarities)),
        "max": float(np.max(similarities)),
        "min": float(np.min(similarities)),
        "pairs": len(similarities)
    }
