#!/usr/bin/env python3
"""
Analyze Top 5 PnL Correlation for each experiment.

This script:
1. Loads top 5 individuals from each experiment (by fitness)
2. Generates PnL curves for each individual
3. Calculates mean pairwise correlation of PnL curves
4. Creates boxplot comparing correlation across experiment groups
"""

import argparse
import json
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from deap import creator, base, gp

# Setup DEAP
if not hasattr(creator, 'FitnessMax'):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gp_quant.evolution.components.gp import operators
from gp_quant.data.loader import load_and_process_data

# Set fonts
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path("/Volumes/X9 Pro/gp_quant_experiments")

# Experiment directories
EXPERIMENTS = {
    "Rolling Signal\n(soft_min)": [
        "rolling_window_signal_niche_20251210_1314",
        "rolling_window_signal_niche_20251210_1343",
        "rolling_window_signal_niche_20251210_1408",
        "rolling_window_signal_niche_20251210_1434",
        "rolling_window_signal_niche_20251210_1504",
        "rolling_window_signal_niche_20251210_1532",
        "rolling_window_signal_niche_20251210_1559",
        "rolling_window_signal_niche_20251210_1624",
        "rolling_window_signal_niche_20251210_1647",
        "rolling_window_signal_niche_20251210_1713",
    ],
    "Rolling Baseline\n(soft_min)": [
        "rolling_window_large_scale_20251209_1631",
        "rolling_window_large_scale_20251209_1647",
        "rolling_window_large_scale_20251209_1703",
        "rolling_window_large_scale_20251209_1718",
        "rolling_window_large_scale_20251209_1734",
        "rolling_window_large_scale_20251209_1750",
        "rolling_window_large_scale_20251209_1806",
        "rolling_window_large_scale_20251209_1821",
        "rolling_window_large_scale_20251209_1836",
        "rolling_window_large_scale_20251209_1851",
    ],
}

# Time periods (8:2 split)
TRAIN_START = "2006-06-25"
TRAIN_END = "2021-06-30"
TEST_START = "2021-07-01"
TEST_END = "2024-12-31"

TOP_K = 1000  # Full population
SAMPLE_MODE = False  # Take all individuals


def load_top_k_individuals(exp_dir: Path, k: int = 5, sample: bool = False) -> List[Any]:
    """Load top k individuals or random sample from the best generation."""
    import random
    
    pop_dir = exp_dir / 'populations'
    stats_file = exp_dir / 'generation_stats.json'
    
    if not pop_dir.exists() or not stats_file.exists():
        return []
    
    try:
        with open(stats_file, 'r') as f:
            gen_stats = json.load(f)
        
        # Find best generation
        best_gen = max(gen_stats, key=lambda g: g.get('best_fitness', -float('inf')))
        best_gen_num = best_gen['generation']
        
        pkl_file = pop_dir / f'generation_{best_gen_num:03d}.pkl'
        with open(pkl_file, 'rb') as f:
            population = pickle.load(f)
        
        if sample:
            # Random sample
            if len(population) <= k:
                return population
            return random.sample(population, k)
        else:
            # Top k by fitness
            population.sort(key=lambda x: x.fitness.values[0] if x.fitness.values else -float('inf'), reverse=True)
            return population[:k]
    except Exception as e:
        print(f"    ⚠️ Error loading {exp_dir.name}: {e}")
        return []


def generate_signals(individual: Any, data: pd.DataFrame, pset) -> np.ndarray:
    """Generate trading signals from a GP individual."""
    import copy
    from gp_quant.evolution.components.gp.operators import NumVector
    
    local_pset = copy.deepcopy(pset)
    
    price_vec = data['Close'].values.astype(float)
    volume_vec = data['Volume'].values.astype(float)
    
    price_vec = price_vec.view(NumVector)
    volume_vec = volume_vec.view(NumVector)
    
    local_pset.terminals[NumVector][0].value = price_vec
    local_pset.terminals[NumVector][1].value = volume_vec
    
    try:
        expr = str(individual)
        func = gp.compile(expr=expr, pset=local_pset)
        
        try:
            signals = func()
        except TypeError:
            signals = func(price_vec, volume_vec)
        
        if isinstance(signals, (bool, np.bool_)):
            signals = np.full(len(data), signals, dtype=bool)
        elif isinstance(signals, np.ndarray):
            if signals.ndim == 0 or signals.size == 1:
                scalar_val = signals.item() if signals.ndim == 0 else signals[0]
                signals = np.full(len(data), bool(scalar_val), dtype=bool)
            else:
                signals = signals.astype(bool)
        else:
            signals = np.full(len(data), bool(signals), dtype=bool)
        
        return signals
    except Exception as e:
        return np.zeros(len(data), dtype=bool)


def calculate_pnl_curve(signals: np.ndarray, data: pd.DataFrame, 
                        initial_capital: float = 100000.0) -> pd.Series:
    """Calculate PnL curve from trading signals."""
    position = 0
    capital = initial_capital
    shares = 0.0
    equity_values = []
    
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    
    for i in range(len(signals) - 1):
        signal = signals[i]
        next_open = open_prices[i + 1]
        
        if position == 0 and signal:
            if next_open > 0:
                shares = capital / next_open
                capital = 0.0
                position = 1
        elif position == 1 and not signal:
            capital = shares * next_open
            shares = 0.0
            position = 0
        
        current_equity = capital + shares * close_prices[i]
        equity_values.append(current_equity)
    
    final_equity = capital + shares * close_prices[-1]
    equity_values.append(final_equity)
    
    return pd.Series(equity_values, index=data.index)


def calculate_mean_pairwise_correlation(pnl_curves: List[pd.Series]) -> float:
    """
    Calculate mean pairwise correlation of PnL curves.
    If two curves are identical (including constant), correlation = 1.0
    Uses vectorized operations for speed.
    """
    from tqdm import tqdm
    
    if len(pnl_curves) < 2:
        return np.nan
    
    # Convert to numpy array for speed
    # Align all curves first
    aligned = pd.concat(pnl_curves, axis=1, join='inner')
    if aligned.shape[1] < 2:
        return np.nan
    
    n = aligned.shape[1]
    data = aligned.values  # shape: (timepoints, n_individuals)
    
    # Precompute means and stds for all columns
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    
    # Identify constant columns (std=0)
    constant_mask = stds == 0
    
    # Normalize non-constant columns for correlation calculation
    # For constant columns, set to 0 (will handle separately)
    data_normalized = np.zeros_like(data, dtype=float)
    non_const_idx = ~constant_mask
    if np.any(non_const_idx):
        data_normalized[:, non_const_idx] = (data[:, non_const_idx] - means[non_const_idx]) / stds[non_const_idx]
    
    # Calculate correlation matrix for non-constant columns
    # corr(i,j) = mean(normalized_i * normalized_j)
    if np.sum(non_const_idx) >= 2:
        corr_matrix = np.corrcoef(data[:, non_const_idx].T)
    else:
        corr_matrix = np.array([[1.0]])
    
    # Build full correlation result
    total_pairs = n * (n - 1) // 2
    correlations = []
    
    # Map non-constant indices
    non_const_indices = np.where(non_const_idx)[0]
    const_indices = np.where(constant_mask)[0]
    
    for i in range(n):
        for j in range(i + 1, n):
            x = data[:, i]
            y = data[:, j]
            
            # Case 1: Both identical
            if np.allclose(x, y):
                correlations.append(1.0)
            # Case 2: Both constant but different
            elif constant_mask[i] and constant_mask[j]:
                correlations.append(0.0)
            # Case 3: One constant, one not
            elif constant_mask[i] or constant_mask[j]:
                correlations.append(0.0)
            # Case 4: Both non-constant - use precomputed correlation
            else:
                # Find indices in reduced matrix
                idx_i = np.searchsorted(non_const_indices, i)
                idx_j = np.searchsorted(non_const_indices, j)
                corr = corr_matrix[idx_i, idx_j]
                correlations.append(corr if not np.isnan(corr) else 0.0)
    
    if len(correlations) == 0:
        return np.nan
    
    return np.mean(correlations)


def analyze_experiment(exp_dir_name: str, data: pd.DataFrame) -> Optional[float]:
    """Analyze a single experiment and return mean correlation of top 5."""
    exp_path = BASE_DIR / exp_dir_name
    
    # Load top 5 individuals
    top_k = load_top_k_individuals(exp_path, TOP_K)
    if len(top_k) < 2:
        print(f"    {exp_dir_name[-4:]}: Not enough individuals")
        return None
    
    # Generate PnL curves for each
    pnl_curves = []
    for ind in top_k:
        signals = generate_signals(ind, data, operators.pset)
        pnl = calculate_pnl_curve(signals, data)
        pnl_curves.append(pnl)
    
    # Calculate mean correlation
    mean_corr = calculate_mean_pairwise_correlation(pnl_curves)
    
    print(f"    {exp_dir_name[-4:]}: Top {len(top_k)} Mean Corr = {mean_corr:.4f}")
    return mean_corr


def analyze_all_experiments(data: pd.DataFrame) -> Dict[str, List[float]]:
    """Analyze all experiments and collect results."""
    results = {}
    
    for group_name, exp_dirs in EXPERIMENTS.items():
        print(f"\n分析 {group_name.replace(chr(10), ' ')}...")
        group_results = []
        
        for exp_dir in exp_dirs:
            corr = analyze_experiment(exp_dir, data)
            if corr is not None and not np.isnan(corr):
                group_results.append(corr)
        
        results[group_name] = group_results
        if group_results:
            print(f"  → {len(group_results)} runs, Mean Corr = {np.mean(group_results):.4f}")
    
    return results


def plot_boxplot(results: Dict[str, List[float]]):
    """Create boxplot of mean correlations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "Rolling Signal\n(soft_min)": "#f39c12",
        "Rolling Baseline\n(soft_min)": "#8e44ad",
    }
    
    labels = list(EXPERIMENTS.keys())
    data = [results.get(name, []) for name in labels]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, name in zip(bp['boxes'], labels):
        patch.set_facecolor(colors.get(name, '#cccccc'))
        patch.set_alpha(0.8)
    
    # Add mean values
    for i, name in enumerate(labels):
        vals = results.get(name, [])
        if vals:
            mean_val = np.mean(vals)
            ax.scatter(i + 1, mean_val, marker='D', color='white', s=50, zorder=5, edgecolors='black')
            ax.annotate(f'{mean_val:.3f}', (i + 1, mean_val), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'Top {TOP_K} PnL Curve Mean Correlation\n(Lower = More Diverse)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Pairwise Correlation', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add reference line at 1.0 (perfect correlation)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Correlation')
    ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='No Correlation')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    output_path = BASE_DIR / "top5_pnl_correlation_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Boxplot 已保存至: {output_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print(f"📊 Top {TOP_K} PnL Correlation Analysis")
    print("=" * 60)
    print(f"\nTrain: {TRAIN_START} → {TRAIN_END}")
    print(f"Test:  {TEST_START} → {TEST_END}")
    
    # Load SPY data
    print("\n載入數據...")
    data_dict = load_and_process_data('history', ['SPY'])
    data = data_dict['SPY']
    print(f"SPY 數據: {len(data)} 天")
    
    # Analyze all experiments
    results = analyze_all_experiments(data)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 統計摘要")
    print("=" * 60)
    
    for name in EXPERIMENTS.keys():
        clean_name = name.replace(chr(10), ' ')
        group_results = results.get(name, [])
        if group_results:
            print(f"\n{clean_name} ({len(group_results)} runs):")
            print(f"  Mean Corr: {np.mean(group_results):.4f} ± {np.std(group_results):.4f}")
    
    # Plot
    plot_boxplot(results)


if __name__ == "__main__":
    main()
