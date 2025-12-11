#!/usr/bin/env python3
"""
Compare Rolling Window Signal Niche experiments

Usage:
    python scripts/compare_signal_niche.py --plot boxplot
    python scripts/compare_signal_niche.py --plot pnl
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
    "Rolling TED\n(soft_min)": [
        "rolling_window_ted_niche_20251210_0504",
        "rolling_window_ted_niche_20251210_0602",
        "rolling_window_ted_niche_20251210_0806",
    ],
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
        "rolling_window_signal_niche_20251210_1713"
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


# =============================================================================
# Step 1: Load best individual from pickle (across ALL generations)
# =============================================================================
def load_best_individual(exp_dir: Path) -> Tuple[Optional[Any], Optional[int]]:
    """
    Load the best individual across ALL generations.
    
    Returns:
        Tuple of (best_individual, best_generation_number)
        Returns (None, None) if loading fails.
    """
    pop_dir = exp_dir / 'populations'
    stats_file = exp_dir / 'generation_stats.json'
    
    if not pop_dir.exists():
        print(f"    ⚠️ No populations directory: {exp_dir.name}")
        return None, None
    
    # Step 1: Read generation_stats.json to find the best generation
    if not stats_file.exists():
        print(f"    ⚠️ No generation_stats.json: {exp_dir.name}")
        return None, None
    
    try:
        with open(stats_file, 'r') as f:
            gen_stats = json.load(f)
    except Exception as e:
        print(f"    ⚠️ Error reading generation_stats.json: {e}")
        return None, None
    
    if not gen_stats:
        print(f"    ⚠️ Empty generation_stats.json: {exp_dir.name}")
        return None, None
    
    # Find the generation with the highest best_fitness
    best_gen_stats = max(gen_stats, key=lambda g: g.get('best_fitness', -float('inf')))
    best_gen_num = best_gen_stats['generation']
    best_fitness = best_gen_stats['best_fitness']
    
    # Step 2: Load the pickle file for the best generation
    pkl_file = pop_dir / f'generation_{best_gen_num:03d}.pkl'
    
    if not pkl_file.exists():
        print(f"    ⚠️ Pickle file not found: {pkl_file.name}")
        return None, None
    
    try:
        with open(pkl_file, 'rb') as f:
            population = pickle.load(f)
        
        # Find individual with highest fitness in this generation
        best = max(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.values else -float('inf'))
        return best, best_gen_num
    except Exception as e:
        print(f"    ⚠️ Error loading pickle: {e}")
        return None, None


# =============================================================================
# Step 2: Generate signals from individual
# =============================================================================
def generate_signals(individual: Any, data: pd.DataFrame, pset) -> np.ndarray:
    """Generate trading signals from a GP individual."""
    import copy
    from gp_quant.evolution.components.gp.operators import NumVector
    
    # Make a copy of pset to avoid modifying global state
    local_pset = copy.deepcopy(pset)
    
    # Prepare data vectors
    price_vec = data['Close'].values.astype(float)
    volume_vec = data['Volume'].values.astype(float)
    
    # Convert to NumVector type
    price_vec = price_vec.view(NumVector)
    volume_vec = volume_vec.view(NumVector)
    
    # Set terminal values in pset
    local_pset.terminals[NumVector][0].value = price_vec
    local_pset.terminals[NumVector][1].value = volume_vec
    
    try:
        # Compile using str(individual) to avoid ephemeral constant issues
        expr = str(individual)
        func = gp.compile(expr=expr, pset=local_pset)
        
        # Execute - try without arguments first
        try:
            signals = func()
        except TypeError as e:
            # If it needs arguments, provide them
            if "missing" in str(e) and "required positional arguments" in str(e):
                signals = func(price_vec, volume_vec)
            else:
                raise
        
        # Handle different return types
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
        print(f"    ⚠️ Signal generation error: {e}")
        return np.zeros(len(data), dtype=bool)


# =============================================================================
# Step 3: Calculate PnL curve from signals
# =============================================================================
def calculate_pnl_curve(signals: np.ndarray, data: pd.DataFrame, 
                        initial_capital: float = 100000.0) -> pd.Series:
    """Calculate PnL curve from trading signals."""
    position = 0  # 0 = flat, 1 = long
    capital = initial_capital
    shares = 0.0
    equity_values = []
    
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    
    for i in range(len(signals) - 1):
        signal = signals[i]
        next_open = open_prices[i + 1]
        
        if position == 0 and signal:  # Buy signal
            if next_open > 0:
                shares = capital / next_open
                capital = 0.0
                position = 1
        elif position == 1 and not signal:  # Sell signal
            capital = shares * next_open
            shares = 0.0
            position = 0
        
        # Record equity at end of day
        current_equity = capital + shares * close_prices[i]
        equity_values.append(current_equity)
    
    # Final day
    final_equity = capital + shares * close_prices[-1]
    equity_values.append(final_equity)
    
    return pd.Series(equity_values, index=data.index)


# =============================================================================
# Step 4: Calculate Sharpe ratio from PnL curve
# =============================================================================
def calculate_sharpe(pnl_curve: pd.Series) -> float:
    """Calculate annualized Sharpe ratio from PnL curve."""
    returns = pnl_curve.pct_change().dropna()
    
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    return np.sqrt(252) * returns.mean() / returns.std()


# =============================================================================
# Step 5: Analyze single experiment
# =============================================================================
def analyze_experiment(exp_dir_name: str, data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze a single experiment and return train/test metrics."""
    exp_path = BASE_DIR / exp_dir_name
    
    # Step 1: Load best individual (from the globally best generation)
    best_ind, best_gen = load_best_individual(exp_path)
    if best_ind is None:
        return None
    
    # Step 2: Generate signals for full data
    signals = generate_signals(best_ind, data, operators.pset)
    
    # Step 3: Calculate full PnL curve
    full_pnl = calculate_pnl_curve(signals, data)
    
    # Step 4: Split into train/test periods
    train_mask = (data.index >= TRAIN_START) & (data.index <= TRAIN_END)
    test_mask = (data.index >= TEST_START) & (data.index <= TEST_END)
    
    train_pnl = full_pnl[train_mask]
    test_pnl = full_pnl[test_mask]
    
    # Normalize PnL to start from 0,
    if len(train_pnl) > 0:
        train_pnl_normalized = train_pnl - train_pnl.iloc[0]
        train_sharpe = calculate_sharpe(train_pnl)
    else:
        train_pnl_normalized = pd.Series()
        train_sharpe = 0.0
    
    if len(test_pnl) > 0:
        test_pnl_normalized = test_pnl - test_pnl.iloc[0]
        test_sharpe = calculate_sharpe(test_pnl)
    else:
        test_pnl_normalized = pd.Series()
        test_sharpe = 0.0
    
    return {
        'exp_name': exp_dir_name,
        'best_generation': best_gen,
        'train_sharpe': train_sharpe,
        'test_sharpe': test_sharpe,
        'train_pnl': train_pnl_normalized,
        'test_pnl': test_pnl_normalized,
        'full_pnl': full_pnl,
    }


# =============================================================================
# Step 6: Analyze all experiments
# =============================================================================
def analyze_all_experiments(data: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Analyze all experiments and collect results."""
    results = {}
    
    for group_name, exp_dirs in EXPERIMENTS.items():
        print(f"\n分析 {group_name.replace(chr(10), ' ')}...")
        group_results = []
        
        for exp_dir in exp_dirs:
            result = analyze_experiment(exp_dir, data)
            if result:
                print(f"    {exp_dir[-4:]}: Gen={result['best_generation']:02d}, Train={result['train_sharpe']:.2f}, Test={result['test_sharpe']:.2f}")
                group_results.append(result)
            else:
                print(f"    {exp_dir[-4:]}: Failed")
        
        results[group_name] = group_results
        print(f"  → {len(group_results)} runs 完成")
    
    return results


# =============================================================================
# Step 7a: Plot boxplot
# =============================================================================
def plot_boxplot(results: Dict[str, List[Dict]], bh_train: float, bh_test: float):
    """Create boxplot comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        "Rolling Signal\n(soft_min)": "#f39c12",
        "Rolling Baseline\n(soft_min)": "#8e44ad",
    }
    
    labels = list(EXPERIMENTS.keys())
    
    # Train subplot
    ax1 = axes[0]
    train_data = [[r['train_sharpe'] for r in results.get(name, [])] for name in labels]
    bp1 = ax1.boxplot(train_data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, name in zip(bp1['boxes'], labels):
        patch.set_facecolor(colors.get(name, '#cccccc'))
        patch.set_alpha(0.8)
    
    for i, name in enumerate(labels):
        data = [r['train_sharpe'] for r in results.get(name, [])]
        if data:
            mean_val = np.mean(data)
            ax1.scatter(i + 1, mean_val, marker='D', color='white', s=50, zorder=5, edgecolors='black')
            ax1.annotate(f'{mean_val:.2f}', (i + 1, mean_val), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    ax1.axhline(y=bh_train, color='red', linestyle='--', alpha=0.7, label=f'Buy & Hold ({bh_train:.2f})')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Train Period Sharpe Ratio (80%)\n({TRAIN_START} → {TRAIN_END})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Test subplot
    ax2 = axes[1]
    test_data = [[r['test_sharpe'] for r in results.get(name, [])] for name in labels]
    bp2 = ax2.boxplot(test_data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, name in zip(bp2['boxes'], labels):
        patch.set_facecolor(colors.get(name, '#cccccc'))
        patch.set_alpha(0.8)
    
    for i, name in enumerate(labels):
        data = [r['test_sharpe'] for r in results.get(name, [])]
        if data:
            mean_val = np.mean(data)
            ax2.scatter(i + 1, mean_val, marker='D', color='white', s=50, zorder=5, edgecolors='black')
            ax2.annotate(f'{mean_val:.2f}', (i + 1, mean_val), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    ax2.axhline(y=bh_test, color='red', linestyle='--', alpha=0.7, label=f'Buy & Hold ({bh_test:.2f})')
    ax2.legend(loc='upper right')
    ax2.set_title(f'Test Period Sharpe Ratio (20%)\n({TEST_START} → {TEST_END})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('soft_min vs mean 聚合方法比較 (含 TED Niche)\n(8:2 Split, pop=1000)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    output_path = BASE_DIR / "signal_niche_vs_baseline_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Boxplot 已保存至: {output_path}")
    
    plt.show()


# =============================================================================
# Step 7b: Plot PnL curves for Signal Niche vs Baseline
# =============================================================================
def plot_pnl_curves(results: Dict[str, List[Dict]], bh_train_pnl: pd.Series, bh_test_pnl: pd.Series):
    """
    Create PnL curve comparison for Signal Niche vs Baseline.
    """
    # Color palette for runs
    run_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Calculate B&H Sharpe ratios
    bh_train_sharpe = calculate_sharpe(bh_train_pnl + 100000)
    bh_test_sharpe = calculate_sharpe(bh_test_pnl + 100000)
    
    # Figure: Signal Niche (top) vs Baseline (bottom)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    groups = [
        ("Rolling Signal\n(soft_min)", "Signal Niche"),
        ("Rolling Baseline\n(soft_min)", "Baseline"),
    ]
    
    for row_idx, (group_key, method_label) in enumerate(groups):
        group_results = results.get(group_key, [])
        
        for col_idx, (period_label, pnl_key, sharpe_key, bh_pnl, bh_sharpe) in enumerate([
            ('Training / Validation Period', 'train_pnl', 'train_sharpe', bh_train_pnl, bh_train_sharpe),
            ('Test Period (Out-of-Sample)', 'test_pnl', 'test_sharpe', bh_test_pnl, bh_test_sharpe)
        ]):
            ax = axes[row_idx, col_idx]
            
            # Plot Buy & Hold (gray dashed)
            if len(bh_pnl) > 0:
                ax.plot(bh_pnl.index, bh_pnl.values, 
                       color='gray', linestyle='--', linewidth=1.5, alpha=0.8,
                       label=f'Buy-and-Hold (Sharpe: {bh_sharpe:.2f})')
            
            # Plot each run with its own color and Sharpe in legend
            for i, r in enumerate(group_results):
                pnl = r[pnl_key]
                sharpe = r[sharpe_key]
                best_gen = r['best_generation']
                
                if len(pnl) > 0:
                    label = f'Run{i+1} (Sharpe: {sharpe:.2f}, Gen{best_gen})'
                    ax.plot(pnl.index, pnl.values, 
                           color=run_colors[i], linewidth=1.2, alpha=0.85,
                           label=label)
            
            # Title and labels
            if row_idx == 0:
                ax.set_title(period_label, fontsize=12, fontweight='bold')
            
            ax.set_ylabel(f'{method_label} - PnL ($)', fontsize=10)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=0)
    
    # Row labels on the left
    for row_idx, (_, method_label) in enumerate(groups):
        axes[row_idx, 0].annotate(
            method_label.upper(), xy=(-0.15, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', rotation=90, va='center', ha='center'
        )
    
    fig.suptitle('Rolling Window: Signal Niche vs Baseline PnL 曲線比較', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    output_path = BASE_DIR / "signal_niche_pnl_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ signal_niche_pnl_comparison.png 已保存至: {output_path}")
    
    plt.show()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Compare soft_min vs mean aggregation methods')
    parser.add_argument('--plot', type=str, choices=['boxplot', 'pnl', 'both'], default='boxplot',
                       help='Plot type: boxplot, pnl, or both')
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 soft_min vs mean 聚合方法比較")
    print("=" * 60)
    print(f"\nTrain: {TRAIN_START} → {TRAIN_END}")
    print(f"Test:  {TEST_START} → {TEST_END}")
    
    # Load SPY data
    print("\n載入數據...")
    data_dict = load_and_process_data('history', ['SPY'])
    data = data_dict['SPY']
    print(f"SPY 數據: {len(data)} 天")
    
    # Calculate Buy & Hold
    train_mask = (data.index >= TRAIN_START) & (data.index <= TRAIN_END)
    test_mask = (data.index >= TEST_START) & (data.index <= TEST_END)
    
    train_data = data[train_mask]
    test_data = data[test_mask]
    
    # B&H PnL curves (assuming start with 100k and buy on first day)
    bh_train_pnl = (train_data['Close'] / train_data['Close'].iloc[0] * 100000) - 100000
    bh_test_pnl = (test_data['Close'] / test_data['Close'].iloc[0] * 100000) - 100000
    
    bh_train_sharpe = calculate_sharpe(bh_train_pnl + 100000)
    bh_test_sharpe = calculate_sharpe(bh_test_pnl + 100000)
    
    print(f"Buy & Hold Sharpe: Train={bh_train_sharpe:.2f}, Test={bh_test_sharpe:.2f}")
    
    # Analyze all experiments
    results = analyze_all_experiments(data)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 統計摘要")
    print("="*60)
    print(f"\nBuy & Hold: Train={bh_train_sharpe:.2f}, Test={bh_test_sharpe:.2f}")
    
    for name in EXPERIMENTS.keys():
        clean_name = name.replace(chr(10), ' ')
        group_results = results.get(name, [])
        if group_results:
            train_sharpes = [r['train_sharpe'] for r in group_results]
            test_sharpes = [r['test_sharpe'] for r in group_results]
            print(f"\n{clean_name} ({len(group_results)} runs):")
            print(f"  Train: Mean={np.mean(train_sharpes):.2f}, Std={np.std(train_sharpes):.2f}")
            print(f"  Test:  Mean={np.mean(test_sharpes):.2f}, Std={np.std(test_sharpes):.2f}")
    
    # Plot
    if args.plot in ['boxplot', 'both']:
        plot_boxplot(results, bh_train_sharpe, bh_test_sharpe)
    
    if args.plot in ['pnl', 'both']:
        plot_pnl_curves(results, bh_train_pnl, bh_test_pnl)


if __name__ == "__main__":
    main()
