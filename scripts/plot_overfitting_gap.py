#!/usr/bin/env python3
"""
Visualize Train-Test Sharpe Gap using boxplots.
Each experiment group shows the distribution of (Test - Train) gaps.
可刪除腳本 - 用完即刪
"""

import json
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
from deap import creator, base, gp

# Setup DEAP
if not hasattr(creator, 'FitnessMax'):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gp_quant.evolution.components.gp import operators
from gp_quant.data.loader import load_and_process_data

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = Path("/Volumes/X9 Pro/gp_quant_experiments")

# 實驗配置（按順序排列）
EXPERIMENTS = {
    "Baseline\n(single)": [
        "baseline_large_scale_20251209_2032",
        "baseline_large_scale_20251209_2037",
        "baseline_large_scale_20251209_2042",
        "baseline_large_scale_20251209_2047",
        "baseline_large_scale_20251209_2053",
        "baseline_large_scale_20251209_2058",
        "baseline_large_scale_20251209_2103",
        "baseline_large_scale_20251209_2108",
        "baseline_large_scale_20251209_2113",
        "baseline_large_scale_20251209_2118",
    ],
    "Rolling\n(mean)": [
        "rolling_window_large_scale_20251209_0830",
        "rolling_window_large_scale_20251209_0854",
        "rolling_window_large_scale_20251209_0921",
        "rolling_window_large_scale_20251209_0945",
        "rolling_window_large_scale_20251209_1011",
        "rolling_window_large_scale_20251209_1035",
        "rolling_window_large_scale_20251209_1059",
        "rolling_window_large_scale_20251209_1125",
        "rolling_window_large_scale_20251209_1150",
        "rolling_window_large_scale_20251209_1215",
    ],
    "Rolling\n(soft_min)": [
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
    "RSFGP\n(mean)": [
        "rsfgp_large_scale_20251209_0537",
        "rsfgp_large_scale_20251209_0554",
        "rsfgp_large_scale_20251209_0611",
        "rsfgp_large_scale_20251209_0629",
        "rsfgp_large_scale_20251209_0646",
        "rsfgp_large_scale_20251209_0703",
        "rsfgp_large_scale_20251209_0720",
        "rsfgp_large_scale_20251209_0737",
        "rsfgp_large_scale_20251209_0756",
        "rsfgp_large_scale_20251209_0813",
    ],
    "RSFGP\n(soft_min)": [
        "rsfgp_large_scale_20251209_1453",
        "rsfgp_large_scale_20251209_1504",
        "rsfgp_large_scale_20251209_1513",
        "rsfgp_large_scale_20251209_1523",
        "rsfgp_large_scale_20251209_1533",
        "rsfgp_large_scale_20251209_1542",
        "rsfgp_large_scale_20251209_1552",
        "rsfgp_large_scale_20251209_1602",
        "rsfgp_large_scale_20251209_1612",
        "rsfgp_large_scale_20251209_1621",
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
        "rolling_window_signal_niche_20251210_1713",
    ],
}

TRAIN_START = "2006-06-25"
TRAIN_END = "2021-06-30"
TEST_START = "2021-07-01"
TEST_END = "2024-12-31"


def load_best_individual(exp_dir: Path):
    """Load the best individual from best generation."""
    pop_dir = exp_dir / 'populations'
    stats_file = exp_dir / 'generation_stats.json'
    
    if not pop_dir.exists() or not stats_file.exists():
        return None, None
    
    try:
        with open(stats_file, 'r') as f:
            gen_stats = json.load(f)
        best_gen = max(gen_stats, key=lambda g: g.get('best_fitness', -float('inf')))
        best_gen_num = best_gen['generation']
        
        pkl_file = pop_dir / f'generation_{best_gen_num:03d}.pkl'
        with open(pkl_file, 'rb') as f:
            population = pickle.load(f)
        
        best = max(population, key=lambda x: x.fitness.values[0] if x.fitness.values else -float('inf'))
        return best, best_gen_num
    except:
        return None, None


def generate_signals(individual, data, pset):
    """Generate trading signals."""
    import copy
    from gp_quant.evolution.components.gp.operators import NumVector
    
    local_pset = copy.deepcopy(pset)
    price_vec = data['Close'].values.astype(float).view(NumVector)
    volume_vec = data['Volume'].values.astype(float).view(NumVector)
    
    local_pset.terminals[NumVector][0].value = price_vec
    local_pset.terminals[NumVector][1].value = volume_vec
    
    try:
        func = gp.compile(expr=str(individual), pset=local_pset)
        try:
            signals = func()
        except TypeError:
            signals = func(price_vec, volume_vec)
        
        if isinstance(signals, (bool, np.bool_)):
            signals = np.full(len(data), signals, dtype=bool)
        elif isinstance(signals, np.ndarray):
            if signals.ndim == 0 or signals.size == 1:
                signals = np.full(len(data), bool(signals.item() if signals.ndim == 0 else signals[0]), dtype=bool)
            else:
                signals = signals.astype(bool)
        else:
            signals = np.full(len(data), bool(signals), dtype=bool)
        return signals
    except:
        return np.zeros(len(data), dtype=bool)


def calculate_pnl_curve(signals, data, initial_capital=100000.0):
    """Calculate PnL curve."""
    position = 0
    capital = initial_capital
    shares = 0.0
    equity_values = []
    
    open_prices = data['Open'].values
    close_prices = data['Close'].values
    
    for i in range(len(signals) - 1):
        if position == 0 and signals[i] and open_prices[i + 1] > 0:
            shares = capital / open_prices[i + 1]
            capital = 0.0
            position = 1
        elif position == 1 and not signals[i]:
            capital = shares * open_prices[i + 1]
            shares = 0.0
            position = 0
        equity_values.append(capital + shares * close_prices[i])
    
    equity_values.append(capital + shares * close_prices[-1])
    return pd.Series(equity_values, index=data.index)


def calculate_sharpe(pnl_curve):
    """Calculate Sharpe ratio."""
    returns = pnl_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return np.sqrt(252) * returns.mean() / returns.std()


def main():
    print("載入數據...")
    data_dict = load_and_process_data('history', ['SPY'])
    data = data_dict['SPY']
    
    train_mask = (data.index >= TRAIN_START) & (data.index <= TRAIN_END)
    test_mask = (data.index >= TEST_START) & (data.index <= TEST_END)
    
    # Collect gaps for each group
    all_gaps = {}
    all_trains = {}
    all_tests = {}
    
    for group_name, exp_dirs in EXPERIMENTS.items():
        print(f"\n分析 {group_name.replace(chr(10), ' ')}...")
        gaps = []
        trains = []
        tests = []
        
        for exp_dir_name in exp_dirs:
            exp_path = BASE_DIR / exp_dir_name
            ind, gen = load_best_individual(exp_path)
            if ind is None:
                continue
            
            signals = generate_signals(ind, data, operators.pset)
            full_pnl = calculate_pnl_curve(signals, data)
            
            train_sharpe = calculate_sharpe(full_pnl[train_mask])
            test_sharpe = calculate_sharpe(full_pnl[test_mask])
            gap = test_sharpe - train_sharpe  # Test - Train (positive = no overfitting)
            
            gaps.append(gap)
            trains.append(train_sharpe)
            tests.append(test_sharpe)
            print(f"    {exp_dir_name[-4:]}: Train={train_sharpe:.2f}, Test={test_sharpe:.2f}, Gap={gap:+.2f}")
        
        all_gaps[group_name] = gaps
        all_trains[group_name] = trains
        all_tests[group_name] = tests
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    labels = list(EXPERIMENTS.keys())
    colors = {
        "Baseline\n(single)": "#e74c3c",
        "Rolling\n(mean)": "#d7bde2",
        "Rolling\n(soft_min)": "#8e44ad",
        "RSFGP\n(mean)": "#85c1e9",
        "RSFGP\n(soft_min)": "#2980b9",
        "Rolling Signal\n(soft_min)": "#f39c12",
    }
    
    # Subplot 1: Train vs Test Bar Chart (means)
    ax1 = axes[0]
    x = np.arange(len(labels))
    width = 0.35
    
    train_means = [np.mean(all_trains.get(name, [0])) for name in labels]
    test_means = [np.mean(all_tests.get(name, [0])) for name in labels]
    
    bars1 = ax1.bar(x - width/2, train_means, width, label='Train Sharpe', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_means, width, label='Test Sharpe', color='#e74c3c', alpha=0.8)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Train vs Test Sharpe Ratio', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.70, color='gray', linestyle='--', alpha=0.5, label='B&H (0.70)')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Gap boxplot (Test - Train)
    ax2 = axes[1]
    gap_data = [all_gaps.get(name, []) for name in labels]
    
    bp3 = ax2.boxplot(gap_data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, name in zip(bp3['boxes'], labels):
        patch.set_facecolor(colors.get(name, '#cccccc'))
        patch.set_alpha(0.8)
    
    # Add mean diamonds
    for i, name in enumerate(labels):
        vals = all_gaps.get(name, [])
        if vals:
            mean_val = np.mean(vals)
            ax2.scatter(i + 1, mean_val, marker='D', color='white', s=50, zorder=5, edgecolors='black')
            ax2.annotate(f'{mean_val:+.2f}', (i + 1, mean_val), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Test - Train Gap', fontsize=11)
    ax2.set_title('Overfitting 程度 (Gap < 0 = Overfitting)', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='--')
    ax2.axhspan(-0.5, 0, alpha=0.1, color='red')
    ax2.axhspan(0, 0.5, alpha=0.1, color='green')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = BASE_DIR / "overfitting_analysis_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 已保存至: {output_path}")
    
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("📊 Overfitting 分析摘要 (Test - Train Gap)")
    print("="*70)
    for name in labels:
        gaps = all_gaps.get(name, [])
        if gaps:
            mean_gap = np.mean(gaps)
            status = "✅ 無" if mean_gap >= 0 else ("⚠️ 中度" if mean_gap > -0.15 else "❌ 嚴重")
            print(f"{name.replace(chr(10), ' '):<25} Gap: {mean_gap:+.2f} ± {np.std(gaps):.2f}  {status}")


if __name__ == "__main__":
    main()
