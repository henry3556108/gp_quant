#!/usr/bin/env python3
"""
Plot Overlaid PnL Curves

Plots the PnL curves of multiple experiments on the same axes for direct comparison.
"""

import json
import pickle
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data
from plot_all_experiments_pnl import load_best_individual, backtest_individual, calculate_buy_and_hold

def plot_overlay(experiments_data, output_path):
    """Plot overlaid PnL curves with Train/Test subplots"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    colors = ['#2E86AB', '#F18F01', '#C73E1D', '#6A994E', '#A23B72']
    periods = ['train', 'test']
    titles = ['In-Sample (Train) Period', 'Out-of-Sample (Test) Period']
    
    for idx, period in enumerate(periods):
        ax = axes[idx]
        
        # Plot Buy and Hold (use the first experiment's B&H)
        if experiments_data:
            bh_key = f'{period}_bh'
            if bh_key in experiments_data[0] and not experiments_data[0][bh_key].empty:
                bh_pnl = experiments_data[0][bh_key]
                ax.plot(bh_pnl.index, bh_pnl.values, label='Buy & Hold', 
                         color='gray', linestyle='--', linewidth=2, alpha=0.7)
        
        # Plot each experiment
        for i, exp in enumerate(experiments_data):
            result_key = f'{period}_result'
            if result_key in exp and 'equity_curve' in exp[result_key] and not exp[result_key]['equity_curve'].empty:
                pnl = exp[result_key]['equity_curve'] - 100000.0
                fitness = exp[f'{period}_fitness']
                color = colors[i % len(colors)]
                
                ax.plot(pnl.index, pnl.values, label=f"{exp['name']} (Fit: {fitness:.4f})", 
                         color=color, linewidth=2.5, alpha=0.9)
                
                # Add final value annotation
                final_val = pnl.iloc[-1]
                ax.annotate(f"${final_val:,.0f}", 
                             xy=(pnl.index[-1], final_val),
                             xytext=(10, 0), textcoords='offset points',
                             color=color, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Data Available', 
                        transform=ax.transAxes, ha='center', va='center', color='gray')

        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Overlaid plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot overlaid PnL curves")
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    # Define experiments (Hardcoded for this specific task)
    experiments = [
        {
            "name": "Baseline (No Niche)",
            "path": "experiment_result/baseline_no_niche_records_20251129_1651",
            "config": "experiment_result/baseline_no_niche_records_20251129_1651/config.json"
        },
        {
            "name": "PnL Niche",
            "path": "experiment_result/pnl_niche_test_records_20251129_1701",
            "config": "experiment_result/pnl_niche_test_records_20251129_1701/config.json"
        }
    ]
    
    experiments_data = []
    
    for exp in experiments:
        print(f"Processing {exp['name']}...")
        with open(exp['config'], 'r') as f:
            config = json.load(f)
            
        best_ind, train_fitness = load_best_individual(exp['path'])
        
        tickers_dir = Path(config['data']['tickers_dir'])
        csv_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        raw_data = load_and_process_data(str(tickers_dir), tickers)
        
        train_data, test_data = split_train_test_data(
            raw_data,
            train_data_start=config['data']['train_data_start'],
            train_backtest_start=config['data']['train_backtest_start'],
            train_backtest_end=config['data']['train_backtest_end'],
            test_data_start=config['data']['test_data_start'],
            test_backtest_start=config['data']['test_backtest_start'],
            test_backtest_end=config['data']['test_backtest_end']
        )
        
        # Train
        train_result = backtest_individual(best_ind, train_data, config, 'train')
        train_bh = calculate_buy_and_hold(train_data, config, 'train')
        
        # Test
        try:
            test_result = backtest_individual(best_ind, test_data, config, 'test')
            test_bh = calculate_buy_and_hold(test_data, config, 'test')
            test_fitness = test_result['metrics']['excess_return']
        except Exception as e:
            print(f"  ⚠️ Test period backtest failed: {e}")
            test_result = {}
            test_bh = pd.Series(dtype=float)
            test_fitness = 0.0
        
        experiments_data.append({
            'name': exp['name'],
            'train_result': train_result,
            'train_bh': train_bh,
            'train_fitness': train_fitness,
            'test_result': test_result,
            'test_bh': test_bh,
            'test_fitness': test_fitness
        })
        
    plot_overlay(experiments_data, args.output)

if __name__ == "__main__":
    main()
