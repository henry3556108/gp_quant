import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from deap import gp
from gp_quant.evolution.components.gp.operators import pset
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data
from plot_all_experiments_pnl import calculate_metrics, calculate_buy_and_hold

def load_final_population(exp_dir: str) -> List:
    """Load the final generation population from an experiment directory."""
    exp_path = Path(exp_dir)
    
    # Try loading final result first to get generation number
    final_result_path = exp_path / 'final_result.json'
    if final_result_path.exists():
        with open(final_result_path, 'r') as f:
            final_result = json.load(f)
        final_gen = final_result.get('final_generation', final_result.get('generation', 0))
    else:
        # Fallback: find latest generation file
        pop_files = list((exp_path / 'populations').glob('generation_*.pkl'))
        if not pop_files:
            raise FileNotFoundError(f"No population files found in {exp_path}")
        
        # Extract generation numbers
        gens = []
        for f in pop_files:
            try:
                gen_num = int(f.stem.split('_')[1])
                gens.append(gen_num)
            except:
                pass
        final_gen = max(gens)
        print(f"  ⚠️ final_result.json not found. Using latest generation: {final_gen}")

    pop_file = exp_path / 'populations' / f'generation_{final_gen:03d}.pkl'
    print(f"  Loading population from {pop_file}")
    
    with open(pop_file, 'rb') as f:
        population = pickle.load(f)
        
    return population

def evaluate_population_robustness(population, data, config, train_fitness_threshold=0.5):
    """
    Evaluate population on test data, filtering for robust individuals.
    Robust = High Train Fitness AND Positive Test PnL.
    """
    
    # Prepare data for engine
    processed_data = {}
    for ticker, ticker_data in data.items():
        if isinstance(ticker_data, dict) and 'data' in ticker_data:
            processed_data[ticker] = ticker_data['data']
        else:
            processed_data[ticker] = ticker_data
            
    start_date = config['data']['test_backtest_start']
    end_date = config['data']['test_backtest_end']
    
    engine = PortfolioBacktestingEngine(
        data=processed_data,
        backtest_start=start_date,
        backtest_end=end_date,
        initial_capital=100000.0,
        pset=pset
    )
    
    robust_candidates = []
    
    # Filter by Train Fitness first
    candidates = [ind for ind in population if ind.fitness.values and ind.fitness.values[0] > train_fitness_threshold]
    print(f"  Found {len(candidates)} candidates with Train Fitness > {train_fitness_threshold}")
    
    if not candidates:
        print("  ⚠️ No candidates met the training fitness threshold. Trying top 10%...")
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.values else -np.inf, reverse=True)
        candidates = sorted_pop[:int(len(population)*0.1)]
        print(f"  Selected top {len(candidates)} individuals instead.")

    print(f"  Evaluating {len(candidates)} candidates on TEST data...")
    for i, ind in enumerate(tqdm(candidates)):
        try:
            # Backtest
            backtest_res = engine.backtest(ind)
            
            # Calculate metrics
            equity_curve = backtest_res['equity_curve']
            final_value = equity_curve.iloc[-1]
            pnl = final_value - 100000.0
            sharpe, _ = calculate_metrics(equity_curve)
            
            robust_candidates.append({
                'individual': ind,
                'test_pnl': pnl,
                'test_sharpe': sharpe,
                'train_fitness': ind.fitness.values[0],
                'equity_curve': equity_curve
            })
        except Exception as e:
            pass
            
    return robust_candidates

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_dir = Path(f"analysis/robust_best_{timestamp}")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        {
            "name": "PnL Niche Elitist (Fixed)",
            "path": "experiment_result/pnl_niche_elitist_records_20251201_1924",
            "config": "experiment_result/pnl_niche_elitist_records_20251201_1924/config.json",
            "threshold": 0.5
        },
        {
            "name": "Baseline Niche Elitist (New)",
            "path": "experiment_result/baseline_niche_elitist_records_20251201_2037",
            "config": "experiment_result/baseline_niche_elitist_records_20251201_2037/config.json",
            "threshold": 0.8 # Higher threshold for the overfitting experiment
        }
    ]
    
    summary_data = []
    
    for exp in experiments:
        print(f"\n🚀 Processing {exp['name']}...")
        
        # Load Config
        with open(exp['config'], 'r') as f:
            config = json.load(f)
            
        # Override dates
        config['data']['train_data_start'] = '2008-01-01'
        config['data']['train_backtest_start'] = '2008-06-25'
        config['data']['train_backtest_end'] = '2019-12-31'
        config['data']['test_data_start'] = '2020-01-01'
        config['data']['test_backtest_start'] = '2020-06-30'
        config['data']['test_backtest_end'] = '2024-12-30'
        
        # Load Data
        tickers_dir = Path(config['data']['tickers_dir'])
        csv_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        raw_data = load_and_process_data(str(tickers_dir), tickers)
        
        _, test_data = split_train_test_data(
            raw_data,
            train_data_start=config['data']['train_data_start'],
            train_backtest_start=config['data']['train_backtest_start'],
            train_backtest_end=config['data']['train_backtest_end'],
            test_data_start=config['data']['test_data_start'],
            test_backtest_start=config['data']['test_backtest_start'],
            test_backtest_end=config['data']['test_backtest_end']
        )
        
        # Load Population
        population = load_final_population(exp['path'])
        
        # Evaluate Robustness
        robust_candidates = evaluate_population_robustness(population, test_data, config, train_fitness_threshold=exp['threshold'])
        
        if not robust_candidates:
            print("  ❌ No robust candidates found.")
            continue
            
        # Sort by Test PnL
        robust_candidates.sort(key=lambda x: x['test_pnl'], reverse=True)
        
        # Take Top 1
        best_robust = robust_candidates[0]
        
        print(f"  🏆 Best Robust Individual Found!")
        print(f"     Test PnL: ${best_robust['test_pnl']:,.2f}")
        print(f"     Test Sharpe: {best_robust['test_sharpe']:.4f}")
        print(f"     Train Fitness: {best_robust['train_fitness']:.4f}")
        
        # Calculate Buy and Hold
        test_bh = calculate_buy_and_hold(test_data, config, 'test')
        test_bh_final = test_bh.iloc[-1]
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot B&H
        plt.plot(test_bh.index, test_bh.values, label=f'Buy & Hold (PnL: ${test_bh_final:,.0f})', color='gray', linestyle='--')
        
        # Plot Best Robust Ind
        pnl_curve = best_robust['equity_curve'] - 100000.0
        plt.plot(pnl_curve.index, pnl_curve.values, label=f'Robust Ind (PnL: ${best_robust["test_pnl"]:,.0f})', color='green', linewidth=2)
        
        plt.title(f"{exp['name']} - Best Robust Individual (Train Fit > {exp['threshold']})")
        plt.xlabel('Date')
        plt.ylabel('PnL ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = analysis_dir / f"robust_pnl_{exp['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(plot_path)
        print(f"  📊 Plot saved to {plot_path}")
        plt.close()
        
        summary_data.append({
            'Experiment': exp['name'],
            'Best Robust Test PnL': f"${best_robust['test_pnl']:,.0f}",
            'Test Sharpe': f"{best_robust['test_sharpe']:.4f}",
            'Train Fitness': f"{best_robust['train_fitness']:.4f}",
            'B&H PnL': f"${test_bh_final:,.0f}",
            'Expression': str(best_robust['individual'])
        })

    # Save Summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = analysis_dir / f"robust_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Summary saved to {csv_path}")
        print(df[['Experiment', 'Best Robust Test PnL', 'Test Sharpe', 'Train Fitness', 'B&H PnL']].to_string(index=False))

if __name__ == "__main__":
    main()
