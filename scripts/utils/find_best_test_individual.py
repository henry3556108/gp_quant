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

def evaluate_population_on_test(population, data, config):
    """Evaluate entire population on test data and return metrics."""
    
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
    
    results = []
    
    print(f"  Evaluating {len(population)} individuals on TEST data...")
    for i, ind in enumerate(tqdm(population)):
        try:
            # Backtest
            backtest_res = engine.backtest(ind)
            
            # Calculate metrics
            equity_curve = backtest_res['equity_curve']
            final_value = equity_curve.iloc[-1]
            pnl = final_value - 100000.0
            sharpe, _ = calculate_metrics(equity_curve)
            
            results.append({
                'individual': ind,
                'pnl': pnl,
                'sharpe': sharpe,
                'equity_curve': equity_curve,
                'train_fitness': ind.fitness.values[0] if ind.fitness.values else -np.inf
            })
        except Exception as e:
            # print(f"Error evaluating individual {i}: {e}")
            pass
            
    return results

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_dir = Path(f"analysis/test_best_{timestamp}")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        {
            "name": "PnL Niche Elitist (Fixed)",
            "path": "experiment_result/pnl_niche_elitist_records_20251201_1924",
            "config": "experiment_result/pnl_niche_elitist_records_20251201_1924/config.json"
        },
        {
            "name": "Baseline Niche Elitist (New)",
            "path": "experiment_result/baseline_niche_elitist_records_20251201_2037",
            "config": "experiment_result/baseline_niche_elitist_records_20251201_2037/config.json"
        }
    ]
    
    summary_data = []
    
    for exp in experiments:
        print(f"\n🚀 Processing {exp['name']}...")
        
        # Load Config
        with open(exp['config'], 'r') as f:
            config = json.load(f)
            
        # Override dates if needed (using same logic as perform_analysis.py)
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
        
        # Evaluate on Test
        results = evaluate_population_on_test(population, test_data, config)
        
        if not results:
            print("  ❌ No valid results found.")
            continue
            
        # Find Best by Test PnL
        best_test_ind = max(results, key=lambda x: x['pnl'])
        
        print(f"  🏆 Best Test Individual Found!")
        print(f"     Test PnL: ${best_test_ind['pnl']:,.2f}")
        print(f"     Test Sharpe: {best_test_ind['sharpe']:.4f}")
        print(f"     Train Fitness: {best_test_ind['train_fitness']:.4f}")
        print(f"     Expression: {best_test_ind['individual']}")
        
        # Calculate Buy and Hold
        test_bh = calculate_buy_and_hold(test_data, config, 'test')
        test_bh_final = test_bh.iloc[-1]
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Plot B&H
        plt.plot(test_bh.index, test_bh.values, label=f'Buy & Hold (PnL: ${test_bh_final:,.0f})', color='gray', linestyle='--')
        
        # Plot Best Test Ind
        pnl_curve = best_test_ind['equity_curve'] - 100000.0
        plt.plot(pnl_curve.index, pnl_curve.values, label=f'Best Test Ind (PnL: ${best_test_ind["pnl"]:,.0f})', color='blue', linewidth=2)
        
        plt.title(f"{exp['name']} - Best Individual on Test Data")
        plt.xlabel('Date')
        plt.ylabel('PnL ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = analysis_dir / f"best_test_pnl_{exp['name'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(plot_path)
        print(f"  📊 Plot saved to {plot_path}")
        plt.close()
        
        summary_data.append({
            'Experiment': exp['name'],
            'Best Test PnL': f"${best_test_ind['pnl']:,.0f}",
            'Test Sharpe': f"{best_test_ind['sharpe']:.4f}",
            'Train Fitness': f"{best_test_ind['train_fitness']:.4f}",
            'B&H PnL': f"${test_bh_final:,.0f}",
            'Expression': str(best_test_ind['individual'])
        })

    # Save Summary CSV
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_path = analysis_dir / f"best_test_summary_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Summary saved to {csv_path}")
        print(df[['Experiment', 'Best Test PnL', 'Test Sharpe', 'Train Fitness', 'B&H PnL']].to_string(index=False))

if __name__ == "__main__":
    main()
