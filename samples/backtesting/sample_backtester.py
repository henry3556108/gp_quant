"""
Sample script to demonstrate the backtesting engine with a simple strategy.

This script performs the following actions:
1. Loads historical stock data.
2. Defines a simple trading rule (e.g., a moving average crossover) using a GP tree.
3. Initializes and runs the BacktestingEngine with this rule.
4. Visualizes the backtest results, including the equity curve and trade markers.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from deap import gp, creator, base
from gp_quant.data.loader import load_and_process_data
from gp_quant.backtesting.engine import BacktestingEngine
from gp_quant.gp.operators import pset
from gp_quant.utils.visualization import plot_backtest_results

def main():
    """Main function to run the backtesting demonstration."""
    print("--- Running Backtesting Engine Sample ---")

    # --- 1. Load Data ---
    data_dir = os.path.join(project_root, 'TSE300_selected')
    ticker = 'RY.TO'
    all_data = load_and_process_data(data_dir, [ticker])
    if ticker not in all_data:
        print(f"Could not load data for {ticker}. Exiting.")
        return
    stock_data = all_data[ticker]

    # --- 2. Define a Trading Rule ---
    # We'll create a simple moving average crossover rule:
    # "Buy if the 10-day MA is greater than the 30-day MA"
    # DEAP string representation: gt(avg(PRICE, 10), avg(PRICE, 30))
    rule_str = "gt(avg(SeriesType.PRICE, rand_int_n), avg(SeriesType.PRICE, rand_int_n))"
    # For a deterministic test, let's manually create the tree
    rule_tree = gp.PrimitiveTree.from_string("gt(avg(SeriesType.PRICE, 10), avg(SeriesType.PRICE, 30))", pset)

    print(f"\nBacktesting with rule: {rule_tree}")

    # DEAP creator setup
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    individual = creator.Individual(rule_tree)

    # --- 3. Run Backtest ---
    engine = BacktestingEngine(stock_data, initial_capital=100000.0)
    results = engine.run_simulation(individual)

    print("\nBacktest Finished.")
    print(f"Final Portfolio Value: ${results['equity_curve'].iloc[-1]:,.2f}")
    print(f"Total P&L: ${results['pnl']:,.2f}")
    print(f"Number of trades: {len(results['trades'])}")

    # --- 4. Visualize Results ---
    output_dir = os.path.join(project_root, 'samples', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'backtest_results.png')

    plot_backtest_results(stock_data, results['equity_curve'], results['trades'], ticker, save_path)

    print("\n--- Sample script finished. ---")

if __name__ == "__main__":
    main()
