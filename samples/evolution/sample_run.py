"""
Sample script to demonstrate a full, end-to-end evolutionary run.

This script performs the following actions:
1. Loads historical stock data for a specified ticker.
2. Sets up the DEAP environment.
3. Runs a small-scale evolution for a few generations.
4. Takes the best individual (trading rule) from the Hall of Fame.
5. Runs this best rule through the backtesting engine to get detailed results.
6. Visualizes the performance of the best rule and saves the plot.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from deap import creator, base, gp

from gp_quant.data.loader import load_and_process_data
from gp_quant.evolution.engine import run_evolution
from gp_quant.backtesting.engine import BacktestingEngine
from gp_quant.gp.operators import pset

def setup_deap_creator():
    """Initializes DEAP's creator with Fitness and Individual types."""
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    # The pset is associated with the individual in the toolbox, not here.

def main():
    """Main function to run the sample evolution."""
    print("--- Running Full Evolution and Backtest Sample ---")

    # --- 1. Configuration ---
    ticker = 'RY.TO'
    population_size = 50  # Smaller population for a quick sample
    n_generations = 5     # Fewer generations for a quick sample

    # --- 2. Setup and Data Loading ---
    setup_deap_creator()
    data_dir = os.path.join(project_root, 'TSE300_selected')
    stock_data = load_and_process_data(data_dir, [ticker])[ticker]

    print(f"\nRunning evolution for {ticker} with population={population_size}, generations={n_generations}...")

    # --- 3. Run Evolution ---
    pop, log, hof = run_evolution(
        data=stock_data,
        population_size=population_size,
        n_generations=n_generations
    )

    # --- 4. Analyze and Backtest the Best Rule ---
    best_individual = hof[0]
    print("\n--- Evolution Finished ---")
    print(f"Best Individual Fitness (Excess Return): {best_individual.fitness.values[0]:.2f}")
    print("\nBest Evolved Trading Rule:")
    print(best_individual)

    print("\n--- Running Detailed Backtest on Best Evolved Rule ---")
    backtester = BacktestingEngine(stock_data)
    results = backtester.run_detailed_simulation(best_individual)

    gp_return = results.get('gp_return', 0)
    bh_return = results.get('buy_and_hold_return', 0)
    trades = results.get('trades', [])

    print("\nBacktest Finished.")
    print(f"GP Strategy Final Return: ${gp_return:,.2f}")
    print(f"Buy-and-Hold Final Return: ${bh_return:,.2f}")
    print(f"Excess Return: ${(gp_return - bh_return):,.2f}")
    print(f"Number of trades: {len(trades)}")

    if trades:
        import pandas as pd
        trades_df = pd.DataFrame(trades)
        print("\n--- Trade Log --- ")
        print(trades_df.head())

    print("\n--- Sample script finished. ---")

if __name__ == "__main__":
    main()
