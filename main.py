"""
Main Entry Point for the GP Quant Trading Project

This script orchestrates the entire process of evolving trading rules.
It handles command-line arguments for configuration, loads data, initializes
the evolutionary engine, and reports the final results.

Usage:
    python main.py --ticker <TICKER> [--generations <N>] [--population <N>]

Example:
    python main.py --ticker RY.TO --generations 50 --population 500
"""
import argparse
import os
import dill
import numpy as np
import pandas as pd
from deap import creator, base, gp

from gp_quant.data.loader import load_and_process_data
from gp_quant.evolution.engine import run_evolution
from gp_quant.gp.operators import pset, NumVector
from gp_quant.backtesting.engine import BacktestingEngine

def setup_deap_creator():
    """Initializes DEAP's creator with Fitness and Individual types."""
    # A single objective to maximize (excess return)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # The individual is a GP tree with the defined fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def fix_loaded_individual(individual, price_vec, volume_vec):
    """
    Fix loaded individual by replacing empty terminal arrays with actual data.
    This is needed because terminals are cleared before saving.
    """
    # Convert the individual to string, replace empty arrays, then back to tree
    individual_str = str(individual)
    
    # Replace empty arrays with ARG0 (price) and ARG1 (volume) references
    # This is a simple approach - in practice you might need more sophisticated parsing
    fixed_str = individual_str.replace('[]', 'ARG0')  # Replace first occurrence with price
    
    # For more complex cases, we might need to analyze the context
    # But for now, let's try a different approach - use the BacktestingEngine method
    return individual

def run_evolution_for_tickers(args, data_dir):
    """Runs the evolution process for the specified tickers."""
    print("--- Starting GP Quant Evolution ---")
    print(f"Tickers: {args.tickers}, Generations: {args.generations}, Population: {args.population}")

    all_stock_data = load_and_process_data(data_dir, args.tickers)

    for ticker in args.tickers:
        if ticker not in all_stock_data:
            print(f"\n--- Skipping {ticker}: Data not found or failed to load. ---")
            continue

        stock_data = all_stock_data[ticker]
        print(f"\n--- Running Evolution for {ticker} (Data shape: {stock_data.shape}) ---")

        pop, log, hof = run_evolution(
            data=stock_data,
            n_generations=args.generations,
            population_size=args.population
        )

        best_individual = hof[0]
        print(f"\n--- Results for {ticker} ---")
        print(f"Best Individual Fitness (Excess Return): {best_individual.fitness.values[0]:.2f}")
        print("Best Evolved Trading Rule:")
        print(best_individual)
        
        # Test fitness consistency before saving
        print(f"\n--- Fitness Consistency Test ---")
        print(f"Individual structure: {str(best_individual)}")
        
        # Check if individual contains actual data arrays
        individual_str = str(best_individual)
        has_data_arrays = '[' in individual_str and len(individual_str) > 200
        print(f"Contains data arrays: {has_data_arrays}")
        
        backtester_test = BacktestingEngine(stock_data)
        fitness_before_save = backtester_test.evaluate(best_individual)
        print(f"Fitness before save: {fitness_before_save[0]:.2f}")
        print(f"Original fitness: {best_individual.fitness.values[0]:.2f}")
        print(f"Match: {abs(fitness_before_save[0] - best_individual.fitness.values[0]) < 0.01}")
        
        if not abs(fitness_before_save[0] - best_individual.fitness.values[0]) < 0.01:
            print("WARNING: Fitness inconsistency detected! Individual may contain hardcoded data.")

        # --- Run Detailed Backtest and Save Trade Log ---
        print("\n--- Running Detailed Backtest on Best Individual ---")
        detailed_backtester = BacktestingEngine(stock_data, initial_capital=100000.0)
        backtest_results = detailed_backtester.run_detailed_simulation(best_individual)

        gp_return = backtest_results['gp_return']
        bh_return = backtest_results['buy_and_hold_return']
        print(f"Buy-and-Hold Final Return: ${bh_return:,.2f}")
        print(f"Excess Return (Fitness): ${(gp_return - bh_return):,.2f}")

        # Save trade details to CSV
        csv_save_path = f"trade_details_{ticker}.csv"
        trades_df = pd.DataFrame(backtest_results['trades'])
        if not trades_df.empty:
            trades_df.to_csv(csv_save_path, index=False)
            print(f"Trade details saved to {csv_save_path}")
        else:
            print("No trades were executed.")

        # --- Save Best Individual ---
        # Get the clean string representation BEFORE modifying the pset for dill saving.
        clean_rule_str = str(best_individual)

        # 1. Save the reliable text representation
        save_path_txt = f"best_individual_{ticker}.txt"
        individual_data = {
            'rule_string': clean_rule_str,
            'fitness': best_individual.fitness.values[0],
            'ticker': ticker
        }
        import json
        with open(save_path_txt, "w", encoding='utf-8') as f:
            json.dump(individual_data, f, indent=2)

        # 2. Save the dill object (optional, less robust)
        # To avoid global state pollution, we create a temporary copy for dill saving.
        save_path_dill = f"best_individual_{ticker}.dill"
        
        # Create a deep copy of the pset to modify for serialization
        import copy
        pset_for_dill = copy.deepcopy(pset)
        pset_for_dill.terminals[NumVector][0].value = np.array([])
        pset_for_dill.terminals[NumVector][1].value = np.array([])

        # Temporarily attach the modified pset to a copy of the individual for saving
        individual_for_dill = copy.deepcopy(best_individual)
        individual_for_dill.pset = pset_for_dill

        with open(save_path_dill, "wb") as f:
            # We dump the individual copy, which now has a clean pset reference
            dill.dump(individual_for_dill, f)

        print(f"Saved best individual for {ticker} to {save_path_dill} and {save_path_txt}")
        print("-------------------------------------")

def load_and_show_signals(args, data_dir):
    """Loads a saved rule from a .txt file, rebuilds the individual, and performs a full backtest."""
    ticker = args.load_best
    print(f"--- Loading and Backtesting Individual for {ticker} ---")

    load_path_txt = f"best_individual_{ticker}.txt"
    if not os.path.exists(load_path_txt):
        print(f"Error: Saved rule file not found at {load_path_txt}")
        return

    # Load the rule string from the text file
    import json
    with open(load_path_txt, "r", encoding='utf-8') as f:
        individual_data = json.load(f)
    
    rule_string = individual_data['rule_string']
    saved_fitness = individual_data['fitness']
    print(f"Loaded rule string: {rule_string}")
    print(f"Original Saved Fitness: {saved_fitness:,.2f}")

    # Rebuild the individual from the string representation.
    try:
        best_individual = creator.Individual.from_string(rule_string, pset)
    except TypeError as e:
        print(f"\nError: Could not rebuild individual from string.")
        print(f"DEAP Error: {e}")
        return

    # Load the corresponding data
    stock_data = load_and_process_data(data_dir, [ticker]).get(ticker)
    if stock_data is None:
        print(f"Error: Could not load data for {ticker}")
        return

    # --- Run Detailed Backtest ---
    print("\n--- Running Detailed Backtest on Loaded Individual ---")
    backtester = BacktestingEngine(stock_data, initial_capital=100000.0)
    results = backtester.run_detailed_simulation(best_individual)

    if 'error' in results:
        print(f"Error during backtest: {results['error']}")
        return

    gp_return = results['gp_return']
    bh_return = results['buy_and_hold_return']
    trades = results['trades']

    print(f"GP Strategy Final Return: ${gp_return:,.2f}")
    print(f"Buy-and-Hold Final Return: ${bh_return:,.2f}")
    print(f"Recalculated Excess Return (Fitness): {(gp_return - bh_return):,.2f}")
    print(f"Number of trades: {len(trades)}")

    # --- Show Signal Changes ---
    signals = backtester.get_signals(best_individual)
    if len(signals) > 0:
        print("\n--- Generated Trading Signal Changes ---")
        dates = stock_data.index
        signal_changes = 0
        for i in range(1, len(signals)):
            if signals[i] != signals[i-1]:
                action = "BUY" if signals[i] else "SELL"
                print(f"{dates[i].date()}: {action}")
                signal_changes += 1
        if signal_changes == 0:
            print("No signal changes detected.")
    else:
        print("Could not generate signals.")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="GP Quant Trading Rule Evolution")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", type=str, nargs='+', help="One or more stock tickers to run evolution for.")
    group.add_argument("--load_best", type=str, help="Load a saved individual and show its signals.")
    
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run.")
    parser.add_argument("--population", type=int, default=500, help="Population size.")
    args = parser.parse_args()

    # --- 1. Setup Environment ---
    setup_deap_creator()
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, 'TSE300_selected')

    if args.tickers:
        run_evolution_for_tickers(args, data_dir)
    elif args.load_best:
        load_and_show_signals(args, data_dir)

if __name__ == "__main__":
    main()
