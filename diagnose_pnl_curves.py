"""
Diagnostic script to investigate PnL curves and correlation issues
"""

import pandas as pd
import numpy as np
import dill
import matplotlib.pyplot as plt
from gp_quant.backtesting.engine import BacktestingEngine
from deap import creator, base, gp

# Setup DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Configuration
RECORDS_DIR = "experiments_results/ABX_TO/individual_records_long_run01"
TICKER = "ABX.TO"
DATA_FILE = f"TSE300_selected/{TICKER}.csv"
BACKTEST_START = '1993-07-02'
BACKTEST_END = '1999-06-25'

# Load market data
print("Loading market data...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
print(f"Data range: {data.index[0].date()} to {data.index[-1].date()}")
print()

# Initialize backtesting engine
engine = BacktestingEngine(
    data=data,
    backtest_start=BACKTEST_START,
    backtest_end=BACKTEST_END
)

def analyze_generation(gen_num, n_individuals=5):
    """Analyze PnL curves for a specific generation"""
    print(f"\n{'='*80}")
    print(f"Generation {gen_num}")
    print(f"{'='*80}")
    
    # Load population
    pop_file = f"{RECORDS_DIR}/generation_{gen_num:03d}/population.pkl"
    with open(pop_file, 'rb') as f:
        population = dill.load(f)
    
    print(f"Population size: {len(population)}")
    
    # Sample individuals
    import random
    sampled_inds = random.sample(population, min(n_individuals, len(population)))
    
    pnl_curves = []
    valid_count = 0
    
    for idx, ind in enumerate(sampled_inds):
        print(f"\nIndividual {idx+1}:")
        print(f"  Tree: {str(ind)[:100]}...")
        print(f"  Height: {ind.height}, Length: {len(ind)}")
        print(f"  Fitness: {ind.fitness.values[0] if ind.fitness.valid else 'Invalid'}")
        
        try:
            pnl_curve = engine.get_pnl_curve(ind)
            
            if len(pnl_curve) > 0:
                print(f"  PnL curve length: {len(pnl_curve)}")
                print(f"  PnL range: [{pnl_curve.min():.2f}, {pnl_curve.max():.2f}]")
                print(f"  PnL final: {pnl_curve.iloc[-1]:.2f}")
                print(f"  PnL std: {pnl_curve.std():.2f}")
                print(f"  PnL mean: {pnl_curve.mean():.2f}")
                
                # Check if PnL has variance
                if pnl_curve.std() > 0:
                    pnl_curves.append(pnl_curve)
                    valid_count += 1
                    print(f"  ✓ Valid PnL curve")
                else:
                    print(f"  ✗ No variance (constant PnL)")
            else:
                print(f"  ✗ Empty PnL curve")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nValid PnL curves: {valid_count} / {n_individuals}")
    
    # Calculate correlations if we have at least 2 valid curves
    if len(pnl_curves) >= 2:
        print(f"\nCalculating correlations between {len(pnl_curves)} curves...")
        
        # Convert to matrix
        pnl_matrix = np.array([curve.values for curve in pnl_curves])
        corr_matrix = np.corrcoef(pnl_matrix)
        
        # Extract upper triangle
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        upper_tri = upper_tri[~np.isnan(upper_tri)]
        
        if len(upper_tri) > 0:
            print(f"Correlation statistics:")
            print(f"  Mean: {np.mean(upper_tri):.4f}")
            print(f"  Std: {np.std(upper_tri):.4f}")
            print(f"  Min: {np.min(upper_tri):.4f}")
            print(f"  Max: {np.max(upper_tri):.4f}")
            print(f"  Median: {np.median(upper_tri):.4f}")
            print(f"\nAll correlations: {upper_tri}")
        else:
            print("No valid correlations calculated")
        
        # Plot PnL curves
        fig, ax = plt.subplots(figsize=(12, 6))
        for idx, pnl_curve in enumerate(pnl_curves):
            ax.plot(pnl_curve.index, pnl_curve.values, alpha=0.7, label=f'Individual {idx+1}')
        
        ax.set_title(f'PnL Curves - Generation {gen_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative PnL ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'pnl_curves_gen{gen_num}.png', dpi=150)
        print(f"\n✓ PnL curves plot saved to: pnl_curves_gen{gen_num}.png")
        plt.close()
    else:
        print("\nNot enough valid curves to calculate correlations")
    
    return valid_count, len(pnl_curves)

# Analyze generations with mean = 0
print("\n" + "="*80)
print("ANALYZING GENERATIONS WITH MEAN CORRELATION = 0")
print("="*80)

zero_corr_gens = [4, 6, 7, 10, 12, 13, 15, 16, 17, 18, 19]
for gen in zero_corr_gens[:3]:  # Analyze first 3
    analyze_generation(gen, n_individuals=10)

# Analyze generations with non-zero mean
print("\n" + "="*80)
print("ANALYZING GENERATIONS WITH NON-ZERO MEAN CORRELATION")
print("="*80)

nonzero_corr_gens = [0, 2, 3, 5]
for gen in nonzero_corr_gens[:2]:  # Analyze first 2
    analyze_generation(gen, n_individuals=10)

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
