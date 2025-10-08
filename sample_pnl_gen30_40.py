"""
Sample and visualize PnL curves from generations 30-40
"""

import pandas as pd
import numpy as np
import dill
import matplotlib.pyplot as plt
from gp_quant.backtesting.engine import BacktestingEngine
from deap import creator, base, gp
import random

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

# Initialize backtesting engine
engine = BacktestingEngine(
    data=data,
    backtest_start=BACKTEST_START,
    backtest_end=BACKTEST_END
)

def get_valid_pnl_curves(gen_num, n_samples=10):
    """Get valid PnL curves from a generation"""
    pop_file = f"{RECORDS_DIR}/generation_{gen_num:03d}/population.pkl"
    with open(pop_file, 'rb') as f:
        population = dill.load(f)
    
    # Sample individuals
    sampled_inds = random.sample(population, min(n_samples, len(population)))
    
    pnl_curves = []
    individuals = []
    
    for ind in sampled_inds:
        try:
            pnl_curve = engine.get_pnl_curve(ind)
            
            # Check if PnL has variance
            if len(pnl_curve) > 0 and pnl_curve.std() > 0:
                pnl_curves.append(pnl_curve)
                individuals.append(ind)
                
        except Exception:
            continue
    
    return pnl_curves, individuals

# Sample from generations 30-40
target_generations = [30, 35, 40]

print("\n" + "="*80)
print("Sampling PnL curves from generations 30, 35, 40")
print("="*80)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for idx, gen_num in enumerate(target_generations):
    print(f"\nGeneration {gen_num}:")
    
    # Try to get at least 5 valid curves
    pnl_curves = []
    individuals = []
    attempts = 0
    max_attempts = 5
    
    while len(pnl_curves) < 5 and attempts < max_attempts:
        curves, inds = get_valid_pnl_curves(gen_num, n_samples=20)
        pnl_curves.extend(curves)
        individuals.extend(inds)
        attempts += 1
    
    # Take first 5-8 valid curves
    pnl_curves = pnl_curves[:8]
    individuals = individuals[:8]
    
    print(f"  Found {len(pnl_curves)} valid PnL curves")
    
    # Calculate correlations
    if len(pnl_curves) >= 2:
        pnl_matrix = np.array([curve.values for curve in pnl_curves])
        corr_matrix = np.corrcoef(pnl_matrix)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        upper_tri = upper_tri[np.isfinite(upper_tri)]
        
        print(f"  Correlation statistics:")
        print(f"    Mean: {np.mean(upper_tri):.4f}")
        print(f"    Std: {np.std(upper_tri):.4f}")
        print(f"    Range: [{np.min(upper_tri):.4f}, {np.max(upper_tri):.4f}]")
    
    # Plot
    ax = axes[idx]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pnl_curves)))
    
    for i, (pnl_curve, ind) in enumerate(zip(pnl_curves, individuals)):
        fitness = ind.fitness.values[0] if ind.fitness.valid else 0
        label = f'Ind {i+1} (fit={fitness:.0f})'
        ax.plot(pnl_curve.index, pnl_curve.values, alpha=0.7, 
               linewidth=2, color=colors[i], label=label)
    
    # Formatting
    ax.set_title(f'Generation {gen_num} - PnL Curves ({len(pnl_curves)} individuals)', 
                fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative PnL ($)', fontsize=11)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    # Add statistics text
    if len(pnl_curves) >= 2:
        stats_text = f'Corr: μ={np.mean(upper_tri):.3f}, σ={np.std(upper_tri):.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('PnL Curves Analysis: Generations 30, 35, 40', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('pnl_curves_gen30_40.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to: pnl_curves_gen30_40.png")
plt.close()

# Also create a comparison plot showing diversity metrics
print("\n" + "="*80)
print("Creating correlation comparison plot")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Collect correlation data for multiple generations
gen_range = range(25, 46)
corr_means = []
corr_stds = []
valid_gens = []

for gen in gen_range:
    pnl_curves, _ = get_valid_pnl_curves(gen, n_samples=30)
    
    if len(pnl_curves) >= 2:
        pnl_matrix = np.array([curve.values for curve in pnl_curves])
        corr_matrix = np.corrcoef(pnl_matrix)
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        upper_tri = upper_tri[np.isfinite(upper_tri)]
        
        if len(upper_tri) > 0:
            corr_means.append(np.mean(upper_tri))
            corr_stds.append(np.std(upper_tri))
            valid_gens.append(gen)
            print(f"Gen {gen}: mean={np.mean(upper_tri):.3f}, std={np.std(upper_tri):.3f}, n_curves={len(pnl_curves)}")

# Plot 1: Mean correlation
ax1.plot(valid_gens, corr_means, 'o-', linewidth=2, markersize=6, color='steelblue')
ax1.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Gen 30')
ax1.axvline(x=35, color='orange', linestyle='--', alpha=0.5, label='Gen 35')
ax1.axvline(x=40, color='green', linestyle='--', alpha=0.5, label='Gen 40')
ax1.set_title('Mean PnL Correlation', fontsize=13, fontweight='bold')
ax1.set_xlabel('Generation', fontsize=11)
ax1.set_ylabel('Mean Correlation', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

# Plot 2: Std correlation
ax2.plot(valid_gens, corr_stds, 'o-', linewidth=2, markersize=6, color='darkorange')
ax2.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Gen 30')
ax2.axvline(x=35, color='orange', linestyle='--', alpha=0.5, label='Gen 35')
ax2.axvline(x=40, color='green', linestyle='--', alpha=0.5, label='Gen 40')
ax2.set_title('Std Dev of PnL Correlation', fontsize=13, fontweight='bold')
ax2.set_xlabel('Generation', fontsize=11)
ax2.set_ylabel('Std Deviation', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle('PnL Correlation Trends (Generations 25-45)', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('pnl_correlation_trends_gen25_45.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Correlation trends plot saved to: pnl_correlation_trends_gen25_45.png")

print("\n" + "="*80)
print("✅ Analysis complete!")
print("="*80)
