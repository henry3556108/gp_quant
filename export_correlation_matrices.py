"""
Export correlation matrices for all generations

This script calculates and saves the PnL correlation matrix for each generation
as separate CSV files for detailed inspection.
"""

import pandas as pd
import numpy as np
import dill
import os
from pathlib import Path
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
OUTPUT_DIR = "correlation_matrices"
SAMPLE_SIZE = 50  # Sample size per generation

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("📊 Exporting PnL Correlation Matrices")
print("="*80)
print(f"Records directory: {RECORDS_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Sample size: {SAMPLE_SIZE} individuals per generation")
print()

# Load market data
print("Loading market data...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
print(f"✓ Data loaded: {data.index[0].date()} to {data.index[-1].date()}")
print()

# Initialize backtesting engine
engine = BacktestingEngine(
    data=data,
    backtest_start=BACKTEST_START,
    backtest_end=BACKTEST_END
)

# Summary data for all generations
summary_data = []

# Process each generation
for gen_num in range(51):  # 0 to 50
    print(f"Processing Generation {gen_num}...", end=" ")
    
    # Load population
    pop_file = f"{RECORDS_DIR}/generation_{gen_num:03d}/population.pkl"
    with open(pop_file, 'rb') as f:
        population = dill.load(f)
    
    # Sample individuals
    import random
    sampled_inds = random.sample(population, min(SAMPLE_SIZE, len(population)))
    
    # Generate PnL curves
    pnl_curves = []
    individual_info = []
    
    for idx, ind in enumerate(sampled_inds):
        try:
            pnl_curve = engine.get_pnl_curve(ind)
            
            # Check if PnL has variance
            if len(pnl_curve) > 0 and pnl_curve.std() > 0:
                pnl_curves.append(pnl_curve.values)
                
                # Store individual info
                fitness = ind.fitness.values[0] if ind.fitness.valid else 0
                individual_info.append({
                    'ind_id': idx,
                    'fitness': fitness,
                    'height': ind.height,
                    'length': len(ind),
                    'pnl_final': pnl_curve.iloc[-1],
                    'pnl_std': pnl_curve.std()
                })
        except Exception:
            continue
    
    n_valid = len(pnl_curves)
    print(f"{n_valid} valid curves", end=" ")
    
    if n_valid >= 2:
        # Calculate correlation matrix
        pnl_matrix = np.array(pnl_curves)
        corr_matrix = np.corrcoef(pnl_matrix)
        
        # Create DataFrame with individual IDs
        ind_ids = [info['ind_id'] for info in individual_info]
        corr_df = pd.DataFrame(
            corr_matrix,
            index=[f"Ind_{id}" for id in ind_ids],
            columns=[f"Ind_{id}" for id in ind_ids]
        )
        
        # Save correlation matrix
        corr_file = f"{OUTPUT_DIR}/gen{gen_num:03d}_correlation_matrix.csv"
        corr_df.to_csv(corr_file)
        
        # Save individual info
        info_df = pd.DataFrame(individual_info)
        info_file = f"{OUTPUT_DIR}/gen{gen_num:03d}_individual_info.csv"
        info_df.to_csv(info_file, index=False)
        
        # Calculate statistics
        upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        upper_tri = upper_tri[np.isfinite(upper_tri)]
        
        if len(upper_tri) > 0:
            summary_data.append({
                'generation': gen_num,
                'n_valid_individuals': n_valid,
                'n_pairs': len(upper_tri),
                'corr_mean': np.mean(upper_tri),
                'corr_std': np.std(upper_tri),
                'corr_min': np.min(upper_tri),
                'corr_max': np.max(upper_tri),
                'corr_median': np.median(upper_tri),
                'corr_q25': np.percentile(upper_tri, 25),
                'corr_q75': np.percentile(upper_tri, 75)
            })
            print(f"→ mean={np.mean(upper_tri):.3f}, std={np.std(upper_tri):.3f} ✓")
        else:
            print("→ No valid correlations")
    else:
        print("→ Not enough valid curves")

print()
print("="*80)

# Save summary
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{OUTPUT_DIR}/correlation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Summary saved to: {summary_file}")
    print(f"✓ {len(summary_data)} generations with valid correlations")
    print()
    
    # Display summary statistics
    print("Summary Statistics:")
    print(f"  Generations with data: {len(summary_data)}")
    print(f"  Mean correlation range: [{summary_df['corr_mean'].min():.3f}, {summary_df['corr_mean'].max():.3f}]")
    print(f"  Overall mean correlation: {summary_df['corr_mean'].mean():.3f}")
    print()
    
    # Show first few rows
    print("First 10 generations:")
    print(summary_df.head(10).to_string(index=False))
    print()
    
    print("Last 10 generations:")
    print(summary_df.tail(10).to_string(index=False))

print()
print("="*80)
print("✅ Export complete!")
print("="*80)
print(f"\nFiles saved in: {OUTPUT_DIR}/")
print("  - gen###_correlation_matrix.csv: Full correlation matrix")
print("  - gen###_individual_info.csv: Individual metadata")
print("  - correlation_summary.csv: Summary statistics for all generations")
