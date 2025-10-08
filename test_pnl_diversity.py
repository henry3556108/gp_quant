"""
Test script for PnL correlation-based diversity analysis

This script demonstrates how to calculate and visualize PnL diversity.
"""

import pandas as pd
from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer

# Configuration
RECORDS_DIR = "experiments_results/ABX_TO/individual_records_long_run01"
TICKER = "ABX.TO"
DATA_FILE = f"TSE300_selected/{TICKER}.csv"

# Backtest period (should match the experiment configuration)
TRAIN_DATA_START = '1992-06-30'
TRAIN_BACKTEST_START = '1993-07-02'
TRAIN_BACKTEST_END = '1999-06-25'

# Sampling configuration (for performance)
SAMPLE_SIZE = 50  # Sample 50 individuals per generation instead of all 500

print("="*80)
print("🧪 Testing PnL Correlation-based Diversity Analysis")
print("="*80)
print()

# Step 1: Load market data
print("1. Loading market data...")
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
print(f"   ✓ Loaded {len(data)} days of data for {TICKER}")
print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
print()

# Step 2: Initialize analyzer
print("2. Initializing diversity analyzer...")
analyzer = DiversityAnalyzer(RECORDS_DIR)
print(f"   ✓ Analyzer created for: {RECORDS_DIR}")
print()

# Step 3: Load populations
print("3. Loading populations...")
populations = analyzer.load_populations(verbose=True)
print(f"   ✓ Loaded {len(populations)} generations")
print()

# Step 4: Calculate PnL diversity (with sampling for performance)
print("4. Calculating PnL correlation diversity...")
print(f"   Using sample size: {SAMPLE_SIZE} individuals per generation")
print(f"   This may take a few minutes...")
print()

pnl_diversity_data = analyzer.calculate_pnl_diversity_trends(
    data=data,
    backtest_start=TRAIN_BACKTEST_START,
    backtest_end=TRAIN_BACKTEST_END,
    sample_size=SAMPLE_SIZE,
    verbose=True
)

print()
print(f"   ✓ Calculated PnL diversity for {len(pnl_diversity_data)} generations")
print()

# Step 5: Display sample results
print("5. Sample PnL diversity data:")
print(pnl_diversity_data.head(10))
print()

# Step 6: Display summary statistics
print("6. Summary statistics:")
print(f"   Mean correlation:")
print(f"      Initial (gen 0): {pnl_diversity_data['pnl_corr_mean'].iloc[0]:.4f}")
print(f"      Final (gen 50):  {pnl_diversity_data['pnl_corr_mean'].iloc[-1]:.4f}")
print(f"      Overall mean:    {pnl_diversity_data['pnl_corr_mean'].mean():.4f}")
print()
print(f"   Correlation std dev:")
print(f"      Initial (gen 0): {pnl_diversity_data['pnl_corr_std'].iloc[0]:.4f}")
print(f"      Final (gen 50):  {pnl_diversity_data['pnl_corr_std'].iloc[-1]:.4f}")
print(f"      Overall mean:    {pnl_diversity_data['pnl_corr_std'].mean():.4f}")
print()

# Step 7: Save results
print("7. Saving results...")
pnl_diversity_data.to_csv("pnl_diversity_test.csv", index=False)
print("   ✓ Data saved to: pnl_diversity_test.csv")
print()

# Step 8: Generate visualization
print("8. Generating visualization...")
DiversityVisualizer.plot_pnl_diversity(
    pnl_diversity_data,
    save_path="pnl_diversity_test.png",
    show=False
)
print("   ✓ Plot saved to: pnl_diversity_test.png")
print()

print("="*80)
print("✅ PnL diversity analysis complete!")
print("="*80)
print()
print("Interpretation:")
print("- Mean correlation close to 1.0 → Individuals have similar trading behavior")
print("- Mean correlation close to 0.0 → Individuals have diverse trading behavior")
print("- High std dev → Mix of similar and diverse individuals in population")
print("- Low std dev → Consistent correlation across all pairs")
