"""
Sample Portfolio Backtesting Script

This script demonstrates how to use the portfolio backtesting engine
with parallel fitness evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from deap import creator, base, gp
from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.parallel.fitness_evaluator import ParallelFitnessEvaluator
import time

# Setup DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*80)
print("📊 Portfolio Backtesting Demo")
print("="*80)
print()

# Configuration
TICKERS = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
BACKTEST_START = '1997-06-25'
BACKTEST_END = '1999-06-25'
INITIAL_CAPITAL = 100000.0

print("Configuration:")
print(f"  Tickers: {TICKERS}")
print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,.0f}")
print()

# Load data
print("1. Loading market data...")
data = {}
for ticker in TICKERS:
    file_path = project_root / f"TSE300_selected/{ticker}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        data[ticker] = df
        print(f"   ✓ {ticker}: {len(df)} days")
    else:
        print(f"   ✗ {ticker}: File not found")

if len(data) != len(TICKERS):
    print("\n❌ Not all data files found. Please check TSE300_selected/ directory.")
    sys.exit(1)

print()

# Create portfolio engine
print("2. Initializing portfolio backtesting engine...")
try:
    engine = PortfolioBacktestingEngine(
        data=data,
        backtest_start=BACKTEST_START,
        backtest_end=BACKTEST_END,
        initial_capital=INITIAL_CAPITAL
    )
    print(f"   ✓ Engine initialized")
    print(f"   ✓ Common trading dates: {len(engine.common_dates)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print()

# Create a simple test individual (buy and hold)
print("3. Creating test individual...")
# Note: This is a placeholder. In real usage, you would create a proper GP individual
# For now, we'll skip the actual backtest since we need the proper primitive set
print("   ⚠️  Skipping backtest - requires proper GP primitive set")
print("   (This will be integrated in the evolution engine)")
print()

# Demonstrate parallel fitness evaluator
print("4. Testing parallel fitness evaluator...")
evaluator = ParallelFitnessEvaluator(n_workers=8, enable_parallel=True)
print(f"   ✓ Evaluator initialized with {evaluator.n_workers} workers")
print()

# Performance comparison
print("5. Performance comparison (Sequential vs Parallel):")
print("   Note: Actual comparison requires a population of GP individuals")
print("   Expected speedup with 8 workers: ~7x")
print()

print("="*80)
print("✅ Portfolio Backtesting Demo Complete!")
print("="*80)
print()
print("Next Steps:")
print("  1. Integrate with EvolutionEngine")
print("  2. Test with real GP individuals")
print("  3. Run full experiments")
print()
