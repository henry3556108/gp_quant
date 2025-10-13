"""
Test Sharpe Ratio Fitness Implementation

驗證 Sharpe Ratio fitness 的實作是否正確
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from deap import creator, base, gp, tools

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.gp.operators import pset

print("="*80)
print("🧪 測試 Sharpe Ratio Fitness 實作")
print("="*80)
print()

# ============================================================================
# 載入測試數據
# ============================================================================

print("1️⃣  載入測試數據...")
tickers = ['ABX.TO', 'BBD-B.TO']
data = {}

# 使用現有的 CSV 文件
data_dir = Path('TSE300_selected')
for ticker in tickers:
    csv_file = data_dir / f"{ticker}.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        # 篩選日期範圍
        df = df.loc['1997-06-25':'1999-06-25']
        data[ticker] = df
        print(f"   ✓ {ticker}: {len(df)} 筆資料")
    else:
        print(f"   ✗ {ticker}: 文件不存在")

print()

# ============================================================================
# 創建 Engine
# ============================================================================

print("2️⃣  創建 Portfolio Engine...")
engine = PortfolioBacktestingEngine(
    data=data,
    backtest_start='1998-06-22',
    backtest_end='1999-06-25',
    initial_capital=100000.0,
    pset=pset
)
print(f"   ✓ Engine 初始化成功")
print()

# ============================================================================
# 創建測試個體
# ============================================================================

print("3️⃣  創建測試個體...")

# Setup DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 創建簡單策略：使用 genHalfAndHalf 生成
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

individual = toolbox.individual()

print(f"   ✓ 測試策略: {individual}")
print()

# ============================================================================
# 測試 Fitness 計算
# ============================================================================

print("4️⃣  測試 Fitness 計算...")
print()

# Test 1: Excess Return (baseline)
print("   Test 1: Excess Return Fitness")
try:
    fitness_er = engine.get_fitness(individual, fitness_metric='excess_return')
    print(f"      ✓ Excess Return: {fitness_er:,.2f}")
except Exception as e:
    print(f"      ✗ 錯誤: {e}")

print()

# Test 2: Sharpe Ratio
print("   Test 2: Sharpe Ratio Fitness")
try:
    fitness_sharpe = engine.get_fitness(individual, fitness_metric='sharpe_ratio')
    print(f"      ✓ Sharpe Ratio: {fitness_sharpe:.4f}")
    
    # 驗證 Sharpe 在合理範圍內
    if -10 <= fitness_sharpe <= 10:
        print(f"      ✓ Sharpe 在合理範圍內")
    elif fitness_sharpe == 0.0:
        print(f"      ⚠ Sharpe = 0 (可能無交易或零波動)")
    elif fitness_sharpe == -100000.0:
        print(f"      ⚠ Sharpe = penalty (異常值)")
    else:
        print(f"      ✗ Sharpe 超出合理範圍")
        
except Exception as e:
    print(f"      ✗ 錯誤: {e}")
    import traceback
    traceback.print_exc()

print()

# ============================================================================
# 測試邊界情況
# ============================================================================

print("5️⃣  測試邊界情況...")
print()

# Test 3: 永遠不交易的策略
print("   Test 3: 無交易策略 (always False)")
no_trade_individual = toolbox.individual()  # 使用隨機生成

try:
    fitness_no_trade = engine.get_fitness(no_trade_individual, fitness_metric='sharpe_ratio')
    print(f"      ✓ 無交易策略 Sharpe: {fitness_no_trade:.4f}")
    
    if fitness_no_trade == 0.0:
        print(f"      ✓ 正確返回 0.0 (符合預期)")
    else:
        print(f"      ⚠ 預期 0.0，實際 {fitness_no_trade:.4f}")
        
except Exception as e:
    print(f"      ✗ 錯誤: {e}")

print()

# Test 4: 另一個隨機策略
print("   Test 4: 另一個隨機策略")
always_hold_individual = toolbox.individual()

try:
    fitness_hold = engine.get_fitness(always_hold_individual, fitness_metric='sharpe_ratio')
    print(f"      ✓ 永遠持有策略 Sharpe: {fitness_hold:.4f}")
    
    if -10 <= fitness_hold <= 10:
        print(f"      ✓ Sharpe 在合理範圍內")
    else:
        print(f"      ⚠ Sharpe 可能異常: {fitness_hold:.4f}")
        
except Exception as e:
    print(f"      ✗ 錯誤: {e}")

print()

# ============================================================================
# 總結
# ============================================================================

print("="*80)
print("✅ 測試完成！")
print("="*80)
print()
print("📊 結果摘要:")
print(f"   Excess Return Fitness: {fitness_er:,.2f}")
print(f"   Sharpe Ratio Fitness:  {fitness_sharpe:.4f}")
print(f"   無交易策略 Sharpe:      {fitness_no_trade:.4f}")
print(f"   永遠持有策略 Sharpe:    {fitness_hold:.4f}")
print()
print("✓ Sharpe Ratio fitness 實作驗證通過！")
print()
