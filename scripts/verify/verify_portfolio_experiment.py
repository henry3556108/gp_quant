"""
簡化的 Portfolio 實驗測試

測試新的 Portfolio Engine 能否正確運行
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from deap import creator, base, gp, tools
import random

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.gp.operators import pset

def main():
    print("="*80)
    print("🧪 Portfolio Engine 完整實驗測試")
    print("="*80)
    print()
    
    # 載入數據
    print("1. 載入數據...")
    data = {}
    tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
    
    for ticker in tickers:
        file_path = project_root / f"TSE300_selected/{ticker}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            data[ticker] = df
            print(f"   ✓ {ticker}: {len(df)} 天")
    
    print()
    
    # 創建 engine（傳入 pset）
    print("2. 創建 Portfolio Engine...")
    engine = PortfolioBacktestingEngine(
        data=data,
        backtest_start='1997-06-25',
        backtest_end='1999-06-25',
        initial_capital=100000.0,
        pset=pset  # 傳入 pset
    )
    print(f"   ✓ 初始化成功")
    print(f"   ✓ 交易日數: {len(engine.common_dates)}")
    print()
    
    # 設置 DEAP
    print("3. 設置 DEAP...")
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    print("   ✓ DEAP 設置完成")
    print()
    
    # 創建一個簡單的個體
    print("4. 測試單個個體評估...")
    individual = toolbox.individual()
    print(f"   個體: {individual}")
    print(f"   深度: {individual.height}, 節點數: {len(individual)}")
    
    # 評估 fitness
    try:
        fitness = engine.get_fitness(individual)
        print(f"   ✓ Fitness 評估成功: ${fitness:,.2f}")
    except Exception as e:
        print(f"   ✗ Fitness 評估失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 詳細回測
    print("5. 詳細回測...")
    try:
        result = engine.backtest(individual)
        metrics = result['metrics']
        
        print(f"   ✓ 回測成功")
        print(f"   Total Return: {metrics['total_return']*100:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        print(f"\n   各股票 PnL:")
        for ticker, pnl in result['per_stock_pnl'].items():
            print(f"     {ticker}: ${pnl:,.2f}")
        
    except Exception as e:
        print(f"   ✗ 回測失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    
    # 測試演化
    print("6. 測試演化（100 個體，5 代）...")
    population = toolbox.population(n=200)
    
    for gen in range(5):
        print(f"\n   Generation {gen + 1}/5:")
        
        # 序列評估（避免 multiprocessing 問題）
        fitness_values = []
        for ind in population:
            try:
                fit = engine.get_fitness(ind)
                fitness_values.append(fit)
            except:
                fitness_values.append(-1000000.0)
        
        # 分配 fitness
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = (fit,)
        
        # 統計
        fits = [ind.fitness.values[0] for ind in population]
        min_fit, avg_fit, max_fit = min(fits), np.mean(fits), max(fits)
        print(f"     Fitness - Min: {min_fit:.4f} ({min_fit*100:.2f}%), "
              f"Avg: {avg_fit:.4f} ({avg_fit*100:.2f}%), "
              f"Max: {max_fit:.4f} ({max_fit*100:.2f}%)")
        print(f"     PnL估算 - Min: ${min_fit*100000:,.0f}, "
              f"Avg: ${avg_fit*100000:,.0f}, "
              f"Max: ${max_fit*100000:,.0f}")
        
        # 選擇和繁殖（如果不是最後一代）
        if gen < 4:
            offspring = tools.selTournament(population, len(population), tournsize=3)
            offspring = list(map(toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:
                    gp.cxOnePoint(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < 0.2:
                    gp.mutUniform(mutant, expr=toolbox.expr, pset=pset)
                    del mutant.fitness.values
            
            population = offspring
    
    print()
    print("="*80)
    print("✅ 所有測試通過！")
    print("="*80)
    print()
    print("測試總結:")
    print("  ✓ Portfolio Engine 初始化")
    print("  ✓ 單個體 Fitness 評估")
    print("  ✓ 詳細回測")
    print("  ✓ 小規模演化")
    print()
    print("Phase 1 核心功能驗證成功！🎉")

if __name__ == '__main__':
    main()
