"""
驗證 Early Stopping 功能的演化實驗

目標: 驗證 Early Stopping 能在演化收斂時正確觸發
配置: 小族群 (50) + 多世代 (30) + Early Stopping (patience=5)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from deap import creator, base, gp, tools
import random
import time
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.gp.operators import pset
from gp_quant.evolution.early_stopping import EarlyStopping

print("="*80)
print("🧪 Early Stopping 功能驗證實驗")
print("="*80)
print()

# ============================================================================
# 實驗配置
# ============================================================================

CONFIG = {
    'tickers': ['ABX.TO', 'BBD-B.TO'],  # 只用 2 支股票加快測試
    'population_size': 50,
    'generations': 30,  # 設定 30 代，但預期會提前停止
    'initial_capital': 100000,
    
    # Early Stopping 配置
    'early_stopping_enabled': True,
    'early_stopping_patience': 5,
    'early_stopping_min_delta': 0.001,
    
    # 演化參數
    'crossover_prob': 0.7,
    'mutation_prob': 0.2,
    'tournament_size': 3,
    
    # Fitness 指標
    'fitness_metric': 'sharpe_ratio',
}

print("📋 實驗配置:")
print(f"  股票: {', '.join(CONFIG['tickers'])}")
print(f"  族群大小: {CONFIG['population_size']}")
print(f"  最大世代: {CONFIG['generations']}")
print(f"  Fitness 指標: {CONFIG['fitness_metric']}")
print(f"  Early Stopping: {'啟用' if CONFIG['early_stopping_enabled'] else '停用'}")
if CONFIG['early_stopping_enabled']:
    print(f"    - Patience: {CONFIG['early_stopping_patience']} 代")
    print(f"    - Min Delta: {CONFIG['early_stopping_min_delta']}")
print()

# ============================================================================
# 載入數據
# ============================================================================

print("1️⃣  載入數據...")
data = {}
for ticker in CONFIG['tickers']:
    df = pd.read_csv(f'TSE300_selected/{ticker}.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # 只用最近 500 天數據加快測試
    df = df.tail(500).reset_index(drop=True)
    # 設置 DatetimeIndex（Portfolio Engine 需要）
    df = df.set_index('Date')
    data[ticker] = df
    print(f"   ✓ {ticker}: {len(df)} 天")
print()

# ============================================================================
# 初始化 Portfolio Engine
# ============================================================================

print("2️⃣  初始化 Portfolio Engine...")

# 使用日期範圍（從 DatetimeIndex 獲取）
dates = data[CONFIG['tickers'][0]].index
backtest_start = dates[250].strftime('%Y-%m-%d')
backtest_end = dates[-1].strftime('%Y-%m-%d')

print(f"   回測期間: {backtest_start} 到 {backtest_end}")

engine = PortfolioBacktestingEngine(
    data=data,
    backtest_start=backtest_start,
    backtest_end=backtest_end,
    initial_capital=CONFIG['initial_capital']
)
print(f"   ✓ 交易日數: {len(engine.common_dates)}")
print()

# ============================================================================
# 設置 DEAP
# ============================================================================

print("3️⃣  設置 DEAP...")

# 清理舊的 creator
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evaluate_individual(individual):
    """評估個體 fitness"""
    try:
        fitness = engine.get_fitness(individual, fitness_metric=CONFIG['fitness_metric'])
        return (fitness,)
    except Exception as e:
        return (-1000000.0,)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=CONFIG['tournament_size'])
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=lambda ind: ind.height, max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: ind.height, max_value=17))

print("   ✓ DEAP 設置完成")
print()

# ============================================================================
# 初始化 Early Stopping
# ============================================================================

early_stopping = None
if CONFIG['early_stopping_enabled']:
    print("4️⃣  初始化 Early Stopping...")
    early_stopping = EarlyStopping(
        patience=CONFIG['early_stopping_patience'],
        min_delta=CONFIG['early_stopping_min_delta'],
        mode='max'
    )
    print(f"   ✓ Patience: {early_stopping.patience}")
    print(f"   ✓ Min Delta: {early_stopping.min_delta}")
    print()

# ============================================================================
# 演化
# ============================================================================

print("5️⃣  開始演化...")
print("="*80)
print()

# 創建初始族群
pop = toolbox.population(n=CONFIG['population_size'])
hof = tools.HallOfFame(10)

# 統計
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# 評估初始族群
print("⏳ 評估初始族群...")
start_time = time.time()
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
eval_time = time.time() - start_time
print(f"✓ 完成 ({eval_time:.1f}s)")
print()

hof.update(pop)
record = stats.compile(pop)

print(f"📊 Generation 0/{CONFIG['generations']}")
print(f"   Fitness - Min: {record['min']:.4f}, Avg: {record['avg']:.4f}, Max: {record['max']:.4f}")
print()

# 演化循環
early_stopped = False
actual_generations = 0

for gen in range(1, CONFIG['generations'] + 1):
    gen_start = time.time()
    
    # 選擇
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    
    # 交叉
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CONFIG['crossover_prob']:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # 變異
    for mutant in offspring:
        if random.random() < CONFIG['mutation_prob']:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # 評估需要評估的個體
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # 更新族群
    pop[:] = offspring
    hof.update(pop)
    record = stats.compile(pop)
    
    gen_time = time.time() - gen_start
    actual_generations = gen
    
    # 顯示進度
    print(f"📊 Generation {gen}/{CONFIG['generations']}")
    print(f"   Fitness - Min: {record['min']:.4f}, Avg: {record['avg']:.4f}, Max: {record['max']:.4f}")
    print(f"   最佳個體: {hof[0].fitness.values[0]:.4f}")
    
    # Early Stopping 檢查
    if early_stopping is not None:
        current_best = hof[0].fitness.values[0]
        should_stop = early_stopping.step(current_best)
        
        status = early_stopping.get_status()
        print(f"   ⏸️  Early Stopping: {status['counter']}/{status['patience']} 代無顯著進步")
        
        if should_stop:
            print()
            print("="*80)
            print("⏹️  Early Stopping 觸發！")
            print("="*80)
            print(f"   連續 {early_stopping.counter} 代無顯著進步")
            print(f"   最佳 fitness: {early_stopping.best_fitness:.4f}")
            print(f"   最終 generation: {gen}/{CONFIG['generations']}")
            print(f"   Early Stopping 狀態: {status}")
            early_stopped = True
            break
    
    print(f"   ⏱️  耗時: {gen_time:.1f}s")
    print()

# ============================================================================
# 結果總結
# ============================================================================

print()
print("="*80)
print("✅ 演化完成！")
print("="*80)
print()

print("📊 實驗結果:")
print(f"   總世代數: {actual_generations}/{CONFIG['generations']}")
print(f"   Early Stopping: {'是（第 {} 代觸發）'.format(actual_generations) if early_stopped else '否（完整運行）'}")
if early_stopped:
    print(f"   節省世代數: {CONFIG['generations'] - actual_generations}")
    print(f"   節省比例: {(CONFIG['generations'] - actual_generations) / CONFIG['generations'] * 100:.1f}%")
print()

print("🏆 最佳個體:")
best = hof[0]
print(f"   Fitness: {best.fitness.values[0]:.4f}")
print(f"   深度: {best.height}")
print(f"   節點數: {len(best)}")
print(f"   規則: {str(best)}")
print()

# 詳細回測最佳個體
print("📈 最佳個體詳細回測:")
result = engine.backtest(best)
metrics = result['metrics']
print(f"   Total Return: {metrics['total_return']:.2%}")
print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
if 'total_trades' in metrics:
    print(f"   總交易次數: {metrics['total_trades']}")
print()

print("="*80)
print("🎉 驗證完成！")
print("="*80)
print()

# ============================================================================
# 結論
# ============================================================================

print("📝 驗證結論:")
print()

if CONFIG['early_stopping_enabled']:
    if early_stopped:
        print("✅ Early Stopping 功能正常！")
        print(f"   - 成功在第 {actual_generations} 代觸發")
        print(f"   - 連續 {CONFIG['early_stopping_patience']} 代無顯著進步")
        print(f"   - 節省了 {CONFIG['generations'] - actual_generations} 代的計算時間")
        print()
        print("💡 建議:")
        print("   - 功能驗證通過，可以用於正式實驗")
        print("   - 可根據實驗需求調整 patience 和 min_delta")
    else:
        print("⚠️  Early Stopping 未觸發")
        print(f"   - 演化持續改進，運行了完整的 {actual_generations} 代")
        print(f"   - 這是正常的，表示族群仍在持續進化")
        print()
        print("💡 建議:")
        print("   - 如果想測試觸發情況，可以:")
        print("     1. 減少 patience (例如 5 → 3)")
        print("     2. 增加 min_delta (例如 0.001 → 0.01)")
        print("     3. 減少族群多樣性（更容易收斂）")
else:
    print("ℹ️  Early Stopping 未啟用")
    print(f"   - 運行了完整的 {actual_generations} 代")

print()
print("="*80)
