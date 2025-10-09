# Portfolio Evaluation 使用指南

## 概述

Portfolio Evaluation 模組提供多股票組合回測和並行 fitness 評估功能。

## 核心組件

### 1. EventDrivenRebalancer

事件驅動的資金再平衡策略。

**特點**:
- 等權重初始資金分配
- 獨立資金池（股票間不互相挪用）
- 事件驅動：只在交易信號時調整
- 完整交易記錄

**使用範例**:
```python
from gp_quant.backtesting.rebalancing import EventDrivenRebalancer

rebalancer = EventDrivenRebalancer(
    tickers=['ABX.TO', 'RY.TO'],
    initial_capital=100000,
    equal_weight=True
)

# 處理買入信號
transaction = rebalancer.handle_buy_signal(
    ticker='ABX.TO',
    date=datetime(2020, 1, 1),
    price=25.50
)

# 處理賣出信號
transaction = rebalancer.handle_sell_signal(
    ticker='ABX.TO',
    date=datetime(2020, 2, 1),
    price=28.00
)

# 獲取組合總值
total_value = rebalancer.get_portfolio_value()
```

### 2. PortfolioBacktestingEngine

多股票組合回測引擎。

**特點**:
- 支援多股票同時回測
- 自動信號生成
- 完整績效分析
- Thread-safe 設計（每個 worker 獨立實例）

**使用範例**:
```python
from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine

# 準備數據
data = {
    'ABX.TO': df_abx,
    'RY.TO': df_ry,
    'TRP.TO': df_trp,
    'BBD-B.TO': df_bbd
}

# 創建引擎
engine = PortfolioBacktestingEngine(
    data=data,
    backtest_start='1997-06-25',
    backtest_end='1999-06-25',
    initial_capital=100000
)

# 回測個體
result = engine.backtest(individual)

# 獲取結果
equity_curve = result['equity_curve']
metrics = result['metrics']
per_stock_pnl = result['per_stock_pnl']
```

### 3. ParallelFitnessEvaluator

並行 fitness 評估器。

**特點**:
- 使用 multiprocessing（避免 GIL）
- 自動負載平衡
- 錯誤處理與 fallback
- 完全隔離（無 race condition）

**使用範例**:
```python
from gp_quant.parallel.fitness_evaluator import ParallelFitnessEvaluator

# 創建評估器
evaluator = ParallelFitnessEvaluator(
    n_workers=8,
    enable_parallel=True
)

# 評估族群
fitness_scores = evaluator.evaluate_population(
    population,
    engine.get_fitness
)
```

## 績效指標

`PortfolioMetrics` 提供以下指標：

- **Total Return**: 總回報率
- **Excess Return**: 超額回報（相對於無風險利率）
- **Sharpe Ratio**: 夏普比率（年化）
- **Max Drawdown**: 最大回撤
- **Volatility**: 波動率（年化）
- **Calmar Ratio**: Calmar 比率
- **Win Rate**: 勝率
- **PnL**: 損益

## Thread Safety 與 Race Condition 防護

### 設計原則

1. **Process Isolation**: 使用 multiprocessing 而非 threading
2. **No Shared State**: 每個 worker 有獨立的引擎實例
3. **Immutable Data**: 只傳遞不可變數據給 workers
4. **Stateless Methods**: `PortfolioMetrics` 所有方法為 stateless

### 安全使用模式

✅ **正確**:
```python
# 每個 worker 創建自己的引擎
def eval_func(individual):
    engine = create_engine()  # 獨立實例
    return engine.get_fitness(individual)

evaluator.evaluate_population(population, eval_func)
```

❌ **錯誤**:
```python
# 共享引擎實例（會有 race condition）
engine = create_engine()  # 共享實例

def eval_func(individual):
    return engine.get_fitness(individual)  # 危險！

evaluator.evaluate_population(population, eval_func)
```

## 資源分配

### CPU 核心分配策略

**序列模式（推薦用於開發）**:
```
Fitness Evaluation: 8 cores
  ↓ 完成後
Tree Similarity: 8 cores
```

**並行模式（用於生產）**:
```
Fitness Evaluation: 4 cores  ┐
                              ├─ 同時執行
Tree Similarity: 4 cores     ┘
```

### 配置範例

```python
# 序列模式
evaluator = ParallelFitnessEvaluator(n_workers=8)

# 並行模式（Phase 4）
from gp_quant.parallel.executor import ParallelExecutor

executor = ParallelExecutor(max_workers=2)
tasks = [
    (evaluate_fitness, (population,), {'n_workers': 4}),
    (calculate_similarity, (population,), {'n_workers': 4})
]
results = executor.execute_concurrent(tasks)
```

## 效能基準

### Population = 500

| 模式 | 時間 | 加速比 |
|------|------|--------|
| 序列 | 8 分鐘 | 1.0x |
| 並行 (8 cores) | 1 分鐘 | 8.0x |

### Population = 5000

| 模式 | 時間 | 加速比 |
|------|------|--------|
| 序列 | 2.9 小時 | 1.0x |
| 並行 (8 cores) | 25 分鐘 | 7.0x |

## 錯誤處理

### 自動 Fallback

```python
evaluator = ParallelFitnessEvaluator(
    n_workers=8,
    enable_parallel=True,
    min_population_for_parallel=50  # 小於 50 自動用序列
)

# 如果並行失敗，自動 fallback 到序列模式
fitness_scores = evaluator.evaluate_population(population, eval_func)
```

### 日誌記錄

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gp_quant.parallel')

# 會記錄：
# - Worker 數量
# - 執行模式（parallel/sequential）
# - 錯誤信息
# - Fallback 事件
```

## 測試

運行測試：

```bash
pytest tests/backtesting/test_portfolio_engine.py -v
```

## 下一步

- [ ] 整合到 EvolutionEngine
- [ ] 實作 Tree Similarity 並行計算
- [ ] 完整實驗流程
- [ ] 效能優化

## 參考

- `gp_quant/backtesting/portfolio_engine.py`
- `gp_quant/parallel/fitness_evaluator.py`
- `samples/portfolio/sample_portfolio_backtest.py`
