# Phase 1: Portfolio-Based GP Evolution Experiment

## 概述

這個實驗使用 **PortfolioBacktestingEngine** 同時評估多個股票的組合表現，進行 GP 演化。

## 主要特點

- ✅ **多股票組合評估**: 同時評估 4 支股票 (ABX.TO, BBD-B.TO, RY.TO, TRP.TO)
- ✅ **大規模演化**: 500 個體，50 代
- ✅ **完整記錄**: 每個 generation 都儲存族群快照
- ✅ **詳細分析**: 包含最佳個體的詳細回測結果

## 使用方法

### 1. 運行完整實驗（500 個體，50 代）

```bash
python run_portfolio_experiment.py
```

**預計耗時**: 約 20-30 分鐘

### 2. 運行測試實驗（10 個體，3 代）

```bash
python test_run_portfolio_experiment.py
```

**預計耗時**: 約 1 秒

## 實驗配置

```python
CONFIG = {
    # 股票組合
    'tickers': ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO'],
    
    # 回測期間
    'backtest_start': '1997-06-25',
    'backtest_end': '1999-06-25',
    'initial_capital': 100000.0,
    
    # GP 參數
    'population_size': 500,
    'generations': 50,
    
    # 演化參數
    'crossover_prob': 0.8,
    'mutation_prob': 0.2,
    'tournament_size': 3,
}
```

## 輸出結構

```
portfolio_experiment_results/
└── portfolio_exp_YYYYMMDD_HHMMSS/
    ├── config.json                    # 實驗配置
    ├── evolution_log.json             # 演化日誌（JSON 格式）
    ├── evolution_log.csv              # 演化日誌（CSV 格式）
    ├── best_individual_result.json    # 最佳個體結果
    ├── best_individual_trades.csv     # 最佳個體交易記錄
    ├── logs/                          # 日誌目錄（預留）
    └── generations/                   # 族群快照
        ├── generation_001.pkl         # Generation 1 族群
        ├── generation_002.pkl         # Generation 2 族群
        ├── ...
        └── generation_050.pkl         # Generation 50 族群
```

## 輸出文件說明

### 1. `config.json`
實驗配置參數

### 2. `evolution_log.json` / `evolution_log.csv`
每個 generation 的統計數據：
- `generation`: 世代編號
- `min_fitness`: 最小 fitness
- `avg_fitness`: 平均 fitness
- `max_fitness`: 最大 fitness
- `std_fitness`: fitness 標準差
- `eval_time`: 評估耗時（秒）
- `timestamp`: 時間戳

### 3. `best_individual_result.json`
最佳個體的詳細結果：
- `individual`: GP 規則字串
- `fitness`: Fitness 值
- `metrics`: 績效指標
  - `total_return`: 總回報率
  - `sharpe_ratio`: Sharpe Ratio
  - `max_drawdown`: 最大回撤
  - `volatility`: 波動率
  - `win_rate`: 勝率
- `per_stock_pnl`: 各股票 PnL 貢獻
- `total_trades`: 總交易數

### 4. `best_individual_trades.csv`
最佳個體的所有交易記錄

### 5. `generations/generation_XXX.pkl`
每個 generation 的完整族群快照，包含：
- `generation`: 世代編號
- `population`: 整個族群（所有個體）
- `hall_of_fame`: 前 10 個最佳個體
- `statistics`: 統計數據
- `timestamp`: 時間戳

## 載入族群快照

```python
import dill

# 載入特定 generation 的族群
with open('portfolio_experiment_results/portfolio_exp_XXX/generations/generation_010.pkl', 'rb') as f:
    data = dill.load(f)

generation = data['generation']
population = data['population']
hall_of_fame = data['hall_of_fame']
statistics = data['statistics']

print(f"Generation {generation}")
print(f"族群大小: {len(population)}")
print(f"最佳個體 fitness: {hall_of_fame[0].fitness.values[0]}")
```

## 與 run_all_experiments.py 的差異

| 特性 | run_all_experiments.py | run_portfolio_experiment.py |
|------|------------------------|----------------------------|
| **評估方式** | 單股票獨立評估 | 多股票組合評估 |
| **Fitness** | 單股票超額回報 | 組合總超額回報 |
| **運行次數** | 每股票 10 次 × 2 期間 | 單次運行 |
| **儲存內容** | 每次運行的最佳個體 | 每個 generation 的族群 |
| **適用場景** | 比較不同股票/期間 | 研究演化過程 |

## 實驗目的

1. **驗證多股票組合評估**: 確認 PortfolioBacktestingEngine 能正確評估組合表現
2. **研究演化動態**: 通過儲存每個 generation，可以分析演化軌跡
3. **發現最佳策略**: 找到在多股票組合上表現最佳的交易規則

## 後續分析

可以使用儲存的 generation 快照進行：
- 演化軌跡分析
- 多樣性分析
- 收斂性分析
- 最佳個體的穩定性分析

## 注意事項

1. **儲存空間**: 500 個體 × 50 代 ≈ 每個 generation 約 2-5 MB，總共約 100-250 MB
2. **運行時間**: 完整實驗約需 20-30 分鐘
3. **記憶體使用**: 族群較大時可能需要較多記憶體

## 範例輸出

```
📊 Generation 50/50
====================================================================================================
⏳ 評估 500 個個體...
✓ 評估完成 (25.3s)

📈 Fitness 統計:
   Min: -0.1234 (-12.34%) | PnL: $-12,340
   Avg: +0.0856 (+8.56%) | PnL: $+8,560
   Max: +0.3675 (+36.75%) | PnL: $+36,750
   Std: 0.0821

💾 儲存 Generation 50 族群...
   ✓ 已儲存: generation_050.pkl (4.52 MB)

🏆 當前最佳個體:
   Fitness: +0.3675 (+36.75%)
   PnL: $+36,750
   深度: 5, 節點數: 23
   規則: gt(lag(vol(ARG1, 39), 61), logical_or(V_FALSE, V_TRUE))
```
