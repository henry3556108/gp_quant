# engine.py 完整說明文檔

**文件**: `gp_quant/evolution/engine.py`  
**總行數**: 194 行  
**核心功能**: GP 演化算法的主引擎  
**創建日期**: 2025-10-06

---

## 📋 目錄

1. [文件概覽](#文件概覽)
2. [函數 1: ranked_selection()](#函數-1-ranked_selection)
3. [函數 2: run_evolution()](#函數-2-run_evolution)
4. [完整演化流程](#完整演化流程)
5. [關鍵設計決策](#關鍵設計決策)
6. [常見問題 FAQ](#常見問題-faq)

---

## 📦 文件概覽

### **文件結構**

```
engine.py (194 行)
├── 導入模塊 (L1-17)
├── ranked_selection() (L20-63)    ← 自定義選擇算子
└── run_evolution() (L65-193)      ← 主演化函數
    ├── 設置階段 (L81-123)
    ├── 初始化階段 (L125-150)
    └── 演化循環 (L153-192)
```

### **依賴模塊**

```python
# 標準庫
import random, operator
import numpy as np
import pandas as pd
from typing import Dict, Union

# 第三方庫
from tqdm import trange, tqdm          # 進度條
from deap import base, creator, tools, gp  # DEAP 演化框架

# 專案內部
from gp_quant.backtesting.engine import BacktestingEngine, PortfolioBacktestingEngine
from gp_quant.gp.operators import pset
```

### **核心職責**

1. ✅ 實現自定義選擇算子（Ranked Selection + SUS）
2. ✅ 配置 DEAP Toolbox（生成器、算子、限制）
3. ✅ 執行完整的演化循環（選擇、交配、變異、評估）
4. ✅ 記錄演化統計和進度
5. ✅ 返回最佳個體和演化日誌

---

## 🎯 函數 1: ranked_selection()

**位置**: L20-63 (44 行)  
**功能**: 實現 Ranked Selection + Stochastic Universal Sampling (SUS)

### **函數簽名**

```python
def ranked_selection(individuals, k, max_rank_fitness=1.8, min_rank_fitness=0.2):
    """
    Custom selection operator implementing Ranked Selection + SUS.
    
    Args:
        individuals: A list of individuals to select from.
        k: The number of individuals to select.
        max_rank_fitness: The fitness value assigned to the best individual (Max in PRD).
        min_rank_fitness: The fitness value assigned to the worst individual (Min in PRD).
    
    Returns:
        A list of selected individuals.
    """
```

### **參數說明**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `individuals` | List[Individual] | - | 要選擇的族群 |
| `k` | int | - | 要選擇的個體數量 |
| `max_rank_fitness` | float | 1.8 | 分配給第 1 名的 rank fitness |
| `min_rank_fitness` | float | 0.2 | 分配給最後一名的 rank fitness |

### **演算法流程**

```
輸入: individuals (500 個), k (500)
  ↓
步驟 1: 排序 (L38)
  sorted_individuals = sorted(individuals, key=fitness, reverse=True)
  結果: [第1名, 第2名, ..., 第500名]
  ↓
步驟 2: 分配 rank_fitness (L42-47)
  for i, ind in enumerate(sorted_individuals):
      rank = i + 1
      ind.rank_fitness = 1.8 - (1.6 * (rank-1) / 499)
  
  結果:
    第 1 名: rank_fitness = 1.8
    第 2 名: rank_fitness = 1.797
    第 3 名: rank_fitness = 1.794
    ...
    第 500 名: rank_fitness = 0.2
  ↓
步驟 3: 暫存原始 fitness (L53)
  original_fitnesses = [ind.fitness.values for ind in sorted_individuals]
  ↓
步驟 4: 替換為 rank_fitness (L54-55)
  for ind in sorted_individuals:
      ind.fitness.values = (ind.rank_fitness,)
  ↓
步驟 5: 執行 SUS (L57)
  chosen = tools.selStochasticUniversalSampling(sorted_individuals, k)
  
  SUS 原理:
    - 計算總 fitness: sum = 1.8 + 1.797 + ... + 0.2
    - 間隔距離: distance = sum / k
    - 隨機起點: start = random(0, distance)
    - 等距選擇: [start, start+distance, start+2*distance, ...]
  ↓
步驟 6: 恢復原始 fitness (L60-61)
  for ind, fit in zip(sorted_individuals, original_fitnesses):
      ind.fitness.values = fit
  ↓
步驟 7: 返回選中的個體 (L63)
  return chosen
```

### **SUS vs 輪盤賭**

| 特性 | 輪盤賭 (Roulette Wheel) | SUS |
|------|------------------------|-----|
| 選擇方式 | 每次隨機旋轉 | 一次旋轉，等距選擇 |
| 隨機性 | 高（每次獨立） | 低（等距確定） |
| 選擇偏差 | 大 | 小 |
| 期望值 | 正確 | 正確 |
| 方差 | 大 | 小 |
| 適用場景 | 一般 | 需要穩定選擇 |

**SUS 示意圖**:
```
Fitness 輪盤:
┌─────────────────────────────────────────────────┐
│ A(1.8) │ B(1.797) │ C(1.794) │ ... │ Z(0.2) │
└─────────────────────────────────────────────────┘
  ↑        ↑          ↑               ↑
  指針1    指針2      指針3           指針k
  (等距分布，一次旋轉確定所有指針位置)
```

---

## 🚀 函數 2: run_evolution()

**位置**: L65-193 (129 行)  
**功能**: 配置並執行完整的 GP 演化算法

### **函數簽名**

```python
def run_evolution(data, population_size=500, n_generations=50, 
                  crossover_prob=0.6, mutation_prob=0.05):
    """
    Configures and runs the main evolutionary algorithm.
    
    Args:
        data: The historical stock data. Can be either:
              - A single Pandas DataFrame (for single ticker evolution)
              - A Dict[str, DataFrame] (for portfolio evolution)
        population_size: The number of individuals in the population.
        n_generations: The number of generations to run.
        crossover_prob: The probability of crossover.
        mutation_prob: The probability of mutation.
    
    Returns:
        A tuple containing the final population, the logbook, and the hall of fame.
    """
```

### **參數說明**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `data` | DataFrame or Dict | - | 股票數據（單 ticker 或多 ticker） |
| `population_size` | int | 500 | 族群大小 |
| `n_generations` | int | 50 | 演化代數 |
| `crossover_prob` | float | 0.6 | 交配機率（60%） |
| `mutation_prob` | float | 0.05 | 變異機率（5%） |

### **返回值**

```python
return pop, logbook, hof

# pop: List[Individual]
#   - 最終一代的完整族群（500 個個體）
#   - 每個個體都有 fitness 值

# logbook: tools.Logbook
#   - 記錄每一代的統計數據
#   - 包含 gen, nevals, avg, std, min, max

# hof: tools.HallOfFame
#   - 保存演化過程中最好的個體
#   - hof[0] 是最佳個體
```

---

## 🔧 3.1 設置階段 (L81-123)

### **步驟 1: 創建 Toolbox** (L82)

```python
toolbox = base.Toolbox()
```

**作用**: 創建 DEAP 的工具箱，用於註冊所有演化算子

---

### **步驟 2: 檢測數據類型並創建回測引擎** (L84-107)

```python
if isinstance(data, dict):
    # Portfolio 模式
    first_ticker = list(data.keys())[0]
    if isinstance(data[first_ticker], dict) and 'data' in data[first_ticker]:
        # 新結構（含 backtest_config）
        data_dict = {ticker: data[ticker]['data'] for ticker in data.keys()}
        backtest_config = {
            ticker: {
                'backtest_start': data[ticker]['backtest_start'],
                'backtest_end': data[ticker]['backtest_end']
            }
            for ticker in data.keys()
        }
        backtester = PortfolioBacktestingEngine(data_dict, backtest_config=backtest_config)
    else:
        # 舊結構（向後兼容）
        backtester = PortfolioBacktestingEngine(data)
    print(f"Running PORTFOLIO evolution with {len(data)} tickers")
else:
    # Single ticker 模式
    backtester = BacktestingEngine(data)
    print(f"Running SINGLE TICKER evolution")
```

**智能檢測邏輯**:

```
data 是 dict？
  ├─ 是 → Portfolio 模式
  │   │
  │   └─ data[ticker] 是 dict 且包含 'data' 鍵？
  │       ├─ 是 → 新結構（重構後）
  │       │   └─ 提取 data_dict 和 backtest_config
  │       │   └─ PortfolioBacktestingEngine(data_dict, backtest_config)
  │       │
  │       └─ 否 → 舊結構（向後兼容）
  │           └─ PortfolioBacktestingEngine(data)
  │
  └─ 否 → Single ticker 模式
      └─ BacktestingEngine(data)
```

**數據結構範例**:

```python
# 新結構（重構後）
data = {
    'ABX.TO': {
        'data': DataFrame,
        'backtest_start': '1998-06-22',
        'backtest_end': '1999-06-25'
    },
    'BBD-B.TO': {
        'data': DataFrame,
        'backtest_start': '1998-06-22',
        'backtest_end': '1999-06-25'
    }
}

# 舊結構（向後兼容）
data = {
    'ABX.TO': DataFrame,
    'BBD-B.TO': DataFrame
}

# Single ticker
data = DataFrame
```

---

### **步驟 3: 註冊生成器** (L109-112)

```python
# Attribute generator
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
```

**層級關係**:

```
toolbox.population(n=500)
  └─ 調用 tools.initRepeat(list, toolbox.individual, n=500)
      └─ 重複調用 toolbox.individual() 500 次
          └─ 調用 tools.initIterate(creator.Individual, toolbox.expr)
              └─ 調用 toolbox.expr()
                  └─ 調用 gp.genHalfAndHalf(pset, min_=2, max_=6)
                      └─ 生成一個深度 2-6 的 GP 樹
```

**genHalfAndHalf 說明**:

```python
gp.genHalfAndHalf(pset=pset, min_=2, max_=6)
```

- **Half**: 50% 使用 `grow` 方法（樹可以不同深度）
- **Half**: 50% 使用 `full` 方法（樹都是最大深度）
- **min_=2**: 最小深度 2 層
- **max_=6**: 最大深度 6 層

**範例生成的樹**:

```
深度 2 (grow):
  gt(ARG0, ARG1)

深度 4 (grow):
  and(gt(SMA(ARG0, 20), ARG0), lt(RSI(ARG0, 14), 30))

深度 6 (full):
  or(
    and(
      gt(add(ARG0, ARG1), mul(ARG0, 2)),
      lt(RSI(ARG0, 14), 70)
    ),
    gt(ARG0, SMA(ARG0, 50))
  )
```

---

### **步驟 4: 註冊演化算子** (L114-119)

```python
# Operator registration
toolbox.register("evaluate", backtester.evaluate)
toolbox.register("select", ranked_selection)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
```

#### **L115: evaluate - 評估算子**

```python
toolbox.register("evaluate", backtester.evaluate)
```

**作用**: 計算個體的 fitness（excess return）

**調用流程**:
```
toolbox.evaluate(individual)
  → backtester.evaluate(individual)
    → 生成交易信號
    → 向量化回測
    → 計算 GP return
    → 計算 B&H return
    → 返回 excess return
```

#### **L116: select - 選擇算子**

```python
toolbox.register("select", ranked_selection)
```

**作用**: 使用自定義的 Ranked Selection + SUS

#### **L117: mate - 交配算子**

```python
toolbox.register("mate", gp.cxOnePoint)
```

**作用**: 單點交叉（One-Point Crossover）

**範例**:
```
父代 1: and(gt(ARG0, 100), lt(RSI(ARG0, 14), 30))
父代 2: or(gt(SMA(ARG0, 20), ARG0), gt(ARG1, 1000))

隨機選擇交叉點:
  父代 1 的子樹: RSI(ARG0, 14)
  父代 2 的子樹: SMA(ARG0, 20)

交換後:
子代 1: and(gt(ARG0, 100), lt(SMA(ARG0, 20), 30))
子代 2: or(gt(RSI(ARG0, 14), ARG0), gt(ARG1, 1000))
```

#### **L118-119: mutate - 變異算子**

```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
```

**作用**: 均勻變異（Uniform Mutation）

**流程**:
1. 隨機選擇一個節點
2. 用 `toolbox.expr_mut()` 生成新子樹（深度 0-2）
3. 替換選中的節點

**範例**:
```
原始: and(gt(ARG0, 100), lt(RSI(ARG0, 14), 30))

隨機選擇變異點: RSI(ARG0, 14)

生成新子樹: SMA(ARG0, 50)

變異後: and(gt(ARG0, 100), lt(SMA(ARG0, 50), 30))
```

---

### **步驟 5: 添加大小限制** (L121-123)

```python
# Decorators for size limit
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
```

**作用**: 限制 GP 樹的最大深度為 17 層

**為什麼需要？**
- **Bloat 問題**: GP 樹會無限增長（交配和變異傾向於增加深度）
- **計算效率**: 過深的樹計算慢
- **過擬合**: 過複雜的規則容易過擬合
- **可解釋性**: 保持策略的可理解性

**如何工作？**
```python
# 交配前
child1.height = 10
child2.height = 12

# 交配
toolbox.mate(child1, child2)

# 交配後
child1.height = 18  # 超過限制！
child2.height = 15  # 正常

# staticLimit 的處理
if child1.height > 17:
    # 拒絕這次交配，恢復原狀
    child1 = original_child1
    child2 = original_child2
```

---

## 🌱 3.2 初始化階段 (L125-150)

### **步驟 1: 創建初始族群** (L126)

```python
pop = toolbox.population(n=population_size)
```

**結果**:
```python
pop = [
    Individual(gt(ARG0, ARG1)),
    Individual(and(lt(RSI(ARG0, 14), 30), gt(ARG0, 100))),
    Individual(or(gt(SMA(ARG0, 20), ARG0), V_TRUE)),
    ...
    # 總共 500 個隨機生成的交易規則
]
```

---

### **步驟 2: 創建 Hall of Fame** (L127)

```python
hof = tools.HallOfFame(1)
```

**作用**: 保存演化過程中最好的 1 個個體

**特性**:
- 即使該個體在後續演化中被淘汰，仍會保留
- 確保不會丟失歷史最佳解

---

### **步驟 3: 配置統計** (L129-133)

```python
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
```

**作用**: 定義每一代要記錄的統計量

**統計量說明**:
- `avg`: 平均 fitness（族群整體水平）
- `std`: 標準差（族群多樣性）
- `min`: 最小 fitness（最差個體）
- `max`: 最大 fitness（最佳個體）

---

### **步驟 4: 創建日誌** (L139-140)

```python
logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields
```

**logbook.header**:
```python
['gen', 'nevals', 'avg', 'std', 'min', 'max']
```

---

### **步驟 5: 評估初始族群** (L142-146)

```python
print("Evaluating initial population...")
fitnesses = list(tqdm(toolbox.map(toolbox.evaluate, pop), total=len(pop), desc="Initial Evaluation"))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
```

**流程**:
```
pop = [ind0, ind1, ind2, ..., ind499]
  ↓
toolbox.map(toolbox.evaluate, pop)
  → [evaluate(ind0), evaluate(ind1), ..., evaluate(ind499)]
  → [fit0, fit1, fit2, ..., fit499]
  ↓
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
  ↓
pop = [
    ind0(fitness=5000),
    ind1(fitness=12000),
    ind2(fitness=-3000),
    ...
]
```

**輸出**:
```
Evaluating initial population...
Initial Evaluation: 100%|████████████| 500/500 [00:05<00:00, 95.23it/s]
```

---

### **步驟 6: 記錄第 0 代** (L148-150)

```python
record = stats.compile(pop)
logbook.record(gen=0, nevals=len(pop), **record)
print(logbook.stream)
```

**record 內容**:
```python
record = {
    'avg': 5178.93,
    'std': 6129.27,
    'min': -50000.00,
    'max': 14109.3
}
```

**輸出**:
```
gen     nevals  avg         std         min         max    
0       500     5178.93     6129.27     -50000.00   14109.3
```

---

## 🔄 3.3 演化循環 (L153-192)

### **主循環結構**

```python
for gen in (pbar := trange(1, n_generations + 1, desc="Generation")):
    # 1. 選擇 (L155-156)
    # 2. 交配 (L159-163)
    # 3. 變異 (L165-168)
    # 4. 評估 (L171-179)
    # 5. 替換 (L183)
    # 6. 更新 HOF (L186)
    # 7. 記錄統計 (L189-192)
```

---

### **步驟 1: 選擇** (L155-156)

```python
offspring = toolbox.select(pop, len(pop))
offspring = list(map(toolbox.clone, offspring))
```

#### **L155: 選擇**

```python
offspring = toolbox.select(pop, len(pop))
```

**調用流程**:
```
toolbox.select(pop, 500)
  → ranked_selection(pop, 500)
    → 排序
    → 分配 rank_fitness
    → SUS 選擇
    → 返回 500 個選中的個體
```

**offspring 結構**:
```python
offspring = [
    ref_to_ind1,  # 可能是 pop 中的第 1 名
    ref_to_ind3,  # 可能是 pop 中的第 3 名
    ref_to_ind1,  # 同一個個體可能被選中多次！
    ref_to_ind5,
    ...
    # 總共 500 個，但可能有重複
]
```

**重要特性**:
- ✅ 長度相同: `len(offspring) == len(pop) == 500`
- ✅ 有重複: 優秀個體可能被選中多次
- ✅ 引用相同: `offspring[0] is pop[3]` 可能為 True

#### **L156: 克隆**

```python
offspring = list(map(toolbox.clone, offspring))
```

**為什麼必須 clone？**

```python
# 不 clone 的問題
offspring = [ref_to_ind1, ref_to_ind3, ref_to_ind1, ...]
                ↓              ↓              ↓
              pop[1]        pop[3]        pop[1]  (同一個對象！)

# 修改 offspring[0]
toolbox.mate(offspring[0], offspring[1])
# 問題 1: pop[1] 也被修改了！
# 問題 2: offspring[2] 也被修改了！（因為指向同一個對象）

# clone 之後
offspring = [copy_of_ind1, copy_of_ind3, copy_of_ind1, ...]
                ↓              ↓              ↓
            新對象1         新對象2         新對象3  (獨立對象)

# 修改 offspring[0]
toolbox.mate(offspring[0], offspring[1])
# ✅ pop[1] 不受影響
# ✅ offspring[2] 不受影響
```

---

### **步驟 2: 交配** (L159-163)

```python
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < crossover_prob:
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values
```

#### **配對邏輯**

```python
offspring[::2]   # 偶數索引: [0, 2, 4, 6, ..., 498]
offspring[1::2]  # 奇數索引: [1, 3, 5, 7, ..., 499]

zip(offspring[::2], offspring[1::2])
# 配對: (0,1), (2,3), (4,5), ..., (498,499)
```

**具體範例**:
```python
offspring = [ind0, ind1, ind2, ind3, ind4, ind5, ..., ind499]

迭代 1: child1 = ind0, child2 = ind1
  → random.random() = 0.45 < 0.6 → 交配 ✅
  → toolbox.mate(ind0, ind1)
  → del ind0.fitness.values
  → del ind1.fitness.values

迭代 2: child1 = ind2, child2 = ind3
  → random.random() = 0.75 > 0.6 → 不交配 ❌

迭代 3: child1 = ind4, child2 = ind5
  → random.random() = 0.23 < 0.6 → 交配 ✅
  → toolbox.mate(ind4, ind5)
  → del ind4.fitness.values
  → del ind5.fitness.values

...

總共 250 對，期望約 150 對會交配（60%）
```

#### **為什麼刪除 fitness？**

```python
del child1.fitness.values
del child2.fitness.values
```

**原因**:
- 交配後的個體已經改變，原來的 fitness 不再有效
- 刪除 fitness 標記為「無效」，需要重新評估
- DEAP 通過 `ind.fitness.valid` 檢查是否有效

```python
# 交配前
child1.fitness.values = (5000,)
child1.fitness.valid = True

# 交配
toolbox.mate(child1, child2)
# child1 的 GP 樹已經改變

# 刪除 fitness
del child1.fitness.values
# child1.fitness.valid = False  (自動設置)

# 後續會重新評估
```

---

### **步驟 3: 變異** (L165-168)

```python
for mutant in offspring:
    if random.random() < mutation_prob:
        toolbox.mutate(mutant)
        del mutant.fitness.values
```

**流程**:
```python
offspring 中的每個個體:
  → 5% 機率發生變異
  → 變異後刪除 fitness

範例:
  ind0: random() = 0.03 < 0.05 → 變異 ✅
  ind1: random() = 0.87 > 0.05 → 不變異 ❌
  ind2: random() = 0.02 < 0.05 → 變異 ✅
  ...
  
總共 500 個個體，期望約 25 個會變異（5%）
```

---

### **步驟 4: 評估無效個體** (L171-179)

```python
# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    # Final safeguard before assigning fitness
    if not np.isfinite(fit[0]) or fit[0] > 1e12:
        ind.fitness.values = (-100000.0,)  # Penalty fitness
    else:
        ind.fitness.values = fit
```

#### **L171: 找出無效個體**

```python
invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
```

**哪些個體無效？**
- 被交配過的個體（約 300 個）
- 被變異過的個體（約 25 個）
- 總共約 300-325 個（有重疊）

**為什麼只評估無效個體？**
- **效率**: 節省 35-40% 的計算時間
- **正確性**: 未被修改的個體 fitness 仍然有效

#### **L172: 評估**

```python
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
```

#### **L173-179: 分配 fitness（含安全檢查）**

```python
for ind, fit in zip(invalid_ind, fitnesses):
    if not np.isfinite(fit[0]) or fit[0] > 1e12:
        ind.fitness.values = (-100000.0,)  # Penalty fitness
    else:
        ind.fitness.values = fit
```

**安全檢查**:
```python
# 檢查 1: NaN 或 Inf
if not np.isfinite(fit[0]):
    # 可能原因: 除以零、對數負數等
    ind.fitness.values = (-100000.0,)

# 檢查 2: 異常大的值
if fit[0] > 1e12:
    # 可能原因: 計算錯誤、溢出
    ind.fitness.values = (-100000.0,)

# 正常情況
else:
    ind.fitness.values = fit
```

**為什麼用 -100000？**
- 確保這些個體在下一代被淘汰
- 比任何正常的負 fitness 都小

---

### **步驟 5: 替換族群** (L183)

```python
pop[:] = offspring
```

**替換策略**: Generational Replacement（世代替換）

```python
# 替換前
pop = [old_ind0, old_ind1, ..., old_ind499]

# 替換後
pop = [new_ind0, new_ind1, ..., new_ind499]

# 舊族群完全被新族群替換
# 不保留任何舊個體（除了 HOF）
```

**為什麼用 `pop[:]` 而不是 `pop =`？**
```python
# 錯誤方式
pop = offspring  # 只是改變引用，不修改原列表

# 正確方式
pop[:] = offspring  # 修改原列表的內容
```

---

### **步驟 6: 更新 Hall of Fame** (L186)

```python
hof.update(pop)
```

**作用**: 更新歷史最佳個體

```python
# hof.update() 的邏輯
if max(pop, key=fitness) > hof[0]:
    hof[0] = max(pop, key=fitness)
```

**範例**:
```
第 10 代: hof[0].fitness = 15000
第 11 代: max(pop).fitness = 18000 → 更新 hof[0]
第 12 代: max(pop).fitness = 17000 → 不更新（沒有更好）
第 13 代: max(pop).fitness = 20000 → 更新 hof[0]
...
```

---

### **步驟 7: 記錄統計** (L189-192)

```python
record = stats.compile(pop)
logbook.record(gen=gen, nevals=len(invalid_ind), **record)
pbar.set_description(f"Gen {gen} | Avg: {record['avg']:.2f} | Best: {record['max']:.2f}")
```

**輸出**:
```
Gen 25 | Avg: 8234.56 | Best: 18500.23: 50%|█████     | 25/50
```

**logbook 內容**:
```python
logbook = [
    {'gen': 0, 'nevals': 500, 'avg': 5178.93, 'std': 6129.27, 'min': -50000, 'max': 14109.3},
    {'gen': 1, 'nevals': 312, 'avg': 6234.56, 'std': 5432.10, 'min': -30000, 'max': 18500.2},
    {'gen': 2, 'nevals': 298, 'avg': 7123.45, 'std': 4876.32, 'min': -20000, 'max': 20123.5},
    ...
    {'gen': 50, 'nevals': 287, 'avg': 11197.80, 'std': 3456.78, 'min': 2000, 'max': 25000.5}
]
```

---

## 📊 完整演化流程

```
開始
  ↓
┌─────────────────────────────────────┐
│ 設置階段 (L81-123)                  │
│ ✅ 創建 Toolbox                     │
│ ✅ 檢測數據類型                     │
│ ✅ 創建回測引擎                     │
│ ✅ 註冊生成器                       │
│ ✅ 註冊演化算子                     │
│ ✅ 添加大小限制                     │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 初始化階段 (L125-150)               │
│ ✅ 創建初始族群 (500 個)            │
│ ✅ 創建 Hall of Fame                │
│ ✅ 配置統計                         │
│ ✅ 創建日誌                         │
│ ✅ 評估初始族群                     │
│ ✅ 記錄第 0 代                      │
└─────────────────────────────────────┘
  ↓
┌─────────────────────────────────────┐
│ 演化循環 (L153-192) × 50 代         │
│                                     │
│ 第 gen 代:                          │
│   1. 選擇 (Ranked + SUS)            │
│      └─ 選出 500 個個體             │
│   2. 克隆                           │
│      └─ 深拷貝避免污染              │
│   3. 交配 (60% 機率)                │
│      └─ 250 對，約 150 對交配       │
│   4. 變異 (5% 機率)                 │
│      └─ 500 個，約 25 個變異        │
│   5. 評估無效個體                   │
│      └─ 約 300-325 個需要評估       │
│   6. 替換族群                       │
│      └─ 新族群完全替換舊族群        │
│   7. 更新 HOF                       │
│      └─ 保存歷史最佳                │
│   8. 記錄統計                       │
│      └─ avg, std, min, max          │
│                                     │
└─────────────────────────────────────┘
  ↓
返回 (pop, logbook, hof)
  ↓
結束
```

---

## 🎯 關鍵設計決策

### **1. 為什麼用 Ranked Selection + SUS？**

**問題**: 直接用原始 fitness 選擇
- 超級優秀個體壟斷繁殖機會
- 族群多樣性快速下降
- 容易過早收斂到局部最優

**解決**: Ranked Selection
- 只看排名，不看絕對值
- 優秀個體仍有優勢，但不會壟斷
- 保持族群多樣性

**為什麼用 SUS 而不是輪盤賭？**
- SUS 選擇偏差更小
- 更接近期望的選擇比例
- 減少隨機性帶來的不穩定

---

### **2. 為什麼用 Generational Replacement？**

**優點**:
- 簡單、易實現
- 演化速度快
- 容易理解和調試

**缺點**:
- 可能丟失優秀個體

**解決**: Hall of Fame
- 保存歷史最佳個體
- 確保不會丟失最優解

---

### **3. 為什麼只評估無效個體？**

**效率考量**:
```
總個體: 500
交配影響: ~300 個
變異影響: ~25 個
總無效: ~325 個

節省計算: (500 - 325) / 500 = 35%
```

**正確性**:
- 未被修改的個體 fitness 仍然有效
- 不需要重新評估

---

### **4. 為什麼需要安全檢查？**

**可能的異常情況**:
```python
# NaN
0 / 0 → NaN
log(-1) → NaN

# Inf
1 / 0 → Inf
exp(1000) → Inf

# 異常大的值
某個計算錯誤 → 1e15
```

**影響**:
- NaN 會污染後續計算
- Inf 會導致選擇失敗
- 異常值會誤導演化方向

**解決**: 分配懲罰 fitness
```python
if not np.isfinite(fit[0]) or fit[0] > 1e12:
    ind.fitness.values = (-100000.0,)
```

---

### **5. 為什麼深度限制是 17？**

**Bloat 問題**:
- GP 樹會無限增長
- 交配和變異傾向於增加深度
- 最終導致計算緩慢、過擬合

**深度 17 的考量**:
- 足夠表達複雜策略
- 不會過度膨脹
- 保持可解釋性

**範例**:
```
深度 5: and(gt(SMA(ARG0, 20), ARG0), lt(RSI(ARG0, 14), 30))
  → 可理解 ✅

深度 20: and(or(and(gt(...), lt(...)), or(...)), and(or(...), ...))
  → 難以理解 ❌
```

---

## ❓ 常見問題 FAQ

### **Q1: 為什麼 offspring 可能有重複個體？**

**A**: 因為 SUS 選擇允許同一個優秀個體被選中多次。

```python
# 選擇前
pop = [ind1(fit=25000), ind2(fit=20000), ..., ind500(fit=100)]

# SUS 選擇後
offspring = [ind1, ind3, ind1, ind5, ind1, ...]
             ↑         ↑         ↑
             同一個優秀個體被選中 3 次
```

這是合理的，因為優秀個體應該有更多繁殖機會。

---

### **Q2: 為什麼要暫存和恢復 fitness？**

**A**: 因為 DEAP 的 SUS 函數從 `ind.fitness.values` 讀取 fitness，但我們想用 rank_fitness 選擇，同時保留原始 fitness 用於日誌記錄。

```python
# 暫存原始 fitness
original_fitnesses = [ind.fitness.values for ind in sorted_individuals]

# 替換為 rank_fitness（用於選擇）
for ind in sorted_individuals:
    ind.fitness.values = (ind.rank_fitness,)

# SUS 選擇
chosen = tools.selStochasticUniversalSampling(sorted_individuals, k)

# 恢復原始 fitness（用於日誌）
for ind, fit in zip(sorted_individuals, original_fitnesses):
    ind.fitness.values = fit
```

---

### **Q3: 為什麼交配和變異後要刪除 fitness？**

**A**: 因為個體已經改變，原來的 fitness 不再有效。

```python
# 交配前
child1 = gt(ARG0, 100)
child1.fitness.values = (5000,)

# 交配
toolbox.mate(child1, child2)
# child1 現在變成: and(gt(ARG0, 100), lt(RSI(ARG0, 14), 30))

# 問題: child1 的 fitness 還是 5000，但這是舊規則的 fitness！

# 解決: 刪除 fitness
del child1.fitness.values
# child1.fitness.valid = False

# 後續重新評估
child1.fitness.values = evaluate(child1)  # 新的 fitness
```

---

### **Q4: 為什麼用 `pop[:] = offspring` 而不是 `pop = offspring`？**

**A**: 因為 `pop[:] = offspring` 修改原列表的內容，而 `pop = offspring` 只是改變引用。

```python
# 錯誤方式
original_pop = pop
pop = offspring
# original_pop 仍然指向舊族群
# 其他引用 pop 的地方不會更新

# 正確方式
original_pop = pop
pop[:] = offspring
# original_pop 也更新了（因為是同一個列表）
# 所有引用 pop 的地方都更新了
```

---

### **Q5: Hall of Fame 如何工作？**

**A**: HOF 保存演化過程中最好的個體，即使該個體在後續被淘汰也會保留。

```python
hof = tools.HallOfFame(1)

# 第 0 代
hof.update(pop)  # hof[0] = best_of_gen0 (fitness=14109)

# 第 1 代
hof.update(pop)  # hof[0] = best_of_gen1 (fitness=18500)

# 第 2 代
hof.update(pop)  # hof[0] 不變（gen2 的最佳 < 18500）

# 第 50 代
hof.update(pop)  # hof[0] = best_of_gen50 (fitness=25000)

# 最終
hof[0]  # 整個演化過程中最好的個體
```

---

### **Q6: 為什麼只評估約 300-325 個個體，而不是全部 500 個？**

**A**: 因為只有被交配或變異的個體需要重新評估。

```python
# 交配影響
250 對 × 60% 交配率 = 150 對 = 300 個個體

# 變異影響
500 個 × 5% 變異率 = 25 個個體

# 重疊
有些個體既被交配又被變異

# 總計
約 300-325 個個體需要評估
約 175-200 個個體不需要評估（節省 35-40% 計算）
```

---

### **Q7: 如何判斷演化是否成功？**

**A**: 觀察 logbook 的統計數據：

```python
# 成功的演化
gen     avg         max
0       5000        14000
10      8000        18000   ← avg 上升
20      10000       22000   ← max 上升
30      11500       24000
40      12000       25000
50      12500       25500   ← 收斂

# 失敗的演化（過早收斂）
gen     avg         max
0       5000        14000
10      13000       14000   ← avg 快速上升到 max
20      13500       14000   ← 停滯
30      13500       14000
40      13500       14000
50      13500       14000   ← 沒有進步

# 失敗的演化（不穩定）
gen     avg         max
0       5000        14000
10      3000        12000   ← 下降
20      8000        18000   ← 上升
30      4000        15000   ← 下降
40      9000        20000   ← 不穩定
```

---

### **Q8: 如何調整演化參數？**

**參數建議**:

| 參數 | 預設值 | 建議範圍 | 影響 |
|------|--------|----------|------|
| `population_size` | 500 | 100-1000 | 越大越好，但計算慢 |
| `n_generations` | 50 | 30-100 | 越多越好，但耗時長 |
| `crossover_prob` | 0.6 | 0.5-0.8 | 太高會破壞好個體 |
| `mutation_prob` | 0.05 | 0.01-0.1 | 太高會破壞收斂 |
| `max_rank_fitness` | 1.8 | 1.5-2.0 | 選擇壓力 |
| `min_rank_fitness` | 0.2 | 0.1-0.5 | 選擇壓力 |

**調整建議**:
- **探索不足**: 增加 `mutation_prob`、減小 `max_rank_fitness`
- **收斂太慢**: 增加 `crossover_prob`、增大 `max_rank_fitness`
- **過早收斂**: 增加 `population_size`、減小選擇壓力
- **計算太慢**: 減小 `population_size`、減少 `n_generations`

---

## 📝 Review Checklist

完成 review 後，確保你能回答：

### **ranked_selection()**
- [ ] Ranked Selection 的公式是什麼？
- [ ] 為什麼要暫存和恢復 fitness？
- [ ] SUS 和輪盤賭的區別？
- [ ] 如何調整選擇壓力？

### **run_evolution() - 設置階段**
- [ ] 如何檢測數據類型？
- [ ] 新舊數據結構的區別？
- [ ] genHalfAndHalf 如何工作？
- [ ] 為什麼需要深度限制？

### **run_evolution() - 初始化階段**
- [ ] 初始族群如何生成？
- [ ] Hall of Fame 的作用？
- [ ] 統計量的意義？

### **run_evolution() - 演化循環**
- [ ] 為什麼要 clone offspring？
- [ ] 交配和變異如何配對？
- [ ] 為什麼只評估無效個體？
- [ ] 如何處理異常 fitness？
- [ ] Generational Replacement 的優缺點？

### **整體理解**
- [ ] 完整演化流程？
- [ ] 各個參數如何影響演化？
- [ ] 如何判斷演化是否成功？
- [ ] 如何調試演化問題？

---

## 🎓 總結

`engine.py` 是整個專案的核心，實現了完整的 GP 演化算法：

1. **自定義選擇**: Ranked Selection + SUS，保持族群多樣性
2. **智能檢測**: 自動識別數據類型和結構
3. **安全機制**: 深度限制、異常檢查、懲罰 fitness
4. **效率優化**: 只評估無效個體，節省 35-40% 計算
5. **完整記錄**: 詳細的統計和日誌，方便分析

理解這個文件，你就理解了整個演化過程！

---

**文檔版本**: 1.0  
**最後更新**: 2025-10-06  
**作者**: Cascade AI Assistant
