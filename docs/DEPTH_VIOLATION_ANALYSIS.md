# GP 深度超限問題分析報告

## 📊 問題概述

根據 `portfolio_depth_violations.csv` 的分析結果，發現以下問題：

### 違規統計

| 實驗 | 違規 Generation 數 | 最大深度 | 平均深度 |
|------|-------------------|---------|---------|
| portfolio_exp_sharpe_20251023_133445 | 39/50 (78%) | **69** | 20.02 |
| portfolio_exp_sharpe_20251023_160709 | 3/31 (9.7%) | 23 | 7.58 |
| portfolio_exp_sharpe_20251023_161559 | 5/24 (20.8%) | 23 | 10.14 |

### 關鍵發現

1. **深度爆炸性增長**
   - portfolio_exp_sharpe_20251023_133445 從 Gen 12 的深度 18 增長到 Gen 47 的深度 **69**
   - 平均深度從 Gen 1 的 2.0 增長到 Gen 50 的 **20.02**
   - 這是一個**指數級增長**的趨勢

2. **違規開始時間點**
   - 大部分違規從 Gen 12-20 開始出現
   - 一旦開始違規，深度會持續增長，無法自我修正

3. **族群大小的影響**
   - population_size=5000 的實驗違規最嚴重（78%）
   - population_size=500 的實驗違規較輕（9.7%-20.8%）

---

## 🔍 根本原因分析

### 1. **缺少深度限制機制**

當前的 GP 配置：

```python
# 初始化
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# Crossover
toolbox.register("mate", gp.cxOnePoint)

# Mutation
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
```

**問題：**
- ✅ 初始族群深度控制正確（max_=3，實際最大深度 3）
- ❌ **Crossover 沒有深度限制**：`gp.cxOnePoint` 可以產生任意深度的子樹
- ❌ **Mutation 沒有深度限制**：`gp.mutUniform` 使用 `toolbox.expr` 生成新子樹，但交叉後的樹可能已經很深

### 2. **Crossover 的深度增長機制**

`gp.cxOnePoint` 的工作原理：
1. 在兩個父代中隨機選擇一個交叉點
2. 交換兩個子樹
3. **沒有檢查結果深度**

**示例：**
```
父代 1 (深度 10):        父代 2 (深度 8):
      +                        *
     / \                      / \
    A   B (深度 9)           C   D (深度 7)

交叉後：
子代 1 (深度 可能 > 17):
      +
     / \
    A   D (深度 7)  ← 如果 A 本身深度就很深，結果會超過限制
```

### 3. **Mutation 的深度累積**

`gp.mutUniform` 的問題：
- 它會用 `toolbox.expr` 生成的新子樹替換某個節點
- `toolbox.expr` 生成的子樹深度最大為 3
- 但如果替換的節點本身在樹的深層，總深度 = 節點深度 + 3，可能超過 17

### 4. **選擇壓力導致深度增長**

- 更複雜的樹（深度更深）可能有更好的 fitness
- Tournament selection 會偏好這些複雜的樹
- 沒有深度懲罰機制，導致深度持續增長

---

## 🎯 改善方案

### 方案 A：使用 DEAP 內建的深度限制裝飾器（推薦）

DEAP 提供了 `gp.staticLimit` 裝飾器來限制深度。

#### 優點
- ✅ 官方推薦方法
- ✅ 自動拒絕超過深度限制的操作
- ✅ 不需要修改核心邏輯
- ✅ 性能開銷小

#### 實作方式

```python
from deap import gp

# 定義深度限制
MAX_DEPTH = 17

# 使用裝飾器包裝 mate 和 mutate
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# 應用深度限制
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH))
```

**工作原理：**
1. 執行 crossover/mutation
2. 檢查結果的深度
3. 如果超過限制，**返回原始個體**（不應用變異）
4. 這樣可以保證族群中所有個體都符合深度限制

---

### 方案 B：使用深度感知的 Crossover/Mutation

使用 DEAP 提供的深度限制版本：

```python
# 使用深度限制的 crossover
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)

# 使用深度限制的 mutation
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 再加上 staticLimit 雙重保險
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=17))
```

---

### 方案 C：添加深度懲罰到 Fitness

在 fitness 計算中加入深度懲罰：

```python
def evaluate_individual(individual):
    try:
        # 原始 fitness 計算
        fitness_value = calculate_fitness(individual)
        
        # 深度懲罰
        depth = individual.height
        if depth > 17:
            # 超過深度限制，嚴重懲罰
            penalty = (depth - 17) * 0.1  # 每超過 1 層，懲罰 0.1
            fitness_value -= penalty
        elif depth > 12:
            # 接近限制，輕微懲罰
            penalty = (depth - 12) * 0.01
            fitness_value -= penalty
        
        return (fitness_value,)
    except Exception as e:
        return (-1000000.0,)
```

**缺點：**
- ❌ 可能影響演化效果
- ❌ 需要調整懲罰係數
- ❌ 不能完全防止超限

---

### 方案 D：後處理修剪（不推薦）

在每個 generation 後檢查並修剪超深的樹：

```python
def prune_tree(individual, max_depth=17):
    """修剪超過深度限制的樹"""
    if individual.height <= max_depth:
        return individual
    
    # 簡單修剪：隨機選擇一個深層節點，替換為終端節點
    # ... 實作邏輯
    return individual

# 在演化循環中
for ind in population:
    if ind.height > 17:
        ind = prune_tree(ind)
```

**缺點：**
- ❌ 破壞了樹的結構
- ❌ 可能產生無效的表達式
- ❌ 需要重新評估 fitness

---

## 📋 推薦實作方案

### **最佳方案：方案 A + 部分方案 B**

結合 DEAP 的最佳實踐：

```python
import operator
from deap import gp

# 1. 定義深度限制常數
MAX_DEPTH_INIT = 6   # 初始族群
MAX_DEPTH_EVOLVE = 17  # 演化過程

# 2. 初始化（保持不變）
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. 註冊演化操作
toolbox.register("select", tools.selTournament, tournsize=CONFIG['tournament_size'])

# 使用 cxOnePoint（標準 crossover）
toolbox.register("mate", gp.cxOnePoint)

# Mutation 使用較小的子樹
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # 生成深度 0-2 的子樹
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 4. 應用深度限制裝飾器（關鍵！）
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH_EVOLVE))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH_EVOLVE))

# 5. 編譯和評估
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_individual)
```

### 關鍵改動

1. **添加 `operator` import**
2. **定義深度限制常數**
3. **修改 mutation 的 expr 生成器**：從 `max_=3` 改為 `max_=2`
4. **添加 `staticLimit` 裝飾器**到 `mate` 和 `mutate`

---

## 🧪 驗證方案

### 1. 單元測試

創建測試腳本驗證深度限制：

```python
def test_depth_limit():
    """測試深度限制是否有效"""
    
    # 創建深度接近限制的個體
    population = toolbox.population(n=100)
    
    # 強制某些個體深度接近 17
    for ind in population[:10]:
        # 通過多次 mutation 增加深度
        for _ in range(10):
            toolbox.mutate(ind)
    
    # 執行多代演化
    for gen in range(20):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Mutation
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        population[:] = offspring
        
        # 檢查深度
        max_depth = max(ind.height for ind in population)
        avg_depth = sum(ind.height for ind in population) / len(population)
        
        print(f"Gen {gen}: max_depth={max_depth}, avg_depth={avg_depth:.2f}")
        
        # 斷言：所有個體深度 <= 17
        assert all(ind.height <= 17 for ind in population), \
            f"Found individual with depth > 17 in generation {gen}"
    
    print("✅ 深度限制測試通過！")
```

### 2. 實際實驗驗證

運行一個小規模實驗（population_size=500, generations=50）：

```bash
# 修改 run_portfolio_experiment.py 後執行
python run_portfolio_experiment.py

# 檢查深度
python check_portfolio_depth.py
```

預期結果：
- ✅ 所有 generation 的 max_depth ≤ 17
- ✅ 平均深度穩定在 5-10 之間
- ✅ 沒有違規記錄

---

## 📊 預期效果

### 修改前（當前狀態）

| Generation | Max Depth | Avg Depth | 違規 |
|-----------|-----------|-----------|------|
| 1 | 3 | 2.0 | ❌ |
| 10 | 11 | 4.85 | ❌ |
| 20 | 24 | 8.22 | ✅ |
| 30 | 39 | 10.87 | ✅ |
| 50 | 69 | 20.02 | ✅ |

### 修改後（預期）

| Generation | Max Depth | Avg Depth | 違規 |
|-----------|-----------|-----------|------|
| 1 | 3 | 2.0 | ❌ |
| 10 | 12 | 5.5 | ❌ |
| 20 | 15 | 7.0 | ❌ |
| 30 | 17 | 8.5 | ❌ |
| 50 | 17 | 9.0 | ❌ |

---

## ⚠️ 潛在影響

### 1. 演化效果

**可能的影響：**
- 限制深度可能會限制表達能力
- 某些複雜的交易策略可能無法表達

**緩解措施：**
- 深度 17 已經足夠表達複雜策略（2^17 = 131,072 個可能的節點）
- 可以通過增加 primitive set 的豐富度來補償

### 2. 收斂速度

**可能的影響：**
- 某些 crossover/mutation 會被拒絕（返回原個體）
- 可能稍微減慢收斂速度

**緩解措施：**
- 調整 crossover_prob 和 mutation_prob
- 使用更小的 mutation 子樹（max_=2）

### 3. 多樣性

**正面影響：**
- ✅ 防止族群被超深的樹主導
- ✅ 保持族群多樣性
- ✅ 避免過擬合

---

## 🎯 實作檢查清單

- [ ] 1. 在 `run_portfolio_experiment.py` 中添加 `import operator`
- [ ] 2. 定義 `MAX_DEPTH_INIT = 6` 和 `MAX_DEPTH_EVOLVE = 17`
- [ ] 3. 修改 mutation 的 expr：`toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)`
- [ ] 4. 修改 mutate 註冊：使用 `expr_mut` 而不是 `expr`
- [ ] 5. 添加 `staticLimit` 裝飾器到 `mate`
- [ ] 6. 添加 `staticLimit` 裝飾器到 `mutate`
- [ ] 7. 創建測試腳本 `test_depth_limit.py`
- [ ] 8. 運行測試驗證
- [ ] 9. 運行小規模實驗驗證
- [ ] 10. 檢查 `check_portfolio_depth.py` 確認無違規

---

## 📚 參考資料

1. **DEAP 官方文檔**
   - [Bloat Control](https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html#bloat-control)
   - [Static Limit Decorator](https://deap.readthedocs.io/en/master/api/gp.html#deap.gp.staticLimit)

2. **論文參考**
   - Koza, J. R. (1992). "Genetic Programming: On the Programming of Computers by Means of Natural Selection"
   - Poli, R., et al. (2008). "A Field Guide to Genetic Programming"

3. **最佳實踐**
   - 使用 `staticLimit` 是 DEAP 推薦的標準做法
   - 大多數 GP 研究都使用深度限制來控制 bloat

---

## ✅ 總結

### 問題根源
- **缺少深度限制機制**：crossover 和 mutation 沒有深度檢查
- **深度累積效應**：每次操作都可能增加深度，無法自我修正

### 推薦方案
- **使用 `gp.staticLimit` 裝飾器**（方案 A）
- **配合較小的 mutation 子樹**（方案 B 的一部分）

### 預期效果
- ✅ 完全消除深度違規
- ✅ 保持演化效果
- ✅ 符合論文要求

### 下一步
1. 等待您確認方案
2. 實作修改
3. 運行測試驗證
4. 重新執行實驗
