# 深度限制實作對比分析

## 🔍 問題：為什麼 `run_all_experiments.py` 不會超過深度限制？

你的疑問非常好！讓我們對比兩個實作：

---

## 📊 對比總覽

| 項目 | `run_all_experiments.py` (main.py) | `run_portfolio_experiment.py` |
|------|-----------------------------------|------------------------------|
| **使用的 engine** | `gp_quant/evolution/engine.py` | 自己實作演化循環 |
| **有 staticLimit** | ✅ **有** (第 208-209 行) | ❌ **沒有** |
| **深度違規** | ❌ 無違規 | ✅ 76% 違規率 |
| **最大深度** | ≤ 17 | 69 |

---

## 🔑 關鍵差異

### 1. `run_all_experiments.py` → `main.py` → `engine.py`

**檔案：`gp_quant/evolution/engine.py`**

```python
# 第 195-209 行
def run_evolution(...):
    # ... 省略其他程式碼 ...
    
    # Operator registration
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # ✅ 關鍵：有深度限制裝飾器！
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
```

**演化循環（第 250-261 行）：**

```python
# ✅ 正確：接收 staticLimit 返回的個體
for i in range(0, len(offspring) - 1, 2):
    if random.random() < crossover_prob:
        # 接收返回值！
        offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
        del offspring[i].fitness.values
        del offspring[i+1].fitness.values

for i in range(len(offspring)):
    if random.random() < mutation_prob:
        # 接收返回值！
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values
```

### 2. `run_portfolio_experiment.py`

**檔案：`run_portfolio_experiment.py`**

```python
# 第 215-223 行
# 註冊 GP 操作
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 註冊演化操作
toolbox.register("select", tools.selTournament, tournsize=CONFIG['tournament_size'])
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)

# ❌ 問題：沒有 staticLimit 裝飾器！
```

**演化循環（第 580-600 行）：**

```python
# ❌ 錯誤：沒有接收返回值（但這裡沒有 staticLimit 所以沒差）
# Crossover
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CONFIG['crossover_prob']:
        toolbox.mate(child1, child2)  # 沒有接收返回值
        del child1.fitness.values
        del child2.fitness.values

# Mutation
for mutant in offspring:
    if random.random() < CONFIG['mutation_prob']:
        toolbox.mutate(mutant)  # 沒有接收返回值
        del mutant.fitness.values
```

---

## 🎯 核心問題

### `run_portfolio_experiment.py` 缺少兩個關鍵要素：

1. **沒有 `staticLimit` 裝飾器**
   ```python
   # ❌ 缺少這兩行
   toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
   toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
   ```

2. **沒有正確接收返回值**（雖然沒有 staticLimit 時這不是問題）
   ```python
   # ❌ 當前寫法
   toolbox.mate(child1, child2)
   
   # ✅ 應該寫成（如果有 staticLimit）
   child1, child2 = toolbox.mate(child1, child2)
   ```

---

## 📋 `staticLimit` 的工作原理

### 裝飾器模式

```python
# 原始函數
def mate(ind1, ind2):
    # 執行 crossover
    return ind1, ind2

# 加上 staticLimit 裝飾器後
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# 實際執行時
def decorated_mate(ind1, ind2):
    # 1. 執行原始 crossover
    result1, result2 = original_mate(ind1, ind2)
    
    # 2. 檢查深度
    if result1.height > 17:
        result1 = ind1  # 超限，返回原個體
    if result2.height > 17:
        result2 = ind2  # 超限，返回原個體
    
    # 3. 返回結果
    return result1, result2
```

### 關於你的疑問

你在文檔中提到：
> "如果超過限制，**返回原始個體**（不應用變異 => 這部分應該改成應該重新嘗試讓他能夠合規變異）"

**DEAP 的設計哲學：**

DEAP 的 `staticLimit` **確實是返回原個體**，而不是重新嘗試。這是有原因的：

#### 為什麼不重新嘗試？

1. **性能考量**
   - 重新嘗試可能需要多次迭代
   - 在某些情況下可能永遠找不到合規的變異
   - 會大幅增加計算時間

2. **演化壓力**
   - 返回原個體相當於「拒絕這次變異」
   - 這個個體仍然會參與選擇
   - 如果它的 fitness 好，會被保留；如果不好，會被淘汰
   - 這是一種**自然的演化壓力**

3. **實務效果**
   - 大部分 crossover/mutation 不會超限
   - 只有少數會被拒絕
   - 整體演化效果不受影響

#### 如果真的要重新嘗試

如果你堅持要重新嘗試，可以這樣實作：

```python
def retry_mate(ind1, ind2, max_retries=3):
    """帶重試機制的 crossover"""
    for _ in range(max_retries):
        # 執行 crossover
        child1, child2 = gp.cxOnePoint(toolbox.clone(ind1), toolbox.clone(ind2))
        
        # 檢查深度
        if child1.height <= 17 and child2.height <= 17:
            return child1, child2  # 成功
    
    # 重試失敗，返回原個體
    return ind1, ind2

toolbox.register("mate", retry_mate)
```

**但這不是推薦做法**，因為：
- ❌ 增加計算開銷
- ❌ 可能陷入無限循環
- ❌ DEAP 社群不推薦
- ✅ `staticLimit` 的「返回原個體」已經足夠有效

---

## 🔧 修復 `run_portfolio_experiment.py`

### 方案 A：完全對齊 `engine.py`（推薦）

```python
import operator
from deap import gp

# 定義深度限制
MAX_DEPTH_EVOLVE = 17

# 註冊操作
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # 改這裡
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # 改這裡

# 添加深度限制裝飾器
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_EVOLVE))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_DEPTH_EVOLVE))

# 演化循環中接收返回值
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CONFIG['crossover_prob']:
        child1, child2 = toolbox.mate(child1, child2)  # 接收返回值
        del child1.fitness.values
        del child2.fitness.values

for i, mutant in enumerate(offspring):
    if random.random() < CONFIG['mutation_prob']:
        offspring[i], = toolbox.mutate(mutant)  # 接收返回值
        del offspring[i].fitness.values
```

### 方案 B：直接使用 `engine.py`

更好的做法是**重構 `run_portfolio_experiment.py`**，讓它也使用 `gp_quant/evolution/engine.py`，而不是自己實作演化循環。

---

## 📊 驗證結果

### `engine.py` 的深度檢查結果

運行 `check_depth_limits.py` 檢查 `experiments_results/`：

```bash
python check_depth_limits.py
```

**結果：**
- ✅ 所有 generation 的 max_depth ≤ 17
- ✅ 0% 違規率
- ✅ 平均深度穩定在 5-10 之間

### `run_portfolio_experiment.py` 的深度檢查結果

運行 `check_portfolio_depth.py` 檢查 `portfolio_experiment_results/`：

```bash
python check_portfolio_depth.py
```

**結果：**
- ❌ 76% 違規率（實驗 133445）
- ❌ 最大深度 69
- ❌ 平均深度持續增長到 20.02

---

## ✅ 總結

### 為什麼 `run_all_experiments.py` 不會超限？

因為它使用的 `gp_quant/evolution/engine.py` **已經正確實作了深度限制**：

1. ✅ 有 `staticLimit` 裝飾器
2. ✅ 正確接收返回值
3. ✅ 使用較小的 mutation 子樹（max_=2）

### 為什麼 `run_portfolio_experiment.py` 會超限？

因為它**自己實作演化循環**，但：

1. ❌ 沒有 `staticLimit` 裝飾器
2. ❌ 沒有接收返回值（雖然沒有 staticLimit 時這不重要）
3. ❌ 使用較大的 mutation 子樹（max_=3）

### 解決方案

**選項 1：修復 `run_portfolio_experiment.py`**
- 添加 `staticLimit` 裝飾器
- 修改演化循環接收返回值
- 改用較小的 mutation 子樹

**選項 2：重構使用 `engine.py`**（更好）
- 讓 `run_portfolio_experiment.py` 也使用 `engine.py`
- 避免重複實作
- 保持一致性

---

## 🎯 下一步

1. **確認方案**：選擇選項 1 還是選項 2？
2. **實作修改**：根據選擇的方案修改程式碼
3. **測試驗證**：運行實驗並檢查深度
4. **對比結果**：確認 0% 違規率

---

## 📚 參考

- **正確實作**：`gp_quant/evolution/engine.py` 第 208-209 行
- **問題實作**：`run_portfolio_experiment.py` 第 215-223 行
- **DEAP 文檔**：https://deap.readthedocs.io/en/master/api/gp.html#deap.gp.staticLimit
