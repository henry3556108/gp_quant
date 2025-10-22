# 深度限制違規問題修復方案

## 問題根源

**演化引擎沒有正確使用 `staticLimit` 裝飾器的返回值**

### 當前錯誤實作（engine.py, lines 244-253）

```python
# Apply crossover and mutation
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < crossover_prob:
        toolbox.mate(child1, child2)  # ❌ 沒有接收返回值
        del child1.fitness.values
        del child2.fitness.values

for mutant in offspring:
    if random.random() < mutation_prob:
        toolbox.mutate(mutant)  # ❌ 沒有接收返回值
        del mutant.fitness.values
```

### staticLimit 的實際行為

```python
def staticLimit(key, max_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))  # 返回新個體列表
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)  # 替換超限個體
            return new_inds  # 返回處理後的個體
        return wrapper
    return decorator
```

**關鍵問題**：`staticLimit` 返回的是**新個體列表**，但當前程式碼沒有接收這個返回值，導致替換後的個體被丟棄，超限個體仍在族群中。

## 修復方案

### 方案 1：修改演化迴圈（推薦）

修改 `gp_quant/evolution/engine.py` 的第 244-253 行：

```python
# Apply crossover and mutation
for i in range(0, len(offspring) - 1, 2):
    if random.random() < crossover_prob:
        # 接收 staticLimit 返回的新個體
        offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
        del offspring[i].fitness.values
        del offspring[i+1].fitness.values

for i in range(len(offspring)):
    if random.random() < mutation_prob:
        # 接收 staticLimit 返回的新個體
        offspring[i], = toolbox.mutate(offspring[i])
        del offspring[i].fitness.values
```

### 方案 2：使用 DEAP 的標準演化算法

使用 DEAP 內建的 `varAnd` 函數，它已經正確處理了返回值：

```python
from deap import algorithms

# 在演化迴圈中使用
offspring = algorithms.varAnd(offspring, toolbox, crossover_prob, mutation_prob)
```

### 方案 3：手動檢查深度（備用方案）

如果不想修改演化迴圈，可以在評估前手動檢查深度：

```python
# 在評估前檢查深度
for ind in offspring:
    if ind.height > 17:
        # 重新生成一個新個體
        ind[:] = toolbox.individual()
```

## 測試驗證

修復後，應該執行以下測試：

1. **小規模測試**：
   ```bash
   # 運行 1 個 ticker，1 次實驗，檢查所有 generation 的深度
   python run_single_test.py
   python check_depth_limits.py
   ```

2. **預期結果**：
   - Generation 0: 100% 符合（深度 <= 6）
   - Generation 1-50: 100% 符合（深度 <= 17）

3. **驗證指標**：
   - 違規率應該從 57.18% 降到 0%
   - 最大深度應該不超過 17

## 實作步驟

1. **備份當前程式碼**
2. **修改 engine.py**（使用方案 1）
3. **運行小規模測試**
4. **檢查深度限制**
5. **確認修復成功後，重新運行完整實驗**

## 注意事項

### 為什麼之前的實驗沒有崩潰？

雖然深度超過 17，但：
- Python 的遞迴深度限制通常是 1000
- 最大深度 129 仍然遠低於這個限制
- 所以程式能正常運行，只是違反了論文的深度限制

### 是否需要重新運行實驗？

**建議重新運行**，因為：
1. 當前結果違反了論文的深度限制要求
2. 深度過大的樹可能導致過擬合
3. 無法與論文結果進行公平比較

### 修復後的預期影響

- **訓練期表現**：可能略微下降（因為樹的複雜度受限）
- **測試期表現**：可能提升（減少過擬合）
- **計算速度**：可能提升（樹更簡單）
- **結果穩定性**：應該提升

## 相關文件

- 深度檢查結果：`depth_limit_check_results.csv`
- 違規記錄：`depth_limit_violations.csv`
- 深度限制分析：`DEPTH_LIMIT_ANALYSIS.md`
