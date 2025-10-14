# Cross-Niche Parent Selection 使用指南

## 概述

`CrossNicheSelector` 實作了跨群（Cross-Niche）親代選擇機制，用於 Niching 策略中維持族群多樣性。

## 核心概念

### 兩階段選擇機制

1. **Stage 1: Within-Niche Tournament Selection**
   - 在每個 niche 內進行 tournament selection
   - 選出每個 niche 的優秀個體
   - 按 niche 大小比例分配配額

2. **Stage 2: Cross-Niche Pairing**
   - 將選出的個體進行跨群配對
   - 可配置跨群比例（預設 80%）
   - 剩餘比例進行群內配對

### 為什麼需要跨群選擇？

- **促進基因交流**: 不同 niche 之間的基因交換可能產生創新解
- **維持多樣性**: 避免各 niche 獨立演化導致整體多樣性下降
- **平衡探索與利用**: 跨群配對增加探索，群內配對保持利用

## 快速開始

### 基本使用

```python
from gp_quant.niching import NichingClusterer, CrossNicheSelector
from gp_quant.similarity import SimilarityMatrix

# 1. 計算相似度矩陣
sim_matrix = SimilarityMatrix(population)
similarity_matrix = sim_matrix.compute()

# 2. 聚類
clusterer = NichingClusterer(n_clusters=5, algorithm='kmeans')
niche_labels = clusterer.fit_predict(similarity_matrix)

# 3. 跨群選擇
selector = CrossNicheSelector(
    cross_niche_ratio=0.8,  # 80% 跨群配對
    tournament_size=3,      # Tournament 大小
    random_state=42         # 隨機種子（可選）
)

# 選擇 k 個親代（k 必須是偶數）
k = 100
parents = selector.select(population, niche_labels, k)

# 4. 查看統計資訊
selector.print_statistics()
```

### 輸出範例

```
============================================================
跨群選擇統計資訊
============================================================
總配對數: 50
跨群配對數: 40 (80.0%)
群內配對數: 10 (20.0%)

配置的跨群比例: 80.0%
實際的跨群比例: 80.0%

各 Niche 配對統計:
  Niche 0 ↔ Niche 1: 5 次
  Niche 0 ↔ Niche 2: 3 次
  Niche 1 ↔ Niche 3: 8 次
  Niche 2 ↔ Niche 4: 6 次
  ...
  Niche 0 內部配對: 2 次
  Niche 1 內部配對: 3 次
============================================================
```

## 參數說明

### CrossNicheSelector 參數

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `cross_niche_ratio` | float | 0.8 | 跨群交配比例，範圍 [0, 1] |
| `tournament_size` | int | 3 | Tournament selection 的大小，必須 >= 2 |
| `random_state` | int | None | 隨機種子，用於可重現性 |

### 跨群比例建議

| 比例 | 適用場景 | 特點 |
|------|---------|------|
| 0.0 | 完全群內配對 | 各 niche 獨立演化，多樣性可能下降 |
| 0.3-0.5 | 保守策略 | 平衡群內競爭與跨群交流 |
| 0.7-0.9 | 推薦策略 | 積極促進基因交流，維持多樣性 |
| 1.0 | 完全跨群配對 | 最大化基因交流，可能削弱 niche 特性 |

**推薦**: 0.8 (80% 跨群，20% 群內)

## 完整範例

### 範例 1: 基本使用

```python
import random
from deap import base, creator, tools, gp
from gp_quant.gp.operators import pset
from gp_quant.similarity import SimilarityMatrix
from gp_quant.niching import NichingClusterer, CrossNicheSelector

# 設置 GP 環境
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 創建族群
population = toolbox.population(n=100)

# 分配 fitness（實際應用中應該是真實評估）
for ind in population:
    ind.fitness.values = (random.uniform(0, 100),)

# 計算相似度並聚類
sim_matrix = SimilarityMatrix(population)
similarity_matrix = sim_matrix.compute(show_progress=True)

clusterer = NichingClusterer(n_clusters=5)
niche_labels = clusterer.fit_predict(similarity_matrix)

# 跨群選擇
selector = CrossNicheSelector(cross_niche_ratio=0.8, tournament_size=3)
parents = selector.select(population, niche_labels, k=50)

# 顯示統計
selector.print_statistics()
```

### 範例 2: 比較不同跨群比例

```python
ratios = [0.0, 0.3, 0.5, 0.8, 1.0]

for ratio in ratios:
    selector = CrossNicheSelector(cross_niche_ratio=ratio, tournament_size=3)
    parents = selector.select(population, niche_labels, k=50)
    stats = selector.get_statistics()
    
    print(f"\n跨群比例: {ratio:.0%}")
    print(f"  實際跨群配對: {stats['cross_niche_pairs']} 對")
    print(f"  實際群內配對: {stats['within_niche_pairs']} 對")
```

### 範例 3: 整合到演化循環（概念）

```python
# 演化循環
for generation in range(num_generations):
    # 1. 評估 fitness
    for ind in population:
        ind.fitness.values = evaluate(ind)
    
    # 2. 計算相似度矩陣（每 N 代計算一次）
    if generation % 5 == 0:
        sim_matrix = SimilarityMatrix(population)
        similarity_matrix = sim_matrix.compute(show_progress=False)
        
        # 3. 聚類
        clusterer = NichingClusterer(n_clusters=5)
        niche_labels = clusterer.fit_predict(similarity_matrix)
    
    # 4. 跨群選擇
    selector = CrossNicheSelector(cross_niche_ratio=0.8, tournament_size=3)
    parents = selector.select(population, niche_labels, k=len(population))
    
    # 5. 變異和交配
    offspring = algorithms.varAnd(parents, toolbox, cxpb=0.8, mutpb=0.2)
    
    # 6. 更新族群
    population[:] = offspring
```

## 統計資訊

### get_statistics() 返回值

```python
stats = selector.get_statistics()
```

返回的字典包含：

| 鍵 | 類型 | 說明 |
|----|------|------|
| `total_pairs` | int | 總配對數 |
| `cross_niche_pairs` | int | 跨群配對數 |
| `within_niche_pairs` | int | 群內配對數 |
| `cross_niche_ratio_actual` | float | 實際跨群比例 |
| `within_niche_ratio_actual` | float | 實際群內比例 |
| `niche_pair_counts` | dict | 各 niche 配對統計 |

## 邊界情況處理

### 1. 只有一個 niche

```python
# 所有個體都在同一個 niche
niche_labels = np.zeros(100, dtype=int)

selector = CrossNicheSelector(cross_niche_ratio=0.8)
parents = selector.select(population, niche_labels, k=50)

# 結果: 所有配對都是群內配對（因為沒有其他 niche）
```

### 2. 每個 niche 只有一個個體

```python
# 每個個體一個 niche
niche_labels = np.arange(100)

selector = CrossNicheSelector(cross_niche_ratio=0.8)
parents = selector.select(population, niche_labels, k=50)

# 結果: 正常運作，每個配對都是跨群配對
```

### 3. 跨群比例為 0 或 1

```python
# 完全群內配對
selector = CrossNicheSelector(cross_niche_ratio=0.0)
parents = selector.select(population, niche_labels, k=50)

# 完全跨群配對
selector = CrossNicheSelector(cross_niche_ratio=1.0)
parents = selector.select(population, niche_labels, k=50)
```

## 性能考量

### 時間複雜度

- **Stage 1 (Tournament Selection)**: O(k × tournament_size)
- **Stage 2 (Pairing)**: O(k)
- **總計**: O(k × tournament_size)

其中 k 是要選擇的個體數量。

### 記憶體使用

- 主要記憶體使用來自統計資訊記錄
- 對於大規模族群（n > 10000），記憶體使用仍然很小（< 1MB）

## 驗證腳本

運行驗證腳本以查看完整演示：

```bash
python scripts/verify/verify_cross_niche_selection.py
```

驗證腳本包含：
1. 基本功能演示
2. 不同跨群比例比較
3. 配對細節可視化
4. 邊界情況測試

## 下一步

### 整合到 EvolutionEngine

待實作：將 `CrossNicheSelector` 整合到主演化引擎中。

### 實驗驗證

需要運行完整實驗來驗證：
- 對演化收斂性的影響
- 多樣性維持效果
- 測試期表現提升

## 參考文獻

1. Mahfoud, S. W. (1995). Niching methods for genetic algorithms. Urbana, 51(95001), 62-94.
2. Sareni, B., & Krahenbuhl, L. (1998). Fitness sharing and niching methods revisited. IEEE transactions on Evolutionary Computation, 2(3), 97-106.

## 常見問題

### Q: 為什麼 k 必須是偶數？

A: 因為選擇的個體會進行配對交配，每對需要兩個個體。

### Q: 如何選擇合適的跨群比例？

A: 建議從 0.8 開始，根據實驗結果調整。如果多樣性下降太快，增加比例；如果收斂太慢，減少比例。

### Q: Tournament size 應該設多大？

A: 預設 3 是一個平衡選擇壓力和多樣性的好值。增大會增加選擇壓力，減小會增加隨機性。

### Q: 多久應該重新計算相似度矩陣？

A: 建議每 5-10 代重新計算一次，平衡計算開銷和 niche 更新頻率。

---

**版本**: 0.2.0  
**最後更新**: 2025-10-14
