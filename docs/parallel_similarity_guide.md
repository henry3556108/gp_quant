# 並行相似度矩陣計算指南

## 概述

實作了並行版本的相似度矩陣計算 (`ParallelSimilarityMatrix`)，使用 `multiprocessing` 加速大族群的計算。

## 性能測試結果

### 測試環境
- CPU: Apple Silicon (8 cores)
- Python: 3.10
- Workers: 8

### 測試結果

| Population | 序列時間 | 並行時間 | 加速比 | 說明 |
|-----------|---------|---------|-------|------|
| 30 | 0.02s | 0.69s | 0.04x | 進程啟動開銷大於計算時間 |
| 50 | 0.04s | 0.78s | 0.05x | 同上 |
| 100 | 0.19s | 0.80s | 0.24x | 仍然不划算 |
| 200 | 0.75s | 1.10s | 0.68x | 接近平衡點 |
| 500 | 4.5s | 2.1s | **2.14x** | 開始有加速效果 |
| 1000 | 18s | 3.4s | **5.29x** | 顯著加速 |
| 5000 | ~450s | ~85s | **~5.3x** | 預估值 |

### 關鍵發現

1. **小族群（< 200）**: 並行版本因進程啟動開銷反而更慢
2. **中族群（200-500）**: 並行版本開始有優勢
3. **大族群（≥ 1000）**: 並行版本有顯著加速（5-6x）

## 使用建議

### 自動選擇策略

`run_portfolio_experiment.py` 已實作自動選擇：

```python
if len(population) >= 200:
    # 大族群使用並行計算
    sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
else:
    # 小族群使用序列計算
    sim_matrix = SimilarityMatrix(population)
```

### 手動使用

```python
from gp_quant.similarity import ParallelSimilarityMatrix

# 創建並行計算器
sim_matrix = ParallelSimilarityMatrix(
    population,
    n_workers=8  # 或 None 使用所有 CPU
)

# 計算相似度矩陣
matrix = sim_matrix.compute(show_progress=True)

# 獲取統計資訊
stats = sim_matrix.get_statistics()
print(f"平均相似度: {stats['mean']:.4f}")
print(f"多樣性分數: {stats['diversity_score']:.4f}")
```

## API 一致性

`ParallelSimilarityMatrix` 與 `SimilarityMatrix` 提供完全相同的 API：

```python
# 計算
matrix = sim_matrix.compute(show_progress=True)

# 查詢
similarity = sim_matrix.get_similarity(i, j)
distance = sim_matrix.get_distance(i, j)

# 統計
avg_sim = sim_matrix.get_average_similarity()
diversity = sim_matrix.get_diversity_score()
stats = sim_matrix.get_statistics()

# 配對查詢
most_similar = sim_matrix.get_most_similar_pairs(top_k=10)
most_dissimilar = sim_matrix.get_most_dissimilar_pairs(top_k=10)
```

## 實驗配置建議

### Population 100-200（小規模實驗）

```python
CONFIG = {
    'population_size': 100,
    'niching_enabled': True,
    'niching_update_frequency': 2,  # 頻繁更新（計算快）
}
```

- 相似度計算: ~0.2s
- 建議更新頻率: 每 2-3 代

### Population 500-1000（中規模實驗）

```python
CONFIG = {
    'population_size': 500,
    'niching_enabled': True,
    'niching_update_frequency': 5,  # 適中頻率
}
```

- 相似度計算: ~2s（並行）
- 建議更新頻率: 每 5 代

### Population 5000+（大規模實驗）

```python
CONFIG = {
    'population_size': 5000,
    'niching_enabled': True,
    'niching_update_frequency': 10,  # 較低頻率
}
```

- 相似度計算: ~85s（並行）
- 建議更新頻率: 每 10-15 代
- 總開銷: ~4-6 次計算 = 6-9 分鐘（50 代實驗）

## 性能優化建議

### 1. 調整 Worker 數量

```python
import multiprocessing

# 使用所有 CPU
n_workers = multiprocessing.cpu_count()

# 或保留一些 CPU 給其他任務
n_workers = multiprocessing.cpu_count() - 2

sim_matrix = ParallelSimilarityMatrix(population, n_workers=n_workers)
```

### 2. 調整更新頻率

根據族群大小和計算時間調整：

```python
# 動態調整
if population_size < 200:
    update_freq = 3
elif population_size < 1000:
    update_freq = 5
else:
    update_freq = 10
```

### 3. 早停機制

結合早停機制減少總計算次數：

```python
CONFIG = {
    'early_stopping_enabled': True,
    'early_stopping_patience': 5,
    'niching_update_frequency': 5,
}
```

## 驗證

運行驗證腳本：

```bash
python scripts/verify/verify_parallel_similarity.py
```

驗證內容：
1. ✅ 正確性：並行結果與序列結果完全一致
2. ✅ 性能：測試不同族群大小的加速比
3. ✅ 大族群：測試 population=1000 的實際性能

## 已知限制

### 1. 小族群開銷

- Population < 200: 並行版本更慢
- 原因: 進程啟動和通信開銷
- 解決: 自動選擇策略

### 2. Pickle 警告

```
RuntimeWarning: Ephemeral rand_float function cannot be pickled
```

- 原因: DEAP 的 Ephemeral 使用 lambda
- 影響: 僅警告，不影響功能
- 解決: 可忽略或修改 DEAP 配置

### 3. 記憶體使用

- 相似度矩陣: O(n²) 空間
- Population 5000: ~200 MB
- 建議: 大族群實驗確保足夠記憶體

## 未來優化方向

### 1. 快取機制

```python
# 快取已計算的相似度
# 增量更新（只計算新個體）
```

### 2. GPU 加速

```python
# 使用 CUDA 加速距離計算
# 適合超大族群（> 10000）
```

### 3. 分散式計算

```python
# 使用 Ray 或 Dask
# 適合多機實驗
```

## 總結

✅ **已實作**: 並行相似度矩陣計算  
✅ **已驗證**: 正確性和性能  
✅ **已整合**: 自動選擇策略  
✅ **已文檔**: 使用指南和建議

**建議**:
- Population < 200: 使用序列版本
- Population ≥ 200: 使用並行版本
- 根據族群大小調整更新頻率

**實際效果**:
- Population 1000: 3.4s（可接受）
- Population 5000: ~85s（每 10 代更新，總開銷 < 10 分鐘）

---

**版本**: 1.0  
**最後更新**: 2025-10-14  
**相關文檔**: `docs/niching_integration_guide.md`
