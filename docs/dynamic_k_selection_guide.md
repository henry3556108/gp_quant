# 動態 Niche 數量 (K) 選擇指南

## 概述

`DynamicKSelector` 提供了靈活的 niche 數量選擇策略，支持：
- ✅ **固定 k 值**（向下兼容舊代碼）
- ✅ **動態選擇**（每代基於 Silhouette Score 選擇最佳 k）
- ✅ **自動上限**（使用 ln(n) 作為 k 的上限）
- ✅ **階段性校準**（前幾代動態選擇，之後使用固定 k）

## 快速開始

### 1. 固定 K 值（向下兼容）

```python
from gp_quant.niching import create_k_selector

# 舊的配置方式（完全兼容）
config = {
    'niching_n_clusters': 3
}

selector = create_k_selector(config)
result = selector.select_k(similarity_matrix, population_size=5000)
print(f"選擇的 k: {result['k']}")  # 輸出: 3
```

### 2. 動態選擇 K 值

```python
# 每代都動態選擇最佳 k
config = {
    'niching_k_selection': 'dynamic',
    'niching_k_min': 2,
    'niching_k_max': 8,
    'niching_algorithm': 'kmeans'
}

selector = create_k_selector(config)
result = selector.select_k(similarity_matrix, population_size=5000)

print(f"選擇的 k: {result['k']}")
print(f"測試的 k 範圍: {result['k_range']}")
print(f"各 k 值的 Silhouette Score: {result['scores']}")
```

### 3. 自動 K 上限（推薦）

```python
# 使用 ln(n) 作為 k 的上限
config = {
    'niching_k_selection': 'auto',
    'niching_k_min': 2,
    'niching_k_max': 'auto',  # 自動計算為 int(ln(population_size))
    'niching_algorithm': 'kmeans'
}

selector = create_k_selector(config)
result = selector.select_k(similarity_matrix, population_size=5000)

# population_size=5000 時，k_max = int(ln(5000)) = 8
print(f"選擇的 k: {result['k']}")
```

### 4. 階段性校準（最佳平衡）⭐

```python
# 前 3 代動態選擇，之後使用校準期的最佳 k
config = {
    'niching_k_selection': 'calibration',
    'niching_k_min': 2,
    'niching_k_max': 'auto',
    'niching_k_calibration_gens': 3,  # 校準期代數
    'niching_algorithm': 'kmeans'
}

selector = create_k_selector(config)

# Generation 1-3: 動態選擇
for gen in range(1, 4):
    result = selector.select_k(similarity_matrix, population_size=5000, generation=gen)
    print(f"Gen {gen}: k={result['k']}, mode={result['mode']}")

# Generation 4+: 使用校準後的固定 k
for gen in range(4, 11):
    result = selector.select_k(similarity_matrix, population_size=5000, generation=gen)
    print(f"Gen {gen}: k={result['k']}, mode={result['mode']}")
```

## 在實驗腳本中使用

### 修改 `run_portfolio_experiment.py`

```python
from gp_quant.niching import create_k_selector, NichingClusterer, CrossNicheSelector

# 配置
CONFIG = {
    # ... 其他配置 ...
    
    # Niching 配置
    'niching_enabled': True,
    
    # 選擇模式 1: 固定 k（向下兼容）
    'niching_n_clusters': 3,
    
    # 或者選擇模式 2: 動態選擇
    # 'niching_k_selection': 'auto',
    # 'niching_k_min': 2,
    # 'niching_k_max': 'auto',
    
    # 或者選擇模式 3: 階段性校準（推薦）
    # 'niching_k_selection': 'calibration',
    # 'niching_k_min': 2,
    # 'niching_k_max': 'auto',
    # 'niching_k_calibration_gens': 3,
    
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 1,
    'niching_algorithm': 'kmeans',
}

# 創建 k 選擇器
k_selector = create_k_selector(CONFIG)

# 在演化循環中
for gen in range(CONFIG['generations']):
    # ... 評估 fitness ...
    
    if CONFIG['niching_enabled'] and gen % CONFIG['niching_update_frequency'] == 0:
        # 計算相似度矩陣
        sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
        similarity_matrix = sim_matrix.compute(show_progress=False)
        
        # 動態選擇 k 值
        k_result = k_selector.select_k(
            similarity_matrix, 
            population_size=len(population),
            generation=gen + 1
        )
        
        selected_k = k_result['k']
        print(f"  選擇的 k 值: {selected_k}")
        
        # 使用選擇的 k 進行聚類
        clusterer = NichingClusterer(
            n_clusters=selected_k,
            algorithm=CONFIG['niching_algorithm']
        )
        niche_labels = clusterer.fit_predict(similarity_matrix)
        
        # ... 跨群選擇 ...
```

## 配置參數說明

### 通用參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `niching_algorithm` | str | 聚類演算法：'kmeans' 或 'hierarchical' |
| `random_state` | int | 隨機種子（可選） |
| `verbose` | bool | 是否顯示詳細資訊（預設 True） |

### 固定模式參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `niching_n_clusters` | int | 固定的 k 值 |

### 動態/自動模式參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `niching_k_selection` | str | 'dynamic' 或 'auto' |
| `niching_k_min` | int | 最小 k 值（預設 2） |
| `niching_k_max` | int/str | 最大 k 值，或 'auto'（使用 ln(n)） |

### 校準模式參數

| 參數 | 類型 | 說明 |
|------|------|------|
| `niching_k_selection` | str | 'calibration' |
| `niching_k_min` | int | 最小 k 值（預設 2） |
| `niching_k_max` | int/str | 最大 k 值，或 'auto'（使用 ln(n)） |
| `niching_k_calibration_gens` | int | 校準期代數（預設 3） |

## 模式比較

| 模式 | 適應性 | 計算開銷 | 適用場景 |
|------|--------|---------|---------|
| **Fixed** | ❌ 無 | ✅ 無 | 已知最佳 k，或快速測試 |
| **Dynamic** | ✅✅✅ 最高 | ❌❌ 高 | 需要最大適應性，計算資源充足 |
| **Auto** | ✅✅✅ 最高 | ❌❌ 高 | 同 Dynamic，但自動設置 k_max |
| **Calibration** | ✅✅ 高 | ✅ 低 | **推薦**：平衡適應性和效率 |

## 計算開銷分析

假設：
- Population size = 5000
- k 範圍 = [2, 8]（7 個值）
- 每個 k 測試時間 ≈ 2 秒
- 總 generations = 50

| 模式 | 每代開銷 | 總開銷 | 說明 |
|------|---------|--------|------|
| Fixed | 0 秒 | 0 秒 | 無額外開銷 |
| Dynamic | ~14 秒 | ~700 秒 (11.7 分鐘) | 每代測試 7 個 k 值 |
| Auto | ~14 秒 | ~700 秒 (11.7 分鐘) | 同 Dynamic |
| Calibration (3 代) | ~14 秒 (前 3 代) | ~42 秒 | **推薦**：只在前 3 代有開銷 |

## 實驗結果參考

根據實驗 `experiment_dynamic_niche_selection.py` 的結果：

| Generation | Population | Best K | Silhouette | ln(5000) |
|------------|-----------|--------|------------|----------|
| 1 | 5000 | **8** | 0.3083 | 8.52 |
| 2 | 5000 | **8** | 0.4770 | 8.52 |
| 3 | 5000 | **8** | 0.4624 | 8.52 |
| 4 | 5000 | **8** | 0.4737 | 8.52 |
| 5 | 5000 | **8** | 0.4570 | 8.52 |
| 6 | 5000 | **8** | 0.4755 | 8.52 |

**觀察**：
- ✅ 所有 generation 都選擇 k=8
- ✅ k=8 ≈ int(ln(5000)) = 8
- ✅ 驗證了 ln(n) 規則的有效性

## 推薦配置

### 推薦 1：階段性校準 + ln(n)（最佳平衡）⭐⭐⭐

```python
CONFIG = {
    'niching_enabled': True,
    'niching_k_selection': 'calibration',
    'niching_k_min': 2,
    'niching_k_max': 'auto',  # ln(n)
    'niching_k_calibration_gens': 3,
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 3,  # 每 3 代更新相似度矩陣
    'niching_algorithm': 'kmeans',
}
```

**優點**：
- 前 3 代探索最佳 k
- 之後使用固定 k，節省計算
- 相似度矩陣每 3 代更新，進一步節省時間

### 推薦 2：完全動態（計算資源充足時）⭐⭐

```python
CONFIG = {
    'niching_enabled': True,
    'niching_k_selection': 'auto',
    'niching_k_min': 2,
    'niching_k_max': 'auto',
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 1,
    'niching_algorithm': 'kmeans',
}
```

**優點**：
- 最大適應性
- 每代都選擇最佳 k

**缺點**：
- 計算開銷較高

### 推薦 3：固定 k=8（基於實驗結果）⭐

```python
CONFIG = {
    'niching_enabled': True,
    'niching_n_clusters': 8,  # 基於實驗結果
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 3,
    'niching_algorithm': 'kmeans',
}
```

**優點**：
- 無額外計算開銷
- 基於實驗驗證的最佳 k

**缺點**：
- 對不同 population size 可能不是最佳

## 向下兼容性

所有舊的配置都完全兼容：

```python
# 舊配置（仍然有效）
CONFIG = {
    'niching_enabled': True,
    'niching_n_clusters': 3,
    'niching_algorithm': 'kmeans',
}

# 自動識別為固定模式
selector = create_k_selector(CONFIG)
# 等同於：
# selector = DynamicKSelector(mode='fixed', fixed_k=3)
```

## 常見問題

### Q1: 應該選擇哪種模式？

**A**: 推薦使用 **Calibration 模式**：
- 前 3 代動態選擇最佳 k
- 之後使用固定 k
- 平衡了適應性和計算效率

### Q2: k_max 應該設為多少？

**A**: 推薦使用 `'auto'`（ln(n)）：
- 有理論基礎
- 實驗驗證有效
- 自動適應 population size

### Q3: 動態選擇會增加多少計算時間？

**A**: 
- Dynamic 模式：每代 ~14 秒，50 代 ~11.7 分鐘
- Calibration 模式：只在前 3 代，總共 ~42 秒
- 相比整個實驗時間（數小時），開銷可接受

### Q4: 如何查看選擇歷史？

**A**:
```python
stats = selector.get_statistics()
print(stats)

# Calibration 模式會包含：
# {
#     'mode': 'calibration',
#     'calibrated_k': 8,
#     'calibration_history': [
#         {'generation': 1, 'best_k': 8, 'score': 0.3083},
#         {'generation': 2, 'best_k': 8, 'score': 0.4770},
#         {'generation': 3, 'best_k': 8, 'score': 0.4624},
#     ]
# }
```

## 示例代碼

完整示例請參考：
- `samples/niching/sample_dynamic_k_selection.py`
- `scripts/analysis/experiment_dynamic_niche_selection.py`

## 參考文獻

1. Silhouette Score: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis"
2. ln(n) 規則: 經典的聚類數量啟發式規則
3. 實驗驗證: `experiment_results/dynamic_niche/EXPERIMENT_REPORT.md`
