# Cluster Labels 儲存與使用指南

## 概述

從現在開始，每個 generation 的 pkl 檔案會儲存每個個體所屬的 niche (cluster) 資訊，包括：
- `cluster_labels`: 每個個體的 niche ID
- `niching_info`: niching 相關資訊（n_clusters, algorithm, silhouette_score）

## 新的 Generation PKL 格式

```python
{
    'generation': 1,
    'population': [ind1, ind2, ...],
    'hall_of_fame': [...],
    'statistics': {...},
    'timestamp': '...',
    
    # 新增的欄位（僅在啟用 niching 時）
    'cluster_labels': [0, 1, 2, 0, 1, ...],  # 每個個體的 niche ID
    'niching_info': {
        'n_clusters': 3,                      # niche 數量
        'algorithm': 'kmeans',                # 聚類演算法
        'silhouette_score': 0.45              # silhouette 分數
    }
}
```

## 向後相容性

- ✅ **舊格式檔案可以正常載入**
- 舊格式檔案的 `cluster_labels` 和 `niching_info` 會是 `None`
- 使用 `has_niching_info()` 函數檢查是否包含 niching 資訊

## 使用方法

### 1. 基本載入

```python
from gp_quant.utils import load_generation, has_niching_info

# 載入 generation
gen_data = load_generation('generations/generation_001.pkl')

# 檢查是否有 niching 資訊
if has_niching_info(gen_data):
    print("包含 niching 資訊")
    print(f"Cluster labels: {gen_data['cluster_labels']}")
    print(f"Niching info: {gen_data['niching_info']}")
else:
    print("舊格式，沒有 niching 資訊")
```

### 2. 獲取特定 Niche 的個體

```python
from gp_quant.utils import get_niche_individuals

# 獲取屬於 niche 0 的所有個體
niche_0_individuals = get_niche_individuals(gen_data, 0)

print(f"Niche 0 有 {len(niche_0_individuals)} 個個體")

# 分析這些個體
for ind in niche_0_individuals[:5]:  # 前 5 個
    print(f"Fitness: {ind.fitness.values[0]:.4f}")
    print(f"Expression: {str(ind)}")
```

### 3. 獲取每個 Niche 的統計資訊

```python
from gp_quant.utils import get_niche_statistics

# 獲取所有 niche 的統計
stats = get_niche_statistics(gen_data)

for niche_id, niche_stats in stats.items():
    print(f"\nNiche {niche_id}:")
    print(f"  Size: {niche_stats['size']}")
    print(f"  Fitness mean: {niche_stats['fitness_mean']:.4f}")
    print(f"  Fitness std: {niche_stats['fitness_std']:.4f}")
    print(f"  Fitness range: [{niche_stats['fitness_min']:.4f}, {niche_stats['fitness_max']:.4f}]")
```

### 4. 載入多個 Generations

```python
from gp_quant.utils import load_multiple_generations

# 載入 generation 1 到 10
generations = load_multiple_generations(
    'generations',
    start_gen=1,
    end_gen=10
)

# 分析每個 generation
for gen_data in generations:
    gen_num = gen_data['generation']
    
    if has_niching_info(gen_data):
        n_clusters = gen_data['niching_info']['n_clusters']
        silhouette = gen_data['niching_info']['silhouette_score']
        print(f"Gen {gen_num}: {n_clusters} niches, silhouette={silhouette:.4f}")
```

### 5. 分析 Niche 演化

```python
from collections import Counter

# 追蹤某個個體在不同 generation 的 niche 變化
individual_index = 0  # 追蹤第一個個體

for gen_data in generations:
    if has_niching_info(gen_data):
        niche_id = gen_data['cluster_labels'][individual_index]
        print(f"Gen {gen_data['generation']}: 個體 {individual_index} 在 niche {niche_id}")
```

### 6. 找出完全相同的 Niche

```python
from collections import Counter

gen_data = load_generation('generations/generation_001.pkl')

if has_niching_info(gen_data):
    population = gen_data['population']
    cluster_labels = gen_data['cluster_labels']
    
    # 按 niche 分組
    niches = {}
    for ind, label in zip(population, cluster_labels):
        if label not in niches:
            niches[label] = []
        niches[label].append(ind)
    
    # 檢查每個 niche 是否完全相同
    for niche_id, individuals in niches.items():
        expressions = [str(ind) for ind in individuals]
        unique_expressions = set(expressions)
        
        if len(unique_expressions) == 1:
            print(f"✅ Niche {niche_id} 完全相同！")
            print(f"   表達式: {list(unique_expressions)[0]}")
            print(f"   個體數: {len(individuals)}")
        else:
            print(f"⚠️  Niche {niche_id} 有 {len(unique_expressions)} 種不同表達式")
```

## 啟用 Niching 實驗

要讓新的實驗儲存 cluster_labels，需要在 `run_portfolio_experiment.py` 中啟用 niching：

```python
CONFIG = {
    ...
    'niching_enabled': True,           # 啟用 niching
    'niching_n_clusters': 3,           # 或使用動態 k 選擇
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 1,
    'niching_algorithm': 'kmeans',
    ...
}
```

## API 參考

### `load_generation(file_path)`
載入 generation pkl 檔案，支援向後相容。

**Returns:**
```python
{
    'generation': int,
    'population': List,
    'hall_of_fame': List,
    'statistics': Dict,
    'timestamp': str,
    'cluster_labels': Optional[List[int]],  # 新格式才有
    'niching_info': Optional[Dict]          # 新格式才有
}
```

### `has_niching_info(gen_data)`
檢查是否包含 niching 資訊。

**Returns:** `bool`

### `get_niche_individuals(gen_data, niche_id)`
獲取屬於特定 niche 的所有個體。

**Returns:** `List` - 個體列表

**Raises:** `ValueError` - 如果沒有 niching 資訊

### `get_niche_statistics(gen_data)`
獲取每個 niche 的統計資訊。

**Returns:** `Dict` - 每個 niche 的統計資訊
```python
{
    niche_id: {
        'size': int,
        'individuals': List,
        'fitness_mean': float,
        'fitness_std': float,
        'fitness_min': float,
        'fitness_max': float
    }
}
```

### `load_multiple_generations(directory, start_gen=1, end_gen=None)`
載入多個 generation 的資料。

**Returns:** `List[Dict]` - generation 資料列表

## 注意事項

1. **記憶體使用**：載入多個 generation 會佔用大量記憶體，建議分批處理
2. **檔案大小**：加入 cluster_labels 後，pkl 檔案大小會略微增加（約 5-10%）
3. **相容性**：所有舊的分析腳本仍然可以正常運作，不受影響

## 範例：更新 analyze_perfect_niches_accurate.py

現在可以直接使用儲存的 cluster_labels，不需要重新計算相似度矩陣：

```python
from gp_quant.utils import load_generation, has_niching_info, get_niche_individuals

gen_data = load_generation('generations/generation_001.pkl')

if has_niching_info(gen_data):
    # 直接使用儲存的 cluster_labels
    niching_info = gen_data['niching_info']
    print(f"N clusters: {niching_info['n_clusters']}")
    print(f"Silhouette score: {niching_info['silhouette_score']}")
    
    # 獲取每個 niche 的個體
    for niche_id in range(niching_info['n_clusters']):
        individuals = get_niche_individuals(gen_data, niche_id)
        print(f"Niche {niche_id}: {len(individuals)} 個個體")
else:
    print("舊格式，需要重新計算相似度矩陣")
```

## 測試

執行測試腳本驗證功能：

```bash
conda run -n gp_quant python test_cluster_labels_storage.py
```

測試內容：
1. ✅ 向後相容性測試（載入舊格式）
2. ✅ 工具函數測試
3. ✅ 新格式使用說明
