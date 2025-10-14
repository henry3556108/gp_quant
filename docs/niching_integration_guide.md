# Niching 策略整合指南

## 概述

Niching 策略已成功整合到 `run_portfolio_experiment.py`，提供完全向下兼容的實作。

## 快速開始

### 啟用 Niching

編輯 `run_portfolio_experiment.py`，修改 `CONFIG` 字典：

```python
CONFIG = {
    # ... 其他配置 ...
    
    # Niching 配置
    'niching_enabled': True,              # 啟用 Niching
    'niching_n_clusters': 5,              # Niche 數量
    'niching_cross_ratio': 0.8,           # 跨群交配比例 (80%)
    'niching_update_frequency': 5,        # 每 5 代更新相似度矩陣
    'niching_algorithm': 'kmeans',        # 聚類演算法
}
```

### 停用 Niching（預設）

```python
CONFIG = {
    # ... 其他配置 ...
    
    'niching_enabled': False,  # 使用傳統 tournament selection
}
```

## 參數詳解

### niching_enabled

- **類型**: `bool`
- **預設值**: `False`
- **說明**: 是否啟用 Niching 策略
  - `False`: 使用傳統 tournament selection
  - `True`: 使用跨群選擇策略

### niching_n_clusters

- **類型**: `int`
- **預設值**: `5`
- **建議範圍**: 3-10
- **說明**: Niche 數量
  - 太少（< 3）: 多樣性不足
  - 適中（3-10）: 平衡多樣性與選擇壓力
  - 太多（> 10）: 每個 niche 太小，選擇壓力不足

**選擇建議**:
- Population 100-500: k = 3-5
- Population 1000-2000: k = 5-7
- Population 5000+: k = 7-10

### niching_cross_ratio

- **類型**: `float`
- **預設值**: `0.8`
- **範圍**: [0.0, 1.0]
- **說明**: 跨群交配比例
  - `0.0`: 完全群內配對（各 niche 獨立演化）
  - `0.3-0.5`: 保守策略
  - `0.7-0.9`: 推薦策略（積極促進基因交流）
  - `1.0`: 完全跨群配對（最大化基因交流）

**推薦**: `0.8` (80% 跨群，20% 群內)

### niching_update_frequency

- **類型**: `int`
- **預設值**: `5`
- **建議範圍**: 5-10
- **說明**: 每 N 代重新計算相似度矩陣
  - 太頻繁（< 3）: 計算開銷大
  - 適中（5-10）: 平衡計算成本與 niche 更新
  - 太稀疏（> 15）: niche 更新不及時

**計算成本估算**:
- Population 100: < 1 秒
- Population 500: ~10 秒
- Population 5000: ~100 秒（未並行化）

### niching_algorithm

- **類型**: `str`
- **預設值**: `'kmeans'`
- **選項**: `'kmeans'` 或 `'hierarchical'`
- **說明**: 聚類演算法
  - `'kmeans'`: K-means 聚類（推薦，速度快）
  - `'hierarchical'`: 層次聚類（更穩定，但較慢）

## 運行範例

### 範例 1: 基本 Niching 實驗

```python
CONFIG = {
    'tickers': ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO'],
    'population_size': 5000,
    'generations': 50,
    
    # 啟用 Niching
    'niching_enabled': True,
    'niching_n_clusters': 5,
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 5,
    'niching_algorithm': 'kmeans',
}
```

運行：
```bash
python run_portfolio_experiment.py
```

### 範例 2: 小規模測試

```python
CONFIG = {
    'tickers': ['ABX.TO', 'BBD-B.TO'],
    'population_size': 100,
    'generations': 10,
    
    # 啟用 Niching（頻繁更新以觀察效果）
    'niching_enabled': True,
    'niching_n_clusters': 3,
    'niching_cross_ratio': 0.8,
    'niching_update_frequency': 2,
    'niching_algorithm': 'kmeans',
}
```

### 範例 3: 對照實驗

**實驗 A（無 Niching）**:
```python
CONFIG = {
    'niching_enabled': False,
    'experiment_name': 'exp_baseline_no_niching'
}
```

**實驗 B（有 Niching）**:
```python
CONFIG = {
    'niching_enabled': True,
    'niching_n_clusters': 5,
    'niching_cross_ratio': 0.8,
    'experiment_name': 'exp_with_niching'
}
```

## 輸出說明

### 控制台輸出

啟用 Niching 後，每 N 代會顯示：

```
🔬 Niching: 計算相似度矩陣...
   ✓ 相似度矩陣計算完成 (12.3s)
   平均相似度: 0.3245
   多樣性分數: 0.6755

🔬 Niching: 聚類（k=5）...
   ✓ 聚類完成
   Silhouette 分數: 0.2841
   各 Niche 大小: {0: 1023, 1: 987, 2: 1045, 3: 956, 4: 989}

🎯 使用跨群選擇...
   ✓ 選擇完成
   跨群配對: 2000 (80%)
   群內配對: 500 (20%)
```

### 日誌文件

`evolution_log.json` 包含 Niching 統計：

```json
{
  "niching": {
    "enabled": true,
    "n_clusters": 5,
    "cross_ratio": 0.8,
    "update_frequency": 5,
    "algorithm": "kmeans",
    "log": [
      {
        "generation": 1,
        "avg_similarity": 0.3245,
        "diversity_score": 0.6755,
        "silhouette_score": 0.2841,
        "niche_sizes": {
          "0": 1023,
          "1": 987,
          "2": 1045,
          "3": 956,
          "4": 989
        },
        "computation_time": 12.3
      },
      ...
    ]
  }
}
```

## 向下兼容性

✅ **完全向下兼容！**

- 所有現有實驗腳本無需修改即可繼續運行
- `niching_enabled = False` 時使用原有的 tournament selection
- 不影響任何現有功能（early stopping、fitness metrics 等）

## 性能考量

### 計算開銷

Niching 的主要開銷來自相似度矩陣計算：

| Population | 時間（序列） | 時間（並行，未實作） |
|-----------|------------|-------------------|
| 100 | < 1s | < 1s |
| 500 | ~10s | ~2s |
| 1000 | ~40s | ~7s |
| 5000 | ~1000s (17min) | ~150s (2.5min) |

**建議**:
- Population < 1000: 可以頻繁更新（每 3-5 代）
- Population 1000-5000: 建議每 5-10 代更新
- Population > 5000: 建議每 10-15 代更新，或實作並行化

### 記憶體使用

- 相似度矩陣: O(n²) 空間
- Population 5000: ~200 MB
- 其他開銷: < 10 MB

## 故障處理

### Niching 計算失敗

如果相似度計算或聚類失敗，系統會自動降級為傳統選擇：

```
✗ Niching 計算失敗: ...
[使用傳統 tournament selection]
```

### 跨群選擇失敗

如果跨群選擇失敗，系統會自動降級為傳統選擇：

```
✗ 跨群選擇失敗: ...
[使用傳統 tournament selection]
```

## 實驗建議

### 初次測試

1. **小規模驗證**（5-10 分鐘）:
   ```python
   population_size = 100
   generations = 10
   niching_enabled = True
   niching_update_frequency = 2
   ```

2. **中規模測試**（1-2 小時）:
   ```python
   population_size = 500
   generations = 20
   niching_enabled = True
   niching_update_frequency = 5
   ```

3. **完整實驗**（數小時）:
   ```python
   population_size = 5000
   generations = 50
   niching_enabled = True
   niching_update_frequency = 10
   ```

### 對照實驗設計

建議運行以下實驗組合：

1. **Baseline**: `niching_enabled = False`
2. **Niching-Conservative**: `cross_ratio = 0.5`
3. **Niching-Aggressive**: `cross_ratio = 0.8`
4. **Niching-Extreme**: `cross_ratio = 1.0`

比較指標：
- 訓練期 fitness
- 測試期 fitness
- 多樣性分數趨勢
- 收斂速度

## 常見問題

### Q: Niching 會讓演化變慢嗎？

A: 會增加一些計算時間，主要來自相似度矩陣計算。建議：
- 調整 `niching_update_frequency` 減少計算頻率
- 未來可實作並行化加速

### Q: 如何判斷 Niching 是否有效？

A: 觀察以下指標：
1. 多樣性分數是否維持較高水平
2. 測試期表現是否優於 baseline
3. Silhouette 分數是否合理（> 0.2）

### Q: k 值如何選擇？

A: 建議：
- 先運行一次實驗，觀察 Silhouette 分數
- 嘗試 k ± 2，選擇 Silhouette 分數最高的

### Q: 為什麼有時跨群比例不精確？

A: 當某些 niche 很小時，可能無法達到精確比例。這是正常的。

## 下一步

1. **運行小規模測試**驗證功能
2. **運行對照實驗**比較有/無 Niching
3. **分析多樣性趨勢**
4. **調整參數**優化效果

---

**版本**: 1.0  
**最後更新**: 2025-10-14  
**相關文檔**: `docs/cross_niche_selection.md`
