# Phase 2: Tree Similarity 計算模組實作計畫

**分支**: `feature/tree-similarity`  
**開始日期**: 2025-10-13  
**預估時間**: 1.5-2 週  
**狀態**: 🚀 開始實作

---

## 📋 目標

實作樹結構相似度計算模組，為 Niching 策略奠定基礎。

### 核心功能
1. **Tree Edit Distance (TED)**: 計算兩棵 GP tree 之間的編輯距離
2. **Similarity Matrix**: 計算族群中所有個體的相似度矩陣
3. **並行計算**: 使用 8 cores 加速相似度計算
4. **視覺化工具**: 相似度矩陣熱圖、分佈圖

---

## 🎯 成功指標

- [ ] TED 計算正確（與已知結果比對）
- [ ] Population=500 時計算時間 < 10 秒
- [ ] Population=5000 時計算時間 < 15 分鐘
- [ ] 加速比 > 6x (8 cores)
- [ ] 記憶體使用 < 2GB (n=5000)
- [ ] 視覺化工具可用
- [ ] 完整文檔和測試

---

## 📂 模組結構

```
gp_quant/similarity/
├── __init__.py                    ✅ 已創建
├── tree_edit_distance.py          ⏳ 待實作
├── similarity_matrix.py           ⏳ 待實作
├── parallel_calculator.py         ⏳ 待實作
├── cache.py                       ⏳ 待實作
└── visualizer.py                  ⏳ 待實作

tests/similarity/
├── __init__.py                    ⏳ 待創建
├── test_tree_edit_distance.py    ⏳ 待創建
├── test_similarity_matrix.py     ⏳ 待創建
└── test_parallel_calculator.py   ⏳ 待創建

samples/similarity/
└── sample_similarity_analysis.py  ⏳ 待創建
```

---

## 📝 實作任務

### Task 2.1: Tree Edit Distance 實作 ⏳

**優先級**: 🔴 最高  
**預估時間**: 2-3 天

#### 子任務
- [ ] 2.1.1 研究 TED 演算法
  - [ ] 閱讀 Zhang-Shasha 論文
  - [ ] 研究 APTED 實作
  - [ ] 決定實作方案
- [ ] 2.1.2 實作基本 TED 計算
  - [ ] 創建 `TreeEditDistance` 類
  - [ ] 實作核心演算法
  - [ ] 處理 DEAP tree 結構
- [ ] 2.1.3 定義編輯成本
  - [ ] 插入成本
  - [ ] 刪除成本
  - [ ] 替換成本（考慮節點類型）
- [ ] 2.1.4 單元測試
  - [ ] 簡單樹測試
  - [ ] 複雜樹測試
  - [ ] 邊界情況測試

**交付物**:
- `gp_quant/similarity/tree_edit_distance.py`
- `tests/similarity/test_tree_edit_distance.py`
- 演算法選擇文檔

---

### Task 2.2: Similarity Matrix 實作 ⏳

**優先級**: 🔴 高  
**預估時間**: 1-2 天

#### 子任務
- [ ] 2.2.1 實作 `SimilarityMatrix` 類
  - [ ] 批次計算相似度
  - [ ] 標準化為 [0, 1]
  - [ ] 對稱矩陣優化
- [ ] 2.2.2 整合 TED 計算
  - [ ] 調用 TreeEditDistance
  - [ ] 距離轉相似度
- [ ] 2.2.3 單元測試
  - [ ] 小族群測試
  - [ ] 對稱性驗證
  - [ ] 標準化驗證

**交付物**:
- `gp_quant/similarity/similarity_matrix.py`
- `tests/similarity/test_similarity_matrix.py`

---

### Task 2.3: 並行計算實作 ⭐ 核心 ⏳

**優先級**: 🔴 最高  
**預估時間**: 2-3 天

#### 子任務
- [ ] 2.3.1 實作 `ParallelSimilarityCalculator`
  - [ ] 使用 multiprocessing.Pool
  - [ ] 任務分配策略
  - [ ] 結果收集與合併
- [ ] 2.3.2 性能優化
  - [ ] 對稱矩陣優化（只計算上三角）
  - [ ] 記憶體管理
  - [ ] 進度追蹤（tqdm）
- [ ] 2.3.3 性能測試
  - [ ] Population=100 測試
  - [ ] Population=500 測試
  - [ ] Population=5000 測試
  - [ ] 加速比測量

**交付物**:
- `gp_quant/similarity/parallel_calculator.py`
- `tests/similarity/test_parallel_calculator.py`
- 性能測試報告

**性能目標**:
| Population | 目標時間 | 加速比 |
|------------|---------|--------|
| 500 | < 10s | > 6x |
| 5000 | < 15min | > 6x |

---

### Task 2.4: 快取機制實作 ⏳

**優先級**: 🟡 中  
**預估時間**: 1 天

#### 子任務
- [ ] 2.4.1 實作 `SimilarityCache` 類
  - [ ] LRU cache 機制
  - [ ] 快取鍵設計
  - [ ] 快取大小限制
- [ ] 2.4.2 整合到計算流程
  - [ ] 查詢快取
  - [ ] 更新快取
- [ ] 2.4.3 測試
  - [ ] 快取命中率測試
  - [ ] 性能提升測試

**交付物**:
- `gp_quant/similarity/cache.py`
- 快取效果分析

---

### Task 2.5: 視覺化工具實作 ⏳

**優先級**: 🟢 低  
**預估時間**: 1-2 天

#### 子任務
- [ ] 2.5.1 相似度矩陣熱圖
  - [ ] 使用 seaborn heatmap
  - [ ] 顏色映射配置
  - [ ] 支援大矩陣（降採樣）
- [ ] 2.5.2 相似度分佈圖
  - [ ] 直方圖
  - [ ] 統計指標顯示
- [ ] 2.5.3 範例腳本
  - [ ] 完整分析流程
  - [ ] 多種視覺化展示

**交付物**:
- `gp_quant/similarity/visualizer.py`
- `samples/similarity/sample_similarity_analysis.py`
- 視覺化範例圖

---

### Task 2.6: 文檔與測試 ⏳

**優先級**: 🟡 中  
**預估時間**: 1 天

#### 子任務
- [ ] 2.6.1 API 文檔
  - [ ] 所有類別的 docstring
  - [ ] 使用範例
  - [ ] 參數說明
- [ ] 2.6.2 使用指南
  - [ ] 快速開始
  - [ ] 進階用法
  - [ ] 性能調優建議
- [ ] 2.6.3 整合測試
  - [ ] 完整流程測試
  - [ ] 與現有系統整合

**交付物**:
- `docs/tree_similarity.md`
- 完整測試套件
- 使用範例

---

## 🔧 技術決策

### 1. Tree Edit Distance 演算法選擇

**選項 A: 使用 `apted` 套件** ⭐ 推薦
- ✅ 成熟的實作
- ✅ 性能優化良好
- ✅ 易於整合
- ❌ 需要適配 DEAP tree 結構

**選項 B: 自行實作 Zhang-Shasha**
- ✅ 完全控制
- ✅ 可針對 GP tree 優化
- ❌ 開發時間長
- ❌ 需要詳細測試

**決策**: 先嘗試 `apted`，如果不適用再考慮自行實作

---

### 2. 並行化策略

**配對數量**:
- Population=500: 124,750 pairs
- Population=5000: 12,497,500 pairs

**分配策略**:
```python
# 對稱矩陣優化：只計算上三角
total_pairs = n * (n - 1) // 2

# 8 workers 均勻分配
pairs_per_worker = total_pairs // 8
```

**記憶體優化**:
- 使用 shared memory 或傳遞索引
- 避免複製整個 population
- 結果矩陣使用 numpy array

---

### 3. 相似度計算公式

**距離轉相似度**:
```python
# 方案 A: 指數衰減
similarity = exp(-distance / max_distance)

# 方案 B: 線性轉換
similarity = 1 - (distance / max_distance)

# 方案 C: 標準化
similarity = 1 / (1 + distance)
```

**決策**: 使用方案 C（標準化），範圍 [0, 1]，易於解釋

---

## 📊 預期效能

### Population = 500
- **序列計算**: ~60 秒
- **並行計算 (8 cores)**: ~8 秒
- **加速比**: 7.5x

### Population = 5000
- **序列計算**: ~6,249 秒 (1.7 小時)
- **並行計算 (8 cores)**: ~860 秒 (14.3 分鐘)
- **加速比**: 7.3x

---

## ⚠️ 風險管理

### 風險 1: TED 計算效能不足
**影響**: 高  
**緩解策略**:
- 使用高效演算法（APTED）
- 實作快取機制
- 平行化計算
- 準備近似演算法 fallback

### 風險 2: 記憶體不足
**影響**: 中  
**緩解策略**:
- 使用 shared memory
- 分批計算
- 監控記憶體使用

### 風險 3: 與 DEAP tree 不兼容
**影響**: 中  
**緩解策略**:
- 先小規模測試
- 實作適配層
- 準備 fallback 方案

---

## 📅 時程規劃

### Week 1 (Day 1-3)
- [x] 創建分支和目錄結構
- [ ] Task 2.1: Tree Edit Distance 實作
- [ ] 初步測試驗證

### Week 1-2 (Day 4-7)
- [ ] Task 2.2: Similarity Matrix 實作
- [ ] Task 2.3: 並行計算實作（開始）
- [ ] 性能測試（小規模）

### Week 2 (Day 8-10)
- [ ] Task 2.3: 並行計算實作（完成）
- [ ] Task 2.4: 快取機制實作
- [ ] 性能測試（大規模）

### Week 2 (Day 11-12)
- [ ] Task 2.5: 視覺化工具實作
- [ ] Task 2.6: 文檔與測試
- [ ] Code review & merge

---

## ✅ 驗收標準

### 功能性
- [ ] TED 計算正確（與已知結果比對）
- [ ] 相似度矩陣正確（對稱性、範圍 [0,1]）
- [ ] 並行計算結果與序列一致
- [ ] 視覺化工具可用

### 性能
- [ ] Population=500: < 10 秒
- [ ] Population=5000: < 15 分鐘
- [ ] 加速比 > 6x
- [ ] 記憶體 < 2GB

### 品質
- [ ] 測試覆蓋率 > 80%
- [ ] 所有測試通過
- [ ] 文檔完整
- [ ] Code review 通過

---

## 📚 參考資料

### 論文
1. Zhang, K., & Shasha, D. (1989). Simple fast algorithms for the editing distance between trees and related problems. SIAM journal on computing, 18(6), 1245-1262.
2. Pawlik, M., & Augsten, N. (2015). Efficient computation of the tree edit distance. ACM Transactions on Database Systems (TODS), 40(1), 1-40.

### 套件
- `apted`: Python implementation of APTED algorithm
- `zss`: Zhang-Shasha tree edit distance
- `multiprocessing`: Python parallel processing
- `numpy`: Numerical computing
- `seaborn`: Statistical visualization

---

## 📝 變更記錄

| 日期 | 變更內容 |
|------|---------|
| 2025-10-13 | 創建 Phase 2 實作計畫，創建分支和目錄結構 |

---

**準備開始實作！** 🚀
