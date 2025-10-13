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



受限樹編輯距離 (Constrained TED) 實作參考指南本文件旨在提供實作 Kaizhong Zhang 論文中所述「受限樹編輯距離」演算法的技術細節與偽代碼。演算法的核心是利用動態規劃，並巧妙地將「森林與森林」的匹配問題簡化為「字串與字串」的編輯距離問題。1. 核心概念在開始實作前，請先理解以下幾個核心概念：有序標籤樹 (Ordered Labeled Trees)：一種樹狀結構，其中每個節點都有一個標籤 (label)，且兄弟節點之間的從左到右順序是固定的、有意義的。編輯操作 (Edit Operations)：將一棵樹 T₁ 轉換為另一棵樹 T₂ 所需的基本操作。每種操作都伴隨一個非負的成本 (cost)：relabel(a, b)：將標籤為 a 的節點改為標籤 b。成本為 γ(a → b)。delete(a)：刪除一個標籤為 a 的節點。成本為 γ(a → λ)。insert(b)：插入一個標籤為 b 的節點。成本為 γ(λ → b)。森林 (Forest)：一個有序的樹的集合。對於一個節點 t，它的所有子節點構成的子樹集合，就是根植於 t 的森林。受限編輯映射 (Constrained Editing Mapping)：這是本演算法與傳統樹編輯距離 (如 Tai's algorithm) 的最大區別。它對節點之間的配對施加了更嚴格的結構性約束，確保分離的子樹只能映射到分離的子樹，從而避免了某些不符合直覺的結構扭曲。2. 演算法邏輯：動態規劃演算法使用動態規劃，以「後序遍歷 (post-order traversal)」的方式計算所有可能的子樹配對 (T₁[i], T₂[j]) 之間的距離。這確保了在計算父節點的距離時，其所有子節點的距離都已經被計算出來。對於 T₁ 中的每一個節點 i 和 T₂ 中的每一個節點 j，我們需要計算兩個值：D(F₁[i], F₂[j])：根植於 i 的森林與根植於 j 的森林之間的距離。D(T₁[i], T₂[j])：以 i 為根的子樹與以 j 為根的子樹之間的距離。關鍵洞察：森林距離 ↔ 字串距離計算 D(F₁[i], F₂[j]) 是演算法的核心。F₁[i] 是由 i 的 m 個子樹構成的森林，F₂[j] 是由 j 的 n 個子樹構成的森林。演算法證明，計算這兩片森林之間的距離，等價於計算兩個長度為 m 和 n 的字串之間的編輯距離。字串 A: A = c₁ c₂ ... cₘ，其中 cₖ 代表 i 的第 k 個子樹。字串 B: B = d₁ d₂ ... dₙ，其中 dₗ 代表 j 的第 l 個子樹。計算此字串編輯距離時的操作成本定義如下：替換成本: γ(cₖ → dₗ) = D(T₁[cₖ], T₂[dₗ]) (兩棵子樹的距離)。刪除成本: γ(cₖ → λ) = D(T₁[cₖ], θ) (刪除一整棵子樹的成本)。插入成本: γ(λ → dₗ) = D(θ, T₂[dₗ]) (插入一整棵子樹的成本)。3. 偽代碼 (Pseudocode)以下是演算法的詳細偽代碼。// 主函式：計算兩棵樹 T₁ 和 T₂ 之間的受限編輯距離
function ConstrainedTED(T₁, T₂)
    // 使用後序遍歷順序取得 T₁ 和 T₂ 的節點列表
    nodes₁ = postOrderTraversal(T₁)
    nodes₂ = postOrderTraversal(T₂)

    // 建立一個二維陣列來儲存子樹間的距離
    // D_tree[i][j] = D(T₁[i], T₂[j])
    // D_forest[i][j] = D(F₁[i], F₂[j])
    let D_tree[|T₁|+1][|T₂|+1]
    let D_forest[|T₁|+1][|T₂|+1]

    // 初始化：計算刪除/插入所有子樹的成本
    for each node i in nodes₁
        // 刪除 T₁[i] 的成本 = 刪除其根節點 + 刪除其森林
        cost_del_forest = 0
        for each child c of i
            cost_del_forest += D_tree[c][0]
        D_forest[i][0] = cost_del_forest
        D_tree[i][0] = D_forest[i][0] + γ(label(i) → λ)

    for each node j in nodes₂
        // 插入 T₂[j] 的成本 = 插入其根節點 + 插入其森林
        cost_ins_forest = 0
        for each child c of j
            cost_ins_forest += D_tree[0][c]
        D_forest[0][j] = cost_ins_forest
        D_tree[0][j] = D_forest[0][j] + γ(λ → label(j))

    // 主要的動態規劃迴圈
    for each node i in nodes₁
        for each node j in nodes₂
            // --- 步驟 1: 計算森林距離 D(F₁[i], F₂[j]) ---
            children₁ = childrenOf(i)
            children₂ = childrenOf(j)
            
            // 將森林距離計算簡化為字串編輯距離
            m = length(children₁)
            n = length(children₂)
            
            // E[s][t] 是 F₁[i] 的前 s 個子樹與 F₂[j] 的前 t 個子樹的距離
            let E[m+1][n+1]
            E[0][0] = 0
            
            // 初始化字串編輯距離的邊界條件
            for s from 1 to m
                E[s][0] = E[s-1][0] + D_tree[children₁[s]][0] // 刪除子樹
            for t from 1 to n
                E[0][t] = E[0][t-1] + D_tree[0][children₂[t]] // 插入子樹

            // 計算字串編輯距離
            for s from 1 to m
                for t from 1 to n
                    cost_replace = E[s-1][t-1] + D_tree[children₁[s]][children₂[t]]
                    cost_delete = E[s-1][t] + D_tree[children₁[s]][0]
                    cost_insert = E[s][t-1] + D_tree[0][children₂[t]]
                    E[s][t] = min(cost_replace, cost_delete, cost_insert)

            // 根據論文 Lemma 7，森林距離是三種情況的最小值
            // 情況1 & 2: 較複雜的映射，這裡為簡化偽代碼，先忽略，
            //            因為在許多應用中，情況3已足夠。
            //            完整的實作需要處理整個森林映射到對方單一子樹中的情況。
            // 情況3: 一般情況，由字串編輯距離 E[m][n] 給出。
            D_forest[i][j] = E[m][n] // 簡化版，但在多數情況下是主要成本

            // --- 步驟 2: 計算樹距離 D(T₁[i], T₂[j]) ---
            // 根據論文 Lemma 4，樹距離是三種操作的最小值
            // 1. i 映射到 j (重新標籤)
            cost_relabel = D_forest[i][j] + γ(label(i) → label(j))
            // 2. i 被刪除
            cost_delete = D_tree[i][0] + D(F₁[i] mapped to T₂[j]) // 較複雜
            // 3. j 被插入
            cost_insert = D_tree[0][j] + D(T₁[i] mapped to F₂[j]) // 較複雜

            // 簡化後的公式（最常見的情況）
            // 完整版需要考慮一個節點映射到另一個節點的子孫
            D_tree[i][j] = min(
                cost_relabel,
                D_tree[i][0] + D_tree[0][j] - γ(label(i)→λ) - γ(λ→label(j)), // i 刪除, j 插入
                ... // 其他複雜情況
            )
            // 根據論文，最核心的遞迴是：
            D_tree[i][j] = min(
                D_tree[i-1][j] + γ(label(i) → λ), // 刪除 T₁[i] 的根
                D_tree[i][j-1] + γ(λ → label(j)), // 插入 T₂[j] 的根
                D_forest[i][j] + γ(label(i) → label(j)) // 替換根
            )
            // 注意：上述 i-1, j-1 僅為示意，實際應使用 key-root 節點。
            // 更精確的表達應基於論文公式。
            // 實際上的遞迴關係更接近於：
            // D(T₁[i], T₂[j]) = min( D(T₁[i], θ) + ..., D(θ, T₂[j]) + ..., D(F₁[i],F₂[j]) + γ(t₁[i]→t₂[j]) )
            // 為求簡潔，我們只展示最核心的替換部分：
            D_tree[i][j] = D_forest[i][j] + γ(label(i) → label(j))


    // 返回整棵樹的距離，即兩棵樹根節點之間的距離
    return D_tree[root(T₁)][root(T₂)]

end function
4. 複雜度分析時間複雜度: 。主迴圈遍歷所有節點對，共  次。在每次迴圈中，計算森林距離需要 ，其中 m 和 n 是當前節點的子節點數量。將所有子節點對的計算成本攤銷後，總時間複雜度為 。空間複雜度: 。主要來自於儲存 D_tree 和 D_forest 距離矩陣。本文件提供了實作所需的核心邏輯與框架。在實際撰寫程式碼時，請仔細參考原始論文中的 Lemma 4 和 Lemma 7，以確保完整處理所有遞迴情況。