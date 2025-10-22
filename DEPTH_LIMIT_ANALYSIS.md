# GP 深度限制分析與建議

## 論文要求
- **初始族群最大深度**: 6
- **後續世代最大深度**: 17

## 當前實作狀態

### ✅ 已正確實作
1. **初始族群深度限制 = 6** ✓
   ```python
   # gp_quant/evolution/engine.py, line 192
   toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
   ```

2. **後續世代深度限制 = 17** ✓
   ```python
   # gp_quant/evolution/engine.py, lines 204-205
   toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
   toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
   ```

### ⚠️ 可能需要調整的部分

**突變子樹深度限制**（當前為 0-2，可能過於保守）

```python
# gp_quant/evolution/engine.py, line 200
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
```

## 問題分析

### 當前設定的影響
- 初始族群：樹深度 2-6
- 突變生成的新子樹：深度 0-2
- 最終限制：整體深度不超過 17

### 潛在問題
1. **探索能力受限**：突變只能生成深度 0-2 的子樹，可能導致：
   - 演化後期難以產生複雜的新結構
   - 族群多樣性下降過快
   - 陷入局部最優

2. **不對稱性**：
   - 初始族群可以有深度 6 的樹
   - 但突變只能插入深度 2 的子樹
   - 這種不對稱可能不利於演化

## 建議改進方案

### 方案 1：標準設定（推薦）
將突變子樹深度提高到 **0-4**，平衡探索與控制：

```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
```

**優點**：
- 保持足夠的探索能力
- 與初始深度 6 較為協調（4 = 6 × 2/3）
- 仍受 staticLimit(17) 保護

### 方案 2：激進設定
將突變子樹深度提高到 **0-6**，與初始深度一致：

```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=6)
```

**優點**：
- 最大化探索能力
- 與初始族群深度一致
- 適合複雜問題

**缺點**：
- 可能產生過於複雜的樹
- 計算成本較高

### 方案 3：保守設定（維持現狀）
保持當前的 **0-2** 設定：

```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
```

**適用情況**：
- 如果當前實驗結果已經很好
- 計算資源有限
- 希望樹結構保持簡單

## 實作建議

### 建議執行步驟

1. **先完成當前實驗**：不要中斷正在進行的實驗
2. **小規模測試**：用 1-2 個 ticker，各跑 2-3 次，比較不同設定
3. **比較指標**：
   - 訓練期超額報酬
   - 測試期超額報酬
   - 樹的平均深度
   - 樹的平均節點數
4. **決定是否採用**：根據測試結果決定是否修改

### 修改位置

**檔案**: `gp_quant/evolution/engine.py`  
**行數**: 200  
**當前程式碼**:
```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
```

**建議修改為**（方案 1）:
```python
toolbox.register("expr_mut", gp.genFull, min_=0, max_=4)
```

## 文獻參考

根據標準 GP 實踐：
- Koza (1992): 突變深度通常為初始深度的 50%-100%
- Poli et al. (2008): 建議突變深度 = 初始深度 × 0.5 到 1.0
- 本專案初始深度 = 6，因此突變深度建議為 3-6

## 結論

**當前實作在深度限制的主要要求上是正確的**，但突變子樹深度可能過於保守。

**建議**：
- 如果當前實驗結果良好 → 可以維持現狀
- 如果希望提升性能 → 建議將突變深度改為 0-4
- 如果遇到過擬合問題 → 維持 0-2 可能更好

**不影響論文複現的核心要求**，因為論文主要規定的是：
- ✅ 初始族群最大深度 6
- ✅ 整體最大深度 17

這兩點都已正確實作。
