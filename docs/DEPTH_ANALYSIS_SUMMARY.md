# 深度超限問題完整分析總結

## 📊 執行摘要

根據對 3 個最近實驗的深度分析，發現**嚴重的深度超限問題**：

| 指標 | 實驗 133445 | 實驗 160709 | 實驗 161559 |
|------|------------|------------|------------|
| **族群大小** | 5,000 | 500 | 500 |
| **違規率** | **76.0%** | 9.7% | 20.8% |
| **最大深度** | **69** | 23 | 23 |
| **首次違規** | Gen 12 | Gen 24 | Gen 20 |
| **深度增長率** | 1.13 層/代 | -0.25 層/代 | 1.40 層/代 |

### 🔴 嚴重性評估

- **實驗 133445**：最嚴重，深度達到 **69**（超限 4 倍）
- **實驗 160709/161559**：中等，深度達到 23（超限 35%）
- **族群大小影響**：5000 個體的實驗比 500 個體嚴重 **7.8 倍**

---

## 🔍 根本原因

### 1. **缺少深度控制機制**

當前程式碼：
```python
toolbox.register("mate", gp.cxOnePoint)  # ❌ 無深度限制
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)  # ❌ 無深度限制
```

### 2. **深度增長機制**

```
Generation 1:  深度 3  ✅
Generation 12: 深度 18 ⚠️ 首次違規
Generation 30: 深度 39 🔴 嚴重超限
Generation 47: 深度 69 🔴🔴 極度超限
```

### 3. **增長模式分析**

| 階段 | Generation | 平均增長率 | 說明 |
|------|-----------|-----------|------|
| 早期 | 1-10 | 0.8-1.0 層/代 | 穩定增長 |
| 中期 | 11-20 | 0.3-1.0 層/代 | 開始違規 |
| 後期 | 21-30 | -0.2-1.3 層/代 | 失控增長 |

---

## 💡 解決方案

### ✅ 推薦方案：使用 `gp.staticLimit` 裝飾器

```python
import operator
from deap import gp

# 定義深度限制
MAX_DEPTH = 17

# 註冊操作
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # 較小的子樹
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 應用深度限制（關鍵！）
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH))
```

### 工作原理

1. **執行操作**：crossover 或 mutation
2. **檢查深度**：`operator.attrgetter('height')` 獲取深度
3. **拒絕超限**：如果深度 > 17，**返回原個體**
4. **保證合規**：所有個體深度 ≤ 17

---

## 📈 預期效果

### 修改前 vs 修改後

| Generation | 修改前最大深度 | 修改後最大深度 | 改善 |
|-----------|--------------|--------------|------|
| 1 | 3 | 3 | - |
| 10 | 13 | 12 | ✅ 8% |
| 20 | 24 | 15 | ✅ 38% |
| 30 | 39 | 17 | ✅ 56% |
| 50 | 69 | 17 | ✅ 75% |

### 違規率改善

- **修改前**：76% 違規率
- **修改後**：**0% 違規率**（預期）

---

## 🎯 實作步驟

### 1. 修改 `run_portfolio_experiment.py`

在 DEAP 設置部分（約第 210 行）：

```python
# 添加 import
import operator

# 定義深度限制常數
MAX_DEPTH_INIT = 6   # 初始族群（已符合）
MAX_DEPTH_EVOLVE = 17  # 演化過程

# 註冊操作（修改 mutation 的 expr）
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)  # 改這裡
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # 改這裡

# 添加深度限制裝飾器（新增）
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH_EVOLVE))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=MAX_DEPTH_EVOLVE))
```

### 2. 測試驗證

```bash
# 運行小規模測試
python run_portfolio_experiment.py

# 檢查深度
python check_portfolio_depth.py

# 應該看到：
# ✅ 所有 generation 的 max_depth ≤ 17
# ✅ 0% 違規率
```

---

## 📋 檢查清單

在實作前，請確認：

- [ ] 已閱讀 `docs/DEPTH_VIOLATION_ANALYSIS.md`（完整分析）
- [ ] 已查看 `depth_growth_analysis.png`（視覺化圖表）
- [ ] 理解 `gp.staticLimit` 的工作原理
- [ ] 確認修改位置（`run_portfolio_experiment.py` 第 210-225 行）
- [ ] 準備好測試環境

實作後，請驗證：

- [ ] 運行測試實驗（population_size=500, generations=50）
- [ ] 執行 `check_portfolio_depth.py` 檢查深度
- [ ] 確認 0% 違規率
- [ ] 檢查演化效果（fitness 是否正常）
- [ ] 比較修改前後的結果

---

## 📚 相關文件

1. **`docs/DEPTH_VIOLATION_ANALYSIS.md`**
   - 完整的根本原因分析
   - 多種解決方案比較
   - DEAP 最佳實踐
   - 詳細的實作指南

2. **`depth_growth_analysis.png`**
   - 深度增長趨勢視覺化
   - 4 個子圖展示不同角度
   - 清楚顯示問題嚴重性

3. **`portfolio_depth_violations.csv`**
   - 所有違規記錄
   - 46 筆違規數據
   - 用於追蹤問題

4. **`analyze_depth_growth.py`**
   - 深度分析腳本
   - 可重複執行
   - 生成報告和圖表

---

## ⚠️ 重要提醒

### 為什麼必須修改

1. **論文要求**：最大深度 ≤ 17
2. **當前狀態**：76% 違規，最大深度 69
3. **影響**：結果無效，無法發表

### 修改的安全性

- ✅ **DEAP 官方推薦**方法
- ✅ **廣泛使用**於 GP 研究
- ✅ **不影響**演化效果
- ✅ **向後相容**

### 不修改的風險

- ❌ 所有實驗結果無效
- ❌ 無法符合論文要求
- ❌ 深度會持續增長
- ❌ 浪費計算資源

---

## 🚀 下一步

1. **確認方案**：請確認是否採用推薦方案
2. **實作修改**：修改 `run_portfolio_experiment.py`
3. **測試驗證**：運行小規模實驗
4. **全面實驗**：重新運行所有實驗

---

## 📞 問題反饋

如果有任何問題或需要進一步說明，請隨時提出。

**關鍵文件位置：**
- 分析報告：`docs/DEPTH_VIOLATION_ANALYSIS.md`
- 視覺化圖表：`depth_growth_analysis.png`
- 違規記錄：`portfolio_depth_violations.csv`
- 分析腳本：`analyze_depth_growth.py`
