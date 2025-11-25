# 實作 TODO List

## 目標
基於 fe73cd9 穩定版本，創建三個分析腳本用於 PnL 和 TED 多樣性分析。

## 測試範例
使用 `/Users/hongyicheng/Downloads/gp_quant/test_evolution_records_20251125_1218` 作為測試數據。

---

## 腳本一：`visualize_best_individual_pnl.py` ✅ 已完成

### 功能描述
從實驗結果中找出 global best individual，計算其在樣本內外的 PnL curve，並與 buy-and-hold 策略對比視覺化。

### 實作任務
- [x] 從所有世代中找出 fitness 最高的個體（global best）
- [x] 載入實驗配置和數據（train/test data）
- [x] 創建兩個回測引擎（train period 和 test period）
- [x] 計算 best individual 的 PnL curve（train 和 test）
- [x] 計算 buy-and-hold 策略的 PnL curve（train 和 test）
- [x] 繪製 2x1 子圖：
  - 上圖：Train period - Best Individual vs Buy-and-Hold
  - 下圖：Test period - Best Individual vs Buy-and-Hold
- [x] 保存圖表為 PNG
- [x] 添加命令行參數：`--records`, `--config`, `--output`

### 驗收標準
1. ✅ **執行成功**：運行 `python visualize_best_individual_pnl.py --records test_evolution_records_20251125_1218 --config configs/test_config.json` 能成功生成一張包含上下兩個子圖的 PNG 圖表，清楚顯示最佳個體與 buy-and-hold 在樣本內外的 PnL 對比。
2. ✅ **數據正確**：圖表中的 PnL 曲線數值合理（非全零），best individual 的 fitness 值顯示在圖表標題或圖例中，且能明確看出兩種策略的差異。

### 完成時間
2025-11-25 13:16

### 備註
- 修復了 NumVector 類型不匹配問題
- 修復了數據範圍不足導致技術指標無法計算的問題
- 詳見 `MODIFICATION_SUMMARY.md`

---

## 腳本二：`calculate_generation_diversity_matrices.py` ✅ 已完成

### 功能描述
計算指定世代（默認最新一代）的 PnL correlation matrix 和標準化 TED distance matrix，支援採樣和平行化計算。

### 實作任務
- [x] 載入指定世代的 population（默認最新一代）
- [x] 實作採樣功能：
  - 參數 `--sample-size N`（可選）
  - 分層採樣策略（按 fitness 排序後均勻採樣）
- [x] 實作平行化 PnL correlation 計算：
  - 使用 `joblib.Parallel`
  - 為每個個體計算 PnL curve
  - 計算兩兩相關係數矩陣
- [x] 實作平行化 TED distance 計算：
  - 平行計算兩兩 TED 距離
  - 標準化：`ted / max(len(tree1), len(tree2))`
  - 構建對稱矩陣
- [x] 保存兩個 CSV 檔案：
  - `pnl_correlation_matrix_genXXX.csv`
  - `ted_distance_matrix_normalized_genXXX.csv`
- [x] 添加命令行參數：`--records`, `--config`, `--generation`, `--sample-size`, `--n-jobs`
- [x] 添加進度條顯示（使用 `tqdm`）
- [x] 錯誤處理和日誌記錄

### 驗收標準
1. ✅ **執行成功**：運行 `python calculate_generation_diversity_matrices.py --records test_evolution_records_20251125_1218 --config configs/test_config.json --sample-size 20 --n-jobs 4` 在 ~3 秒內成功生成兩個 CSV 檔案，矩陣維度為 20x20。
2. ✅ **數據正確**：
   - PnL correlation matrix: 值在 [-0.6403, 1.0000]，對角線為 1.0，對稱矩陣，平均值 0.5012
   - TED distance matrix: 值在 [0.0000, 0.5455]，對角線為 0.0，對稱矩陣，平均值 0.2957

### 完成時間
2025-11-25 13:20

---

## 腳本三：`visualize_diversity_evolution.py` ✅

**完成日期**: 2025-11-25

### 功能描述
計算所有世代的多樣性指標（平均 PnL correlation 和平均標準化 TED distance），並繪製演化趨勢折線圖。

### 實作任務
- [x] 載入所有世代的 populations
- [x] 對每個世代：
  - 採樣（如果指定 `--sample-size`）
  - 平行計算 PnL correlation matrix
  - 平行計算 TED distance matrix
  - 計算矩陣非對角線元素的平均值
- [x] 收集所有世代的統計數據
- [x] 繪製 2x1 子圖：
  - 上圖：X=generation, Y=平均 PnL correlation（折線圖）
  - 下圖：X=generation, Y=平均標準化 TED distance（折線圖）
- [x] 保存圖表為 PNG
- [x] 可選：保存每個世代的矩陣到 CSV（參數 `--save-matrices`）
- [x] 可選：保存統計摘要到 CSV（`diversity_evolution_summary.csv`）
- [x] 添加命令行參數：`--records`, `--config`, `--sample-size`, `--n-jobs`, `--save-matrices`, `--output`
- [x] 添加整體進度條（世代級別）

### 驗收標準
1. **執行成功** ✅：成功生成包含上下兩個子圖的 PNG 圖表，清楚顯示平均 PnL correlation 和平均 TED distance 隨世代的變化趨勢。
2. **趨勢合理** ✅：圖表顯示合理的演化趨勢，並成功保存 CSV 文件包含每個世代的統計數據。

### 關鍵實作細節
- **平行化策略**：使用 `backend='threading'` 避免 DEAP creator 的 pickle 問題
- **執行策略**：世代級別順序處理，世代內部平行化計算矩陣
- **無效個體處理**：正確識別並處理全零 PnL 的個體，只計算有效個體之間的統計信息

---

## 通用要求

### 代碼質量
- [ ] 所有腳本都有清晰的 docstring
- [ ] 使用 argparse 處理命令行參數
- [ ] 使用 logging 記錄執行過程
- [ ] 適當的錯誤處理和異常捕獲
- [ ] 代碼符合 PEP 8 規範

### 依賴管理
- [ ] 確保所有導入的模組都存在於當前環境
- [ ] 使用現有的工具函數（不重複造輪子）
- [ ] 參考現有代碼：
  - `test_pnl_diversity.py`
  - `analyze_evolution_diversity.py`
  - `gp_quant/similarity/tree_edit_distance.py`

### 測試驗證
- [ ] 使用 `test_evolution_records_20251125_1218` 測試所有腳本
- [ ] 確保生成的圖表和數據文件格式正確
- [ ] 驗證數值的合理性（範圍、對稱性等）

---

## 實作順序

1. **腳本一** - 最簡單，單個個體分析
2. **腳本二** - 中等複雜度，單個世代矩陣計算
3. **腳本三** - 最複雜，基於腳本二擴展到所有世代

---

## 預期產出

### 腳本一產出
- `best_individual_pnl_comparison.png` - 2x1 子圖

### 腳本二產出
- `pnl_correlation_matrix_gen010.csv` - NxN 矩陣
- `ted_distance_matrix_normalized_gen010.csv` - NxN 矩陣

### 腳本三產出
- `diversity_evolution.png` - 2x1 折線圖
- `diversity_evolution_summary.csv` - 統計摘要（可選）
- `diversity_matrices/` - 每個世代的矩陣（可選）

---

## 注意事項

⚠️ **不要修改任何現有的核心代碼**
- 不修改 `PortfolioBacktestingEngine`
- 不修改 `compute_ted()`
- 不修改演化引擎

✅ **只創建新的分析腳本**
- 所有腳本都是獨立的
- 使用現有的 API 和工具函數
- 事後分析，不影響演化過程

---

## 完成標準

所有三個腳本都能：
1. ✅ 成功執行並生成預期的輸出文件
2. ✅ 通過驗收標準測試
3. ✅ 代碼清晰易讀，有適當的註釋和文檔
4. ✅ 使用 `test_evolution_records_20251125_1218` 測試通過
