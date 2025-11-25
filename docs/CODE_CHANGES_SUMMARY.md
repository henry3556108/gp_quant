# 程式碼變更總結

**日期**: 2025-11-25  
**版本**: v1.0

---

## 目錄

1. [核心引擎修復](#核心引擎修復)
2. [新增分析腳本](#新增分析腳本)
3. [除錯腳本](#除錯腳本)
4. [配置變更](#配置變更)
5. [清理指令](#清理指令)

---

## 核心引擎修復

### portfolio_engine.py 修復

**檔案**: `gp_quant/evolution/components/backtesting/portfolio_engine.py`

**問題**: 最佳個體的 PnL 曲線為全零，無法產生交易訊號

**根本原因**:
1. 使用 backtest_data 而非完整 data 來計算技術指標
2. backtest_data 只包含回測期間，缺少技術指標所需的歷史數據
3. NumVector 類型不一致導致 pset 終端值設定失敗

**修復 1: 使用完整數據**
- 修改前: `df = self.backtest_data[ticker]`
- 修改後: `df = self.data[ticker]`
- 影響: 確保技術指標有足夠的歷史數據

**修復 2: 動態獲取 NumVector 類型**
- 從 pset.terminals.keys() 動態獲取 NumVector 類型
- 確保類型一致性，避免終端值設定失敗

**驗證結果**:
- 最佳個體成功產生交易訊號
- PnL 曲線正常計算
- 訓練期 fitness: 1.1263

---

## 新增分析腳本

### 1. visualize_best_individual_pnl.py

**功能**: 視覺化最佳個體的 PnL 表現與 Buy-and-Hold 策略比較

**主要特性**:
- 自動載入最後一代的最佳個體
- 計算訓練期和測試期的 PnL 曲線
- 與 Buy-and-Hold 策略對比
- 標註進出場時間點（綠色三角=進場，紅色三角=出場）
- 生成 2x1 子圖（訓練期 + 測試期）

**使用方式**:
```bash
python visualize_best_individual_pnl.py \
  --records test_evolution_records_20251125_1335 \
  --config configs/test_config.json
```

**輸出**: `{records_dir}/best_individual_pnl_comparison.png`

---

### 2. calculate_generation_diversity_matrices.py

**功能**: 計算單一世代的多樣性矩陣

**計算指標**:
1. PnL Correlation Matrix（表型多樣性）
   - 計算所有個體 PnL 曲線的相關性
   - 對角線為 1，對稱矩陣
   - 值域: [-1, 1]

2. TED Distance Matrix（基因型多樣性）
   - 計算樹編輯距離
   - 標準化: TED / max(len(tree1), len(tree2))
   - 對角線為 0，對稱矩陣
   - 值域: [0, 1]

**特殊處理**:
- 識別無效個體（全零 PnL）
- 無效個體的相關性設為 0（對角線為 1）
- 只計算有效個體之間的統計信息

**使用方式**:
```bash
python calculate_generation_diversity_matrices.py \
  --records test_evolution_records_20251125_1335 \
  --config configs/test_config.json \
  --generation 10 \
  --sample-size 20 \
  --n-jobs 4
```

**輸出**:
- `pnl_correlation_matrix_gen{N:03d}.csv`
- `ted_distance_matrix_normalized_gen{N:03d}.csv`

---

### 3. visualize_diversity_evolution.py

**功能**: 分析並視覺化所有世代的多樣性演化趨勢

**計算流程**:
1. 載入所有世代的族群
2. 對每個世代計算 PnL correlation 和 TED distance
3. 生成演化趨勢圖

**關鍵技術 - 平行化策略**:
- 使用 threading backend 避免 DEAP creator 的 pickle 問題
- 世代級別: 順序處理
- 世代內部: 平行化處理

**使用方式**:
```bash
python visualize_diversity_evolution.py \
  --records test_evolution_records_20251125_1335 \
  --config configs/test_config.json \
  --sample-size 20 \
  --n-jobs 4
```

**輸出**:
- `diversity_evolution.png` (2x1 折線圖)
- `diversity_evolution_summary.csv` (統計數據)

**圖表內容**:
- 上圖: PnL Correlation 演化趨勢（表型收斂）
- 下圖: TED Distance 演化趨勢（基因型多樣性）
- 包含趨勢線（2階多項式擬合）
- 顯示統計信息（範圍、平均值、標準差）

---

## 除錯腳本

### 1. debug_signal_generation.py
- 隔離測試單一個體的訊號生成過程
- 詳細輸出每個步驟
- 用於診斷訊號生成問題

### 2. debug_pset.py
- 檢查 pset 的深拷貝行為
- 驗證 NumVector 類型一致性
- 用於診斷類型不匹配問題

### 3. analyze_fitness_distribution.py
- 分析族群的 fitness 分佈
- 識別異常值
- 視覺化 fitness 演化趨勢

---

## 配置變更

### .gitignore
新增: `test_evolution_*`
目的: 排除實驗記錄目錄，避免大量數據文件進入版本控制

---

## 變更統計

### 新增文件
- visualize_best_individual_pnl.py (417 行)
- calculate_generation_diversity_matrices.py (332 行)
- visualize_diversity_evolution.py (528 行)
- debug_signal_generation.py (104 行)
- debug_pset.py (26 行)
- analyze_fitness_distribution.py

### 修改文件
- gp_quant/evolution/components/backtesting/portfolio_engine.py
  - 修改 _generate_signals_for_all_stocks 方法
  - 約 30 行變更

### 配置文件
- .gitignore (1 行新增)

---

## 核心改進總結

### 1. 修復關鍵 Bug
- 問題: 最佳個體 PnL 為零
- 原因: 技術指標計算缺少歷史數據 + NumVector 類型不匹配
- 解決: 使用完整數據 + 動態類型獲取
- 影響: 所有依賴 PnL 計算的功能恢復正常

### 2. 完整分析工具鏈
- 腳本一: 個體表現分析（PnL vs Buy-and-Hold）
- 腳本二: 單世代多樣性分析（矩陣計算）
- 腳本三: 多世代演化分析（趨勢視覺化）

### 3. 平行化優化
- 使用 threading backend 解決 DEAP 序列化問題
- 世代內部平行化，提升計算效率
- 26 個世代分析僅需 50 秒

---

## 清理指令

### 刪除除錯腳本
```bash
rm debug_signal_generation.py debug_pset.py
```

### 刪除臨時文檔
```bash
rm BUG_ANALYSIS_REPORT.md \
   MODIFICATION_SUMMARY.md \
   PNL_CORRELATION_ZERO_VALUES_EXPLANATION.md \
   PNL_DIVERSITY_DIAGNOSIS_REPORT.md \
   SAFE_ENHANCEMENT_PLAN.md
```

### 清理測試數據（已在 .gitignore）
```bash
# 不需要手動刪除，已被 git 忽略
# 如需清理磁盤空間：
# rm -rf test_evolution_*
```

### 一鍵清理所有非必要文件
```bash
rm debug_signal_generation.py debug_pset.py \
   BUG_ANALYSIS_REPORT.md \
   MODIFICATION_SUMMARY.md \
   PNL_CORRELATION_ZERO_VALUES_EXPLANATION.md \
   PNL_DIVERSITY_DIAGNOSIS_REPORT.md \
   SAFE_ENHANCEMENT_PLAN.md
```
