# PnL Diversity 問題診斷報告

## 📋 執行日期
2025-11-25

## 🎯 問題描述
測試腳本無法從實驗結果中的個體計算出有效的 PnL 曲線，所有個體的 PnL 曲線都是全零（或接近零的浮點數誤差）。

## 🔍 診斷過程

### 步驟 1: 檢查數據載入
- ✅ 數據字典正確載入
- ✅ 包含 11 個 ticker: XLE, XLF, XLP, XLC, XLB, XLU, XLV, XLY, XLK, XLI, XLRE
- ✅ 數據形狀正確: (636, 8) for train data

### 步驟 2: 檢查 PortfolioBacktestingEngine 初始化
- ✅ Engine 成功初始化
- ✅ backtest_start: 2019-01-01
- ✅ backtest_end: 2020-12-31

### 步驟 3: 檢查個體回測結果
對於所有測試的個體（generation 0 和 generation 10）：
- ❌ **Transactions: 0 rows** - 沒有任何交易記錄
- ❌ **Equity curve: 全部為 100000.0** - 始終等於初始資金
- ❌ **PnL curve: 全部為 -1.45519152e-11** - 浮點數精度誤差，實際為 0

### 步驟 4: 檢查實驗結果中的 best individual
從 `test_evolution_records_20251125_0000/best_signals/generation_000/` 檢查：
- ✅ **entry_exit_points.csv 存在且有 70 筆交易記錄**
- ✅ **backtest_summary.json 顯示:**
  - fitness: 0.1049
  - total_transactions: 70
  - final_value: 110490.37
  - pnl: 10490.37

## 🚨 根本原因

### **數據集不匹配！**

1. **實驗結果中的交易記錄使用的 tickers:**
   - `ABX.TO` (加拿大股票)
   - `BBD-B.TO` (加拿大股票)
   - `RY.TO` (加拿大股票)
   - `TRP.TO` (加拿大股票)

2. **測試腳本載入的 tickers (從 `history/` 目錄):**
   - `XLE`, `XLF`, `XLP`, `XLC`, `XLB`, `XLU`, `XLV`, `XLY`, `XLK`, `XLI`, `XLRE` (美國 ETF)

3. **結論:**
   - 實驗結果 `test_evolution_records_20251125_0000` 是用**加拿大股票數據**運行的
   - 但當前 `configs/test_config.json` 指向的 `history/` 目錄包含的是**美國 ETF 數據**
   - 當我們用美國 ETF 數據回測訓練在加拿大股票上的個體時，個體的交易規則無法產生任何交易信號

## 💡 解決方案

### 方案 1: 使用正確的數據集
找到實驗運行時使用的加拿大股票數據目錄，並在測試腳本中使用相同的數據。

### 方案 2: 使用當前數據重新運行實驗
使用 `history/` 目錄（美國 ETF）重新運行演化實驗，生成新的實驗結果。

### 方案 3: 直接從實驗結果計算 PnL 相關性
不重新回測，而是從已保存的 `best_signals/` 中讀取每個 generation 的 equity curve 或 PnL 數據，直接計算相關性。

## 📊 技術細節

### PortfolioBacktestingEngine 行為
當個體的交易規則無法在給定的數據上產生任何交易信號時：
- `transactions` DataFrame 為空 (0 rows)
- `equity_curve` 保持為初始資金 (100000.0)
- `pnl_curve` 為 0 (或浮點數精度誤差)

這是**預期行為**，不是 bug。問題在於數據集不匹配。

### 為什麼個體有 fitness 但沒有交易？
- 個體的 `fitness.values[0]` 是在**訓練時**（使用加拿大股票數據）計算並保存的
- 當我們用**不同的數據**（美國 ETF）回測時，個體無法產生交易信號
- 這證實了數據集不匹配的結論

## ✅ 驗證
可以通過以下方式驗證：
1. 檢查實驗運行時的日誌，確認使用的數據目錄
2. 查找包含 `ABX.TO`, `BBD-B.TO`, `RY.TO`, `TRP.TO` 的數據目錄
3. 使用正確的數據重新測試

## 📝 建議
1. 在實驗配置中明確記錄使用的 ticker 列表
2. 在實驗結果目錄中保存數據配置的副本
3. 添加數據集驗證機制，確保測試時使用的數據與訓練時一致
