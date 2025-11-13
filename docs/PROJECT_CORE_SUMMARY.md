# GP Quant 專案核心內容總結

## 📋 專案概述

**專案名稱**: GP Quant - 遺傳編程股票交易規則生成系統  
**核心目標**: 基於 Potvin et al. (2004) 論文實作遺傳編程自動生成股票交易規則  
**當前狀態**: 重構階段，存在代碼重複問題需要解決

## 🏗️ 專案架構

### 核心組件
```
gp_quant/
├── evolution/engine.py          # 核心演化引擎 (✅ 已修復深度限制)
├── gp/operators.py             # GP 運算子和函式集
├── backtesting/engine.py       # 回測引擎
├── data/loader.py              # 數據載入器
├── niching/                    # Niching 策略模組
└── similarity/                 # 樹相似度計算模組
```

### 兩個主要入口點
1. **`main.py`** → 調用 `engine.py` (標準 GP 實驗)
2. **`run_portfolio_experiment.py`** → 獨立實作演化循環 (Portfolio + Niching 實驗)

## 🔴 核心問題：代碼重複

### 問題描述
- `run_portfolio_experiment.py` 重新實作了完整的演化循環 (~150-200 行重複代碼)
- 沒有重用 `engine.py` 中的核心邏輯和 bug 修復
- 導致深度超限問題：`engine.py` 有 `staticLimit` 修復，但 `run_portfolio_experiment.py` 沒有

### 影響
- **深度違規率**: `engine.py` 0% vs `run_portfolio_experiment.py` 76%
- **維護困難**: 需要同步修改兩個文件
- **結果不一致**: Portfolio 實驗結果可能無效

## 🎯 重構方案

### 推薦路徑 (基於 `docs/REFACTORING_PROPOSAL.md`)

1. **立即修復 (方案 A)**:
   - 讓 `run_portfolio_experiment.py` 調用 `engine.py`
   - 通過 `generation_callback` 實作 Niching 邏輯
   - 立即獲得深度限制修復

2. **中期優化 (方案 B)**:
   - 擴展 `engine.py` 支援 Niching 配置參數
   - 統一演化引擎接口

3. **長期架構 (方案 C)**:
   - 組件化設計，可插拔策略模式

## 📊 實驗配置與結果

### 論文參數設定
- **族群大小**: 500
- **世代數**: 50
- **複製率**: 35%, **交配率**: 60%, **突變率**: 5%
- **最大深度**: 初始 6, 後續 17
- **選擇方法**: 排名 + SUS

### 最新實驗結果 (基於 `EXPERIMENT_SUMMARY.md`)
- **總實驗數**: 80 (4 股票 × 2 訓練期 × 10 次運行)
- **成功率**: 92.5% (74/80)
- **樣本外勝率**: 38.8% (31/80)
- **最佳表現**: ABX.TO 短訓練期 Run 5 ($59,658.31 超額報酬)

## 🔧 關鍵技術特性

### GP 函式集
- **布林運算子**: and, or, not
- **算術運算子**: +, -, /, ×
- **關係運算子**: <, >
- **技術指標**: RSI, ROC, avg, max, min, lag, volatility

### 特殊功能
- **Niching 策略**: 基於樹結構相似度的族群多樣性維持
- **Early Stopping**: 防止過度擬合
- **Portfolio 評估**: 多股票組合策略演化
- **並行處理**: 支援多核心加速

## 📁 檔案分類建議

### 🟢 核心保留檔案
**技術文檔**:
- `PRD.md` - 產品需求文檔
- `docs/REFACTORING_PROPOSAL.md` - 重構方案
- `docs/CODE_DUPLICATION_ANALYSIS.md` - 代碼重複分析
- `docs/DEPTH_*.md` (3個) - 深度超限問題分析
- `docs/*_guide.md` (6個) - 技術使用指南

**核心代碼**:
- `gp_quant/` - 主要程式碼模組
- `main.py` - 標準 GP 入口
- `run_portfolio_experiment.py` - Portfolio 實驗入口 (待重構)
- `requirements.txt` - 依賴管理

### 🟡 可封存檔案 (歷史記錄)
- `IMPLEMENTATION_PLAN.md` - 實作計劃
- `EXPERIMENT_SUMMARY.md` - 實驗總結
- `NICHING_*_SUMMARY.md` (2個) - Niching 整合總結
- `PARALLEL_SIMILARITY_SUMMARY.md` - 並行相似度總結
- `PHASE2*.md` (3個) - 階段性計劃
- `PROJECT_STATUS.md` - 專案狀態
- `PORTFOLIO_EXPERIMENT_README.md` - Portfolio 實驗說明

### 🔴 可安全移除檔案
- `docs/CPU_CORE_MANAGEMENT.md` - CPU 管理指南 (環境特定)
- `docs/CPU_TEMPERATURE_GUIDE.md` - 溫度管理指南 (環境特定)

## 🚀 下一步行動

### 優先級 1: 修復代碼重複
1. 實作重構方案 A (快速修復)
2. 驗證深度限制問題解決
3. 確認 Niching 功能正常運作

### 優先級 2: 整理專案
1. 封存歷史文檔到 `archive/` 目錄
2. 移除環境特定檔案
3. 更新 `.gitignore` 忽略實驗結果

### 優先級 3: 持續優化
1. 實作重構方案 B (統一接口)
2. 改善測試覆蓋率
3. 文檔標準化

## 📈 專案價值

### 學術價值
- 忠實實作經典 GP 交易論文
- 擴展支援現代技術 (Niching, Portfolio)
- 提供可重現的實驗框架

### 技術價值
- 模組化 GP 演化引擎
- 高效能並行處理
- 完整的回測和評估系統

### 商業價值
- 自動化交易策略生成
- 多股票組合優化
- 風險管理和多樣性維持

---

**建立時間**: 2025-11-13  
**基於分析**: 專案結構、代碼重複問題、重構方案、實驗結果  
**目的**: 為專案重構和檔案整理提供核心參考
