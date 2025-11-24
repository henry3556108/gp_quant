# 🧬 GP Quant - Genetic Programming for Quantitative Trading

基於遺傳編程的量化交易策略演化系統

---

## 📋 專案概述

本專案使用遺傳編程（Genetic Programming）自動演化交易策略，支持多股票投資組合回測和並行評估。

### 核心功能
- ✅ 遺傳編程策略演化
- ✅ 多股票投資組合回測
- ✅ 並行評估（6 核心）
- ✅ 自動保存每世代最佳策略訊號
- ✅ 完整的交易記錄追蹤
- ✅ 實驗狀態可重載

---

## 🚀 快速開始

### 1. 運行測試實驗
```bash
python main_evolution.py --config configs/test_config.json --test
```

### 2. 運行大規模實驗
```bash
python main_evolution.py --config configs/large_scale_experiment.json
```

### 3. 監控實驗進度
```bash
./monitor_experiment.sh
```

### 4. 繼續未完成的實驗
```bash
python continue_evolution.py --records large_scale_records_YYYYMMDD_HHMM --generations 10
```

---

## 📁 專案結構

```
gp_quant/
├── main_evolution.py              # 主程序
├── continue_evolution.py          # 繼續演化
├── monitor_experiment.sh          # 監控腳本
│
├── configs/                       # 配置文件
│   ├── test_config.json          # 測試配置
│   └── large_scale_experiment.json  # 大規模實驗配置
│
├── gp_quant/                      # 核心代碼庫
│   ├── data/                     # 數據處理
│   ├── evolution/                # 演化引擎
│   │   └── components/          # 組件化架構
│   │       ├── backtesting/     # 回測引擎
│   │       ├── evaluators/      # 適應度評估
│   │       ├── handlers/        # 事件處理
│   │       ├── strategies/      # 演化策略
│   │       └── gp/              # GP 原語集
│   └── backtesting/              # 回測工具
│
├── TSE300_selected/               # 股票數據
│
├── docs/                          # 文檔
│   ├── NEW_FEATURES_SUMMARY.md   # 新功能總結
│   ├── TRADING_LOGIC_FIX_REPORT.md  # 交易邏輯修復
│   ├── SIGNAL_VS_TRANSACTION_EXPLANATION.md  # 訊號說明
│   └── LARGE_SCALE_EXPERIMENT_GUIDE.md  # 實驗指南
│
└── archive/                       # 歸檔文件
    ├── test_scripts/             # 測試腳本
    ├── old_docs/                 # 舊文檔
    └── old_records/              # 舊實驗記錄
```

---

## ⚙️ 配置說明

### 測試配置 (`configs/test_config.json`)
- 族群大小: 2000
- 演化世代: 10
- 適合快速測試

### 大規模配置 (`configs/large_scale_experiment.json`)
- 族群大小: 10000
- 演化世代: 25
- 適合正式實驗

### 主要參數
```json
{
  "evolution": {
    "population_size": 10000,    // 族群大小
    "generations": 25,           // 演化世代
    "max_processors": 6          // 並行核心數
  },
  "fitness": {
    "function": "excess_return", // 適應度函數
    "parameters": {
      "max_processors": 6        // 評估並行數
    }
  }
}
```

---

## 📊 實驗輸出

### 記錄目錄結構
```
large_scale_records_YYYYMMDD_HHMM/
├── config.json                    # 實驗配置
├── generation_stats.json          # 世代統計
├── final_result.json              # 最終結果
├── engine_state.pkl               # 演化狀態（可重載）
│
├── populations/                   # 族群數據
│   ├── generation_000.pkl        # 完整個體
│   └── generation_000_stats.json # 統計數據
│
├── genealogy/                     # 譜系數據
│   └── generation_000.json
│
└── best_signals/                  # 最佳個體訊號
    └── generation_000/
        ├── backtest_summary.json  # 回測摘要
        ├── entry_exit_points.csv  # 交易記錄
        └── signals_*.csv          # 每日訊號
```

---

## 🔍 結果分析

### 查看最終結果
```bash
cat large_scale_records_*/final_result.json | python -m json.tool
```

### 查看最佳策略
```bash
# 回測摘要
cat large_scale_records_*/best_signals/generation_025/backtest_summary.json

# 交易記錄
cat large_scale_records_*/best_signals/generation_025/entry_exit_points.csv

# 訊號數據
head large_scale_records_*/best_signals/generation_025/signals_ABX.TO.csv
```

### 繪製演化曲線
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 載入統計數據
with open('large_scale_records_*/generation_stats.json') as f:
    stats = json.load(f)

df = pd.DataFrame(stats)

# 繪製適應度趨勢
plt.figure(figsize=(12, 6))
plt.plot(df['generation'], df['best_fitness'], label='Best')
plt.plot(df['generation'], df['avg_fitness'], label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.savefig('evolution_progress.png')
```

---

## 🔧 核心功能

### 1. 並行評估
- 使用 6 核心並行評估個體
- 通過傳遞文件路徑解決 DataFrame 序列化問題
- 顯著提升評估速度

### 2. 交易邏輯
- 正確處理買入和賣出訊號
- 支持多次進出場
- 完整的交易記錄追蹤

### 3. 自動保存訊號
- 每世代自動保存最佳個體的交易訊號
- 包含完整的回測摘要和交易記錄
- 便於分析策略演化過程

### 4. 時間流水號
- 自動為記錄目錄添加時間戳記
- 避免數據覆蓋
- 便於管理多個實驗

---

## 📝 重要文檔

### 功能說明
- [新功能總結](NEW_FEATURES_SUMMARY.md) - 最新功能介紹
- [並行實現計劃](PARALLEL_IMPLEMENTATION_PLAN.md) - 並行評估實現
- [交易邏輯修復](TRADING_LOGIC_FIX_REPORT.md) - Bug 修復報告
- [訊號說明](SIGNAL_VS_TRANSACTION_EXPLANATION.md) - 訊號與交易對應

### 實驗指南
- [大規模實驗指南](LARGE_SCALE_EXPERIMENT_GUIDE.md) - 完整實驗流程
- [實驗狀態](EXPERIMENT_STATUS.md) - 當前實驗狀態

### 產品文檔
- [產品需求文檔](PRD.md) - 完整的產品需求

---

## 🐛 已知問題與修復

### ✅ 已修復
1. **交易邏輯 Bug** - 賣出訊號無法執行
   - 問題：訊號為 0 時不會觸發賣出
   - 修復：正確處理 `signal == 0` 的情況
   - 詳見：[TRADING_LOGIC_FIX_REPORT.md](TRADING_LOGIC_FIX_REPORT.md)

2. **DataFrame 序列化問題** - 並行評估失敗
   - 問題：DataFrame 無法序列化傳遞給子進程
   - 修復：傳遞文件路徑，子進程自行讀取
   - 詳見：[PARALLEL_IMPLEMENTATION_PLAN.md](PARALLEL_IMPLEMENTATION_PLAN.md)

---

## 💡 使用建議

### 測試階段
1. 使用 `test_config.json` 進行小規模測試
2. 驗證配置和數據正確性
3. 檢查輸出文件格式

### 正式實驗
1. 使用 `large_scale_experiment.json`
2. 預留足夠時間（約 1-1.5 小時）
3. 使用監控腳本追蹤進度
4. 實驗完成後分析結果

### 結果驗證
1. 檢查最終適應度
2. 驗證交易記錄的合理性
3. 分析訊號與交易的對應關係
4. 在測試集上驗證策略

---

## 📞 支援

如有問題，請查看：
1. [實驗指南](LARGE_SCALE_EXPERIMENT_GUIDE.md)
2. [清理摘要](CLEANUP_SUMMARY.md)
3. 歸檔文件：`archive/`

---

## 📜 版本歷史

### v2.0 (2024-11-24)
- ✅ 修復交易邏輯 Bug
- ✅ 實現並行評估
- ✅ 自動保存每世代訊號
- ✅ 添加時間流水號
- ✅ 整理專案結構

### v1.0
- 基礎演化框架
- 投資組合回測
- 單進程評估

---

**專案已整理完成，核心功能正常運行！** ✅
