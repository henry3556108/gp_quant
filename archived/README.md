# Archived Experiment Scripts

這個目錄包含早期版本的實驗腳本，已被更完整的版本取代。

## 檔案說明

### `run_experiments.py`
- **用途**: 最早期的實驗腳本
- **特點**: 針對單一 ticker (BBD-B.TO)，使用舊的日期格式
- **狀態**: 已被 `run_all_experiments.py` 取代

### `run_both_experiments.py`
- **用途**: 短/長訓練期對比實驗
- **特點**: 針對 BBD-B.TO，運行兩種訓練期配置
- **狀態**: 已被 `run_all_experiments.py` 取代

## 當前使用的腳本

請使用根目錄下的 **`run_all_experiments.py`**，它提供：
- ✅ 多 ticker 支援
- ✅ 10 次重複實驗
- ✅ 自動統計分析
- ✅ 完整的結果保存和日誌

## 執行指令

```bash
# 在專案根目錄執行
python run_all_experiments.py
```
