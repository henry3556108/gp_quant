# 並行執行指南

## 功能說明

`run_all_experiments.py` 現在支持並行執行，可以同時運行多個實驗，大幅縮短總執行時間。

## 使用方式

### 方法 1: 直接運行（默認 8 個並行）
```bash
python run_all_experiments.py
```

### 方法 2: 在 Python 中指定並行數
```python
from run_all_experiments import run_all_experiments

# 使用 4 個並行工作
results_df = run_all_experiments(max_workers=4)

# 使用 8 個並行工作（默認）
results_df = run_all_experiments(max_workers=8)

# 使用 16 個並行工作（如果 CPU 核心足夠）
results_df = run_all_experiments(max_workers=16)
```

## 並行數選擇建議

| CPU 核心數 | 建議並行數 | 說明 |
|-----------|-----------|------|
| 4 核心 | 2-4 | 保守配置 |
| 8 核心 | 4-8 | 平衡配置 |
| 16 核心 | 8-16 | 高性能配置 |

**注意**: 每個實驗會佔用一定的 CPU 和內存，建議不要超過 CPU 核心數。

## 性能提升

### 順序執行（原版本）
- 80 次實驗 × 平均 5 分鐘 = **400 分鐘（6.7 小時）**

### 並行執行（8 個工作）
- 80 次實驗 ÷ 8 並行 × 平均 5 分鐘 = **50 分鐘**
- **速度提升**: 約 8 倍

### 並行執行（16 個工作）
- 80 次實驗 ÷ 16 並行 × 平均 5 分鐘 = **25 分鐘**
- **速度提升**: 約 16 倍

## 輸出格式

並行執行時的輸出更加簡潔：

```
🔬 開始: ABX.TO | 短訓練期 | Run 1
🔬 開始: ABX.TO | 短訓練期 | Run 2
✓ 完成: ABX.TO | 短訓練期 | Run 1 | 超額: $18,076 ✅ | 35.0s
✓ 完成: ABX.TO | 短訓練期 | Run 2 | 超額: $-9,583 ❌ | 29.7s

📊 總進度: 2/80 (2.5%) | 剛完成: ABX.TO 短訓練期 Run 2
```

## 技術實現

使用 Python 的 `concurrent.futures.ProcessPoolExecutor` 實現：
- ✅ 真正的並行執行（多進程）
- ✅ 自動任務調度
- ✅ 異常處理和錯誤隔離
- ✅ 進度追蹤

## 優勢

1. **不修改源文件** - 使用命令行參數傳遞配置
2. **支持並行** - 可以同時運行多個實驗
3. **無文件衝突** - 每個實驗有獨立的輸出目錄
4. **錯誤隔離** - 單個實驗失敗不影響其他實驗
5. **進度可視化** - 實時顯示完成進度

## 注意事項

1. **內存使用**: 並行數越多，內存佔用越大
2. **CPU 負載**: 建議監控 CPU 使用率，避免過載
3. **文件 I/O**: 大量並行可能導致磁盤 I/O 瓶頸
4. **隨機性**: 每次運行結果可能略有不同（GP 的隨機性）

## 測試

測試小規模並行執行：
```bash
python test_parallel.py
```

## 查看 CPU 核心數

```bash
# macOS
sysctl -n hw.ncpu

# Linux
nproc

# Python
python -c "import os; print(os.cpu_count())"
```
