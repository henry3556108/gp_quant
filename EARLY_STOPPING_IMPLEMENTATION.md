# 早停機制實作文檔

## 📋 論文要求

> GP 會在達到最大世代數（50 世代）外，在**連續 15 個世代沒有對最佳解進行任何改善後停止**。

## ✅ 實作方案

採用**方案 3：回調函數方式**

### 優點
- ✅ **最低耦合**：只在 `engine.py` 添加一個可選參數
- ✅ **完全向後兼容**：不傳回調時，行為與原來完全相同
- ✅ **靈活性高**：可用於任何需要在每代後執行的邏輯
- ✅ **實作簡單**：只需修改幾行程式碼

## 🔧 修改內容

### 1. engine.py 修改

**位置**: `gp_quant/evolution/engine.py`

#### 修改 1: 添加參數

```python
def run_evolution(
    data, 
    population_size=500, 
    n_generations=50, 
    crossover_prob=0.6, 
    mutation_prob=0.05,
    individual_records_dir: Optional[str] = None,
    generation_callback=None  # 新增參數
):
    """
    Args:
        generation_callback: Optional callback function called after each generation.
                           Signature: callback(gen, pop, hof, logbook) -> bool
                           If returns True, evolution stops early.
    """
```

#### 修改 2: 添加回調檢查

```python
# 在演化迴圈末尾（save_population 之後）
for gen in (pbar := trange(1, n_generations + 1, desc="Generation")):
    # ... 演化邏輯 ...
    
    save_population(pop, gen, individual_records_dir)
    
    # 新增：回調檢查
    if generation_callback is not None:
        should_stop = generation_callback(gen, pop, hof, logbook)
        if should_stop:
            print(f"\n⏹️  Evolution stopped by callback at generation {gen}")
            break
```

### 2. main.py 修改

**位置**: `main.py` 的 `run_portfolio_evolution()` 函數

#### 修改 1: 導入 EarlyStopping

```python
from gp_quant.evolution.early_stopping import EarlyStopping
```

#### 修改 2: 創建早停物件和回調函數

```python
# 初始化早停（論文要求：連續 15 代無改善）
early_stopping = EarlyStopping(
    patience=15,      # 論文要求
    min_delta=0.0,    # 任何改善都算
    mode='max'        # fitness 越大越好
)

def early_stop_callback(gen, pop, hof, logbook):
    """早停回調函數"""
    current_best = hof[0].fitness.values[0]
    if early_stopping.step(current_best):
        print(f"\n⏹️  Early Stopping Triggered!")
        print(f"   No improvement for {early_stopping.patience} consecutive generations")
        print(f"   Best fitness: ${early_stopping.best_fitness:,.2f}")
        print(f"   Stopped at generation {gen}/{args.generations}")
        return True  # 停止演化
    return False  # 繼續演化
```

#### 修改 3: 傳入回調函數

```python
pop, log, hof = run_evolution(
    data=train_data,
    n_generations=args.generations,
    population_size=args.population,
    individual_records_dir=individual_records_dir,
    generation_callback=early_stop_callback  # 傳入早停回調
)
```

## 📊 工作原理

### 早停邏輯流程

```
每個 generation 結束後:
  1. 取得當前最佳 fitness
  2. 與歷史最佳 fitness 比較
  3. 如果有改善:
     - 更新歷史最佳 fitness
     - 重置計數器為 0
  4. 如果無改善:
     - 計數器 +1
  5. 如果計數器 >= 15:
     - 觸發早停
     - 停止演化
```

### EarlyStopping 類別

**位置**: `gp_quant/evolution/early_stopping.py`

**主要方法**:
- `__init__(patience, min_delta, mode)`: 初始化
- `step(current_fitness)`: 檢查是否應該停止
- `get_status()`: 返回當前狀態
- `reset()`: 重置狀態

**參數說明**:
- `patience=15`: 連續無改善的世代數（論文要求）
- `min_delta=0.0`: 最小改善閾值（任何改善都算）
- `mode='max'`: 最大化 fitness（超額報酬）

## 🧪 測試

### 運行測試腳本

```bash
python test_early_stopping.py
```

### 預期結果

1. **早停觸發**（族群收斂）:
   ```
   ⏹️  Early Stopping Triggered!
      No improvement for 15 consecutive generations
      Best fitness: $12,345.67
      Stopped at generation 35/50
   ```

2. **早停未觸發**（持續改善）:
   ```
   ℹ️  早停機制未觸發（演化持續改善或達到最大世代數）
      ✓ 完成所有 50 個世代
   ```

## 📈 預期影響

### 計算效率
- **節省時間**: 如果在第 35 代停止，節省 30% 計算時間
- **自動優化**: 不需要手動判斷何時停止

### 結果品質
- **防止過擬合**: 避免過度演化導致訓練期過擬合
- **符合論文**: 完全符合論文的實驗設定

### 實驗一致性
- **標準化**: 所有實驗使用相同的停止條件
- **可重現**: 結果更容易重現

## 🔍 與其他方案的比較

| 特性 | 方案 1<br/>手動迴圈 | 方案 2<br/>包裝函數 | 方案 3<br/>回調函數 ✅ |
|------|-------------------|-------------------|---------------------|
| 修改 engine.py | ❌ 不修改 | ❌ 不修改 | ✅ 最小修改 |
| 耦合度 | 低 | 低 | **最低** |
| 靈活性 | 中 | 高 | **最高** |
| 實作難度 | 高 | 高 | **最低** |
| 向後兼容 | ✅ | ✅ | ✅ |
| 可擴展性 | 中 | 高 | **最高** |

## 📝 使用範例

### 啟用早停

```python
from gp_quant.evolution.early_stopping import EarlyStopping

early_stopping = EarlyStopping(patience=15, min_delta=0.0, mode='max')

def callback(gen, pop, hof, logbook):
    return early_stopping.step(hof[0].fitness.values[0])

pop, log, hof = run_evolution(
    data=data,
    generation_callback=callback
)
```

### 禁用早停（向後兼容）

```python
# 不傳 generation_callback，行為與原來完全相同
pop, log, hof = run_evolution(data=data)
```

### 自定義回調（靈活性）

```python
def custom_callback(gen, pop, hof, logbook):
    # 可以實作任何邏輯
    # 例如：每 10 代儲存檢查點
    if gen % 10 == 0:
        save_checkpoint(pop, hof)
    
    # 例如：達到目標 fitness 就停止
    if hof[0].fitness.values[0] > 50000:
        print("Target fitness reached!")
        return True
    
    return False
```

## ✅ 驗證清單

- [x] engine.py 添加 generation_callback 參數
- [x] main.py 整合 EarlyStopping
- [x] 早停條件設為 15 代（符合論文）
- [x] 保持向後兼容（不傳回調時正常運行）
- [x] 創建測試腳本
- [x] 文檔完整

## 🚀 下一步

1. **運行測試**: `python test_early_stopping.py`
2. **驗證功能**: 確認早停機制正常工作
3. **重新運行實驗**: 使用早停機制重新運行完整實驗
4. **比較結果**: 對比有無早停的結果差異

## 📚 相關文件

- `gp_quant/evolution/engine.py`: 演化引擎（添加回調支持）
- `gp_quant/evolution/early_stopping.py`: 早停機制實作
- `main.py`: 主程式（整合早停）
- `test_early_stopping.py`: 測試腳本
- `DEPTH_VIOLATION_FIX.md`: 深度限制修復文檔
