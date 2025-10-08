# Pickle Population Storage 實作總結

## 實作概述

本次實作在 `feature/pickle-population-storage` 分支中完成，目的是在演化過程中儲存每個世代的完整族群（population），以便後續分析族群多樣性。

## 修改內容

### 1. 演化引擎 (`gp_quant/evolution/engine.py`)

**新增功能**:
- 使用 `dill` 替代 `pickle` 進行序列化（因為 DEAP 的 ephemeral 常數使用 lambda 函數）
- 新增 `save_population()` 函數，在每個世代結束後儲存族群
- `run_evolution()` 函數新增 `individual_records_dir` 參數

**關鍵程式碼**:
```python
def save_population(population, generation, individual_records_dir):
    """Save the current population to a pickle file using dill."""
    if individual_records_dir is None:
        return
    
    gen_dir = os.path.join(individual_records_dir, f"generation_{generation:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    
    pickle_file = os.path.join(gen_dir, "population.pkl")
    with open(pickle_file, 'wb') as f:
        dill.dump(population, f)
```

### 2. 主程式 (`main.py`)

**修改內容**:
- 在 `run_portfolio_evolution()` 中設置 `individual_records_dir`
- 目錄結構: `experiments_results/{ticker}/individual_records/generation_{N:03d}/population.pkl`

## 測試結果

### 小規模測試配置
- **標的**: ABX.TO
- **世代數**: 10
- **族群大小**: 100
- **執行時間**: 2.1 秒

### 儲存空間統計
- **總儲存空間**: 232 KB (11 個世代，包含 generation 0)
- **平均每世代**: ~20 KB
- **檔案格式**: `.pkl` (使用 dill 序列化)

### 完整實驗預估

**實驗規模**:
- 4 個標的 (ABX.TO, BBD-B.TO, RY.TO, TRP.TO)
- 2 種訓練期 (短期、長期)
- 每種配置 10 次運行
- 每次運行 50 個世代
- 族群大小 500

**儲存空間預估**:
- **每世代大小** (500 個體): ~99 KB
- **總世代數**: 4,000
- **預估總空間**: **~386 MB**

## 檔案結構

```
experiments_results/
└── ABX_TO/
    ├── individual_records/
    │   ├── generation_000/
    │   │   └── population.pkl (23 KB)
    │   ├── generation_001/
    │   │   └── population.pkl (22 KB)
    │   ├── generation_002/
    │   │   └── population.pkl (19 KB)
    │   └── ... (每個世代一個資料夾)
    ├── short_run01_train_trades.csv
    ├── short_run01_test_trades.csv
    ├── short_run01_result.json
    └── ...
```

## 讀取儲存的族群

**範例程式碼**:
```python
import dill
from deap import creator, base, gp

# 必須先設置 DEAP creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 讀取族群
with open("experiments_results/ABX_TO/individual_records/generation_000/population.pkl", 'rb') as f:
    population = dill.load(f)

# 分析族群
for ind in population:
    print(f"Fitness: {ind.fitness.values}")
    print(f"Structure: {str(ind)}")
    print(f"Height: {ind.height}, Length: {len(ind)}")
```

## 驗證結果

✅ **成功驗證**:
- 族群可以完整儲存和讀取
- 個體的 fitness 值正確保存
- 個體的樹狀結構完整保存
- 可以訪問個體的所有屬性（height, length 等）

## 注意事項

1. **dill vs pickle**: 使用 `dill` 是因為 DEAP 的 ephemeral 常數使用 lambda 函數，標準 `pickle` 無法序列化
2. **讀取前需設置 creator**: 反序列化前必須先創建 DEAP 的 `FitnessMax` 和 `Individual` 類別
3. **儲存空間**: 完整實驗約需 400 MB，請確保有足夠的磁碟空間
4. **目錄衝突**: 目前每個 ticker 只有一個 `individual_records` 目錄，多次運行會覆蓋

## 後續改進建議

1. **區分不同運行**: 為每次運行創建獨立的目錄，例如 `individual_records_short_run01/`
2. **壓縮儲存**: 使用 gzip 壓縮 pickle 檔案可節省約 50-70% 空間
3. **選擇性儲存**: 只儲存特定世代（如每 5 代）以節省空間
4. **多樣性指標**: 在儲存時同時計算並記錄族群多樣性指標

## 測試檔案

- `test_pickle_storage.py`: 小規模測試腳本
- `verify_pickle_load.py`: 驗證讀取功能
- `debug_pickle.py`: 除錯用腳本

## Git 分支

- **分支名稱**: `feature/pickle-population-storage`
- **基於**: `master`
- **狀態**: 實作完成，測試通過
