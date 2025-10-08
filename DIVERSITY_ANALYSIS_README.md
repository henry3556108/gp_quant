# Population Diversity Analysis Module

## 模組概述

`gp_quant/diversity` 模組提供完整的族群多樣性分析功能，用於分析遺傳規劃（GP）演化過程中族群的多樣性變化趨勢。

## 模組架構

```
gp_quant/
├── diversity/
│   ├── __init__.py          # 模組入口，匯出主要類別
│   ├── metrics.py           # 多樣性指標計算函數
│   ├── analyzer.py          # 分析器主類別
│   └── visualizer.py        # 視覺化工具
└── scripts/
    └── analyze_diversity.py # 命令列工具
```

## 四種多樣性指標

### 1. Structural Diversity (結構多樣性)

衡量族群中個體樹結構的變異程度。

**指標**：
- `height_std`: 樹高度的標準差
- `height_mean`: 平均樹高度
- `length_std`: 樹長度的標準差  
- `length_mean`: 平均樹長度
- `complexity_mean`: 平均樹複雜度 (height × length)
- `complexity_std`: 樹複雜度的標準差

**意義**：
- 標準差越大 → 結構越多樣化
- 標準差越小 → 結構趨於一致

### 2. Genotypic Diversity (基因型多樣性)

衡量族群中不同基因型（樹結構）的數量。

**指標**：
- `unique_ratio`: 唯一個體比例 (unique_count / total_count)
- `unique_count`: 唯一個體數量（透過字串表示去重）
- `duplicate_ratio`: 重複個體比例
- `total_count`: 總個體數

**意義**：
- `unique_ratio` 接近 1.0 → 高度多樣化，幾乎沒有重複
- `unique_ratio` 接近 0.0 → 低多樣性，大量重複個體

### 3. Fitness Diversity (適應度多樣性)

衡量族群中個體適應度值的分散程度。

**指標**：
- `fitness_std`: 適應度標準差
- `fitness_mean`: 平均適應度
- `fitness_cv`: 變異係數 (Coefficient of Variation = std / mean)
- `fitness_range`: 適應度範圍 (max - min)
- `fitness_min`: 最小適應度
- `fitness_max`: 最大適應度

**意義**：
- `fitness_cv` 高 → 適應度差異大，族群中有優劣差異明顯的個體
- `fitness_cv` 低 → 適應度趨於一致，族群收斂

### 4. Phenotypic Diversity (表現型多樣性)

衡量族群使用的運算子和終端符號的多樣性。

**指標**：
- `unique_primitives`: 使用的不同 primitives（函數）種類數
- `unique_terminals`: 使用的不同 terminals（變數/常數）種類數
- `primitive_usage`: Primitives 使用次數分佈（字典）
- `terminal_usage`: Terminals 使用次數分佈（字典）
- `total_primitives_used`: Primitives 總使用次數
- `total_terminals_used`: Terminals 總使用次數

**意義**：
- 種類數越多 → 族群探索更多不同的運算組合
- 種類數越少 → 族群集中使用特定運算子

## 使用方式

### 方法 1: Python 模組

```python
from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer

# 1. 初始化分析器
analyzer = DiversityAnalyzer(
    "experiments_results/ABX_TO/individual_records_long_run01"
)

# 2. 載入所有世代的 populations
analyzer.load_populations(verbose=True)

# 3. 計算多樣性指標
diversity_data = analyzer.calculate_diversity_trends()

# 4. 查看資料
print(diversity_data.head())

# 5. 生成視覺化（預設顯示四種類別各一個代表指標）
DiversityVisualizer.plot_diversity_trends(
    diversity_data,
    save_path="diversity_analysis.png"
)

# 6. 儲存 CSV 資料
analyzer.save_results("diversity_data.csv")

# 7. 取得統計摘要
summary = analyzer.get_summary_statistics()
print(summary)
```

### 方法 2: 命令列工具

```bash
# 基本用法：分析並生成圖表
python -m gp_quant.scripts.analyze_diversity \
    --records_dir experiments_results/ABX_TO/individual_records_long_run01 \
    --output diversity_plot.png

# 同時儲存 CSV 資料
python -m gp_quant.scripts.analyze_diversity \
    --records_dir experiments_results/ABX_TO/individual_records_long_run01 \
    --output diversity_plot.png \
    --csv diversity_data.csv

# 指定要繪製的特定指標
python -m gp_quant.scripts.analyze_diversity \
    --records_dir experiments_results/ABX_TO/individual_records_long_run01 \
    --metrics structural_height_std genotypic_unique_ratio fitness_cv \
    --output custom_plot.png

# 顯示詳細輸出
python -m gp_quant.scripts.analyze_diversity \
    --records_dir experiments_results/ABX_TO/individual_records_long_run01 \
    --output diversity_plot.png \
    --verbose
```

## API 文件

### DiversityAnalyzer

主要的分析器類別，負責載入 population 資料並計算多樣性指標。

#### 初始化

```python
analyzer = DiversityAnalyzer(records_dir: str)
```

**參數**：
- `records_dir`: individual_records 目錄的路徑

**範例**：
```python
analyzer = DiversityAnalyzer("experiments_results/ABX_TO/individual_records_long_run01")
```

#### 主要方法

##### `load_populations(verbose=True)`

載入所有世代的 population 資料。

**參數**：
- `verbose`: 是否顯示載入進度

**返回**：
- `Dict[int, List]`: 字典，key 為世代編號，value 為該世代的 population

**範例**：
```python
populations = analyzer.load_populations(verbose=True)
print(f"Loaded {len(populations)} generations")
```

##### `calculate_diversity_trends(metrics=None)`

計算所有世代的多樣性指標。

**參數**：
- `metrics`: 要計算的指標類別列表，可選 `['structural', 'genotypic', 'fitness', 'phenotypic']`。若為 `None` 則計算全部。

**返回**：
- `pd.DataFrame`: 包含所有世代和指標的 DataFrame

**範例**：
```python
diversity_data = analyzer.calculate_diversity_trends()
print(diversity_data.columns)  # 查看所有可用的指標
```

##### `get_summary_statistics()`

取得多樣性趨勢的統計摘要。

**返回**：
- `Dict`: 包含每個指標的初始值、最終值、平均值、標準差等統計資訊

**範例**：
```python
summary = analyzer.get_summary_statistics()
print(summary['metrics']['genotypic_unique_ratio'])
```

##### `save_results(output_path)`

將多樣性資料儲存為 CSV 檔案。

**參數**：
- `output_path`: CSV 檔案路徑

**範例**：
```python
analyzer.save_results("diversity_analysis.csv")
```

#### 類別方法

##### `from_experiment_result(experiment_dir, period, run_number)`

從實驗結果目錄創建分析器。

**參數**：
- `experiment_dir`: 實驗基礎目錄（如 `"experiments_results/ABX_TO"`）
- `period`: `'short'` 或 `'long'`
- `run_number`: 運行編號 (1-10)

**範例**：
```python
analyzer = DiversityAnalyzer.from_experiment_result(
    "experiments_results/ABX_TO",
    "long",
    1
)
```

### DiversityMetrics

提供靜態方法計算各種多樣性指標。

#### 靜態方法

##### `structural_diversity(population)`

計算結構多樣性指標。

**參數**：
- `population`: DEAP individuals 列表

**返回**：
- `Dict`: 包含 `height_std`, `length_std`, `complexity_mean` 等

##### `genotypic_diversity(population)`

計算基因型多樣性指標。

**參數**：
- `population`: DEAP individuals 列表

**返回**：
- `Dict`: 包含 `unique_ratio`, `unique_count`, `duplicate_ratio` 等

##### `fitness_diversity(population)`

計算適應度多樣性指標。

**參數**：
- `population`: DEAP individuals 列表

**返回**：
- `Dict`: 包含 `fitness_std`, `fitness_cv`, `fitness_range` 等

##### `phenotypic_diversity(population)`

計算表現型多樣性指標。

**參數**：
- `population`: DEAP individuals 列表

**返回**：
- `Dict`: 包含 `unique_primitives`, `unique_terminals` 等

##### `calculate_all(population)`

一次計算所有類別的多樣性指標。

**參數**：
- `population`: DEAP individuals 列表

**返回**：
- `Dict[str, Dict]`: 巢狀字典，第一層 key 為類別名稱

### DiversityVisualizer

提供視覺化工具。

#### 靜態方法

##### `plot_diversity_trends(diversity_data, metrics=None, ...)`

繪製多樣性趨勢圖。

**參數**：
- `diversity_data`: 從 `DiversityAnalyzer` 取得的 DataFrame
- `metrics`: 要繪製的指標列表。若為 `None`，預設繪製四種類別各一個代表指標
- `figsize`: 圖表大小 (width, height)
- `save_path`: 儲存路徑
- `show`: 是否顯示圖表

**預設指標**（四種類別各一）：
- `structural_height_std`: 結構多樣性
- `genotypic_unique_ratio`: 基因型多樣性
- `fitness_cv`: 適應度多樣性
- `phenotypic_unique_primitives`: 表現型多樣性

**範例**：
```python
# 使用預設指標（四種類別）
DiversityVisualizer.plot_diversity_trends(
    diversity_data,
    save_path="diversity_4metrics.png"
)

# 自訂指標
DiversityVisualizer.plot_diversity_trends(
    diversity_data,
    metrics=['structural_height_std', 'structural_length_std', 'fitness_std'],
    save_path="custom_metrics.png"
)
```

## 輸出格式

### CSV 檔案格式

```csv
generation,structural_height_std,structural_length_std,genotypic_unique_ratio,fitness_std,fitness_cv,phenotypic_unique_primitives,...
0,1.437,3.245,0.976,12345.67,0.234,15,...
1,1.523,3.412,0.968,11234.56,0.221,14,...
2,1.678,3.589,0.952,10987.45,0.215,14,...
...
```

### 視覺化輸出

生成的圖表為 2×2 網格，展示四種多樣性類別：

```
┌─────────────────────┬─────────────────────┐
│ Structural          │ Genotypic           │
│ Diversity           │ Diversity           │
│ (Tree Height Std)   │ (Unique Ratio)      │
├─────────────────────┼─────────────────────┤
│ Fitness             │ Phenotypic          │
│ Diversity           │ Diversity           │
│ (CV)                │ (Unique Primitives) │
└─────────────────────┴─────────────────────┘
```

每個子圖包含：
- 藍色實線：實際數值
- 紅色虛線：趨勢線
- 趨勢方向指示（↗ 或 ↘）

## 測試與驗證

執行測試腳本：

```bash
python test_diversity_analysis.py
```

**預期輸出**：
- 成功載入 51 個世代的 population
- 計算 21 個多樣性指標
- 生成視覺化圖表
- 儲存 CSV 資料

## 整合範例

### 批次分析多個實驗

```python
from gp_quant.diversity import DiversityAnalyzer
import pandas as pd

tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
periods = ['short', 'long']
results = []

for ticker in tickers:
    ticker_clean = ticker.replace('.', '_')
    for period in periods:
        for run in range(1, 11):
            try:
                analyzer = DiversityAnalyzer.from_experiment_result(
                    f"experiments_results/{ticker_clean}",
                    period,
                    run
                )
                analyzer.load_populations(verbose=False)
                diversity_data = analyzer.calculate_diversity_trends()
                
                # 取得最終世代的多樣性
                final_diversity = diversity_data.iloc[-1].to_dict()
                final_diversity['ticker'] = ticker
                final_diversity['period'] = period
                final_diversity['run'] = run
                results.append(final_diversity)
                
            except Exception as e:
                print(f"Error processing {ticker} {period} run{run}: {e}")

# 匯總結果
summary_df = pd.DataFrame(results)
summary_df.to_csv("all_experiments_diversity_summary.csv", index=False)
```

## 注意事項

1. **DEAP Creator 設置**：模組會自動設置 DEAP creator，無需手動設置
2. **記憶體使用**：載入大量 population（如 500 個體 × 51 世代）可能佔用數百 MB 記憶體
3. **檔案完整性**：確保每個 `generation_XXX/` 目錄都包含 `population.pkl` 檔案
4. **序列化格式**：必須使用 `dill` 序列化的檔案（標準 `pickle` 無法處理 DEAP 的 lambda 函數）

## 常見問題

**Q: 為什麼某些指標沒有出現在圖表中？**

A: 預設只顯示四種類別各一個代表指標。若要顯示更多指標，使用 `metrics` 參數指定。

**Q: 如何新增自訂的多樣性指標？**

A: 在 `metrics.py` 中新增靜態方法：

```python
@staticmethod
def custom_diversity(population):
    # 您的計算邏輯
    return {'custom_metric': value}
```

**Q: 可以分析單一世代嗎？**

A: 可以，直接使用 `DiversityMetrics`：

```python
from gp_quant.diversity.metrics import DiversityMetrics
metrics = DiversityMetrics.calculate_all(population)
```

## 版本資訊

- **版本**: 1.0.0
- **建立日期**: 2025-10-08
- **相依套件**: `pandas`, `matplotlib`, `numpy`, `dill`, `deap`
