# Diversity Analysis - Quick Reference

## 四種多樣性指標

### 1️⃣ Structural Diversity (結構多樣性)
衡量樹結構的變異程度

| 指標 | 說明 | 意義 |
|------|------|------|
| `height_std` | 樹高度標準差 | 越大 = 結構越多樣 |
| `length_std` | 樹長度標準差 | 越大 = 大小越多樣 |
| `complexity_mean` | 平均複雜度 (height × length) | 樹的平均複雜程度 |

### 2️⃣ Genotypic Diversity (基因型多樣性)
衡量不同基因型的數量

| 指標 | 說明 | 意義 |
|------|------|------|
| `unique_ratio` | 唯一個體比例 | 1.0 = 完全不重複<br>0.0 = 全部重複 |
| `unique_count` | 唯一個體數量 | 透過字串表示去重 |

### 3️⃣ Fitness Diversity (適應度多樣性)
衡量適應度值的分散程度

| 指標 | 說明 | 意義 |
|------|------|------|
| `fitness_std` | 適應度標準差 | 越大 = 適應度差異越大 |
| `fitness_cv` | 變異係數 (std/mean) | 標準化的變異程度 |
| `fitness_range` | 適應度範圍 (max - min) | 最優與最劣的差距 |

### 4️⃣ Phenotypic Diversity (表現型多樣性)
衡量使用的運算子種類

| 指標 | 說明 | 意義 |
|------|------|------|
| `unique_primitives` | 使用的函數種類數 | 越多 = 探索更多運算 |
| `unique_terminals` | 使用的終端符號種類數 | 越多 = 使用更多變數 |

## 快速使用

### Python 模組

```python
from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer

# 分析
analyzer = DiversityAnalyzer("experiments_results/ABX_TO/individual_records_long_run01")
analyzer.load_populations()
diversity_data = analyzer.calculate_diversity_trends()

# 繪圖（預設顯示四種類別）
DiversityVisualizer.plot_diversity_trends(diversity_data, save_path="diversity.png")
```

### 命令列

```bash
python -m gp_quant.scripts.analyze_diversity \
    --records_dir experiments_results/ABX_TO/individual_records_long_run01 \
    --output diversity.png \
    --csv diversity.csv
```

## 預設圖表佈局

```
┌──────────────────────────┬──────────────────────────┐
│  Structural Diversity    │  Genotypic Diversity     │
│  (Tree Height Std)       │  (Unique Ratio)          │
│                          │                          │
│  [趨勢線圖]              │  [趨勢線圖]              │
├──────────────────────────┼──────────────────────────┤
│  Fitness Diversity       │  Phenotypic Diversity    │
│  (Coefficient of Var)    │  (Unique Primitives)     │
│                          │                          │
│  [趨勢線圖]              │  [趨勢線圖]              │
└──────────────────────────┴──────────────────────────┘
```

每個子圖包含：
- 📈 藍色實線：實際數值
- 📉 紅色虛線：趨勢線
- ↗/↘ 趨勢方向指示

## 輸出檔案

### CSV 格式
```csv
generation,structural_height_std,genotypic_unique_ratio,fitness_cv,phenotypic_unique_primitives,...
0,1.437,0.976,0.234,15,...
1,1.523,0.968,0.221,14,...
...
```

### 圖表格式
- PNG/PDF 格式
- 300 DPI 高解析度
- 2x2 網格佈局

## 典型分析流程

1. **載入資料** → `load_populations()`
2. **計算指標** → `calculate_diversity_trends()`
3. **視覺化** → `plot_diversity_trends()`
4. **儲存結果** → `save_results()`

## 完整範例

```python
from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer

# Step 1: 初始化
analyzer = DiversityAnalyzer("experiments_results/ABX_TO/individual_records_long_run01")

# Step 2: 載入
populations = analyzer.load_populations(verbose=True)
print(f"✓ Loaded {len(populations)} generations")

# Step 3: 計算
diversity_data = analyzer.calculate_diversity_trends()
print(f"✓ Calculated {len(diversity_data.columns)-1} metrics")

# Step 4: 摘要
summary = analyzer.get_summary_statistics()
for metric in ['genotypic_unique_ratio', 'fitness_cv']:
    stats = summary['metrics'][metric]
    print(f"{metric}: {stats['initial']:.3f} → {stats['final']:.3f} ({stats['trend']})")

# Step 5: 視覺化（四種類別）
DiversityVisualizer.plot_diversity_trends(
    diversity_data,
    save_path="diversity_4categories.png",
    show=True
)

# Step 6: 儲存
analyzer.save_results("diversity_data.csv")
print("✓ Results saved")
```

## 測試

```bash
python test_diversity_analysis.py
```

預期看到四種類別的圖表和完整的 CSV 資料。
