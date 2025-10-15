# 多樣性視覺化工具使用指南

本指南介紹如何使用多樣性視覺化工具分析 GP 實驗的族群多樣性演化。

## 📋 目錄

- [快速開始](#快速開始)
- [工具概覽](#工具概覽)
- [詳細使用](#詳細使用)
- [範例](#範例)
- [常見問題](#常見問題)

---

## 🚀 快速開始

### 最簡單的使用方式

```bash
# 一鍵分析實驗
python scripts/analysis/analyze_experiment.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --key_generations 1 10 25 50
```

這會：
1. 自動計算所有世代的多樣性指標
2. 繪製多樣性演化曲線
3. 分析關鍵世代（熱圖、分佈圖、t-SNE）

---

## 🛠️ 工具概覽

### 1. `compute_diversity_metrics.py` - 批次計算多樣性

**功能**：計算所有世代的多樣性指標並儲存

**輸入**：`generations/*.pkl`  
**輸出**：`diversity_metrics.json`

**使用方式**：
```bash
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 8
```

**時間估算**：
- Population 5000, 50 代：~9 分鐘（批次並行）
- Population 1000, 50 代：~3 分鐘

---

### 2. `analyze_experiment.py` - 一鍵分析

**功能**：完整分析實驗的多樣性

**使用方式**：
```bash
# 基本分析（只繪製演化曲線）
python scripts/analysis/analyze_experiment.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353

# 完整分析（包含關鍵世代）
python scripts/analysis/analyze_experiment.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --key_generations 1 10 25 50
```

**輸出**：
- `diversity_evolution.png` - 多樣性演化曲線
- `similarity_heatmap_genXXX.png` - 相似度矩陣熱圖
- `similarity_distribution_genXXX.png` - 相似度分佈
- `population_tsne_genXXX.png` - t-SNE 降維圖

---

### 3. `compare_experiments.py` - 比較實驗

**功能**：比較多個實驗的多樣性演化

**使用方式**：
```bash
python scripts/analysis/compare_experiments.py \
    --exp_dirs exp1 exp2 exp3 \
    --labels "With Niching" "Without Niching" "Baseline" \
    --output comparison.png
```

**輸出**：
- 多條曲線對比圖
- 統計比較表

---

## 📊 視覺化函數

### Python API

```python
from gp_quant.similarity import (
    plot_diversity_evolution,
    plot_similarity_heatmap,
    plot_similarity_distribution,
    plot_population_tsne
)

# 1. 繪製演化曲線
plot_diversity_evolution(
    'diversity_metrics.json',
    save_path='evolution.png'
)

# 2. 繪製相似度矩陣熱圖
plot_similarity_heatmap(
    'generations/generation_050.pkl',
    generation=50,
    save_path='heatmap.png'
)

# 3. 繪製相似度分佈
plot_similarity_distribution(
    'generations/generation_050.pkl',
    generation=50,
    save_path='distribution.png'
)

# 4. 繪製 t-SNE 降維圖
plot_population_tsne(
    'generations/generation_050.pkl',
    generation=50,
    method='tsne',  # 或 'pca'
    save_path='tsne.png'
)
```

---

## 📈 詳細使用

### 步驟 1: 運行實驗

```bash
python run_portfolio_experiment.py
```

實驗會儲存：
- `generations/generation_001.pkl` ~ `generation_050.pkl`
- `evolution_log.csv`
- `evolution_log.json`

---

### 步驟 2: 計算多樣性指標

```bash
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 8
```

**參數說明**：
- `--exp_dir`: 實驗目錄
- `--n_workers`: 並行工作進程數（預設 8）
- `--no_batch_parallel`: 不使用批次並行（一次只處理一個世代）
- `--output`: 自訂輸出文件路徑

**輸出**：`diversity_metrics.json`
```json
{
  "experiment": "portfolio_exp_sharpe_20251014_191353",
  "total_generations": 50,
  "population_size": 5000,
  "metrics": [
    {
      "generation": 1,
      "avg_similarity": 0.3124,
      "diversity_score": 0.6876,
      "std_similarity": 0.1234,
      ...
    },
    ...
  ]
}
```

---

### 步驟 3: 視覺化分析

#### 方案 A：快速查看演化曲線

```bash
python -c "
from gp_quant.similarity import plot_diversity_evolution
plot_diversity_evolution(
    'portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353/diversity_metrics.json',
    save_path='evolution.png'
)
"
```

時間：<1 秒

---

#### 方案 B：完整分析

```bash
python scripts/analysis/analyze_experiment.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --key_generations 1 10 25 50
```

時間：~6 分鐘（4 個關鍵世代）

---

### 步驟 4: 比較實驗

```bash
python scripts/analysis/compare_experiments.py \
    --exp_dirs \
        portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
        portfolio_experiment_results/portfolio_exp_sharpe_20251014_234417 \
    --labels "Exp1" "Exp2" \
    --output comparison.png
```

時間：<1 秒

---

## 💡 範例

### 範例 1: 分析單一實驗

```bash
# 完整分析流程
cd /path/to/gp_paper

# 1. 計算多樣性
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353

# 2. 繪製演化曲線
python -c "
from gp_quant.similarity import plot_diversity_evolution
plot_diversity_evolution(
    'portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353/diversity_metrics.json',
    save_path='evolution.png'
)
"

# 3. 分析關鍵世代
python scripts/analysis/analyze_experiment.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --key_generations 1 25 50
```

---

### 範例 2: 比較有/無 Niching

```bash
# 假設你有兩個實驗
# - exp1: niching_enabled = True
# - exp2: niching_enabled = False

# 比較多樣性演化
python scripts/analysis/compare_experiments.py \
    --exp_dirs exp1 exp2 \
    --labels "With Niching" "Without Niching" \
    --output niching_comparison.png
```

---

### 範例 3: 使用 Python API

```python
from pathlib import Path
from gp_quant.similarity import (
    plot_diversity_evolution,
    plot_similarity_heatmap
)

# 設定路徑
exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353')

# 1. 演化曲線
plot_diversity_evolution(
    exp_dir / 'diversity_metrics.json',
    save_path=exp_dir / 'my_evolution.png',
    figsize=(16, 10),
    dpi=300
)

# 2. 分析最終世代
plot_similarity_heatmap(
    exp_dir / 'generations' / 'generation_050.pkl',
    generation=50,
    save_path=exp_dir / 'my_heatmap.png',
    sample_size=1000  # 抽樣 1000 個個體（如果族群太大）
)
```

---

## ❓ 常見問題

### Q1: 計算多樣性需要多久？

**答**：取決於族群大小和世代數

| Population | 世代數 | 時間（批次並行） |
|-----------|-------|----------------|
| 1000 | 50 | ~3 分鐘 |
| 5000 | 50 | ~9 分鐘 |
| 10000 | 50 | ~20 分鐘 |

---

### Q2: 如何只分析部分世代？

**答**：手動選擇世代文件

```python
from gp_quant.similarity import plot_similarity_heatmap

# 只分析 Gen 1, 10, 50
for gen in [1, 10, 50]:
    pkl_file = f'generations/generation_{gen:03d}.pkl'
    plot_similarity_heatmap(pkl_file, generation=gen, 
                           save_path=f'heatmap_gen{gen:03d}.png')
```

---

### Q3: 族群太大，熱圖繪製很慢怎麼辦？

**答**：使用抽樣

```python
plot_similarity_heatmap(
    'generations/generation_050.pkl',
    generation=50,
    sample_size=1000,  # 隨機抽樣 1000 個個體
    save_path='heatmap_sampled.png'
)
```

---

### Q4: 如何自訂圖表樣式？

**答**：使用參數調整

```python
plot_diversity_evolution(
    'diversity_metrics.json',
    save_path='evolution.png',
    figsize=(20, 12),  # 更大的圖表
    dpi=600            # 更高的解析度
)
```

---

### Q5: 可以在 Jupyter Notebook 中使用嗎？

**答**：可以！不指定 `save_path` 即可

```python
from gp_quant.similarity import plot_diversity_evolution

# 在 Notebook 中顯示
plot_diversity_evolution('diversity_metrics.json')
```

---

## 📚 進階使用

### 自訂分析流程

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
with open('diversity_metrics.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data['metrics'])

# 自訂分析
# 例如：找出多樣性最低的世代
min_diversity_gen = df.loc[df['diversity_score'].idxmin()]
print(f"多樣性最低的世代: {min_diversity_gen['generation']}")
print(f"多樣性分數: {min_diversity_gen['diversity_score']:.4f}")

# 自訂繪圖
plt.figure(figsize=(12, 6))
plt.plot(df['generation'], df['diversity_score'], marker='o')
plt.axhline(df['diversity_score'].mean(), color='r', linestyle='--', 
            label=f"平均值: {df['diversity_score'].mean():.4f}")
plt.xlabel('世代')
plt.ylabel('多樣性分數')
plt.title('自訂多樣性分析')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('custom_analysis.png', dpi=300)
```

---

## 🔧 故障排除

### 問題 1: ImportError

```
ImportError: cannot import name 'plot_diversity_evolution'
```

**解決**：確保已安裝所有依賴

```bash
pip install -r requirements.txt
```

---

### 問題 2: 找不到 diversity_metrics.json

```
FileNotFoundError: diversity_metrics.json
```

**解決**：先運行計算腳本

```bash
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir your_experiment_dir
```

---

### 問題 3: 記憶體不足

```
MemoryError: Unable to allocate array
```

**解決**：使用抽樣或減少並行數

```bash
# 減少並行數
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir your_experiment_dir \
    --n_workers 4

# 或使用序列計算
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir your_experiment_dir \
    --no_batch_parallel
```

---

## 📞 支援

如有問題，請查看：
- 範例腳本：`samples/similarity/sample_diversity_analysis.py`
- 測試文件：`tests/similarity/test_visualizer.py`
- 原始碼：`gp_quant/similarity/visualizer.py`

---

## 📝 更新日誌

### v0.1.0 (2025-10-15)
- ✅ 初始版本
- ✅ 批次並行計算
- ✅ 4 種視覺化工具
- ✅ 一鍵分析腳本
- ✅ 實驗比較工具
