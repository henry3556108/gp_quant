# CPU 溫度管理指南

## 問題描述

多樣性計算是 CPU 密集型任務，特別是對於大族群（5000 個體）：
- 需要計算 ~1250 萬次相似度比較
- 使用多進程並行計算
- 可能導致 CPU 溫度升高

## 解決方案

### 1. 減少並行工作數 ⭐ 推薦

**從 8 個降到 2-4 個工作進程**：

```bash
# 使用 2 個工作進程（最溫和）
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2

# 使用 4 個工作進程（平衡）
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 4
```

**效果**：
- ✅ 降低 CPU 使用率 50-75%
- ⏱️ 增加計算時間約 2-4 倍
- 🌡️ 顯著降低溫度

### 2. 添加冷卻時間 ⭐⭐ 最有效

**在每個世代計算後暫停**：

```bash
# 每個世代後暫停 5 秒
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --cooldown 5

# 每個世代後暫停 10 秒（更保守）
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --cooldown 10
```

**效果**：
- ✅ CPU 有時間冷卻
- ⏱️ 增加總時間（25 世代 × 冷卻時間）
- 🌡️ 防止持續高溫

### 3. 使用序列計算

**一次只處理一個世代**：

```bash
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 5
```

**效果**：
- ✅ 更平穩的 CPU 使用
- ⏱️ 計算時間最長
- 🌡️ 溫度最穩定

### 4. 組合策略 ⭐⭐⭐ 最推薦

**結合多種方法**：

```bash
# 低溫模式：2 workers + 序列 + 10 秒冷卻
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 10
```

**預估時間**：
- 每個世代：~200 秒計算 + 10 秒冷卻 = 210 秒
- 25 個世代：~5250 秒 = **約 87 分鐘**
- 但 CPU 溫度會保持在安全範圍

## 監控 CPU 溫度

### macOS 監控工具

```bash
# 安裝 osx-cpu-temp
brew install osx-cpu-temp

# 實時監控
while true; do 
    osx-cpu-temp
    sleep 2
done
```

### 在計算時監控

```bash
# 終端 1：運行計算
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --cooldown 5

# 終端 2：監控溫度
watch -n 2 "ps aux | grep python | grep -v grep; echo '---'; osx-cpu-temp"
```

## 建議配置

### 根據情況選擇

| 情況 | Workers | Cooldown | Batch | 預估時間 | 溫度影響 |
|------|---------|----------|-------|----------|----------|
| 🔥 電腦很燙 | 2 | 10s | 否 | ~90 分鐘 | 最低 |
| 🌡️ 溫度偏高 | 2 | 5s | 否 | ~60 分鐘 | 低 |
| ⚖️ 平衡 | 4 | 3s | 否 | ~45 分鐘 | 中等 |
| ⚡ 快速（風險） | 8 | 0s | 是 | ~20 分鐘 | 高 |

### 實驗 1 (無 niching) 建議

```bash
# 推薦配置：平衡速度與溫度
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 5
```

## 其他降溫建議

### 1. 環境改善
- 🌬️ 確保通風良好
- ❄️ 使用散熱墊或支架
- 🔇 清理風扇灰塵

### 2. 系統設置
- 💻 關閉其他耗 CPU 的應用
- 🔋 使用電源適配器（不要用電池）
- 🌙 在較涼爽的時段運行（晚上/清晨）

### 3. 分批處理
如果還是太燙，可以分批處理世代：

```bash
# 只處理前 10 個世代
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --cooldown 10
    # 然後手動修改腳本處理後 15 個世代
```

## 安全溫度範圍

### MacBook Pro
- ✅ 正常：< 70°C
- ⚠️ 偏高：70-85°C
- 🔥 危險：> 85°C

如果溫度超過 85°C，建議：
1. 立即停止計算
2. 讓電腦冷卻 10-15 分鐘
3. 使用更保守的配置重新開始

## 快速參考

```bash
# 🔥 電腦很燙時使用
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir <實驗目錄> \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 10

# ⚖️ 一般情況使用
python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir <實驗目錄> \
    --n_workers 4 \
    --cooldown 5
```
