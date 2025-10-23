# CPU 核心管理指南

## 🖥️ 您的系統配置

根據檢測，您的 Mac 有：
- **物理核心**: 8 個
- **邏輯核心**: 8 個（無超線程）

## 📊 當前實驗的 CPU 使用

### 進程分佈
您的多樣性分析使用：
- **主進程**: 1 個（PID 3183）
- **Worker 進程**: 2 個（PID 13621, 13622）
- **CPU 使用率**: 每個 worker ~74%

### 實際核心占用
- 2 個 worker ≈ 使用 **1.5-2 個核心**
- 剩餘 **6 個核心** 可用於其他任務

## 🔍 如何查看 CPU 使用情況

### 1. 使用我們的腳本
```bash
bash scripts/utils/check_cpu_affinity.sh <PID>

# 範例
bash scripts/utils/check_cpu_affinity.sh 3183
```

### 2. 使用活動監視器（圖形界面）
```bash
open -a "Activity Monitor"
```
- 點擊 "CPU" 標籤
- 查看 "% CPU" 欄位
- 可以看到每個核心的使用情況

### 3. 使用 top 命令
```bash
# 實時監控特定進程
top -pid 3183

# 查看所有 Python 進程
top | grep python
```

### 4. 使用 htop（需要安裝）
```bash
# 安裝
brew install htop

# 運行（顯示所有核心）
sudo htop
```

## ⚙️ macOS 的 CPU 親和性限制

### ❌ 不支持的功能
macOS **不支持** Linux 的 `taskset` 命令來綁定進程到特定核心：

```bash
# Linux 上可以這樣做（macOS 不行）
taskset -c 0,1 python script.py  # ❌ macOS 不支持
```

### ✅ macOS 的調度機制
- **自動調度**: macOS 內核自動在所有核心間分配負載
- **動態平衡**: 系統會根據溫度和負載動態調整
- **效能核心優先**: M1/M2 Mac 會優先使用效能核心

## 🎛️ 如何控制 CPU 使用

雖然不能指定核心，但可以通過以下方式控制：

### 1. 控制 Worker 數量 ⭐ 最有效

```bash
# 使用 2 個 worker（當前配置）
--n_workers 2

# 使用 4 個 worker
--n_workers 4

# 使用 8 個 worker（使用所有核心）
--n_workers 8
```

**建議**：
- 2 workers = 使用 25% CPU（2/8 核心）
- 4 workers = 使用 50% CPU（4/8 核心）
- 8 workers = 使用 100% CPU（8/8 核心）

### 2. 使用 nice 降低優先級

```bash
# 降低進程優先級（讓其他任務優先）
nice -n 10 python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir <實驗目錄> \
    --n_workers 2

# nice 值範圍: -20（最高優先級）到 19（最低優先級）
```

### 3. 使用 cpulimit 限制 CPU 使用率

```bash
# 安裝 cpulimit
brew install cpulimit

# 限制進程只使用 50% CPU
cpulimit -p 3183 -l 50 &

# 限制新啟動的進程
cpulimit -l 50 -- python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir <實驗目錄> \
    --n_workers 2
```

### 4. 使用進程組限制（需要 root）

```bash
# 創建 CPU 限制組（需要 sudo）
sudo cgcreate -g cpu:/limited
sudo cgset -r cpu.shares=512 limited  # 50% CPU

# 在限制組中運行
sudo cgexec -g cpu:limited python script.py
```

## 📈 實際應用範例

### 場景 1: 最小 CPU 使用（保持電腦流暢）
```bash
nice -n 15 conda run -n gp_quant python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 10
```
- 使用 2 workers
- 低優先級
- 長冷卻時間
- **預估時間**: ~2 小時
- **CPU 使用**: ~25%

### 場景 2: 平衡模式（當前配置）
```bash
conda run -n gp_quant python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 2 \
    --no_batch_parallel \
    --cooldown 5
```
- 使用 2 workers
- 正常優先級
- 中等冷卻時間
- **預估時間**: ~85 分鐘
- **CPU 使用**: ~40%

### 場景 3: 快速模式（高 CPU 使用）
```bash
conda run -n gp_quant python scripts/analysis/compute_diversity_metrics.py \
    --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
    --n_workers 8 \
    --cooldown 0
```
- 使用 8 workers
- 批次並行
- 無冷卻時間
- **預估時間**: ~20 分鐘
- **CPU 使用**: ~100%
- ⚠️ 可能導致高溫

## 🌡️ 監控溫度

### 安裝溫度監控工具
```bash
# 方法 1: osx-cpu-temp
brew install osx-cpu-temp

# 使用
osx-cpu-temp

# 方法 2: istats
sudo gem install iStats

# 使用
istats
```

### 實時監控腳本
```bash
# 同時監控進程和溫度
watch -n 5 "echo '=== CPU 溫度 ===' && osx-cpu-temp && echo '' && echo '=== 進程狀態 ===' && ps -p 3183 -o pid,%cpu,%mem,etime,command"
```

## 💡 最佳實踐

### 1. 根據任務選擇配置

| 任務類型 | Workers | Cooldown | 預估時間 | CPU 使用 |
|---------|---------|----------|----------|----------|
| 背景運行 | 2 | 10s | ~2h | 25% |
| 正常運行 | 2-4 | 5s | ~1h | 40-50% |
| 快速完成 | 8 | 0s | ~20min | 100% |

### 2. 溫度管理
- 🌡️ < 70°C: 安全，可以增加 workers
- 🌡️ 70-85°C: 注意，保持當前配置
- 🔥 > 85°C: 危險，減少 workers 或添加冷卻

### 3. 多任務場景
如果需要同時做其他工作：
```bash
# 使用較少的 workers，留出核心給其他任務
--n_workers 2  # 使用 2 核心，留 6 核心給其他任務
```

### 4. 夜間運行
如果可以讓電腦整夜運行：
```bash
# 使用最保守的配置，確保穩定
--n_workers 2 --no_batch_parallel --cooldown 10
```

## 🔧 故障排除

### 問題 1: CPU 使用率過高
```bash
# 解決方案：減少 workers
kill <PID>  # 停止當前進程
# 重新運行，使用更少的 workers
--n_workers 2
```

### 問題 2: 進程占用特定核心
```bash
# macOS 會自動平衡，無需手動干預
# 如果某個核心過熱，系統會自動遷移進程
```

### 問題 3: 想要更精細的控制
```bash
# 考慮使用虛擬機或 Docker
# 在容器中可以設置 CPU 限制
docker run --cpus=2 ...
```

## 📚 相關資源

- [macOS 進程管理文檔](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/)
- [Python multiprocessing 文檔](https://docs.python.org/3/library/multiprocessing.html)
- [htop 使用指南](https://htop.dev/)

## 🎯 快速參考

```bash
# 查看系統核心數
sysctl -n hw.ncpu

# 查看進程 CPU 使用
ps -p <PID> -o pid,%cpu,%mem,command

# 查看所有核心使用情況
top -l 1 | grep "CPU usage"

# 降低進程優先級
nice -n 10 <command>

# 限制 CPU 使用率
cpulimit -p <PID> -l 50
```
