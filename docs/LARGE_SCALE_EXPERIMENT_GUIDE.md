# 🚀 大規模演化實驗指南

## 📋 **實驗配置**

### **基本參數**
- **族群大小**: 10,000 個體
- **演化世代**: 25 代
- **並行處理器**: 6 核心
- **隨機種子**: 42（可重現）

### **演化策略**
- **初始化**: Ramped Half-and-Half (深度 3-8)
- **選擇**: Ranked SUS (錦標賽大小 7，精英 50)
- **交配**: One-Point Leaf-Biased (機率 75%，最大深度 17)
- **變異**: Uniform (機率 5%，最大深度 17)
- **替換**: Elitist (保留前 50 名)

### **適應度函數**
- **函數**: Excess Return（超額報酬）
- **並行評估**: 6 核心
- **緩存**: 啟用

### **終止條件**
- **早停**: 啟用
- **耐心值**: 10 代
- **最小改善**: 0.0001

---

## ⏱️ **執行時間估算**

### **基於測試結果推算**

測試結果（100 個體 × 10 代）：
- 初始評估: ~1 秒
- 總時間: ~1.5 秒

大規模實驗（10000 個體 × 25 代）：
- **初始評估** (10000 個體): ~150 秒 (2.5 分鐘)
- **每世代評估** (~7500 個體): ~112 秒 (1.9 分鐘)
- **25 世代總計**: ~2800 秒 (47 分鐘)
- **保存訊號時間**: ~800 秒 (13 分鐘)

**預計總時間**: 約 **60 分鐘**（1 小時）

### **時間分配**
```
初始評估:     2.5 分鐘  (4%)
世代 1-25:   47.0 分鐘 (78%)
保存訊號:    13.0 分鐘 (22%)
─────────────────────────
總計:        60.0 分鐘
```

---

## 💾 **預期資源使用**

### **磁盤空間**
- **族群數據**: ~500 MB (26 世代 × 10000 個體)
- **訊號數據**: ~200 MB (26 世代 × 4 股票 × 訊號)
- **統計數據**: ~10 MB
- **總計**: 約 **700-800 MB**

### **內存使用**
- **主進程**: ~500 MB
- **6 個子進程**: ~300 MB × 6 = 1.8 GB
- **總計**: 約 **2.3 GB**

### **CPU 使用**
- **並行評估**: 6 核心 100% 使用率
- **單進程操作**: 1 核心使用率

---

## 🚀 **執行步驟**

### **方法 1: 使用啟動腳本（推薦）**

```bash
python run_large_scale_experiment.py
```

腳本會：
1. ✅ 檢查前置條件
2. ⏱️  顯示時間估算
3. ❓ 要求確認
4. 🚀 運行實驗
5. 📊 顯示結果摘要

### **方法 2: 直接運行**

```bash
python main_evolution.py --config configs/large_scale_experiment.json --verbose
```

### **方法 3: 背景運行（長時間實驗）**

```bash
nohup python main_evolution.py --config configs/large_scale_experiment.json --verbose > experiment.log 2>&1 &

# 查看進度
tail -f experiment.log

# 查看進程
ps aux | grep main_evolution
```

---

## 📊 **生成的文件結構**

```
large_scale_records_20251124_1730/
├── config.json                          # 實驗配置
├── generation_stats.json                # 世代統計
├── final_result.json                    # 最終結果
├── engine_state.pkl                     # 演化引擎狀態
├── experiment_summary.json              # 實驗摘要
│
├── populations/                         # 族群數據
│   ├── generation_000.pkl               # 完整個體（可重載）
│   ├── generation_000_stats.json        # 統計數據（可讀）
│   ├── generation_001.pkl
│   ├── generation_001_stats.json
│   └── ...
│
├── genealogy/                           # 譜系數據
│   ├── generation_000.json
│   ├── generation_001.json
│   └── ...
│
└── best_signals/                        # 最佳個體訊號
    ├── generation_000/
    │   ├── backtest_summary.json        # 回測摘要
    │   ├── entry_exit_points.csv        # 交易記錄
    │   ├── signals_ABX.TO.csv           # 訊號
    │   ├── signals_BBD-B.TO.csv
    │   ├── signals_RY.TO.csv
    │   └── signals_TRP.TO.csv
    ├── generation_001/
    └── ...
    └── generation_025/                  # 最終世代
        └── ...
```

---

## 📈 **監控進度**

### **實時監控**

```bash
# 查看當前世代
tail -f large_scale_records_*/generation_stats.json

# 查看最佳適應度
cat large_scale_records_*/generation_stats.json | jq '.[] | {generation, best_fitness}'

# 查看進程狀態
ps aux | grep main_evolution

# 查看 CPU 使用率
top -pid $(pgrep -f main_evolution)
```

### **進度指標**

每個世代會顯示：
```
🔄 第 5/25 世代
評估個體: 100%|██████████| 7500/7500 [01:52<00:00, 66.67個體/s]
💾 保存第 5 世代數據...
   📊 最佳個體訊號已保存: generation_005
   📊 最佳適應度: 0.2345
```

---

## 🔍 **結果分析**

### **1. 查看最終結果**

```bash
# 最終結果摘要
cat large_scale_records_*/final_result.json | python -m json.tool

# 最佳適應度
cat large_scale_records_*/final_result.json | jq '.best_fitness'

# 收斂世代
cat large_scale_records_*/final_result.json | jq '.convergence_generation'
```

### **2. 分析演化趨勢**

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
plt.plot(df['generation'], df['best_fitness'], label='Best', marker='o')
plt.plot(df['generation'], df['avg_fitness'], label='Average', marker='s')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Evolution (10000 individuals × 25 generations)')
plt.legend()
plt.grid(True)
plt.savefig('fitness_evolution.png', dpi=300)
```

### **3. 分析最佳策略**

```bash
# 查看最佳個體的回測摘要
cat large_scale_records_*/best_signals/generation_025/backtest_summary.json | python -m json.tool

# 查看交易記錄
cat large_scale_records_*/best_signals/generation_025/entry_exit_points.csv | head -20

# 統計交易次數
cat large_scale_records_*/best_signals/generation_025/entry_exit_points.csv | wc -l
```

### **4. 比較不同世代**

```python
import json
from pathlib import Path

# 比較第 0、10、20、25 代的最佳個體
generations = [0, 10, 20, 25]

for gen in generations:
    summary_file = f'large_scale_records_*/best_signals/generation_{gen:03d}/backtest_summary.json'
    with open(summary_file) as f:
        data = json.load(f)
    
    print(f"\nGeneration {gen}:")
    print(f"  Fitness: {data['fitness']:.4f}")
    print(f"  Total Return: {data['metrics']['total_return']:.4f}")
    print(f"  Sharpe Ratio: {data['metrics']['sharpe_ratio']:.4f}")
    print(f"  Transactions: {data['total_transactions']}")
```

---

## ⚠️ **注意事項**

### **1. 系統資源**
- 確保有足夠的磁盤空間（至少 1 GB）
- 確保有足夠的內存（至少 3 GB 可用）
- 建議關閉其他耗資源的程序

### **2. 執行時間**
- 預計需要約 1 小時
- 不要在實驗進行中關閉電腦
- 可以使用 `nohup` 在背景運行

### **3. 中斷處理**
- 如果需要中斷，使用 `Ctrl+C`
- 已完成的世代數據會被保存
- 可以使用 `EvolutionLoader` 重新載入並繼續

### **4. 結果驗證**
- 檢查 `final_result.json` 確認實驗完成
- 驗證 `best_signals/generation_025/` 存在
- 確認交易記錄有買入和賣出

---

## 🎯 **成功標準**

實驗成功的標誌：
- ✅ 完成 25 世代演化
- ✅ 每世代保存最佳個體訊號
- ✅ 最佳適應度 > 0（正報酬）
- ✅ 交易記錄包含買入和賣出
- ✅ 所有文件正確生成

---

## 📝 **實驗後分析清單**

- [ ] 查看最終適應度和收斂世代
- [ ] 分析適應度演化趨勢
- [ ] 檢查最佳策略的交易記錄
- [ ] 驗證訊號與交易的對應關係
- [ ] 比較不同世代的策略特徵
- [ ] 分析族群多樣性變化
- [ ] 評估策略的穩定性
- [ ] 準備測試集驗證

---

## 🚀 **快速開始**

```bash
# 1. 確認配置
cat configs/large_scale_experiment.json

# 2. 運行實驗
python run_large_scale_experiment.py

# 3. 等待完成（約 1 小時）

# 4. 查看結果
cat large_scale_records_*/final_result.json | python -m json.tool

# 5. 分析最佳策略
cat large_scale_records_*/best_signals/generation_025/backtest_summary.json
```

**準備好了嗎？讓我們開始這次大規模實驗！** 🚀
