# 實驗結果文件結構說明

## 📁 文件組織

所有實驗結果按照以下結構組織：

```
gp_paper/
├── all_experiments_results.csv          # 所有實驗的匯總表格
├── all_experiments_results.json         # 所有實驗的匯總JSON
│
└── experiments_results/                 # 各股票詳細結果目錄
    ├── ABX_TO/                          # ABX.TO 股票的所有實驗
    │   ├── short_run01_train_trades.csv # 短訓練期第1次 - 訓練期交易記錄
    │   ├── short_run01_test_trades.csv  # 短訓練期第1次 - 測試期交易記錄
    │   ├── short_run01_result.json      # 短訓練期第1次 - 結果摘要
    │   ├── short_run01_output.log       # 短訓練期第1次 - 完整輸出日誌
    │   ├── short_run02_*.csv/json/log   # 短訓練期第2次
    │   ├── ...                          # 短訓練期第3-10次
    │   ├── long_run01_train_trades.csv  # 長訓練期第1次 - 訓練期交易記錄
    │   ├── long_run01_test_trades.csv   # 長訓練期第1次 - 測試期交易記錄
    │   ├── long_run01_result.json       # 長訓練期第1次 - 結果摘要
    │   ├── long_run01_output.log        # 長訓練期第1次 - 完整輸出日誌
    │   └── ...                          # 長訓練期第2-10次
    │
    ├── BBD-B_TO/                        # BBD-B.TO 股票的所有實驗
    │   └── (同上結構)
    │
    ├── RY_TO/                           # RY.TO 股票的所有實驗
    │   └── (同上結構)
    │
    └── TRP_TO/                          # TRP.TO 股票的所有實驗
        └── (同上結構)
```

## 📊 文件說明

### 1. 匯總文件

#### `all_experiments_results.csv`
包含所有80次實驗的關鍵指標：
- ticker: 股票代碼
- period: 訓練期類型（短訓練期/長訓練期）
- run_number: 運行次數（1-10）
- train_gp_return: 訓練期GP總報酬
- train_bh_return: 訓練期Buy-and-Hold報酬
- train_excess_return: 訓練期超額報酬
- test_gp_return: 測試期GP總報酬
- test_bh_return: 測試期Buy-and-Hold報酬
- test_excess_return: 測試期超額報酬
- best_fitness: 最佳個體適應度
- duration: 執行時間（秒）
- timestamp: 執行時間戳

### 2. 個股詳細文件

每個股票有獨立的資料夾，包含20次實驗（10次短訓練期 + 10次長訓練期）

#### `*_train_trades.csv` / `*_test_trades.csv`
交易記錄，包含：
- entry_date: 進場日期
- exit_date: 出場日期
- entry_price: 進場價格
- exit_price: 出場價格
- shares: 股數
- pnl: 損益

#### `*_result.json`
該次實驗的結果摘要，包含所有關鍵指標

#### `*_output.log`
該次實驗的完整輸出日誌，包含：
- 演化過程
- 最佳交易規則
- 詳細的回測結果

## 🔍 文件命名規則

- `short`: 短訓練期（1998-06-22 至 1999-06-25）
- `long`: 長訓練期（1993-07-02 至 1999-06-25）
- `run01` 到 `run10`: 第1次到第10次運行
- `train`: 訓練期（樣本內）
- `test`: 測試期（樣本外）

## 📈 如何使用這些數據

### 1. 查看匯總統計
```python
import pandas as pd

# 讀取所有結果
df = pd.read_csv('all_experiments_results.csv')

# 按股票和訓練期分組統計
summary = df.groupby(['ticker', 'period'])['test_excess_return'].agg(['mean', 'std', 'min', 'max'])
print(summary)
```

### 2. 分析特定股票的交易記錄
```python
# 讀取 ABX.TO 短訓練期第1次的測試期交易
trades = pd.read_csv('experiments_results/ABX_TO/short_run01_test_trades.csv')

# 計算勝率
winning_trades = (trades['pnl'] > 0).sum()
total_trades = len(trades)
win_rate = winning_trades / total_trades * 100
print(f"勝率: {win_rate:.1f}%")
```

### 3. 比較不同運行的穩定性
```python
# 查看某股票某訓練期的10次運行結果
abx_short = df[(df['ticker'] == 'ABX.TO') & (df['period'] == '短訓練期')]
print(f"平均超額報酬: ${abx_short['test_excess_return'].mean():,.2f}")
print(f"標準差: ${abx_short['test_excess_return'].std():,.2f}")
```

## 🎯 實驗配置

- **股票數量**: 4 (ABX.TO, BBD-B.TO, RY.TO, TRP.TO)
- **訓練期類型**: 2 (短訓練期, 長訓練期)
- **每個配置運行次數**: 10
- **總實驗數**: 80

### 短訓練期
- 訓練期: 1998-06-22 至 1999-06-25 (約256天)
- 測試期: 1999-06-28 至 2000-06-30 (約256天)

### 長訓練期
- 訓練期: 1993-07-02 至 1999-06-25 (約1498天)
- 測試期: 1999-06-28 至 2000-06-30 (約256天)

### GP 參數
- Generations: 50
- Population: 500
- Crossover Rate: 60%
- Mutation Rate: 5%

## 📝 注意事項

1. 所有金額單位為美元（USD）
2. 初始資金: $100,000
3. 測試期超額報酬 > 0 表示策略超越 Buy-and-Hold
4. 由於GP的隨機性，每次運行結果會有差異
5. 10次運行的統計數據更能反映策略的穩健性
