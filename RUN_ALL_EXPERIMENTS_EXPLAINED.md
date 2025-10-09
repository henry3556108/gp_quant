# run_all_experiments.py 完整解析

**檔案**: `run_all_experiments.py` (372 行)  
**用途**: 大規模自動化實驗系統 - 運行 80 次完整實驗  
**優先級**: 🟡 中等（實驗自動化）

---

## 📋 目錄

1. [核心功能概覽](#1-核心功能概覽)
2. [函數詳解](#2-函數詳解)
3. [實驗配置](#3-實驗配置)
4. [文件輸出結構](#4-文件輸出結構)
5. [執行流程圖](#5-執行流程圖)
6. [關鍵問題解答](#6-關鍵問題解答)

---

## 1. 核心功能概覽

這個腳本是整個專案的**「實驗編排器」**，負責：

- ✅ 自動修改 `main.py` 的日期配置
- ✅ 批量運行多個實驗
- ✅ 提取和保存實驗結果
- ✅ 生成統計分析報告
- ✅ 組織文件輸出結構

### 實驗規模

- **4 個股票**: `ABX.TO`, `BBD-B.TO`, `RY.TO`, `TRP.TO`
- **2 種訓練期**: 短訓練期 (1年) vs 長訓練期 (6年)
- **每個配置運行 10 次**
- **總計**: 4 × 2 × 10 = **80 次實驗**

---

## 2. 函數詳解

### 2.1 `modify_main_py()` - 動態修改配置

```python
def modify_main_py(train_data_start, train_backtest_start, train_backtest_end,
                   test_data_start, test_backtest_start, test_backtest_end)
```

**功能**: 使用正則表達式動態修改 `main.py` 中的日期配置

**參數說明**:
- `train_data_start` - 訓練初始期開始日期（用於計算技術指標）
- `train_backtest_start` - 訓練回測期開始日期
- `train_backtest_end` - 訓練回測期結束日期
- `test_data_start` - 測試初始期開始日期
- `test_backtest_start` - 測試回測期開始日期
- `test_backtest_end` - 測試回測期結束日期

**實現方式**:
1. 讀取 `main.py` 的完整內容
2. 使用 `re.sub()` 替換 6 個日期變量
3. 寫回 `main.py`

**正則表達式模式**:
```python
r"train_data_start = '[0-9]{4}-[0-9]{2}-[0-9]{2}'"
```
匹配格式: `train_data_start = 'YYYY-MM-DD'`

**關鍵設計**:
- ⚠️ 直接修改 `main.py` 文件（非參數傳遞）
- ⚠️ 每次實驗前都會重寫配置
- ✅ 使用正則確保只替換目標變量

---

### 2.2 `extract_results()` - 結果提取器

```python
def extract_results(output)
```

**功能**: 從 `main.py` 的標準輸出中提取關鍵數據

**提取的指標 (7 個)**:
- ✓ `train_gp_return` - 訓練期 GP 策略報酬
- ✓ `train_bh_return` - 訓練期 Buy-and-Hold 報酬
- ✓ `train_excess_return` - 訓練期超額報酬
- ✓ `test_gp_return` - 測試期 GP 策略報酬
- ✓ `test_bh_return` - 測試期 Buy-and-Hold 報酬
- ✓ `test_excess_return` - 測試期超額報酬 ⭐ **最重要**
- ✓ `best_fitness` - 最佳個體的 fitness

**正則表達式模式**:
```python
r'Total GP Return: \$([0-9,.-]+)'
r'Total Buy-and-Hold Return: \$([0-9,.-]+)'
r'Total Excess Return: \$([0-9,.-]+)'
r'Best Individual Fitness \(Total Excess Return\): \$([0-9,.-]+)'
```

**處理邏輯**:
- 使用 `re.findall()` 找到所有匹配
- 第 1 個匹配 = 訓練期數據
- 第 2 個匹配 = 測試期數據
- 移除千分位逗號後轉換為 `float`

**錯誤處理**:
- 如果找不到匹配，對應值保持為 `None`
- 不會中斷程序執行

---

### 2.3 `run_single_experiment()` - 單次實驗執行器

```python
def run_single_experiment(ticker, period_name, 
                         train_data_start, train_backtest_start, train_backtest_end,
                         test_data_start, test_backtest_start, test_backtest_end,
                         run_number)
```

**功能**: 執行一次完整的實驗並保存所有結果

**執行流程**:
1. 創建股票專屬目錄 `experiments_results/{ticker}/`
2. 調用 `modify_main_py()` 修改配置
3. 使用 `subprocess.run()` 執行 `main.py`
4. 提取結果並添加元數據
5. 移動交易記錄 CSV 文件
6. 保存結果 JSON
7. 保存完整輸出日誌
8. 打印摘要

**subprocess 配置**:
```python
subprocess.run(
    ['python', 'main.py', 
     '--tickers', ticker, 
     '--mode', 'portfolio', 
     '--generations', '50', 
     '--population', '500'],
    capture_output=True,  # 捕獲標準輸出/錯誤
    text=True,            # 以文本模式返回
    cwd='...'             # 指定工作目錄
)
```

**文件命名規則**:
```
{period_short}_run{run_number:02d}_{file_type}
```

例如:
- `short_run01_train_trades.csv`
- `short_run01_test_trades.csv`
- `short_run01_result.json`
- `short_run01_output.log`
- `long_run05_train_trades.csv`

**返回值**: 包含所有指標 + 元數據的字典
- 7 個財務指標
- `duration` (執行時間)
- `ticker`
- `period`
- `run_number`
- `timestamp`
- `train_trades_file` (路徑)
- `test_trades_file` (路徑)

---

### 2.4 `run_all_experiments()` - 主控制器

```python
def run_all_experiments()
```

**功能**: 編排所有 80 次實驗的執行

**配置定義**:
```python
tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
n_runs = 10

experiments = [
    {
        'name': '短訓練期',
        'train_data_start': '1997-06-25',      # 初始期 1 年
        'train_backtest_start': '1998-06-22',  # 回測期 1 年
        'train_backtest_end': '1999-06-25',
        'test_data_start': '1998-07-07',       # 初始期 1 年
        'test_backtest_start': '1999-06-28',   # 回測期 1 年
        'test_backtest_end': '2000-06-30'
    },
    {
        'name': '長訓練期',
        'train_data_start': '1992-06-30',      # 初始期 1 年
        'train_backtest_start': '1993-07-02',  # 回測期 6 年 ⭐
        'train_backtest_end': '1999-06-25',
        'test_data_start': '1998-07-07',       # 初始期 1 年
        'test_backtest_start': '1999-06-28',   # 回測期 1 年
        'test_backtest_end': '2000-06-30'
    }
]
```

**執行順序**:
```python
for ticker in tickers:              # 外層循環: 股票
    for exp in experiments:         # 中層循環: 訓練期類型
        for run in range(1, 11):    # 內層循環: 重複次數
            run_single_experiment(...)
```

**進度追蹤**:
- 計算總實驗數: 4 × 2 × 10 = 80
- 每完成一次實驗，更新進度百分比
- 打印: `📊 總進度: 23/80 (28.8%)`

**錯誤處理**:
```python
try:
    result = run_single_experiment(...)
    all_results.append(result)
except Exception as e:
    print(f"❌ 錯誤: {e}")
    continue  # 跳過失敗的實驗，繼續執行
```

**最終輸出**:
1. `all_experiments_results.csv` - 所有結果的表格
2. `all_experiments_results.json` - 所有結果的 JSON
3. 調用 `generate_summary()` 生成統計報告

---

### 2.5 `generate_summary()` - 統計分析器

```python
def generate_summary(df, total_duration)
```

**功能**: 生成詳細的統計分析報告

**分析內容**:

#### 1️⃣ 基本信息
- 總執行時間
- 總實驗數

#### 2️⃣ 分組統計 (按 ticker 和 period)
```python
df.groupby(['ticker', 'period']).agg({
    'test_excess_return': ['mean', 'std', 'min', 'max'],
    'train_excess_return': ['mean', 'std'],
    'duration': 'mean'
})
```

輸出示例:
```
                      test_excess_return                    train_excess_return
                      mean    std    min     max            mean    std
ticker    period
ABX.TO    短訓練期    1234.5  567.8  -200.0  2500.0        890.2   123.4
ABX.TO    長訓練期    2345.6  678.9  500.0   3500.0        1234.5  234.5
```

#### 3️⃣ 勝率分析
計算每個 (ticker, period) 組合超越 Buy-and-Hold 的比例:

```python
wins = (subset['test_excess_return'] > 0).sum()
win_rate = (wins / total) * 100
```

輸出示例:
```
ABX.TO:
  短訓練期: 7/10 (70%) ✅ | 平均超額: $1,234.56
  長訓練期: 9/10 (90%) ✅ | 平均超額: $2,345.67
```

#### 4️⃣ 最佳/最差表現
```python
best_idx = df['test_excess_return'].idxmax()
worst_idx = df['test_excess_return'].idxmin()
```

顯示:
- 股票名稱
- 訓練期類型
- 運行次數
- 樣本外超額報酬

#### 5️⃣ 總體結論
比較短訓練期 vs 長訓練期的整體勝率:

```python
short_win_rate = (short_period['test_excess_return'] > 0).sum() / len(short_period) * 100
long_win_rate = (long_period['test_excess_return'] > 0).sum() / len(long_period) * 100
```

判斷哪種訓練期配置更優

#### 6️⃣ 文件結構說明
打印完整的輸出目錄結構

---

## 3. 實驗配置

### 3.1 時間配置對比

**短訓練期**:
- 訓練初始期: `1997-06-25` → `1998-06-22` (約 1 年，用於計算技術指標)
- 訓練回測期: `1998-06-22` → `1999-06-25` (約 1 年，演化 fitness 評估)
- 測試初始期: `1998-07-07` → `1999-06-28` (約 1 年，用於計算技術指標)
- 測試回測期: `1999-06-28` → `2000-06-30` (約 1 年，樣本外測試)

**長訓練期**:
- 訓練初始期: `1992-06-30` → `1993-07-02` (約 1 年，用於計算技術指標)
- 訓練回測期: `1993-07-02` → `1999-06-25` (約 6 年，演化 fitness 評估) ⭐
- 測試初始期: `1998-07-07` → `1999-06-28` (約 1 年，用於計算技術指標)
- 測試回測期: `1999-06-28` → `2000-06-30` (約 1 年，樣本外測試)

**關鍵差異**:
- ✅ 測試期完全相同（公平比較）
- ✅ 訓練回測期長度不同（1年 vs 6年）
- ✅ 研究問題：更長的訓練期是否能產生更好的泛化能力？

### 3.2 GP 參數配置

**固定參數**（在 subprocess 命令中）:
- `--generations 50` - 演化 50 代
- `--population 500` - 族群大小 500
- `--mode portfolio` - 使用 Portfolio 模式

**其他參數**（在 `main.py` 中配置）:
- 交配率 (CXPB)
- 變異率 (MUTPB)
- 選擇算子 (`ranked_selection`)
- 交配算子 (`cxOnePoint`)
- 變異算子 (`mutUniform`)

---

## 4. 文件輸出結構

### 專案根目錄
```
all_experiments_results.csv       # 匯總表格（80 行 × 多列）
all_experiments_results.json      # 匯總 JSON（完整數據）
```

### experiments_results/
```
experiments_results/
├── ABX_TO/
│   ├── short_run01_train_trades.csv
│   ├── short_run01_test_trades.csv
│   ├── short_run01_result.json
│   ├── short_run01_output.log
│   ├── short_run02_train_trades.csv
│   ├── ... (run03 到 run10)
│   ├── long_run01_train_trades.csv
│   ├── long_run01_test_trades.csv
│   ├── long_run01_result.json
│   ├── long_run01_output.log
│   └── ... (run02 到 run10)
├── BBD-B_TO/
│   └── ... (同上結構)
├── RY_TO/
│   └── ... (同上結構)
└── TRP_TO/
    └── ... (同上結構)
```

### 文件統計
- 每個股票目錄包含: 20 個 CSV + 20 個 JSON + 20 個 LOG = 60 個文件
- 總文件數: 4 股票 × 60 文件/股票 = 240 個文件 + 2 個匯總文件 = **242 個文件**

---

## 5. 執行流程圖

```
開始
  │
  ├─→ 初始化配置
  │   ├─ tickers = [ABX.TO, BBD-B.TO, RY.TO, TRP.TO]
  │   ├─ experiments = [短訓練期, 長訓練期]
  │   └─ n_runs = 10
  │
  ├─→ 三層嵌套循環
  │   │
  │   └─→ for ticker in tickers:                    # 4 次
  │       │
  │       └─→ for exp in experiments:               # 2 次
  │           │
  │           └─→ for run in range(1, 11):          # 10 次
  │               │
  │               ├─→ modify_main_py()              # 修改配置
  │               ├─→ subprocess.run(main.py)       # 執行實驗
  │               ├─→ extract_results()             # 提取結果
  │               ├─→ 保存 CSV/JSON/LOG             # 保存文件
  │               └─→ all_results.append()          # 累積結果
  │
  ├─→ 保存匯總結果
  │   ├─ all_experiments_results.csv
  │   └─ all_experiments_results.json
  │
  ├─→ generate_summary()
  │   ├─ 分組統計
  │   ├─ 勝率分析
  │   ├─ 最佳/最差表現
  │   └─ 總體結論
  │
  └─→ 結束
```

**總執行時間估計**:
- 假設每次實驗 5 分鐘
- 80 次 × 5 分鐘 = 400 分鐘 ≈ **6.7 小時**

---

## 6. 關鍵問題解答

### Q1: 為什麼要修改 main.py 而不是傳遞參數？

**A1**: 
- `main.py` 的日期配置是硬編碼的變量，不是命令行參數
- 使用正則替換是最簡單的自動化方式
- **缺點**: 會修改源文件，需要 git 管理
- **改進方案**: 可以將 `main.py` 重構為接受日期參數

### Q2: 如何確保實驗可重現？

**A2**:
- ✅ 固定的隨機種子（在 `main.py` 中設置）
- ✅ 固定的 GP 參數（generations, population）
- ✅ 固定的日期配置
- ✅ 保存完整的輸出日誌
- ⚠️ 注意：DEAP 的隨機性可能導致結果略有不同

### Q3: 如何處理實驗失敗？

**A3**:
- 使用 `try-except` 捕獲異常
- 打印錯誤信息但不中斷整體流程
- `continue` 跳過失敗的實驗
- **缺點**: 失敗的實驗不會重試
- **改進**: 可以添加重試機制或失敗記錄

### Q4: 結果如何匯總分析？

**A4**:
1. 使用 pandas DataFrame 存儲所有結果
2. `groupby()` 進行分組統計
3. 計算均值、標準差、最小值、最大值
4. 計算勝率（超越 Buy-and-Hold 的比例）
5. 比較短訓練期 vs 長訓練期

### Q5: 為什麼運行 10 次？

**A5**:
- GP 是隨機算法，單次結果不穩定
- 10 次重複可以評估算法的穩定性
- 可以計算均值和標準差
- 可以識別異常值
- 符合統計學的最小樣本量要求

### Q6: 如何選擇最佳策略？

**A6**: 優先級排序
1. `test_excess_return`（樣本外超額報酬）⭐ **最重要**
2. 勝率（10 次中有多少次盈利）
3. 穩定性（標準差越小越好）
4. `train_excess_return`（訓練期表現，參考用）

### Q7: 如何解讀統計結果？

**A7**: 

**好的結果**:
- ✅ `test_excess_return > 0`（超越 Buy-and-Hold）
- ✅ 勝率 > 50%（多數情況下盈利）
- ✅ 標準差小（結果穩定）
- ✅ 訓練期和測試期表現一致（無過擬合）

**警告信號**:
- ⚠️ 訓練期很好但測試期很差（過擬合）
- ⚠️ 標準差很大（不穩定）
- ⚠️ 勝率 < 50%（多數情況下虧損）

### Q8: 長訓練期 vs 短訓練期，哪個更好？

**A8**: 

**理論預期**:
- **長訓練期**: 更多數據，更好的泛化能力
- **短訓練期**: 更接近測試期，更相關的市場環境

**實際結果需要看 `generate_summary()` 的輸出**:
- 比較兩者的平均 `test_excess_return`
- 比較兩者的勝率
- 比較兩者的穩定性

**可能的結論**:
- ✅ 長訓練期明顯優於短訓練期
- ⚠️ 短訓練期表現優於長訓練期（市場環境變化）
- ⚠️ 兩者表現相當（訓練期長度影響不大）

---

## 7. 與其他模塊的關係

### 依賴關係
```
run_all_experiments.py
  │
  ├─→ main.py                     # 執行單次實驗
  │   │
  │   ├─→ gp_quant/evolution/engine.py
  │   ├─→ gp_quant/backtesting/engine.py
  │   ├─→ gp_quant/data/loader.py
  │   └─→ gp_quant/gp/operators.py
  │
  └─→ subprocess                  # 進程管理
      └─→ pandas                  # 數據分析
```

### 輸出被使用於
- 論文結果分析
- 可視化（`plot_pnl.py`）
- 進一步的統計檢驗

---

## 8. 使用示例

### 執行完整實驗
```bash
python run_all_experiments.py
```

### 預期輸出
```
🚀🚀🚀...
開始大規模實驗
股票數量: 4
訓練期類型: 2
每個配置運行次數: 10
總實驗數: 80
🚀🚀🚀...

################################################################################
# 開始處理股票: ABX.TO
################################################################################

================================================================================
配置: 短訓練期
訓練初始期: 1997-06-25 至 1998-06-22
訓練回測期: 1998-06-22 至 1999-06-25
測試初始期: 1998-07-07 至 1999-06-28
測試回測期: 1999-06-28 至 2000-06-30
================================================================================

================================================================================
🔬 運行: ABX.TO | 短訓練期 | 第 1/10 次
================================================================================
樣本外超額報酬: $1,234.56 ✅ 盈利
執行時間: 287.45 秒
📁 文件已保存至: experiments_results/ABX_TO/

📊 總進度: 1/80 (1.2%)
...
```

---

## ✅ Review Checklist

完成 review 後，確保你能回答：

- [ ] 這個腳本的主要目的是什麼？
- [ ] 總共會運行多少次實驗？
- [ ] 如何動態修改 `main.py` 的配置？
- [ ] 如何從輸出中提取結果？
- [ ] 文件輸出結構是什麼？
- [ ] 如何處理實驗失敗？
- [ ] 如何計算勝率？
- [ ] 短訓練期和長訓練期的差異是什麼？
- [ ] 如何判斷哪種配置更好？
- [ ] 總執行時間大約多久？

---

**最後更新**: 2025-10-07  
**Review 狀態**: ✅ 已完成
