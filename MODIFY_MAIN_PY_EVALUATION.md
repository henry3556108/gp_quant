# modify_main_py() 方法評估報告

**評估對象**: `run_all_experiments.py` 中的 `modify_main_py()` 函數  
**評估日期**: 2025-10-07  
**結論**: ⚠️ **不合理，建議重構**

---

## 📋 目錄

1. [當前實現分析](#1-當前實現分析)
2. [問題清單](#2-問題清單)
3. [風險評估](#3-風險評估)
4. [替代方案](#4-替代方案)
5. [推薦方案](#5-推薦方案)
6. [重構實施計劃](#6-重構實施計劃)

---

## 1. 當前實現分析

### 1.1 當前做法

```python
def modify_main_py(train_data_start, train_backtest_start, train_backtest_end,
                   test_data_start, test_backtest_start, test_backtest_end):
    """Modify main.py with new date ranges including initial periods"""
    with open('main.py', 'r') as f:
        content = f.read()
    
    # 使用正則表達式替換 6 個日期變量
    content = re.sub(
        r"train_data_start = '[0-9]{4}-[0-9]{2}-[0-9]{2}'",
        f"train_data_start = '{train_data_start}'",
        content
    )
    # ... (重複 5 次)
    
    with open('main.py', 'w') as f:
        f.write(content)
```

### 1.2 設計意圖

- **目的**: 在運行 80 次實驗時，動態修改 `main.py` 中的日期配置
- **原因**: `main.py` 的日期是硬編碼變量（Line 69-75），不是命令行參數
- **使用場景**: 每次實驗前修改配置，然後執行 `subprocess.run(['python', 'main.py', ...])`

---

## 2. 問題清單

### ❌ 問題 1: 修改源代碼文件

**嚴重程度**: 🔴 高

**問題描述**:
- 直接修改 `main.py` 源文件，每次實驗都會重寫
- 如果實驗中斷，`main.py` 會保持在最後一次修改的狀態
- Git 會顯示 `main.py` 有未提交的修改

**實際影響**:
```bash
$ git status
modified:   main.py  # 每次運行實驗後都會顯示
```

**風險**:
- 可能誤提交修改後的 `main.py`
- 多人協作時會產生衝突
- 無法同時運行多個實驗（文件競爭）

---

### ❌ 問題 2: 缺乏原子性

**嚴重程度**: 🟡 中

**問題描述**:
- 如果在修改 `main.py` 和執行實驗之間程序崩潰，文件會處於不一致狀態
- 沒有備份或恢復機制

**場景**:
```python
modify_main_py(...)  # 修改成功
# 如果這裡崩潰，main.py 已被修改但實驗未運行
result = subprocess.run(['python', 'main.py', ...])
```

---

### ❌ 問題 3: 不支持並行執行

**嚴重程度**: 🟡 中

**問題描述**:
- 無法同時運行多個實驗腳本
- 多個進程會互相覆蓋 `main.py`

**限制**:
```bash
# 無法同時執行
Terminal 1: python run_all_experiments.py  # 修改 main.py
Terminal 2: python run_all_experiments.py  # 也修改 main.py → 衝突
```

---

### ❌ 問題 4: 正則表達式脆弱

**嚴重程度**: 🟡 中

**問題描述**:
- 依賴特定的字符串格式: `train_data_start = 'YYYY-MM-DD'`
- 如果格式改變（例如使用雙引號、添加註釋），正則會失效

**脆弱示例**:
```python
# 這些格式會導致正則失效
train_data_start = "1992-06-30"  # 使用雙引號
train_data_start = '1992-06-30'  # 添加註釋
train_data_start='1992-06-30'    # 沒有空格
```

---

### ❌ 問題 5: 違反單一職責原則

**嚴重程度**: 🟢 低

**問題描述**:
- `main.py` 應該是執行入口，不應該被其他腳本修改
- 配置應該通過參數傳遞，而不是修改源代碼

**設計原則**:
- ✅ 好的設計: `main.py` 接受參數 → 實驗腳本傳遞參數
- ❌ 當前設計: 實驗腳本修改 `main.py` → `main.py` 執行

---

### ❌ 問題 6: 難以測試和調試

**嚴重程度**: 🟢 低

**問題描述**:
- 無法輕易驗證修改是否正確
- 調試時需要檢查文件內容
- 單元測試困難

---

## 3. 風險評估

### 3.1 風險矩陣

| 風險 | 概率 | 影響 | 風險等級 |
|------|------|------|----------|
| 誤提交修改後的 main.py | 高 | 中 | 🔴 高 |
| 實驗中斷導致文件不一致 | 中 | 中 | 🟡 中 |
| 多進程衝突 | 低 | 高 | 🟡 中 |
| 正則表達式失效 | 低 | 中 | 🟢 低 |
| 維護困難 | 中 | 低 | 🟢 低 |

### 3.2 實際發生過的問題

根據 git 歷史，可以看到：
- `main.py` 經常出現在未提交的修改中
- 需要手動 `git restore main.py` 來恢復

---

## 4. 替代方案

### 方案 A: 命令行參數（推薦）⭐

**實現方式**:
```python
# main.py 修改
parser.add_argument("--train_data_start", type=str, default='1992-06-30')
parser.add_argument("--train_backtest_start", type=str, default='1993-07-02')
# ... 添加所有日期參數

# run_all_experiments.py 調用
subprocess.run([
    'python', 'main.py',
    '--tickers', ticker,
    '--train_data_start', train_data_start,
    '--train_backtest_start', train_backtest_start,
    # ... 傳遞所有參數
])
```

**優點**:
- ✅ 不修改源文件
- ✅ 支持並行執行
- ✅ 清晰的參數傳遞
- ✅ 易於測試和調試
- ✅ 符合標準實踐

**缺點**:
- ⚠️ 需要修改 `main.py` 的 argparse 配置
- ⚠️ 命令行會變長（6 個額外參數）

**工作量**: 🟡 中等（約 30 分鐘）

---

### 方案 B: 配置文件

**實現方式**:
```python
# config.json
{
    "train_data_start": "1992-06-30",
    "train_backtest_start": "1993-07-02",
    ...
}

# main.py 修改
parser.add_argument("--config", type=str, help="Path to config JSON")
if args.config:
    with open(args.config) as f:
        config = json.load(f)

# run_all_experiments.py 調用
config_file = f"temp_config_{ticker}_{run_number}.json"
with open(config_file, 'w') as f:
    json.dump(config_dict, f)

subprocess.run(['python', 'main.py', '--config', config_file])
os.remove(config_file)
```

**優點**:
- ✅ 不修改源文件
- ✅ 支持複雜配置
- ✅ 可重用配置
- ✅ 易於版本控制

**缺點**:
- ⚠️ 需要管理臨時文件
- ⚠️ 增加文件 I/O
- ⚠️ 需要更多代碼

**工作量**: 🟡 中等（約 45 分鐘）

---

### 方案 C: 環境變量

**實現方式**:
```python
# run_all_experiments.py
env = os.environ.copy()
env['TRAIN_DATA_START'] = train_data_start
env['TRAIN_BACKTEST_START'] = train_backtest_start

subprocess.run(['python', 'main.py', ...], env=env)

# main.py 修改
train_data_start = os.getenv('TRAIN_DATA_START', '1992-06-30')
```

**優點**:
- ✅ 不修改源文件
- ✅ 簡單實現
- ✅ 支持並行執行

**缺點**:
- ⚠️ 環境變量不夠明確
- ⚠️ 調試困難
- ⚠️ 不是標準實踐

**工作量**: 🟢 低（約 20 分鐘）

---

### 方案 D: 函數化 main.py

**實現方式**:
```python
# main.py 重構
def run_portfolio_evolution_with_dates(
    tickers, generations, population,
    train_data_start, train_backtest_start, ...
):
    # 原來的 run_portfolio_evolution 邏輯
    pass

if __name__ == "__main__":
    # 命令行入口
    args = parser.parse_args()
    run_portfolio_evolution_with_dates(...)

# run_all_experiments.py 直接調用
from main import run_portfolio_evolution_with_dates
result = run_portfolio_evolution_with_dates(
    tickers=[ticker],
    generations=50,
    population=500,
    train_data_start=train_data_start,
    ...
)
```

**優點**:
- ✅ 不修改源文件
- ✅ 不需要 subprocess
- ✅ 更快的執行速度
- ✅ 更好的錯誤處理
- ✅ 易於測試

**缺點**:
- ⚠️ 需要大幅重構 `main.py`
- ⚠️ 改變程序架構
- ⚠️ 需要處理全局狀態（DEAP creator）

**工作量**: 🔴 高（約 2-3 小時）

---

## 5. 推薦方案

### 🏆 推薦: 方案 A（命令行參數）

**理由**:
1. **最符合標準實踐** - 命令行參數是配置程序的標準方式
2. **最小侵入性** - 只需修改 argparse 配置，不改變程序結構
3. **最佳可維護性** - 清晰、明確、易於理解
4. **支持並行** - 多個進程可以同時運行
5. **工作量適中** - 約 30 分鐘即可完成

### 實施優先級

**短期（立即）**:
- 實施方案 A（命令行參數）
- 移除 `modify_main_py()` 函數

**中期（可選）**:
- 如果配置變得複雜，考慮方案 B（配置文件）

**長期（可選）**:
- 如果需要更好的性能和測試性，考慮方案 D（函數化）

---

## 6. 重構實施計劃

### 步驟 1: 修改 main.py 的 argparse

```python
# main.py Line 408 附近
parser.add_argument("--train_data_start", type=str, default='1992-06-30',
                    help="Training initial period start date")
parser.add_argument("--train_backtest_start", type=str, default='1993-07-02',
                    help="Training backtest period start date")
parser.add_argument("--train_backtest_end", type=str, default='1999-06-25',
                    help="Training backtest period end date")
parser.add_argument("--test_data_start", type=str, default='1998-07-07',
                    help="Testing initial period start date")
parser.add_argument("--test_backtest_start", type=str, default='1999-06-28',
                    help="Testing backtest period start date")
parser.add_argument("--test_backtest_end", type=str, default='2000-06-30',
                    help="Testing backtest period end date")
```

### 步驟 2: 修改 main.py 使用 args

```python
# main.py Line 69-75 修改為
train_data_start = args.train_data_start
train_backtest_start = args.train_backtest_start
train_backtest_end = args.train_backtest_end
test_data_start = args.test_data_start
test_backtest_start = args.test_backtest_start
test_backtest_end = args.test_backtest_end
```

### 步驟 3: 修改 run_all_experiments.py

```python
# 移除 modify_main_py() 函數

# 修改 run_single_experiment()
def run_single_experiment(ticker, period_name, 
                         train_data_start, train_backtest_start, train_backtest_end,
                         test_data_start, test_backtest_start, test_backtest_end,
                         run_number):
    """Run a single experiment"""
    # ... 前面的代碼保持不變
    
    # 移除: modify_main_py(...)
    
    # 修改 subprocess 調用
    result = subprocess.run([
        'python', 'main.py',
        '--tickers', ticker,
        '--mode', 'portfolio',
        '--generations', '50',
        '--population', '500',
        '--train_data_start', train_data_start,
        '--train_backtest_start', train_backtest_start,
        '--train_backtest_end', train_backtest_end,
        '--test_data_start', test_data_start,
        '--test_backtest_start', test_backtest_start,
        '--test_backtest_end', test_backtest_end
    ], capture_output=True, text=True)
    
    # ... 後面的代碼保持不變
```

### 步驟 4: 測試

```bash
# 測試單次運行
python main.py \
  --tickers ABX.TO \
  --mode portfolio \
  --generations 2 \
  --population 10 \
  --train_data_start 1997-06-25 \
  --train_backtest_start 1998-06-22 \
  --train_backtest_end 1999-06-25 \
  --test_data_start 1998-07-07 \
  --test_backtest_start 1999-06-28 \
  --test_backtest_end 2000-06-30

# 檢查 git status
git status  # 應該不顯示 main.py 被修改

# 測試完整實驗腳本（小規模）
# 修改 run_all_experiments.py 的 n_runs = 2 進行測試
python run_all_experiments.py
```

### 步驟 5: 更新文檔

- 更新 `RUN_ALL_EXPERIMENTS_EXPLAINED.md`
- 更新 `README.md` 的使用說明
- 添加新參數的文檔

---

## 7. 總結

### 當前方法的評分

| 評估維度 | 評分 | 說明 |
|---------|------|------|
| 正確性 | 🟡 6/10 | 功能正常但有風險 |
| 可維護性 | 🔴 3/10 | 修改源文件，難以維護 |
| 可擴展性 | 🟡 5/10 | 添加新參數需要修改正則 |
| 安全性 | 🔴 4/10 | 可能誤提交、文件衝突 |
| 標準實踐 | 🔴 2/10 | 違反常見設計原則 |
| **總分** | **🔴 4/10** | **不推薦使用** |

### 推薦方案的評分

| 評估維度 | 評分 | 說明 |
|---------|------|------|
| 正確性 | ✅ 10/10 | 標準的參數傳遞 |
| 可維護性 | ✅ 9/10 | 清晰明確 |
| 可擴展性 | ✅ 9/10 | 易於添加新參數 |
| 安全性 | ✅ 10/10 | 不修改源文件 |
| 標準實踐 | ✅ 10/10 | 符合業界標準 |
| **總分** | **✅ 9.6/10** | **強烈推薦** |

### 最終建議

**結論**: 當前的 `modify_main_py()` 方法是一個**臨時的權宜之計**，在生產環境中**不合理**。

**行動建議**:
1. ✅ **立即重構** - 使用命令行參數方案（工作量約 30 分鐘）
2. ✅ **測試驗證** - 確保重構後功能正常
3. ✅ **更新文檔** - 記錄新的使用方式
4. ✅ **提交代碼** - 完成後提交到 git

**預期收益**:
- 消除 git 衝突風險
- 支持並行執行實驗
- 提高代碼質量和可維護性
- 符合軟件工程最佳實踐

---

**評估人**: AI Assistant  
**評估日期**: 2025-10-07  
**建議優先級**: 🔴 高（建議立即重構）
