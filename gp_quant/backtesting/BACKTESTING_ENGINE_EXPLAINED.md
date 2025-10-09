# backtesting/engine.py 完整說明文檔

**文件**: `gp_quant/backtesting/engine.py`  
**總行數**: 477 行  
**核心功能**: 回測引擎 - 評估交易策略的 Fitness  
**創建日期**: 2025-10-06

---

## 📋 目錄

1. [文件概覽](#文件概覽)
2. [Numba JIT 函數: _numba_simulation_loop()](#numba-jit-函數-_numba_simulation_loop)
3. [類 1: BacktestingEngine](#類-1-backtestingengine)
4. [類 2: PortfolioBacktestingEngine](#類-2-portfoliobacktestingengine)
5. [完整回測流程](#完整回測流程)
6. [關鍵設計決策](#關鍵設計決策)
7. [常見問題 FAQ](#常見問題-faq)

---

## 📦 文件概覽

### **文件結構**

```
engine.py (477 行)
├── 導入模塊 (L1-14)
├── _numba_simulation_loop() (L17-45)      ← JIT 編譯的交易模擬
├── BacktestingEngine 類 (L48-365)        ← 單 ticker 回測引擎
│   ├── __init__() (L53-95)
│   ├── evaluate() (L97-192)              ← Fitness 評估
│   ├── _run_vectorized_simulation() (L194-233)
│   ├── run_detailed_simulation() (L235-322)
│   └── get_signals() (L324-365)
└── PortfolioBacktestingEngine 類 (L368-476) ← 多 ticker 組合回測
    ├── __init__() (L377-413)
    ├── evaluate() (L415-442)
    └── run_detailed_simulation() (L444-475)
```

### **依賴模塊**

```python
import pandas as pd
import numpy as np
import numba                    # JIT 編譯加速
from deap import gp
from typing import Callable, Dict, List

from gp_quant.gp.operators import pset, NumVector
```

### **核心職責**

1. ✅ **Fitness 評估**: 計算 GP 個體的 excess return
2. ✅ **交易模擬**: 向量化的高速回測
3. ✅ **初始期支持**: 分離技術指標計算期和回測期
4. ✅ **詳細記錄**: 生成完整的交易日誌
5. ✅ **Portfolio 支持**: 多 ticker 組合評估

---

## ⚡ Numba JIT 函數: _numba_simulation_loop()

**位置**: L17-45 (29 行)  
**功能**: JIT 編譯的交易模擬循環（極速）

### **函數簽名**

```python
@numba.jit(nopython=True)
def _numba_simulation_loop(signals, open_prices, close_prices, initial_capital):
    """
    A Numba-JIT compiled function to run the trading simulation at high speed.
    This function only works with NumPy arrays.
    """
```

### **參數說明**

| 參數 | 類型 | 說明 |
|------|------|------|
| `signals` | np.ndarray (bool) | 交易信號陣列 (True=買入, False=賣出) |
| `open_prices` | np.ndarray (float) | 開盤價陣列 |
| `close_prices` | np.ndarray (float) | 收盤價陣列 |
| `initial_capital` | float | 初始資金 |

### **返回值**

```python
return capital - initial_capital  # 總報酬（可正可負）
```

### **交易邏輯**

```python
position = 0  # 0 = 空倉, 1 = 持倉
capital = initial_capital
shares = 0.0

for i in range(len(signals)):
    signal = signals[i]
    next_day_open_price = open_prices[i + 1]
    
    # 買入邏輯
    if position == 0 and signal == True:
        if next_day_open_price > 0:
            shares = capital / next_day_open_price  # 全倉買入
            capital = 0.0
            position = 1
    
    # 賣出邏輯
    elif position == 1 and signal == False:
        capital = shares * next_day_open_price  # 全倉賣出
        shares = 0.0
        position = 0

# 最後如果還持倉，用收盤價結算
if position == 1:
    capital = shares * close_prices[-1]

return capital - initial_capital
```

### **交易規則**

1. **全倉交易**: 每次買入用全部資金，賣出全部股票
2. **次日開盤執行**: 今天信號，明天開盤價成交
3. **只做多**: 只有買入和賣出，沒有做空
4. **無交易成本**: 不考慮手續費和滑點

### **為什麼用 Numba JIT？**

```python
# 沒有 JIT
def slow_simulation(signals, ...):
    for i in range(len(signals)):  # Python 循環很慢
        ...
# 500 個體 × 50 代 × 1000 天數據 = 25,000,000 次循環
# 耗時: ~30 秒

# 有 JIT
@numba.jit(nopython=True)
def fast_simulation(signals, ...):
    for i in range(len(signals)):  # 編譯成機器碼，超快
        ...
# 耗時: ~0.5 秒

# 加速: 60 倍！
```

### **nopython=True 的限制**

```python
# ✅ 可以用
- NumPy 陣列操作
- 基本數學運算
- 簡單的 if/for/while
- 基本數據類型 (int, float, bool)

# ❌ 不能用
- Pandas DataFrame
- Python list/dict
- 字串操作
- 類和對象
```

---

## 🎯 類 1: BacktestingEngine

**位置**: L48-365 (318 行)  
**功能**: 單 ticker 回測引擎

### **類結構**

```
BacktestingEngine
├── __init__()                    ← 初始化，設置回測期
├── evaluate()                    ← 評估 fitness (核心)
├── _run_vectorized_simulation()  ← 運行向量化模擬
├── run_detailed_simulation()     ← 生成詳細交易記錄
└── get_signals()                 ← 提取交易信號
```

---

### **方法 1: __init__()** (L53-95)

```python
def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0,
             backtest_start: str = None, backtest_end: str = None):
```

#### **參數說明**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `data` | pd.DataFrame | - | 歷史數據（含初始期） |
| `initial_capital` | float | 100000.0 | 初始資金 |
| `backtest_start` | str | None | 回測起始日 (可選) |
| `backtest_end` | str | None | 回測結束日 (可選) |

#### **初始化流程**

```
1. 保存參數 (L67-70)
   self.data = data
   self.initial_capital = initial_capital
   self.backtest_start = backtest_start
   self.backtest_end = backtest_end

2. 處理回測期 (L72-91)
   if backtest_start or backtest_end:
       ├─ 檢查 backtest_start 是否早於數據開始 (L76-80)
       │  └─ 如果是，調整並警告
       ├─ 創建 mask (L83-87)
       └─ 提取 backtest_data (L88)
   else:
       └─ 使用全部數據 (向後兼容) (L91)

3. 深拷貝 pset (L93-95)
   self.pset = copy.deepcopy(pset)
   └─ 避免修改全局 pset
```

#### **backtest_data 的作用**

```python
# 範例數據
data.index: [1997-06-25, ..., 1999-06-25]  # 完整數據（含初始期）
backtest_start: '1998-06-22'
backtest_end: '1999-06-25'

# 結果
self.data:          1997-06-25 到 1999-06-25  # 完整數據（用於計算技術指標）
self.backtest_data: 1998-06-22 到 1999-06-25  # 回測數據（用於計算報酬）
```

#### **為什麼需要分離？**

```python
# 技術指標需要歷史數據
RSI(ARG0, 14)  # 需要前 14 天的數據
SMA(ARG0, 50)  # 需要前 50 天的數據

# 如果從 1998-06-22 開始
# 前 50 天沒有數據 → RSI/SMA 無法計算 → 策略失效

# 解決: 提供初始期
data:          1997-06-25 開始（提供 250 天初始期）
backtest:      1998-06-22 開始（技術指標已經穩定）
```

---

### **方法 2: evaluate()** (L97-192) ⭐ **核心方法**

```python
def evaluate(self, individual: gp.PrimitiveTree) -> tuple[float]:
    """
    Evaluates the fitness of a single GP individual using vectorization.
    """
```

#### **完整流程**

```
輸入: individual (GP 樹)
  ↓
步驟 1: 注入數據到 pset (L101-105)
  price_vec = self.data['Close'].to_numpy()
  volume_vec = self.data['Volume'].to_numpy()
  self.pset.terminals[NumVector][0].value = price_vec
  self.pset.terminals[NumVector][1].value = volume_vec
  ↓
步驟 2: 編譯並執行 GP 樹 (L107-146)
  try:
      rule = gp.compile(expr=individual, pset=self.pset)
      signals = rule()  # 生成交易信號
  except:
      return -100000.0,  # 懲罰 fitness
  ↓
步驟 3: 清理信號 (L112-149)
  ├─ 處理單一布林值 (L114-115)
  ├─ 處理 NaN/Inf (L118-120)
  └─ 轉換為布林陣列 (L149)
  ↓
步驟 4: 提取回測期信號 (L152-162)
  if backtest_start or backtest_end:
      backtest_signals = signals[mask]
  else:
      backtest_signals = signals
  ↓
步驟 5: 運行模擬 (L164)
  gp_return = self._run_vectorized_simulation(backtest_signals, self.backtest_data)
  ↓
步驟 6: 計算 B&H 報酬 (L166-173)
  start_price = self.backtest_data['Close'].iloc[0]
  end_price = self.backtest_data['Close'].iloc[-1]
  buy_and_hold_return = (end_price / start_price - 1) * initial_capital
  ↓
步驟 7: 計算 Excess Return (L175-176)
  excess_return = gp_return - buy_and_hold_return
  ↓
步驟 8: 合理性檢查 (L178-188)
  if not reasonable:
      return -100000.0,
  ↓
輸出: (excess_return,)
```

#### **步驟 1: 注入數據到 pset**

```python
# L101-105
price_vec = self.data['Close'].to_numpy()    # 完整數據的收盤價
volume_vec = self.data['Volume'].to_numpy()  # 完整數據的成交量

# 注入到 pset 的 terminals
self.pset.terminals[NumVector][0].value = price_vec   # ARG0 = price
self.pset.terminals[NumVector][1].value = volume_vec  # ARG1 = volume
```

**為什麼用完整數據？**
```python
# 技術指標需要完整歷史
individual = RSI(ARG0, 14)

# 編譯後
rule = lambda: RSI(price_vec, 14)

# 執行
signals = rule()
# signals[0:249] = 初始期的信號（不用於回測）
# signals[250:505] = 回測期的信號（用於計算報酬）
```

#### **步驟 2: 編譯並執行**

```python
# L109-110
rule: Callable = gp.compile(expr=individual, pset=self.pset)
signals = rule()
```

**範例**:
```python
# individual
and(gt(SMA(ARG0, 20), ARG0), lt(RSI(ARG0, 14), 30))

# 編譯後的 rule
def rule():
    sma_20 = SMA(price_vec, 20)
    rsi_14 = RSI(price_vec, 14)
    cond1 = sma_20 > price_vec
    cond2 = rsi_14 < 30
    return cond1 & cond2

# 執行
signals = rule()
# signals = [False, False, True, True, False, ...]  (505 個布林值)
```

#### **步驟 3: 異常處理**

```python
# L122-146: 多層異常處理

try:
    # 正常執行
    rule = gp.compile(expr=individual, pset=self.pset)
    signals = rule()
    
except TypeError as e:
    # 處理載入的個體（需要參數）
    if "missing" in str(e) and "required positional arguments" in str(e):
        try:
            rule = gp.compile(expr=individual, pset=self.pset)
            signals = rule(price_vec, volume_vec)  # 傳入參數
        except:
            return -100000.0,
    else:
        return -100000.0,
        
except (OverflowError, ValueError, FloatingPointError, Exception) as e:
    # 任何其他錯誤
    return -100000.0,
```

**為什麼需要這麼多異常處理？**
```python
# 可能的錯誤
1. 除以零: div(ARG0, 0)
2. 對數負數: log(sub(ARG0, ARG0))
3. 溢出: exp(mul(ARG0, 1000))
4. 類型錯誤: 載入的個體結構不同
5. 數組長度不匹配: 技術指標計算錯誤
```

#### **步驟 4: 清理信號**

```python
# L114-115: 處理單一布林值
if not isinstance(signals, np.ndarray):
    signals = np.full(self.data.shape[0], signals, dtype=np.bool_)

# 範例
individual = V_TRUE  # 常數終端
signals = True       # 單一布林值
# 轉換後
signals = [True, True, True, ..., True]  # 505 個 True
```

```python
# L118-120: 處理 NaN/Inf
if not np.all(np.isfinite(signals)):
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

# 範例
signals = [True, False, NaN, Inf, True, ...]
# 轉換後
signals = [True, False, False, False, True, ...]
```

#### **步驟 5: 提取回測期信號**

```python
# L153-162
if self.backtest_start or self.backtest_end:
    mask = pd.Series(False, index=self.data.index)
    if self.backtest_start:
        mask |= (self.data.index >= self.backtest_start)
    if self.backtest_end:
        mask &= (self.data.index <= self.backtest_end)
    backtest_signals = signals[mask.values]
else:
    backtest_signals = signals
```

**範例**:
```python
# 完整數據
data.index: [1997-06-25, ..., 1999-06-25]  # 505 天
signals:    [F, F, T, T, F, ..., T, F, T]  # 505 個信號

# 回測期
backtest_start: '1998-06-22'
backtest_end:   '1999-06-25'

# mask
mask: [False, False, ..., True, True, ..., True]  # 前 250 個 False，後 255 個 True

# 結果
backtest_signals: [T, T, F, ..., T, F, T]  # 只有 255 個信號
```

#### **步驟 6: 計算 B&H 報酬**

```python
# L168-173
start_price = self.backtest_data['Close'].iloc[0]  # 回測期第一天收盤價
end_price = self.backtest_data['Close'].iloc[-1]   # 回測期最後一天收盤價

if start_price > 0:
    buy_and_hold_return = (end_price / start_price - 1) * self.initial_capital
else:
    buy_and_hold_return = 0
```

**範例**:
```python
initial_capital = 100000
start_price = 100  # 1998-06-22
end_price = 120    # 1999-06-25

buy_and_hold_return = (120 / 100 - 1) * 100000
                    = 0.2 * 100000
                    = 20000  # 賺 20%
```

#### **步驟 7: 計算 Excess Return**

```python
# L176
excess_return = gp_return - buy_and_hold_return
```

**範例**:
```python
gp_return = 35000           # GP 策略賺 35%
buy_and_hold_return = 20000 # B&H 賺 20%
excess_return = 15000       # Excess return = 15%

# 這就是 fitness！
```

#### **步驟 8: 合理性檢查**

```python
# L181-188
MAX_REASONABLE_FITNESS = self.initial_capital * 1000  # 100,000,000
MIN_REASONABLE_FITNESS = -self.initial_capital * 2    # -200,000

if not np.isfinite(excess_return) or \
   excess_return > MAX_REASONABLE_FITNESS or \
   excess_return < MIN_REASONABLE_FITNESS:
    return -100000.0,
```

**為什麼需要？**
```python
# 異常情況
1. NaN: 計算錯誤
2. Inf: 溢出
3. 過大: 100,000,000+ (不可能賺 1000 倍)
4. 過小: -200,000- (不可能虧超過 2 倍本金)

# 這些都是計算錯誤，給予懲罰 fitness
```

---

### **方法 3: _run_vectorized_simulation()** (L194-233)

```python
def _run_vectorized_simulation(self, signals: np.ndarray, data: pd.DataFrame = None) -> float:
    """
    Runs the simulation using the fast Numba JIT-compiled loop.
    """
```

#### **流程**

```python
# L202-208: 準備數據
if data is None:
    data = self.data

if not hasattr(signals, '__len__'):
    signals = np.full(data.shape[0], signals, dtype=np.bool_)

open_prices_np = data['Open'].to_numpy()
close_prices_np = data['Close'].to_numpy()

# L215-220: 調用 Numba JIT 函數
gp_return = _numba_simulation_loop(
    signals,
    open_prices_np,
    close_prices_np,
    self.initial_capital
)

# L222-231: 合理性檢查
MAX_REASONABLE_RETURN = self.initial_capital * 1000
MIN_REASONABLE_RETURN = -self.initial_capital * 2

if not np.isfinite(gp_return) or \
   gp_return > MAX_REASONABLE_RETURN or \
   gp_return < MIN_REASONABLE_RETURN:
    return -self.initial_capital  # 虧光

return gp_return
```

---

### **方法 4: run_detailed_simulation()** (L235-322)

```python
def run_detailed_simulation(self, individual: gp.PrimitiveTree) -> dict:
    """
    Runs a full simulation and returns detailed trade logs and performance metrics.
    Only records trades within the backtest period.
    """
```

#### **功能**: 生成詳細的交易記錄

#### **返回值結構**

```python
{
    'gp_return': 35000.0,
    'buy_and_hold_return': 20000.0,
    'trades': [
        {
            'entry_date': '1998-07-15',
            'exit_date': '1998-08-20',
            'entry_price': 105.50,
            'exit_price': 112.30,
            'shares': 947.87,
            'pnl': 6443.52
        },
        {
            'entry_date': '1998-09-10',
            'exit_date': '1998-10-05',
            'entry_price': 108.20,
            'exit_price': 115.80,
            'shares': 924.03,
            'pnl': 7022.63
        },
        ...
    ]
}
```

#### **交易記錄邏輯**

```python
# L263-296: 主循環
trades = []
position = 0
capital = initial_capital
shares = 0.0

for i in range(len(backtest_signals) - 1):
    signal = backtest_signals[i]
    next_day_open_price = open_prices[i + 1]
    
    # 買入
    if position == 0 and signal == True and capital > 0:
        shares = capital / next_day_open_price
        capital = 0.0
        position = 1
        entry_price = next_day_open_price
        entry_date = dates[i + 1]
    
    # 賣出
    elif position == 1 and signal == False:
        capital = shares * next_day_open_price
        pnl = (next_day_open_price - entry_price) * shares
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': dates[i + 1].strftime('%Y-%m-%d'),
            'entry_price': round(entry_price, 2),
            'exit_price': round(next_day_open_price, 2),
            'shares': round(shares, 2),
            'pnl': round(pnl, 2)
        })
        shares = 0.0
        position = 0

# L298-309: 處理最後持倉
if position == 1:
    last_close_price = data_to_use['Close'].iloc[-1]
    capital = shares * last_close_price
    pnl = (last_close_price - entry_price) * shares
    trades.append({...})
```

---

### **方法 5: get_signals()** (L324-365)

```python
def get_signals(self, individual: gp.PrimitiveTree) -> np.ndarray:
    """
    Extract trading signals from a GP individual without running full evaluation.
    Returns the boolean signal array.
    """
```

#### **功能**: 只提取信號，不計算 fitness

#### **用途**

```python
# 在 run_detailed_simulation() 中使用
signals = self.get_signals(individual)

# 在 main.py 的 load_and_show_signals() 中使用
signals = backtester.get_signals(best_individual)
for i in range(1, len(signals)):
    if signals[i] != signals[i-1]:
        print(f"{dates[i].date()}: {'BUY' if signals[i] else 'SELL'}")
```

---

## 🎯 類 2: PortfolioBacktestingEngine

**位置**: L368-476 (109 行)  
**功能**: 多 ticker 組合回測引擎

### **類結構**

```
PortfolioBacktestingEngine
├── __init__()                ← 初始化多個 BacktestingEngine
├── evaluate()                ← 評估組合 fitness
└── run_detailed_simulation() ← 生成組合詳細記錄
```

---

### **方法 1: __init__()** (L377-413)

```python
def __init__(self, data_dict: Dict[str, pd.DataFrame], total_capital: float = 100000.0,
             backtest_config: Dict[str, Dict] = None):
```

#### **參數說明**

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `data_dict` | Dict[str, DataFrame] | - | ticker → DataFrame 映射 |
| `total_capital` | float | 100000.0 | 總資金 |
| `backtest_config` | Dict[str, Dict] | None | ticker → backtest 配置 |

#### **初始化流程**

```python
# L388-392: 保存參數
self.data_dict = data_dict
self.total_capital = total_capital
self.tickers = list(data_dict.keys())
self.n_tickers = len(self.tickers)
self.backtest_config = backtest_config or {}

# L394-395: 平均分配資金
self.capital_per_ticker = total_capital / self.n_tickers

# L397-410: 為每個 ticker 創建 BacktestingEngine
self.engines = {}
for ticker, data in data_dict.items():
    config = self.backtest_config.get(ticker, {})
    backtest_start = config.get('backtest_start', None)
    backtest_end = config.get('backtest_end', None)
    
    self.engines[ticker] = BacktestingEngine(
        data, 
        self.capital_per_ticker,
        backtest_start=backtest_start,
        backtest_end=backtest_end
    )
```

#### **範例**

```python
# 輸入
data_dict = {
    'ABX.TO': DataFrame(...),
    'BBD-B.TO': DataFrame(...),
    'RY.TO': DataFrame(...)
}
total_capital = 300000
backtest_config = {
    'ABX.TO': {'backtest_start': '1998-06-22', 'backtest_end': '1999-06-25'},
    'BBD-B.TO': {'backtest_start': '1998-06-22', 'backtest_end': '1999-06-25'},
    'RY.TO': {'backtest_start': '1998-06-22', 'backtest_end': '1999-06-25'}
}

# 結果
self.n_tickers = 3
self.capital_per_ticker = 100000  # 每個 ticker 分配 10 萬
self.engines = {
    'ABX.TO': BacktestingEngine(data, 100000, '1998-06-22', '1999-06-25'),
    'BBD-B.TO': BacktestingEngine(data, 100000, '1998-06-22', '1999-06-25'),
    'RY.TO': BacktestingEngine(data, 100000, '1998-06-22', '1999-06-25')
}
```

---

### **方法 2: evaluate()** (L415-442)

```python
def evaluate(self, individual: gp.PrimitiveTree) -> tuple[float]:
    """
    Evaluates the fitness of a GP individual across all tickers in the portfolio.
    
    The fitness is calculated as the sum of excess returns from all tickers:
    fitness = sum(excess_return_ticker_i for all tickers)
    """
```

#### **流程**

```python
# L428-437: 評估每個 ticker
total_excess_return = 0.0
ticker_results = {}

for ticker in self.tickers:
    engine = self.engines[ticker]
    excess_return = engine.evaluate(individual)[0]
    
    ticker_results[ticker] = excess_return
    total_excess_return += excess_return

# L442: 返回總 excess return
return total_excess_return,
```

#### **範例**

```python
# 同一個 GP 規則在 3 個 ticker 上評估
individual = and(gt(SMA(ARG0, 20), ARG0), lt(RSI(ARG0, 14), 30))

# 評估結果
ABX.TO:    excess_return = 15000
BBD-B.TO:  excess_return = 8000
RY.TO:     excess_return = 12000

# Portfolio fitness
total_excess_return = 15000 + 8000 + 12000 = 35000

return (35000,)
```

---

### **方法 3: run_detailed_simulation()** (L444-475)

```python
def run_detailed_simulation(self, individual: gp.PrimitiveTree) -> Dict:
    """
    Runs detailed simulation for all tickers and returns comprehensive results.
    """
```

#### **返回值結構**

```python
{
    'tickers': {
        'ABX.TO': {
            'gp_return': 25000,
            'buy_and_hold_return': 10000,
            'trades': [...]
        },
        'BBD-B.TO': {
            'gp_return': 18000,
            'buy_and_hold_return': 10000,
            'trades': [...]
        },
        'RY.TO': {
            'gp_return': 22000,
            'buy_and_hold_return': 10000,
            'trades': [...]
        }
    },
    'portfolio_summary': {
        'total_gp_return': 65000,
        'total_bh_return': 30000,
        'total_excess_return': 35000,
        'capital_per_ticker': 100000,
        'total_capital': 300000
    }
}
```

---

## 📊 完整回測流程

### **單 Ticker 回測流程**

```
輸入: individual, data, initial_capital, backtest_start, backtest_end
  ↓
1. 初始化 BacktestingEngine
   ├─ 保存完整數據 (self.data)
   ├─ 提取回測數據 (self.backtest_data)
   └─ 深拷貝 pset
  ↓
2. 評估 (evaluate)
   ├─ 注入數據到 pset
   │  └─ ARG0 = price_vec (完整數據)
   │  └─ ARG1 = volume_vec (完整數據)
   ├─ 編譯並執行 GP 樹
   │  └─ signals = rule()  (完整數據的信號)
   ├─ 清理信號
   │  ├─ 處理單一布林值
   │  ├─ 處理 NaN/Inf
   │  └─ 轉換為布林陣列
   ├─ 提取回測期信號
   │  └─ backtest_signals = signals[mask]
   ├─ 運行模擬
   │  └─ gp_return = _numba_simulation_loop(...)
   ├─ 計算 B&H 報酬
   │  └─ buy_and_hold_return = (end/start - 1) * capital
   ├─ 計算 Excess Return
   │  └─ excess_return = gp_return - buy_and_hold_return
   └─ 合理性檢查
      └─ return (excess_return,)
  ↓
輸出: (excess_return,)
```

### **Portfolio 回測流程**

```
輸入: individual, data_dict, total_capital, backtest_config
  ↓
1. 初始化 PortfolioBacktestingEngine
   ├─ 計算每個 ticker 的資金
   │  └─ capital_per_ticker = total_capital / n_tickers
   └─ 為每個 ticker 創建 BacktestingEngine
      └─ engines[ticker] = BacktestingEngine(...)
  ↓
2. 評估 (evaluate)
   ├─ 對每個 ticker 評估
   │  └─ excess_return_i = engines[ticker].evaluate(individual)
   └─ 加總
      └─ total_excess_return = sum(excess_return_i)
  ↓
輸出: (total_excess_return,)
```

---

## 🎯 關鍵設計決策

### **1. 為什麼用 Numba JIT？**

**性能對比**:
```python
# Python 循環
def python_loop(signals, prices, capital):
    for i in range(len(signals)):
        # 交易邏輯
        ...
# 500 個體 × 50 代 × 1000 天 = 25,000,000 次循環
# 耗時: ~30 秒

# Numba JIT
@numba.jit(nopython=True)
def numba_loop(signals, prices, capital):
    for i in range(len(signals)):
        # 交易邏輯
        ...
# 耗時: ~0.5 秒
# 加速: 60 倍！
```

**代價**: 只能用 NumPy 陣列，不能用 Pandas

---

### **2. 為什麼分離初始期和回測期？**

**問題**: 技術指標需要歷史數據

```python
# 沒有初始期
data: 1998-06-22 到 1999-06-25  # 256 天

individual = SMA(ARG0, 50)
# 前 50 天的 SMA 無法計算 → NaN
# 策略失效

# 有初始期
data: 1997-06-25 到 1999-06-25  # 506 天（含 250 天初始期）
backtest: 1998-06-22 到 1999-06-25  # 256 天

individual = SMA(ARG0, 50)
# 在 1998-06-22 時，SMA 已經有 250 天歷史數據
# 策略正常運作
```

---

### **3. 為什麼用 Excess Return 作為 Fitness？**

**定義**:
```python
excess_return = gp_return - buy_and_hold_return
```

**原因**:
1. **公平比較**: 不同時期、不同 ticker 的市場表現不同
2. **相對表現**: 我們要的是「比 B&H 好多少」，不是絕對報酬
3. **風險調整**: 考慮了市場整體走勢

**範例**:
```python
# 情況 1: 牛市
gp_return = 50000  (50%)
bh_return = 40000  (40%)
excess_return = 10000  (10%)  ← fitness

# 情況 2: 熊市
gp_return = -10000  (-10%)
bh_return = -30000  (-30%)
excess_return = 20000  (20%)  ← fitness 更高！

# GP 策略在熊市中虧得少，fitness 反而更高
```

---

### **4. 為什麼需要多層異常處理？**

**可能的錯誤**:
```python
1. 除以零
   div(ARG0, 0)
   
2. 對數負數
   log(sub(ARG0, ARG0))
   
3. 溢出
   exp(mul(ARG0, 1000))
   
4. 數組長度不匹配
   技術指標計算錯誤
   
5. 類型錯誤
   載入的個體結構不同
```

**處理策略**:
```python
try:
    # 正常執行
except TypeError:
    # 處理載入的個體
    try:
        # 傳入參數重試
    except:
        return -100000.0,  # 懲罰
except (OverflowError, ValueError, ...):
    # 任何其他錯誤
    return -100000.0,  # 懲罰
```

---

### **5. 為什麼需要合理性檢查？**

**問題**: 計算錯誤可能產生異常值

```python
# 異常情況
1. NaN: 0/0, log(-1)
2. Inf: 1/0, exp(1000)
3. 過大: 100,000,000+ (不可能賺 1000 倍)
4. 過小: -200,000- (不可能虧超過 2 倍本金)
```

**檢查**:
```python
MAX_REASONABLE_FITNESS = initial_capital * 1000  # 100,000,000
MIN_REASONABLE_FITNESS = -initial_capital * 2    # -200,000

if not np.isfinite(excess_return) or \
   excess_return > MAX_REASONABLE_FITNESS or \
   excess_return < MIN_REASONABLE_FITNESS:
    return -100000.0,  # 懲罰
```

---

### **6. 為什麼用全倉交易？**

**簡化假設**:
```python
# 買入: 用全部資金買股票
shares = capital / price
capital = 0

# 賣出: 賣掉全部股票
capital = shares * price
shares = 0
```

**原因**:
1. **簡單**: 容易實現和理解
2. **一致性**: 所有策略用相同的資金管理
3. **可比性**: 不同策略的結果可以直接比較

**現實中**: 可能需要倉位管理、風險控制等

---

## ❓ 常見問題 FAQ

### **Q1: 為什麼 evaluate() 用完整數據，但只計算回測期的報酬？**

**A**: 因為技術指標需要完整歷史數據。

```python
# 完整數據用於計算技術指標
price_vec = self.data['Close'].to_numpy()  # 1997-06-25 到 1999-06-25
signals = rule()  # 使用完整數據計算 RSI, SMA 等

# 回測期用於計算報酬
backtest_signals = signals[mask]  # 只取 1998-06-22 到 1999-06-25
gp_return = _numba_simulation_loop(backtest_signals, ...)
```

---

### **Q2: 為什麼要深拷貝 pset？**

**A**: 避免修改全局 pset，導致多線程問題。

```python
# 不拷貝的問題
self.pset = pset  # 引用全局 pset
self.pset.terminals[NumVector][0].value = price_vec_A  # 修改全局 pset

# 另一個 BacktestingEngine
other_engine.pset.terminals[NumVector][0].value = price_vec_B  # 覆蓋！

# 結果: 兩個 engine 都用 price_vec_B

# 拷貝後
self.pset = copy.deepcopy(pset)  # 獨立副本
self.pset.terminals[NumVector][0].value = price_vec_A  # 只修改自己的
```

---

### **Q3: 為什麼交易在次日開盤執行？**

**A**: 避免前視偏差（Look-Ahead Bias）。

```python
# 錯誤: 當天信號當天執行
for i in range(len(signals)):
    signal = signals[i]
    price = prices[i]  # 當天價格
    if signal:
        buy(price)  # 用當天價格買入

# 問題: 信號是用當天收盤價計算的，但買入用當天開盤價
# 這在現實中不可能（你不知道未來的收盤價）

# 正確: 當天信號次日執行
for i in range(len(signals)):
    signal = signals[i]
    next_price = prices[i + 1]  # 次日價格
    if signal:
        buy(next_price)  # 用次日開盤價買入
```

---

### **Q4: Portfolio 模式的 fitness 為什麼是加總？**

**A**: 因為是等權重配置，總報酬就是各 ticker 報酬的和。

```python
# 3 個 ticker，每個分配 100,000
ABX.TO:   excess_return = 15,000
BBD-B.TO: excess_return = 8,000
RY.TO:    excess_return = 12,000

# Portfolio 總 excess return
total = 15,000 + 8,000 + 12,000 = 35,000

# 這相當於
# 總資金 300,000，總報酬 35,000
# 報酬率 = 35,000 / 300,000 = 11.67%
```

---

### **Q5: 為什麼 Numba 函數不能用 Pandas？**

**A**: Numba 的 `nopython=True` 模式只支持 NumPy 和基本類型。

```python
# ❌ 不能用
@numba.jit(nopython=True)
def bad_function(df):
    return df['Close'].mean()  # Pandas 不支持

# ✅ 可以用
@numba.jit(nopython=True)
def good_function(arr):
    return arr.mean()  # NumPy 支持

# 解決: 轉換為 NumPy
df_close = df['Close'].to_numpy()
result = good_function(df_close)
```

---

### **Q6: 如何調試 evaluate() 中的錯誤？**

**A**: 取消註釋 debug 語句。

```python
# L119: 取消註釋
if not np.all(np.isfinite(signals)):
    print(f"\n[DIAGNOSTIC] Sanitizing non-finite signals for: {individual}")
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

# L137: 取消註釋
except Exception as e2:
    print(f"[ERROR] Could not evaluate individual with arguments {individual}: {e2}")
    return -100000.0,

# L187: 取消註釋
if not np.isfinite(excess_return) or ...:
    print(f"[WARNING] Unreasonable fitness detected: {excess_return:.2e}, assigning penalty")
    return -100000.0,
```

---

### **Q7: 為什麼有時候 fitness 是 -100000？**

**A**: 這是懲罰 fitness，表示個體無效。

**可能原因**:
```python
1. 編譯錯誤 (L138, L141, L146)
2. 異常 fitness (L188)
3. 異常 return (L231)
```

**如何找出原因**: 取消註釋 debug 語句，查看錯誤信息。

---

### **Q8: 如何提高回測速度？**

**A**: 已經使用 Numba JIT，速度已經很快了。進一步優化：

```python
1. 減少族群大小
   population_size = 500 → 300

2. 減少演化代數
   n_generations = 50 → 30

3. 使用多進程
   from multiprocessing import Pool
   # 但要注意 Numba JIT 的線程安全

4. 簡化技術指標
   # 避免過於複雜的計算
```

---

## 📝 Review Checklist

完成 review 後，確保你能回答：

### **Numba JIT**
- [ ] 為什麼用 Numba JIT？
- [ ] nopython=True 的限制？
- [ ] 交易邏輯的細節？
- [ ] 為什麼次日開盤執行？

### **BacktestingEngine**
- [ ] 初始期和回測期的區別？
- [ ] evaluate() 的完整流程？
- [ ] 為什麼用完整數據計算信號？
- [ ] 異常處理的策略？
- [ ] Excess Return 的計算？
- [ ] 合理性檢查的標準？

### **PortfolioBacktestingEngine**
- [ ] 如何管理多個 ticker？
- [ ] fitness 如何計算？
- [ ] backtest_config 如何傳遞？

### **整體理解**
- [ ] 完整回測流程？
- [ ] 為什麼用 Excess Return？
- [ ] 如何調試錯誤？
- [ ] 如何優化性能？

---

## 🎓 總結

`backtesting/engine.py` 是評估交易策略的核心：

1. **高性能**: Numba JIT 加速 60 倍
2. **初始期支持**: 技術指標有足夠歷史數據
3. **穩健性**: 多層異常處理和合理性檢查
4. **靈活性**: 支持單 ticker 和 portfolio 模式
5. **詳細記錄**: 可生成完整交易日誌

理解這個文件，你就理解了如何評估一個 GP 交易策略！

---

**文檔版本**: 1.0  
**最後更新**: 2025-10-06  
**作者**: Cascade AI Assistant
