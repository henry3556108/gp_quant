# GP 組件完整說明文檔

**文件**: `gp_quant/gp/operators.py` + `gp_quant/gp/primitives.py`  
**總行數**: 84 + 175 = 259 行  
**核心功能**: 定義 GP 交易規則的語法和原語  
**創建日期**: 2025-10-06

---

## 📋 目錄

1. [文件概覽](#文件概覽)
2. [operators.py - 類型系統與原語集](#operatorspy---類型系統與原語集)
3. [primitives.py - 自定義原語實現](#primitivespy---自定義原語實現)
4. [完整原語目錄](#完整原語目錄)
5. [類型系統工作原理](#類型系統工作原理)
6. [常見問題 FAQ](#常見問題-faq)

---

## 📦 文件概覽

### **兩個文件的關係**

```
operators.py (84 行)
  ├─ 定義類型系統 (NumVector, BoolVector)
  ├─ 創建 pset (PrimitiveSetTyped)
  ├─ 註冊所有原語
  └─ 導入 primitives.py 的函數
      ↓
primitives.py (175 行)
  ├─ 實現技術指標 (RSI, SMA, ROC, etc.)
  ├─ 實現安全運算 (protected_div, mul, etc.)
  └─ 處理邊界條件和異常
```

### **核心職責**

#### **operators.py**:
1. ✅ 定義類型系統（強類型約束）
2. ✅ 配置 DEAP 的 PrimitiveSetTyped
3. ✅ 註冊所有可用的原語和終端
4. ✅ 確保生成的 GP 樹合法

#### **primitives.py**:
1. ✅ 實現技術指標（RSI, SMA, Volatility, etc.）
2. ✅ 實現安全運算（防止除零、溢出）
3. ✅ 向量化操作（高效處理時間序列）
4. ✅ 異常處理（NaN, Inf, 邊界條件）

---

## 🎯 operators.py - 類型系統與原語集

**位置**: `gp_quant/gp/operators.py` (84 行)

### **文件結構**

```
operators.py
├── 導入模塊 (L1-14)
├── 類型系統定義 (L16-20)
├── 原語集初始化 (L22-25)
├── 註冊原語 (L27-54)
│   ├── 布林運算 (L29-32)
│   ├── 關係運算 (L34-36)
│   ├── 算術運算 (L38-42)
│   ├── 技術指標 (L44-51)
│   └── 輔助函數 (L53-54)
└── 註冊終端 (L60-81)
    ├── 布林常數 (L64-65)
    ├── Ephemeral 常數 (L72-75)
    └── 固定常數 (L78-81)
```

---

### **1. 類型系統定義** (L16-20)

```python
# --- Type System Definition ---
class NumVector(np.ndarray): pass
class BoolVector(np.ndarray): pass
```

#### **為什麼需要類型系統？**

**問題：沒有類型系統**
```python
# 可能生成的非法規則
add(gt(ARG0, 100), RSI(ARG0, 14))
     ↑ BoolVector   ↑ NumVector
# 布林值 + 數值 → 無意義！

# 可能生成的非法規則
if root_is_NumVector:
    # 返回數值，但需要布林信號
    return SMA(ARG0, 20)  # ❌ 無法用於交易決策
```

**解決：強類型系統**
```python
# 類型約束確保合法
gt(ARG0, SMA(ARG0, 20))
   ↑ NumVector  ↑ NumVector → BoolVector ✅

# 類型約束防止非法
add(gt(ARG0, 100), RSI(ARG0, 14))
    ↑ BoolVector   ↑ NumVector
# DEAP 不會生成這種樹！
```

#### **兩種類型**

| 類型 | 繼承自 | 用途 | 範例 |
|------|--------|------|------|
| `NumVector` | np.ndarray | 數值向量（價格、指標） | `[100, 105, 103, ...]` |
| `BoolVector` | np.ndarray | 布林向量（交易信號） | `[True, False, True, ...]` |

---

### **2. 原語集初始化** (L22-25)

```python
# --- Primitive Set Initialization ---
pset = gp.PrimitiveSetTyped("MAIN", [NumVector, NumVector], BoolVector)
```

#### **參數解釋**

```python
gp.PrimitiveSetTyped(
    "MAIN",                      # 名稱
    [NumVector, NumVector],      # 輸入類型：ARG0, ARG1
    BoolVector                   # 輸出類型：必須返回布林信號
)
```

#### **含義**

```
輸入:
  ARG0: NumVector  ← 價格序列 [100, 105, 103, ...]
  ARG1: NumVector  ← 成交量序列 [1000000, 1200000, ...]

輸出:
  BoolVector  ← 交易信號 [True, False, True, ...]
```

#### **範例 GP 樹**

```python
# 合法的 GP 樹
gt(SMA(ARG0, 20), ARG0)
   ↑ NumVector    ↑ NumVector → BoolVector ✅

# 編譯後
def rule(ARG0, ARG1):
    sma_20 = SMA(ARG0, 20)  # NumVector
    return sma_20 > ARG0     # BoolVector

# 執行
signals = rule(price_vec, volume_vec)
# signals = [False, False, True, True, ...]
```

---

### **3. 註冊原語** (L27-54)

#### **3.1 布林運算** (L29-32)

```python
# Boolean operators: These operate on and return boolean vectors.
pset.addPrimitive(np.logical_and, [BoolVector, BoolVector], BoolVector, name="logical_and")
pset.addPrimitive(np.logical_or, [BoolVector, BoolVector], BoolVector, name="logical_or")
pset.addPrimitive(prim.logical_not, [BoolVector], BoolVector, name="logical_not")
```

**類型簽名**:
```
logical_and: (BoolVector, BoolVector) → BoolVector
logical_or:  (BoolVector, BoolVector) → BoolVector
logical_not: (BoolVector) → BoolVector
```

**範例**:
```python
# GP 樹
logical_and(gt(ARG0, 100), lt(RSI(ARG0, 14), 30))

# 編譯後
def rule(ARG0, ARG1):
    cond1 = ARG0 > 100           # BoolVector
    cond2 = RSI(ARG0, 14) < 30   # BoolVector
    return cond1 & cond2          # BoolVector

# 執行
signals = rule(price_vec, volume_vec)
# signals = [False, False, True, False, ...]
```

---

#### **3.2 關係運算** (L34-36)

```python
# Relational operators: These are the bridge.
pset.addPrimitive(operator.lt, [NumVector, NumVector], BoolVector, name="lt")
pset.addPrimitive(operator.gt, [NumVector, NumVector], BoolVector, name="gt")
```

**類型簽名**:
```
lt: (NumVector, NumVector) → BoolVector  (小於)
gt: (NumVector, NumVector) → BoolVector  (大於)
```

**為什麼是橋樑？**
```
NumVector (數值世界)
    ↓
  gt/lt (關係運算)
    ↓
BoolVector (布林世界)
```

**範例**:
```python
# GP 樹
gt(SMA(ARG0, 20), ARG0)

# 編譯後
def rule(ARG0, ARG1):
    sma_20 = SMA(ARG0, 20)  # NumVector: [102, 103, 104, ...]
    price = ARG0             # NumVector: [100, 105, 103, ...]
    return sma_20 > price    # BoolVector: [True, False, True, ...]
```

---

#### **3.3 算術運算** (L38-42)

```python
# Arithmetic operators: These operate on and return numerical vectors.
pset.addPrimitive(prim.add, [NumVector, NumVector], NumVector, name="add")
pset.addPrimitive(prim.sub, [NumVector, NumVector], NumVector, name="sub")
pset.addPrimitive(prim.mul, [NumVector, NumVector], NumVector, name="mul")
pset.addPrimitive(prim.protected_div, [NumVector, NumVector], NumVector, name="div")
```

**類型簽名**:
```
add: (NumVector, NumVector) → NumVector
sub: (NumVector, NumVector) → NumVector
mul: (NumVector, NumVector) → NumVector
div: (NumVector, NumVector) → NumVector
```

**範例**:
```python
# GP 樹
gt(add(ARG0, ARG1), mul(ARG0, 2))

# 編譯後
def rule(ARG0, ARG1):
    sum_vec = ARG0 + ARG1    # NumVector
    double_vec = ARG0 * 2    # NumVector
    return sum_vec > double_vec  # BoolVector

# 執行
price = [100, 105, 103]
volume = [1000, 1200, 1100]
sum_vec = [1100, 1305, 1203]
double_vec = [200, 210, 206]
signals = [True, True, True]
```

---

#### **3.4 技術指標** (L44-51)

```python
# Financial primitives
pset.addPrimitive(prim.moving_average, [NumVector, int], NumVector, name="avg")
pset.addPrimitive(prim.moving_max, [NumVector, int], NumVector, name="max")
pset.addPrimitive(prim.moving_min, [NumVector, int], NumVector, name="min")
pset.addPrimitive(prim.lag, [NumVector, int], NumVector, name="lag")
pset.addPrimitive(prim.volatility, [NumVector, int], NumVector, name="vol")
pset.addPrimitive(prim.rate_of_change, [NumVector, int], NumVector, name="ROC")
pset.addPrimitive(prim.relative_strength_index, [NumVector, int], NumVector, name="RSI")
```

**類型簽名**:
```
avg: (NumVector, int) → NumVector  (移動平均)
max: (NumVector, int) → NumVector  (移動最大值)
min: (NumVector, int) → NumVector  (移動最小值)
lag: (NumVector, int) → NumVector  (滯後)
vol: (NumVector, int) → NumVector  (波動率)
ROC: (NumVector, int) → NumVector  (變化率)
RSI: (NumVector, int) → NumVector  (相對強弱指標)
```

**範例**:
```python
# GP 樹
lt(RSI(ARG0, 14), 30)

# 編譯後
def rule(ARG0, ARG1):
    rsi_14 = RSI(ARG0, 14)  # NumVector: [45, 32, 28, 65, ...]
    threshold = 30           # int
    return rsi_14 < threshold  # BoolVector: [False, False, True, False, ...]

# 交易邏輯: RSI < 30 時買入（超賣）
```

---

#### **3.5 輔助函數** (L53-54)

```python
# Add a harmless identity primitive for integers
pset.addPrimitive(prim.identity_int, [int], int, name="id_int")
```

**類型簽名**:
```
id_int: (int) → int
```

**為什麼需要？**
```python
# 問題: DEAP 生成器需要 int → int 的函數
# 如果沒有，生成器可能卡住

# 解決: 提供一個無害的恆等函數
def identity_int(x: int) -> int:
    return x

# 用途: 滿足生成器需求，實際上不影響策略
```

---

### **4. 註冊終端** (L60-81)

#### **4.1 布林常數** (L64-65)

```python
# Add boolean constant terminals
pset.addTerminal(True, BoolVector, name="V_TRUE")
pset.addTerminal(False, BoolVector, name="V_FALSE")
```

**為什麼需要？**
```python
# 問題: 生成深度 0 的 BoolVector 樹
# 如果沒有布林終端，生成器會報錯

# 解決: 提供布林常數
V_TRUE   # 永遠買入
V_FALSE  # 永遠不買入

# 範例 GP 樹
logical_or(gt(ARG0, 100), V_TRUE)
# 只要價格 > 100 或永遠為真 → 永遠買入
```

---

#### **4.2 Ephemeral 常數** (L72-75)

```python
# Ephemeral constants for generating random values at runtime
pset.addEphemeralConstant("rand_float", lambda: random.uniform(-1, 1), float)
pset.addEphemeralConstant("rand_int_n", lambda: random.randint(5, 200), int)
```

**什麼是 Ephemeral 常數？**
```python
# 每次生成個體時，隨機產生一個常數
# 不同個體有不同的常數值

# 個體 1
gt(ARG0, 105.3)  # rand_float = 105.3

# 個體 2
gt(ARG0, 98.7)   # rand_float = 98.7

# 個體 3
RSI(ARG0, 47)    # rand_int_n = 47

# 個體 4
RSI(ARG0, 123)   # rand_int_n = 123
```

**範圍**:
- `rand_float`: -1.0 到 1.0
- `rand_int_n`: 5 到 200（技術指標的回看期）

---

#### **4.3 固定常數** (L78-81)

```python
# Add some fixed common lookback periods as terminals
pset.addTerminal(10, int)
pset.addTerminal(20, int)
pset.addTerminal(50, int)
pset.addTerminal(100, int)
```

**為什麼需要固定常數？**
```python
# 常用的技術指標參數
SMA(ARG0, 10)   # 10 日均線
SMA(ARG0, 20)   # 20 日均線
SMA(ARG0, 50)   # 50 日均線
SMA(ARG0, 100)  # 100 日均線

RSI(ARG0, 14)   # 14 日 RSI（標準）

# 固定常數增加這些常用值出現的機率
```

---

## 🔧 primitives.py - 自定義原語實現

**位置**: `gp_quant/gp/primitives.py` (175 行)

### **文件結構**

```
primitives.py
├── 導入和配置 (L1-18)
├── 輔助函數 (L21-34)
│   ├── identity_int (L21-23)
│   └── protected_div (L25-34)
├── 技術指標 (L36-131)
│   ├── moving_average (L38-46)
│   ├── moving_max (L48-56)
│   ├── moving_min (L58-66)
│   ├── lag (L68-74)
│   ├── volatility (L76-96)
│   ├── rate_of_change (L98-108)
│   └── relative_strength_index (L110-131)
└── 安全運算 (L133-173)
    ├── add (L135-137)
    ├── sub (L139-141)
    ├── logical_not (L143-145)
    └── mul (L147-173)
```

---

### **1. 輔助函數**

#### **identity_int()** (L21-23)

```python
def identity_int(x: int) -> int:
    """Returns the integer unchanged. Used to satisfy DEAP's generator."""
    return x
```

**用途**: 滿足 DEAP 生成器對 int → int 函數的需求

---

#### **protected_div()** (L25-34)

```python
def protected_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized protected division that returns 1.0 in case of division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(left, right)
    # Where the divisor is close to zero, the result is 1.0
    result[np.abs(right) < 1e-6] = 1.0
    result[np.isinf(result)] = 1.0
    result = np.nan_to_num(result, nan=1.0)
    return result
```

**為什麼需要保護？**
```python
# 問題: 除以零
price = [100, 105, 0, 103]
volume = [1000, 1200, 0, 1100]
result = price / volume
# result = [0.1, 0.0875, inf, 0.0936]  ← inf 會破壞計算

# 解決: protected_div
result = protected_div(price, volume)
# result = [0.1, 0.0875, 1.0, 0.0936]  ← 安全
```

**保護策略**:
1. 除數接近 0 (< 1e-6) → 返回 1.0
2. 結果是 Inf → 返回 1.0
3. 結果是 NaN → 返回 1.0

---

### **2. 技術指標**

#### **moving_average()** (L38-46)

```python
def moving_average(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving average."""
    if n == 0:
        return series
    try:
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).mean().to_numpy()
    except Exception:
        return np.full_like(series, np.nan)
```

**功能**: 計算 n 日移動平均

**範例**:
```python
price = [100, 105, 103, 108, 110]
sma_3 = moving_average(price, 3)

# 計算過程
# Day 0: (100) / 1 = 100.0          (min_periods=1)
# Day 1: (100 + 105) / 2 = 102.5
# Day 2: (100 + 105 + 103) / 3 = 102.67
# Day 3: (105 + 103 + 108) / 3 = 105.33
# Day 4: (103 + 108 + 110) / 3 = 107.0

sma_3 = [100.0, 102.5, 102.67, 105.33, 107.0]
```

**關鍵參數**:
- `window=n`: 窗口大小
- `min_periods=1`: 最少需要 1 個數據點（避免前期 NaN）

---

#### **moving_max()** (L48-56) & **moving_min()** (L58-66)

```python
def moving_max(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving maximum."""
    if n == 0:
        return series
    try:
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).max().to_numpy()
    except Exception:
        return np.full_like(series, np.nan)
```

**功能**: 計算 n 日移動最大值/最小值

**範例**:
```python
price = [100, 105, 103, 108, 110]
max_3 = moving_max(price, 3)
min_3 = moving_min(price, 3)

# max_3 = [100, 105, 105, 108, 110]
# min_3 = [100, 100, 100, 103, 103]
```

---

#### **lag()** (L68-74)

```python
def lag(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized lag."""
    if n <= 0:
        return series
    result = np.full_like(series, np.nan)
    result[n:] = series[:-n]
    return result
```

**功能**: 將序列向後移動 n 個位置

**範例**:
```python
price = [100, 105, 103, 108, 110]
lag_2 = lag(price, 2)

# lag_2 = [NaN, NaN, 100, 105, 103]
#          ↑    ↑    ↑    ↑    ↑
#          前2個是NaN，後面是原始數據向後移2位
```

**用途**:
```python
# 計算價格變化
price_change = price - lag(price, 1)
# [NaN, 5, -2, 5, 2]

# 計算 n 日前的價格
price_5_days_ago = lag(price, 5)
```

---

#### **volatility()** (L76-96)

```python
def volatility(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized volatility."""
    if n < 2:
        return np.zeros_like(series)
    
    try:
        # Calculate returns
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(series) / series[:-1]
        returns = np.concatenate([[0.0], returns])
        
        # Calculate rolling std
        returns_series = pd.Series(returns, dtype=np.float64)
        result = returns_series.rolling(window=n, min_periods=1).std().to_numpy()
        
        # Handle inf/nan
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
        return result
    except Exception:
        return np.zeros_like(series)
```

**功能**: 計算 n 日波動率（收益率的標準差）

**範例**:
```python
price = [100, 105, 103, 108, 110]

# 步驟 1: 計算收益率
returns = [0.0, 0.05, -0.019, 0.049, 0.019]
#         [0%, 5%, -1.9%, 4.9%, 1.9%]

# 步驟 2: 計算滾動標準差
vol_3 = volatility(price, 3)
# vol_3 = [0.0, 0.035, 0.036, 0.035, 0.024]
```

**用途**: 衡量價格波動程度，高波動 = 高風險

---

#### **rate_of_change()** (L98-108)

```python
def rate_of_change(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized Rate of Change (ROC)."""
    if n < 1:
        return np.zeros_like(series)
    
    lagged_series = lag(series, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        roc = np.divide(series - lagged_series, lagged_series) * 100
    return np.nan_to_num(roc, nan=0.0, posinf=1e6, neginf=-1e6)
```

**功能**: 計算 n 日變化率（百分比）

**公式**:
```
ROC = (Price_today - Price_n_days_ago) / Price_n_days_ago * 100
```

**範例**:
```python
price = [100, 105, 103, 108, 110]
roc_2 = rate_of_change(price, 2)

# 計算過程
# Day 0: (100 - NaN) / NaN * 100 = 0.0
# Day 1: (105 - NaN) / NaN * 100 = 0.0
# Day 2: (103 - 100) / 100 * 100 = 3.0%
# Day 3: (108 - 105) / 105 * 100 = 2.86%
# Day 4: (110 - 103) / 103 * 100 = 6.80%

roc_2 = [0.0, 0.0, 3.0, 2.86, 6.80]
```

---

#### **relative_strength_index()** (L110-131)

```python
def relative_strength_index(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized Relative Strength Index (RSI)."""
    if n < 1:
        return np.full_like(series, 50.0)
    
    try:
        s = pd.Series(series, dtype=np.float64)
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n, min_periods=1).mean()

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)

        rsi = 100 - (100 / (1 + rs))
        result = rsi.to_numpy()
        return np.clip(result, 0, 100)
    except Exception:
        return np.full_like(series, 50.0)
```

**功能**: 計算 n 日相對強弱指標（RSI）

**公式**:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```

**範例**:
```python
price = [100, 105, 103, 108, 110, 107, 112]

# 步驟 1: 計算價格變化
delta = [NaN, 5, -2, 5, 2, -3, 5]

# 步驟 2: 分離漲跌
gain = [0, 5, 0, 5, 2, 0, 5]  # 只保留正值
loss = [0, 0, 2, 0, 0, 3, 0]  # 只保留負值的絕對值

# 步驟 3: 計算平均漲跌（假設 n=3）
avg_gain = rolling_mean(gain, 3)
avg_loss = rolling_mean(loss, 3)

# 步驟 4: 計算 RS 和 RSI
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# rsi = [50, 100, 83.3, 100, 100, 75, 100]
#        ↑ 中性  ↑ 超買      ↑ 超買
```

**解讀**:
- RSI > 70: 超買（可能下跌）
- RSI < 30: 超賣（可能上漲）
- RSI = 50: 中性

---

### **3. 安全運算**

#### **add(), sub(), logical_not()** (L135-145)

```python
def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized addition."""
    return np.add(a, b)

def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized subtraction."""
    return np.subtract(a, b)

def logical_not(a: np.ndarray) -> np.ndarray:
    """Vectorized logical NOT."""
    return np.logical_not(a)
```

**功能**: 基本向量運算

---

#### **mul()** (L147-173) ⭐ **特殊處理**

```python
def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized multiplication with overflow protection."""
    finfo = np.finfo(np.float64)
    with np.errstate(over='ignore'):
        abs_b_safe = np.abs(b) + 1e-9
        problematic_indices = np.where(np.abs(a) > finfo.max / abs_b_safe)
    
    if len(problematic_indices[0]) > 0:
        idx = problematic_indices[0][0]
        error_msg = (
            f"Overflow detected in mul primitive!\n"
            f"Index: {idx}\n"
            f"Value a[{idx}]: {a[idx]:.2e}\n"
            f"Value b[{idx}]: {b[idx]:.2e}\n"
            f"Result would exceed: {finfo.max:.2e}"
        )
        print("\n--- ASSERTION TRIGGERED ---")
        print(error_msg)
        print("--- END ASSERTION ---\n")
        raise AssertionError(error_msg)
        
    return np.multiply(a, b)
```

**為什麼需要溢出檢查？**
```python
# 問題: 乘法溢出
a = [1e308, 100, 105]
b = [10, 1.5, 2]
result = a * b
# result = [inf, 150, 210]  ← inf 會破壞計算

# 解決: 提前檢查
if abs(a) * abs(b) > max_float:
    raise AssertionError("Overflow!")
```

**檢查邏輯**:
```python
# 檢查條件: |a| * |b| > max_float
# 等價於: |a| > max_float / |b|

finfo.max = 1.7976931348623157e+308  # float64 最大值

if abs(a[i]) > finfo.max / abs(b[i]):
    # 會溢出！拋出錯誤
    raise AssertionError(...)
```

---

## 📚 完整原語目錄

### **函數原語（Functions）**

| 類別 | 名稱 | 類型簽名 | 功能 |
|------|------|----------|------|
| **布林運算** | `logical_and` | (Bool, Bool) → Bool | 邏輯與 |
| | `logical_or` | (Bool, Bool) → Bool | 邏輯或 |
| | `logical_not` | (Bool) → Bool | 邏輯非 |
| **關係運算** | `lt` | (Num, Num) → Bool | 小於 |
| | `gt` | (Num, Num) → Bool | 大於 |
| **算術運算** | `add` | (Num, Num) → Num | 加法 |
| | `sub` | (Num, Num) → Num | 減法 |
| | `mul` | (Num, Num) → Num | 乘法（含溢出檢查） |
| | `div` | (Num, Num) → Num | 除法（保護） |
| **技術指標** | `avg` | (Num, int) → Num | 移動平均 |
| | `max` | (Num, int) → Num | 移動最大值 |
| | `min` | (Num, int) → Num | 移動最小值 |
| | `lag` | (Num, int) → Num | 滯後 |
| | `vol` | (Num, int) → Num | 波動率 |
| | `ROC` | (Num, int) → Num | 變化率 |
| | `RSI` | (Num, int) → Num | 相對強弱指標 |
| **輔助** | `id_int` | (int) → int | 恆等函數 |

### **終端原語（Terminals）**

| 類別 | 名稱 | 類型 | 值/範圍 |
|------|------|------|---------|
| **輸入** | `ARG0` | NumVector | 價格序列 |
| | `ARG1` | NumVector | 成交量序列 |
| **布林常數** | `V_TRUE` | BoolVector | True |
| | `V_FALSE` | BoolVector | False |
| **Ephemeral** | `rand_float` | float | [-1.0, 1.0] |
| | `rand_int_n` | int | [5, 200] |
| **固定常數** | `10` | int | 10 |
| | `20` | int | 20 |
| | `50` | int | 50 |
| | `100` | int | 100 |

---

## 🎯 類型系統工作原理

### **類型約束如何工作？**

```python
# DEAP 的類型檢查
pset = gp.PrimitiveSetTyped("MAIN", [NumVector, NumVector], BoolVector)

# 生成 GP 樹時
def generate_tree(return_type, depth):
    if depth == 0:
        # 選擇一個返回 return_type 的終端
        return random.choice(terminals_of_type[return_type])
    else:
        # 選擇一個返回 return_type 的函數
        func = random.choice(functions_of_type[return_type])
        
        # 遞歸生成子樹（類型匹配）
        children = []
        for arg_type in func.arg_types:
            child = generate_tree(arg_type, depth - 1)
            children.append(child)
        
        return func(*children)
```

### **範例：生成合法的 GP 樹**

```
目標: 生成返回 BoolVector 的樹，深度 3

步驟 1: 選擇返回 BoolVector 的函數
  可選: logical_and, logical_or, logical_not, lt, gt
  選擇: gt (需要 2 個 NumVector 參數)

步驟 2: 生成第 1 個 NumVector 子樹（深度 2）
  可選: add, sub, mul, div, avg, max, min, lag, vol, ROC, RSI
  選擇: avg (需要 1 個 NumVector 和 1 個 int)
  
  步驟 2.1: 生成 NumVector 子樹（深度 1）
    可選: add, sub, mul, div, avg, ..., ARG0, ARG1
    選擇: ARG0
  
  步驟 2.2: 生成 int 子樹（深度 1）
    可選: 10, 20, 50, 100, rand_int_n, id_int
    選擇: 20

步驟 3: 生成第 2 個 NumVector 子樹（深度 2）
  可選: add, sub, mul, div, avg, ..., ARG0, ARG1
  選擇: ARG0

結果: gt(avg(ARG0, 20), ARG0)
```

### **類型約束防止的錯誤**

```python
# ❌ 不會生成（類型不匹配）
add(gt(ARG0, 100), RSI(ARG0, 14))
    ↑ BoolVector   ↑ NumVector
# add 需要兩個 NumVector，但 gt 返回 BoolVector

# ❌ 不會生成（返回類型錯誤）
SMA(ARG0, 20)
↑ 返回 NumVector，但需要 BoolVector

# ✅ 會生成（類型正確）
gt(SMA(ARG0, 20), ARG0)
   ↑ NumVector    ↑ NumVector → BoolVector ✅
```

---

## ❓ 常見問題 FAQ

### **Q1: 為什麼需要兩種類型（NumVector 和 BoolVector）？**

**A**: 確保生成的 GP 樹可以用於交易決策。

```python
# 沒有類型系統
可能生成: SMA(ARG0, 20)
返回: [102, 103, 104, ...]  # 數值
問題: 無法用於交易決策（需要 True/False）

# 有類型系統
必須生成: gt(SMA(ARG0, 20), ARG0)
返回: [True, False, True, ...]  # 布林
✅ 可以用於交易決策
```

---

### **Q2: 為什麼需要 protected_div？**

**A**: 防止除以零導致 Inf 或 NaN。

```python
# 普通除法
price = [100, 105, 0, 103]
volume = [1000, 1200, 0, 1100]
result = price / volume
# result = [0.1, 0.0875, nan, 0.0936]  ← nan 會破壞後續計算

# 保護除法
result = protected_div(price, volume)
# result = [0.1, 0.0875, 1.0, 0.0936]  ← 安全
```

---

### **Q3: 為什麼 mul 需要溢出檢查？**

**A**: 防止乘法溢出導致 Inf。

```python
# 問題
a = [1e308, 100]
b = [10, 2]
result = a * b
# result = [inf, 200]  ← inf 會破壞計算

# 解決
mul(a, b)  # 檢測到溢出，拋出 AssertionError
# 這個個體會被淘汰，不會污染族群
```

---

### **Q4: 為什麼技術指標用 min_periods=1？**

**A**: 避免前期數據不足時產生 NaN。

```python
# min_periods=n（預設）
price = [100, 105, 103, 108, 110]
sma_3 = moving_average(price, 3)
# sma_3 = [NaN, NaN, 102.67, 105.33, 107.0]
#          ↑    ↑ 前兩個是 NaN（數據不足）

# min_periods=1
sma_3 = moving_average(price, 3)
# sma_3 = [100.0, 102.5, 102.67, 105.33, 107.0]
#          ↑ 用 1 個數據計算  ↑ 用 2 個  ↑ 用 3 個
```

---

### **Q5: Ephemeral 常數和固定常數的區別？**

**A**: Ephemeral 每次生成時隨機，固定常數永遠不變。

```python
# Ephemeral 常數
individual1 = RSI(ARG0, 47)   # rand_int_n = 47
individual2 = RSI(ARG0, 123)  # rand_int_n = 123
individual3 = RSI(ARG0, 89)   # rand_int_n = 89
# 每個個體不同

# 固定常數
individual1 = RSI(ARG0, 20)   # 固定 20
individual2 = RSI(ARG0, 20)   # 固定 20
individual3 = RSI(ARG0, 20)   # 固定 20
# 所有個體相同
```

**用途**:
- Ephemeral: 增加多樣性，探索不同參數
- 固定: 增加常用參數（10, 20, 50, 100）的出現機率

---

### **Q6: 為什麼需要 V_TRUE 和 V_FALSE？**

**A**: 滿足 DEAP 生成器對深度 0 的 BoolVector 樹的需求。

```python
# 問題: 生成深度 0 的 BoolVector 樹
# 如果沒有布林終端，生成器會報錯

# 解決: 提供布林常數
V_TRUE   # 永遠買入
V_FALSE  # 永遠不買入

# 實際用途
logical_or(gt(ARG0, 100), V_TRUE)
# 只要價格 > 100 或永遠為真 → 永遠買入
```

---

### **Q7: 如何添加新的技術指標？**

**步驟**:

```python
# 1. 在 primitives.py 實現函數
def bollinger_bands(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates Bollinger Bands."""
    sma = moving_average(series, n)
    std = pd.Series(series).rolling(window=n, min_periods=1).std().to_numpy()
    upper_band = sma + 2 * std
    return upper_band

# 2. 在 operators.py 註冊
pset.addPrimitive(prim.bollinger_bands, [NumVector, int], NumVector, name="BB")

# 3. 現在可以在 GP 樹中使用
gt(ARG0, BB(ARG0, 20))  # 價格突破布林帶上軌
```

---

### **Q8: 為什麼所有技術指標都有 try/except？**

**A**: 防止任何異常導致演化中斷。

```python
def moving_average(series: np.ndarray, n: int) -> np.ndarray:
    try:
        # 正常計算
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).mean().to_numpy()
    except Exception:
        # 任何錯誤都返回 NaN
        return np.full_like(series, np.nan)

# 可能的錯誤
1. 數組長度不匹配
2. 類型轉換失敗
3. 內存不足
4. Pandas 版本不兼容

# 策略: 寧可返回 NaN，也不要讓演化中斷
```

---

## 📝 Review Checklist

完成 review 後，確保你能回答：

### **operators.py**
- [ ] 為什麼需要類型系統？
- [ ] NumVector 和 BoolVector 的區別？
- [ ] pset 的輸入和輸出類型？
- [ ] 關係運算為什麼是橋樑？
- [ ] Ephemeral 常數如何工作？
- [ ] 為什麼需要布林常數終端？

### **primitives.py**
- [ ] protected_div 如何防止除零？
- [ ] mul 為什麼需要溢出檢查？
- [ ] 技術指標的 min_periods 作用？
- [ ] RSI 的計算邏輯？
- [ ] 為什麼所有函數都有 try/except？
- [ ] 如何添加新的技術指標？

### **整體理解**
- [ ] 類型約束如何防止非法樹？
- [ ] 完整的原語目錄？
- [ ] 如何生成合法的 GP 樹？
- [ ] 如何調試類型錯誤？

---

## 🎓 總結

`operators.py` 和 `primitives.py` 定義了 GP 交易規則的語法：

1. **強類型系統**: 確保生成的規則合法且可執行
2. **豐富的原語**: 7 個技術指標 + 4 個算術運算 + 3 個布林運算
3. **安全保護**: protected_div, 溢出檢查, 異常處理
4. **向量化**: 所有操作都是向量化的，高效處理時間序列
5. **靈活性**: 易於添加新的技術指標和運算

理解這兩個文件，你就理解了 GP 如何生成和評估交易規則！

---

**文檔版本**: 1.0  
**最後更新**: 2025-10-06  
**作者**: Cascade AI Assistant
