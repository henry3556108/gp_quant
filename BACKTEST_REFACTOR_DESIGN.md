# 回測引擎重構設計文件

## 📋 需求背景

### 當前問題
目前的回測系統在計算報酬率時，直接使用 `train_start` 到 `train_end` 的數據。這導致：

1. **技術指標計算不準確**：在 `train_start` 時間點，技術指標（如 RSI、移動平均）沒有足夠的歷史數據
2. **不符合 PRD 規範**：PRD 明確定義了「初始期」和「訓練/測試期」的區別
3. **結果可能失真**：前期的技術指標值不可靠，影響策略評估

### PRD 規範

根據 PRD Section 7，實驗設計包含兩個階段，每個階段有兩個時期：

#### 7.1 短訓練週期

**訓練階段 (Train)**：
- **訓練初始期**：1997-06-25 至 1998-06-22 (250 天) → 只用於計算技術指標
- **訓練期**：1998-06-22 至 1999-06-25 (256 天) → 計算技術指標 + 計算報酬

**測試階段 (Test)**：
- **測試初始期**：1998-07-07 至 1999-06-25 (250 天) → 只用於計算技術指標
- **測試期**：1999-06-28 至 2000-06-30 (256 天) → 計算技術指標 + 計算報酬

#### 7.2 長訓練週期

**訓練階段 (Train)**：
- **訓練初始期**：1992-06-30 至 1993-07-02 (250 天) → 只用於計算技術指標
- **訓練期**：1993-07-02 至 1999-06-25 (1498 天) → 計算技術指標 + 計算報酬

**測試階段 (Test)**：
- **測試初始期**：1998-07-07 至 1999-06-25 (250 天) → 只用於計算技術指標
- **測試期**：1999-06-28 至 2000-06-30 (256 天) → 計算技術指標 + 計算報酬

---

## 🎯 設計目標

### 核心概念

引入三個時間點來明確區分「數據範圍」和「回測範圍」：

1. **`data_start_date`**（數據起始日）
   - 用於提供技術指標計算所需的歷史數據
   - 對應 PRD 的「初始期開始」

2. **`backtest_start`**（回測起始日）
   - 開始計算報酬率的日期
   - 對應 PRD 的「訓練期/測試期開始」

3. **`backtest_end`**（回測結束日）
   - 結束計算報酬率的日期
   - 對應 PRD 的「訓練期/測試期結束」

### 數據使用邏輯

```
時間軸：
|-------- 初始期 --------|-------- 回測期 --------|
data_start          backtest_start        backtest_end

用途：
|-- 只計算技術指標 --|-- 技術指標 + 報酬計算 --|
```

---

## 🔧 需要修改的模塊

### 1. `gp_quant/data/loader.py`

#### 1.1 修改 `split_train_test_data()` 函數

**當前簽名**：
```python
def split_train_test_data(
    data: Dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
```

**新簽名**：
```python
def split_train_test_data(
    data: Dict[str, pd.DataFrame],
    train_data_start: str,      # 訓練初始期開始
    train_backtest_start: str,  # 訓練期開始
    train_backtest_end: str,    # 訓練期結束
    test_data_start: str,       # 測試初始期開始
    test_backtest_start: str,   # 測試期開始
    test_backtest_end: str      # 測試期結束
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
```

**修改內容**：
- 訓練數據：從 `train_data_start` 到 `train_backtest_end` 的完整數據
- 測試數據：從 `test_data_start` 到 `test_backtest_end` 的完整數據
- 同時記錄 `backtest_start` 信息，供回測引擎使用

**返回值結構**：
```python
train_data = {
    'ticker': {
        'data': DataFrame,  # 完整數據（含初始期）
        'backtest_start': '1998-06-22',  # 回測起始日
        'backtest_end': '1999-06-25'     # 回測結束日
    }
}
```

---

### 2. `gp_quant/backtesting/engine.py`

#### 2.1 修改 `BacktestingEngine.__init__()`

**當前簽名**：
```python
def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0)
```

**新簽名**：
```python
def __init__(
    self, 
    data: pd.DataFrame, 
    initial_capital: float = 100000.0,
    backtest_start: str = None,  # 新增：回測起始日
    backtest_end: str = None     # 新增：回測結束日（可選）
)
```

**修改內容**：
- 接收完整數據（含初始期）
- 記錄 `backtest_start` 和 `backtest_end`
- 如果未提供 `backtest_start`，則使用全部數據（向後兼容）

#### 2.2 修改 `evaluate()` 方法

**修改邏輯**：
1. 使用**完整數據**計算技術指標（從 `data_start_date` 開始）
2. 只計算 `backtest_start` 到 `backtest_end` 期間的報酬率

**實現方式**：
```python
def evaluate(self, individual: gp.PrimitiveTree) -> tuple[float]:
    # Step 1: 使用完整數據計算技術指標
    price_vec = self.data['Close'].to_numpy()
    volume_vec = self.data['Volume'].to_numpy()
    
    # Step 2: 生成完整的信號序列
    signals = rule()  # 基於完整數據
    
    # Step 3: 只在 backtest_start 到 backtest_end 期間計算報酬
    if self.backtest_start:
        backtest_mask = (self.data.index >= self.backtest_start)
        if self.backtest_end:
            backtest_mask &= (self.data.index <= self.backtest_end)
        
        # 只使用回測期的數據計算報酬
        backtest_data = self.data[backtest_mask]
        backtest_signals = signals[backtest_mask.values]
        
        gp_return = self._run_vectorized_simulation(
            backtest_signals, 
            backtest_data
        )
    else:
        # 向後兼容：使用全部數據
        gp_return = self._run_vectorized_simulation(signals, self.data)
    
    # Step 4: 計算 Buy-and-Hold（也只在回測期）
    # ...
```

#### 2.3 修改 `_run_vectorized_simulation()` 方法

**當前簽名**：
```python
def _run_vectorized_simulation(self, signals: np.ndarray) -> float
```

**新簽名**：
```python
def _run_vectorized_simulation(
    self, 
    signals: np.ndarray,
    data: pd.DataFrame = None  # 新增：可指定數據範圍
) -> float
```

**修改內容**：
- 如果提供 `data` 參數，使用該數據範圍
- 否則使用 `self.data`（向後兼容）

#### 2.4 修改 `run_detailed_simulation()` 方法

**修改邏輯**：
- 同樣只在 `backtest_start` 到 `backtest_end` 期間記錄交易
- 確保交易記錄的日期都在回測期內

---

### 3. `gp_quant/backtesting/portfolio_engine.py`

#### 3.1 修改 `PortfolioBacktestingEngine.__init__()`

**當前簽名**：
```python
def __init__(
    self, 
    data: Dict[str, pd.DataFrame], 
    initial_capital: float = 100000.0
)
```

**新簽名**：
```python
def __init__(
    self, 
    data: Dict[str, pd.DataFrame], 
    initial_capital: float = 100000.0,
    backtest_config: Dict[str, Dict] = None  # 新增：回測配置
)
```

**backtest_config 結構**：
```python
backtest_config = {
    'ticker1': {
        'backtest_start': '1998-06-22',
        'backtest_end': '1999-06-25'
    },
    'ticker2': {
        'backtest_start': '1998-06-22',
        'backtest_end': '1999-06-25'
    }
}
```

#### 3.2 修改 `evaluate()` 方法

**修改邏輯**：
- 為每個 ticker 創建 `BacktestingEngine` 時，傳入對應的 `backtest_start` 和 `backtest_end`
- 確保所有 ticker 在相同的回測期內計算報酬

---

### 4. `main.py`

#### 4.1 修改 `run_portfolio_evolution()` 函數

**當前代碼**：
```python
train_start = '1998-06-22'
train_end = '1999-06-25'
test_start = '1999-06-28'
test_end = '2000-06-30'

train_data, test_data = split_train_test_data(
    all_stock_data, train_start, train_end, test_start, test_end
)
```

**新代碼**：
```python
# 短訓練週期配置
train_data_start = '1997-06-25'      # 訓練初始期開始
train_backtest_start = '1998-06-22'  # 訓練期開始
train_backtest_end = '1999-06-25'    # 訓練期結束

test_data_start = '1998-07-07'       # 測試初始期開始
test_backtest_start = '1999-06-28'   # 測試期開始
test_backtest_end = '2000-06-30'     # 測試期結束

train_data, test_data = split_train_test_data(
    all_stock_data,
    train_data_start, train_backtest_start, train_backtest_end,
    test_data_start, test_backtest_start, test_backtest_end
)
```

#### 4.2 修改回測引擎調用

**當前代碼**：
```python
train_backtester = PortfolioBacktestingEngine(train_data)
test_backtester = PortfolioBacktestingEngine(test_data)
```

**新代碼**：
```python
# 提取回測配置
train_backtest_config = {
    ticker: {
        'backtest_start': train_data[ticker]['backtest_start'],
        'backtest_end': train_data[ticker]['backtest_end']
    }
    for ticker in train_data.keys()
}

test_backtest_config = {
    ticker: {
        'backtest_start': test_data[ticker]['backtest_start'],
        'backtest_end': test_data[ticker]['backtest_end']
    }
    for ticker in test_data.keys()
}

# 創建回測引擎時傳入配置
train_backtester = PortfolioBacktestingEngine(
    {ticker: train_data[ticker]['data'] for ticker in train_data.keys()},
    backtest_config=train_backtest_config
)

test_backtester = PortfolioBacktestingEngine(
    {ticker: test_data[ticker]['data'] for ticker in test_data.keys()},
    backtest_config=test_backtest_config
)
```

---

### 5. `run_all_experiments.py`

#### 5.1 修改 `modify_main_py()` 函數

**當前簽名**：
```python
def modify_main_py(train_start, train_end, test_start, test_end)
```

**新簽名**：
```python
def modify_main_py(
    train_data_start, train_backtest_start, train_backtest_end,
    test_data_start, test_backtest_start, test_backtest_end
)
```

#### 5.2 更新實驗配置

**短訓練週期**：
```python
{
    'name': '短訓練期',
    'train_data_start': '1997-06-25',
    'train_backtest_start': '1998-06-22',
    'train_backtest_end': '1999-06-25',
    'test_data_start': '1998-07-07',
    'test_backtest_start': '1999-06-28',
    'test_backtest_end': '2000-06-30'
}
```

**長訓練週期**：
```python
{
    'name': '長訓練期',
    'train_data_start': '1992-06-30',
    'train_backtest_start': '1993-07-02',
    'train_backtest_end': '1999-06-25',
    'test_data_start': '1998-07-07',
    'test_backtest_start': '1999-06-28',
    'test_backtest_end': '2000-06-30'
}
```

---

## 📊 預期影響

### 正面影響

1. **技術指標更準確**
   - 在回測起始日，所有技術指標都有 250 天的歷史數據支撐
   - RSI、移動平均等指標值更可靠

2. **符合 PRD 規範**
   - 完全遵循 PRD Section 7 的實驗設計
   - 初始期和回測期明確分離

3. **結果更可信**
   - 避免前期技術指標不準確導致的策略失真
   - 報酬率計算更符合實際情況

### 可能的變化

1. **報酬率數值變化**
   - 由於只計算回測期的報酬，數值可能與之前不同
   - 但這是**更準確**的結果

2. **訓練期天數變化**
   - 短訓練期：從 256 天（實際計算報酬）
   - 長訓練期：從 1498 天（實際計算報酬）
   - 但技術指標使用更多歷史數據

3. **需要重新運行實驗**
   - 之前的 80 次實驗結果需要重新計算
   - 以獲得基於正確初始期的結果

---

## ✅ 向後兼容性

### 兼容策略

1. **可選參數**
   - `backtest_start` 和 `backtest_end` 設為可選參數
   - 如果未提供，使用全部數據（舊行為）

2. **單 Ticker 模式**
   - `run_evolution_for_tickers()` 函數保持不變
   - 只修改 portfolio 模式

3. **測試覆蓋**
   - 確保舊的測試案例仍然通過
   - 添加新的測試案例驗證新功能

---

## 🧪 測試計劃

### 單元測試

1. **測試 `split_train_test_data()`**
   - 驗證數據範圍正確
   - 驗證 backtest_start/end 正確記錄

2. **測試 `BacktestingEngine`**
   - 驗證只在回測期計算報酬
   - 驗證技術指標使用完整數據

3. **測試 `PortfolioBacktestingEngine`**
   - 驗證多 ticker 的回測配置
   - 驗證報酬計算正確

### 集成測試

1. **短訓練週期測試**
   - 運行一次完整實驗
   - 驗證報酬率計算正確
   - 檢查交易記錄日期範圍

2. **長訓練週期測試**
   - 運行一次完整實驗
   - 驗證初始期數據正確使用
   - 對比新舊結果差異

### 驗證方法

1. **手動驗證**
   - 檢查第一筆交易日期 >= backtest_start
   - 檢查最後一筆交易日期 <= backtest_end
   - 驗證技術指標在 backtest_start 時已有合理值

2. **數據驗證**
   - 訓練期天數：256 天（短）/ 1498 天（長）
   - 測試期天數：256 天
   - 初始期天數：250 天

---

## 📝 實施步驟

### Phase 1: 核心修改（優先）

1. ✅ 修改 `loader.py` 的 `split_train_test_data()`
2. ✅ 修改 `BacktestingEngine.__init__()` 和 `evaluate()`
3. ✅ 修改 `PortfolioBacktestingEngine`
4. ✅ 單元測試

### Phase 2: 主程序修改

5. ✅ 修改 `main.py` 的 `run_portfolio_evolution()`
6. ✅ 更新日期配置
7. ✅ 集成測試

### Phase 3: 實驗腳本修改

8. ✅ 修改 `run_all_experiments.py`
9. ✅ 更新實驗配置
10. ✅ 運行測試實驗

### Phase 4: 驗證與文檔

11. ✅ 重新運行完整實驗（80次）
12. ✅ 對比新舊結果
13. ✅ 更新文檔和 README

---

## 🚨 注意事項

### 關鍵點

1. **數據完整性**
   - 確保初始期數據存在且完整
   - 處理數據缺失的情況

2. **邊界條件**
   - backtest_start 必須在數據範圍內
   - 確保至少有 250 天初始期數據

3. **性能考慮**
   - 技術指標計算使用完整數據，可能稍慢
   - 但報酬計算只在回測期，應該更快

4. **結果解讀**
   - 新結果與舊結果不可直接比較
   - 需要重新建立基準

### 風險

1. **代碼複雜度增加**
   - 需要傳遞更多參數
   - 需要更仔細的測試

2. **可能的 Bug**
   - 日期範圍錯誤
   - 索引對齊問題
   - 邊界條件處理

3. **實驗時間**
   - 需要重新運行所有實驗
   - 預計需要 1-2 小時

---

## 📚 參考

- PRD Section 7: 實驗設計
- 當前實驗結果: `experiments_results/`
- 相關代碼: `gp_quant/backtesting/engine.py`, `gp_quant/data/loader.py`

---

**文件版本**: 1.0  
**創建日期**: 2025-10-01  
**狀態**: 待審核 → 待實施
