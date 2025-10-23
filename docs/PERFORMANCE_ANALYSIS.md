# 性能問題分析：為什麼一個 Generation 從 5 分鐘變成 15 分鐘？

## 🔍 問題描述

**觀察到的現象**：
- **之前**：一個 generation 約 5 分鐘
- **現在**：一個 generation 約 15 分鐘
- **差異**：慢了 **3 倍**！

**特別慢的部分**：Similarity Matrix 計算

---

## 🎯 根本原因

### 原因 1：`sharpe_ratio` 比 `excess_return` 慢很多 ⚠️

#### 計算複雜度對比

| Fitness Metric | 計算步驟 | 複雜度 |
|---------------|---------|--------|
| **excess_return** | 1. 運行向量化模擬<br>2. 計算 B&H return<br>3. 相減 | **O(n)** |
| **sharpe_ratio** | 1. 運行向量化模擬<br>2. **生成完整 equity curve**<br>3. 計算每日 returns<br>4. 計算 mean/std<br>5. 年化 Sharpe | **O(n) + 額外開銷** |

#### Portfolio 版本更慢！

對於 **PortfolioBacktestingEngine**，使用 `sharpe_ratio` 時：

```python
def _calculate_portfolio_sharpe(self, individual):
    equity_curves = []
    
    for ticker in self.tickers:  # 4 個股票
        engine = self.engines[ticker]
        
        # 每個股票都要：
        # 1. get_signals(individual) - 編譯並執行 GP tree
        # 2. _run_simulation_with_equity_curve() - 生成完整 equity curve
        
        equity_curve = engine._run_simulation_with_equity_curve(
            engine.get_signals(individual),
            engine.backtest_data
        )
        equity_curves.append(equity_curve)
    
    # 然後合併並計算 Sharpe
    combined_equity = pd.concat(equity_curves, axis=1).sum(axis=1)
    returns = combined_equity.pct_change().dropna()
    sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
```

**關鍵問題**：
- 每個個體評估時，**每個股票都要調用 `get_signals()`**
- `get_signals()` 會**重新編譯並執行 GP tree**
- 對於 4 個股票的組合，這意味著**每個個體要編譯執行 4 次**！

#### 時間估算

假設：
- Population size: 5000
- Tickers: 4
- 每次 `get_signals()` + `_run_simulation_with_equity_curve()`: 0.01 秒

**使用 sharpe_ratio**：
```
每個 generation = 5000 individuals × 4 tickers × 0.01s = 200 秒 ≈ 3.3 分鐘
```

**使用 excess_return**（舊版本）：
```
每個 generation = 5000 individuals × 1 次評估 × 0.005s = 25 秒
```

但實際上還有其他開銷（Niching、聚類等），所以：
- **sharpe_ratio**: 3.3 分鐘 + 開銷 → **約 5-7 分鐘**
- **excess_return**: 25 秒 + 開銷 → **約 1-2 分鐘**

---

### 原因 2：Similarity Matrix 計算變慢

你從 `n_workers=8` 改成 `n_workers=6`，這會讓相似度矩陣計算變慢約 **25-33%**。

#### 計算量

對於 population_size = 5000：
```
相似度矩陣大小 = 5000 × 5000 = 25,000,000 個比較
```

即使使用並行計算，這仍然是一個巨大的計算量。

#### 時間估算

根據你的日誌（archive/portfolio_exp_sharpe_20251023_125111）：
- Generation 1 eval_time: 121 秒（約 2 分鐘）

但你說總共要 15 分鐘，這意味著：
```
15 分鐘 - 2 分鐘（評估）= 13 分鐘（其他開銷）
```

這 13 分鐘很可能花在：
1. **Similarity Matrix 計算**（最大開銷）
2. Niching 聚類
3. 跨群選擇
4. 族群儲存

---

## 🔧 解決方案

### 方案 1：優化 `sharpe_ratio` 計算（推薦）⭐

**問題**：每個股票都重新編譯執行 GP tree

**解決**：快取 signals

```python
def _calculate_portfolio_sharpe(self, individual):
    # 只編譯一次 GP tree
    price_vec = self.engines[self.tickers[0]].data['Close'].to_numpy()
    volume_vec = self.engines[self.tickers[0]].data['Volume'].to_numpy()
    
    # 編譯規則（只做一次）
    rule = gp.compile(expr=individual, pset=self.pset)
    
    equity_curves = []
    for ticker in self.tickers:
        engine = self.engines[ticker]
        
        # 使用已編譯的規則直接計算 signals
        engine.pset.terminals[NumVector][0].value = engine.data['Close'].to_numpy()
        engine.pset.terminals[NumVector][1].value = engine.data['Volume'].to_numpy()
        signals = rule()
        
        # 切片到回測期
        if engine.backtest_start or engine.backtest_end:
            mask = pd.Series(True, index=engine.data.index)
            if engine.backtest_start:
                mask &= (engine.data.index >= engine.backtest_start)
            if engine.backtest_end:
                mask &= (engine.data.index <= engine.backtest_end)
            backtest_signals = signals[mask.values]
        else:
            backtest_signals = signals
        
        equity_curve = engine._run_simulation_with_equity_curve(
            backtest_signals,
            engine.backtest_data
        )
        equity_curves.append(equity_curve)
    
    # 合併並計算 Sharpe
    combined_equity = pd.concat(equity_curves, axis=1).sum(axis=1)
    returns = combined_equity.pct_change().dropna()
    sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
    return sharpe
```

**預期改善**：
- 從 4 次編譯 → 1 次編譯
- 速度提升：**約 3-4 倍**
- Generation 時間：15 分鐘 → **約 4-5 分鐘**

---

### 方案 2：減少 Similarity Matrix 計算頻率

**當前設置**：
```python
'niching_update_frequency': 1,  # 每 1 代重新計算
```

**建議**：
```python
'niching_update_frequency': 3,  # 每 3 代重新計算
```

**理由**：
- 族群結構不會每代都劇烈變化
- 相似度矩陣計算非常昂貴（5000×5000）
- 每 3 代更新一次足夠

**預期改善**：
- 減少 66% 的相似度矩陣計算
- Generation 時間：15 分鐘 → **約 7-8 分鐘**（平均）

---

### 方案 3：使用採樣相似度矩陣

對於大族群（>1000），不需要計算完整的 5000×5000 矩陣。

**建議**：
```python
if len(pop) > 1000:
    # 採樣 1000 個代表性個體
    sample_indices = np.random.choice(len(pop), 1000, replace=False)
    sample_pop = [pop[i] for i in sample_indices]
    
    # 只計算 1000×1000 矩陣
    sim_matrix = ParallelSimilarityMatrix(sample_pop, n_workers=6)
    similarity_matrix = sim_matrix.compute(show_progress=False)
    
    # 使用 KNN 將其他個體分配到最近的 cluster
    # ...
```

**預期改善**：
- 計算量：25,000,000 → 1,000,000（減少 96%）
- Similarity Matrix 時間：13 分鐘 → **約 30 秒**
- Generation 時間：15 分鐘 → **約 2.5 分鐘**

---

### 方案 4：改回 `excess_return`（最簡單）

如果你不需要 Sharpe Ratio 作為 fitness：

```python
'fitness_metric': 'excess_return',
```

**預期改善**：
- 立即恢復到之前的速度
- Generation 時間：15 分鐘 → **約 5 分鐘**

---

## 📊 方案比較

| 方案 | 難度 | 預期改善 | 最終時間 | 推薦度 |
|------|------|---------|---------|--------|
| **1. 優化 sharpe_ratio** | 中 | 3-4x | 4-5 分鐘 | ⭐⭐⭐⭐⭐ |
| **2. 減少更新頻率** | 低 | 2x（平均） | 7-8 分鐘 | ⭐⭐⭐ |
| **3. 採樣相似度矩陣** | 高 | 6x | 2.5 分鐘 | ⭐⭐⭐⭐ |
| **4. 改回 excess_return** | 極低 | 3x | 5 分鐘 | ⭐⭐ |
| **組合 1+2** | 中 | 6-8x | 2-3 分鐘 | ⭐⭐⭐⭐⭐ |
| **組合 1+3** | 高 | 12-15x | 1-2 分鐘 | ⭐⭐⭐⭐⭐ |

---

## 🎯 推薦行動方案

### 短期（立即可做）

1. **減少 Niching 更新頻率**
   ```python
   'niching_update_frequency': 3,  # 從 1 改成 3
   ```

2. **恢復 n_workers**
   ```python
   sim_matrix = ParallelSimilarityMatrix(pop, n_workers=8)  # 從 6 改回 8
   ```

**預期改善**：15 分鐘 → **約 6-7 分鐘**

### 中期（1-2 小時實作）

3. **優化 sharpe_ratio 計算**
   - 實作方案 1 的快取邏輯
   - 避免重複編譯 GP tree

**預期改善**：15 分鐘 → **約 3-4 分鐘**

### 長期（未來優化）

4. **實作採樣相似度矩陣**
   - 對大族群使用採樣
   - 使用 KNN 分配

**預期改善**：15 分鐘 → **約 1-2 分鐘**

---

## 📝 總結

你的性能問題主要來自兩個原因：

1. **`sharpe_ratio` 對 Portfolio 特別慢**（約 3-4 倍）
   - 每個股票都重新編譯執行 GP tree
   - 需要生成完整 equity curve

2. **Similarity Matrix 計算非常昂貴**
   - 5000×5000 = 25,000,000 次比較
   - 每代都計算（frequency=1）
   - 只用 6 個 workers

**最佳解決方案**：
- 短期：調整 `niching_update_frequency` 和 `n_workers`
- 中期：優化 `sharpe_ratio` 計算（快取編譯結果）
- 長期：實作採樣相似度矩陣

這樣可以將 generation 時間從 **15 分鐘降到 2-3 分鐘**！🚀
