# 專案分析與重構方案

## 1. 總結 (Executive Summary)

本專案的核心問題在於**代碼重複**與**邏輯不一致**。主要體現在兩個實驗入口：

1.  `run_all_experiments.py` (透過 `main.py` 調用 `gp_quant/evolution/engine.py`)
2.  `run_portfolio_experiment.py`

其中，`run_portfolio_experiment.py` **重新實作了完整的演化循環**，而沒有重用 `engine.py` 中的核心邏輯。這不僅造成了約 150-200 行的重複代碼，更重要的是，它遺漏了 `engine.py` 中已經實現的關鍵修正，例如**樹深度限制 (`staticLimit`)**。

這個遺漏導致 `run_portfolio_experiment.py` 的實驗結果存在嚴重瑕疵（如深度超限），可能使其**無效**。

本方案旨在**統一演化邏輯**，消除重複代碼，修復現有 bug，並提高專案的長期可維護性。

---

## 2. 深入分析：問題在哪裡？

### 2.1. 代碼重複：兩個演化引擎

| 功能 | `gp_quant/evolution/engine.py` | `run_portfolio_experiment.py` | 狀態 |
| :--- | :--- | :--- | :--- |
| **DEAP Toolbox 設置** | ✅ | ✅ | 🔴 **重複** |
| **演化主循環** | ✅ | ✅ | 🔴 **重複** |
| **選擇、交配、變異** | ✅ | ✅ | 🔴 **重複** |
| **Fitness 評估** | ✅ | ✅ | 🔴 **重複** |
| **日誌與統計** | ✅ | ✅ | 🔴 **重複** |
| **樹深度限制 (`staticLimit`)** | ✅ **已實現** | ❌ **缺失** | 💣 **嚴重問題** |
| **Niching 策略** | ❌ **不支持** | ✅ **已實現** | 🟡 **功能分歧** |
| **Early Stopping** | ❌ **不支持** | ✅ **已實現** | 🟡 **功能分歧** |

`run_portfolio_experiment.py` 基本上是 `engine.py` 的一個早期副本，後續在其中添加了 Niching 等新功能，但卻沒有同步 `engine.py` 的 bug 修復和改進。

### 2.2. 根本原因：為何會這樣？

這種情況在快速迭代的專案中很常見：

1.  **歷史演進**：`engine.py` 是最初的標準 GP 引擎。為了快速實現 Portfolio 和 Niching 實驗，開發者可能直接複製了 `engine.py` 的代碼到 `run_portfolio_experiment.py` 並在其基礎上修改。
2.  **功能分歧**：`run_portfolio_experiment.py` 需要 Niching、Early Stopping 等 `engine.py` 當時沒有的功能。最快的實現方式是在腳本層級直接加入這些邏輯，而不是重構核心引擎。
3.  **缺乏整合**：在 `engine.py` 中修復了深度超限的 bug（通過 `staticLimit`），但忘記將此修復同步到 `run_portfolio_experiment.py`。

### 2.3. 重複的危害：深度超限 Bug

這個問題的嚴重性在 `docs/DEPTH_VIOLATION_ANALYSIS.md` 中有詳細記錄。

-   **`engine.py`**：正確使用了 `gp.staticLimit`，確保所有 GP 樹的深度不超過 17。
-   **`run_portfolio_experiment.py`**：**沒有**使用 `gp.staticLimit`，導致在交配和變異過程中，樹的深度失控，某些實驗的違規率高達 **76%**，最大深度達到 **69**。

**這意味著 `run_portfolio_experiment.py` 產生的實驗結果不符合論文要求，是無效的。**

---

## 3. 重構方案

我們的目標是讓 `run_portfolio_experiment.py` 重用 `engine.py` 的核心演化邏輯，同時保留其特殊功能（如 Niching）。

### 方案 A：快速修復 (推薦立即執行)

**思路**：讓 `run_portfolio_experiment.py` 調用 `engine.py` 的 `run_evolution` 函數，並將 Niching、Early Stopping 等特殊邏輯通過 `generation_callback` 傳入。

**優點**：
-   ✅ **快速修復**：預計 1-2 小時內可完成。
-   ✅ **立即解決 Bug**：深度超限問題會立刻被修復。
-   ✅ **減少代碼**：刪除 `run_portfolio_experiment.py` 中約 150 行的重複演化循環。

**缺點**：
-   ⚠️ `generation_callback` 的邏輯可能會變得比較複雜。
-   ⚠️ 不是最優雅的架構，但作為過渡方案極佳。

**實作細節**：請參考第 5 節。

### 方案 B：擴展核心引擎 (推薦的中期方案)

**思路**：將 Niching、Early Stopping 等功能作為可選參數整合進 `gp_quant/evolution/engine.py` 的 `run_evolution` 函數中。

```python
# in gp_quant/evolution/engine.py
def run_evolution(
    ...,
    niching_config: dict = None,
    early_stopping_config: dict = None
):
    # ...
    for gen in range(n_generations):
        # ...
        if niching_config and niching_config['enabled']:
            # 執行 Niching 選擇
            offspring = perform_niching_selection(...)
        else:
            # 標準選擇
            offspring = toolbox.select(...)
        
        # ...
        if early_stopping_config and early_stopping_config['enabled']:
            if should_stop(...):
                break
```

**優點**：
-   ✅ **單一真相來源**：所有演化邏輯集中在 `engine.py`。
-   ✅ **接口清晰**：實驗腳本只需傳遞配置，無需關心實現細節。
-   ✅ **易於維護**：未來任何修改只需在一個地方進行。

**缺點**：
-   ⚠️ 需要對 `engine.py` 進行較大的重構。
-   ⚠️ 函數參數會變多。

### 方案 C：組件化架構 (推薦的長期方案)

**思路**：將演化過程中的各個策略（選擇、評估、變異、Niching）抽象成可插拔的組件（類）。

```python
# 概念代碼
from gp_quant.evolution import EvolutionEngine, NichingStrategy, EarlyStoppingHandler

engine = EvolutionEngine(config)
engine.add_strategy(NichingStrategy(niching_config))
engine.add_handler(EarlyStoppingHandler(es_config))
results = engine.run()
```

**優點**：
-   ✅ **高度靈活**：可以自由組合不同的策略。
-   ✅ **職責分離**：每個類只做一件事，代碼清晰。
-   ✅ **易於擴展**：添加新策略只需實現一個新類。

**缺點**：
-   ⚠️ **重構工作量最大**，需要全面的架構設計。
-   ⚠️ 對於當前專案規模可能過度設計。

---

## 4. 推薦路徑圖

1.  **立即 (Today)**：**實施方案 A (快速修復)**。
    -   **目標**：立即修復深度超限的 bug，確保後續實驗結果的有效性。
    -   **動作**：重構 `run_portfolio_experiment.py`，使其調用 `engine.py`。

2.  **中期 (Next 1-2 Weeks)**：**演進到方案 B (擴展核心引擎)**。
    -   **目標**：將 `generation_callback` 中的邏輯遷移到 `engine.py` 內部，實現更清晰的接口。
    -   **動作**：為 `run_evolution` 添加 `niching_config` 等參數。

3.  **長期 (Future)**：**評估是否需要方案 C (組件化)**。
    -   **目標**：如果專案需要支持更多、更複雜的演化策略，則考慮重構為組件化架構。

---

## 5. 方案 A 實作細節

以下是如何修改 `run_portfolio_experiment.py` 以調用 `engine.py` 的範例。

### 步驟 1：修改 `run_portfolio_experiment.py`

刪除原有的演化主循環（`for gen in range(...)`），替換為對 `run_evolution` 的調用。

```python
# run_portfolio_experiment.py

# ... (保留 CONFIG, 數據載入, engine 初始化等) ...

from gp_quant.evolution.engine import run_evolution
from gp_quant.niching import NichingClusterer, CrossNicheSelector, create_k_selector
from gp_quant.similarity import ParallelSimilarityMatrix

def main():
    # ... (保留現有的初始化代碼) ...

    # 1. 初始化 Niching 和 Early Stopping
    early_stopping = None
    if CONFIG['early_stopping_enabled']:
        early_stopping = EarlyStopping(...)

    niching_selector = None
    k_selector = None
    if CONFIG['niching_enabled']:
        niching_selector = CrossNicheSelector(...)
        k_selector = create_k_selector(CONFIG)

    # 2. 定義 generation_callback
    def generation_callback(gen, pop, hof, logbook, record):
        """
        在每個世代結束後執行的回調函數。
        處理 Niching、Early Stopping 和日誌記錄。
        """
        print(f"--- Generation {gen} Callback ---")

        # 早停檢查
        if early_stopping and early_stopping.step(hof[0].fitness.values[0]):
            print("🛑 早停觸發！")
            return True  # 返回 True 以停止演化

        # Niching 邏輯
        custom_selector = None
        if niching_selector and gen % CONFIG['niching_update_frequency'] == 0:
            print("🔬 執行 Niching...")
            # 計算相似度矩陣
            sim_matrix_calculator = ParallelSimilarityMatrix(pop, n_workers=6)
            similarity_matrix = sim_matrix_calculator.compute(show_progress=False)
            
            # 動態選擇 k
            k_result = k_selector.select_k(similarity_matrix, len(pop), gen)
            selected_k = k_result['k']
            
            # 聚類
            clusterer = NichingClusterer(n_clusters=selected_k, algorithm=CONFIG['niching_algorithm'])
            niche_labels = clusterer.fit_predict(similarity_matrix)
            
            # 創建一個使用當前 niche_labels 的選擇器函數
            def niching_selection_func(population, k):
                return niching_selector.select(population, niche_labels, k)
            
            custom_selector = niching_selection_func
            print(f"   ✓ Niching 選擇器已準備就緒 (k={selected_k})")

        # 返回自定義選擇器或 False
        # engine.py 會檢查返回值，如果是 callable，則用它作為下一代的選擇器
        return custom_selector or False

    # 3. 調用核心演化引擎
    print("🚀 開始調用核心演化引擎...")
    
    # 準備訓練數據
    train_data_for_engine = {
        ticker: {
            'data': data[ticker],
            'backtest_start': CONFIG['train_backtest_start'],
            'backtest_end': CONFIG['train_backtest_end']
        }
        for ticker in CONFIG['tickers']
    }

    pop, log, hof = run_evolution(
        data=train_data_for_engine,
        population_size=CONFIG['population_size'],
        n_generations=CONFIG['generations'],
        crossover_prob=CONFIG['crossover_prob'],
        mutation_prob=CONFIG['mutation_prob'],
        individual_records_dir=str(generations_dir),
        generation_callback=generation_callback,
        fitness_metric=CONFIG['fitness_metric'],
        tournament_size=CONFIG['tournament_size']
    )

    # ... (保留後續的分析和儲存邏輯) ...

if __name__ == '__main__':
    main()
```

### 步驟 2：修改 `gp_quant/evolution/engine.py`

確保 `run_evolution` 能夠接收並使用 `generation_callback` 返回的自定義選擇器。

```python
# in gp_quant/evolution/engine.py

def run_evolution(...):
    # ... (現有代碼) ...

    for gen in (pbar := trange(1, n_generations + 1, desc="Generation")):
        
        # 這裡的 toolbox.select 是默認的 ranked_selection
        # 如果 callback 返回了自定義選擇器，我們會替換它
        
        # ... (選擇、交配、變異) ...
        
        # 在循環的末尾調用 callback
        if generation_callback:
            callback_result = generation_callback(gen, pop, hof, logbook, record)
            
            if callback_result is True:
                print(f"Evolution stopped early at generation {gen}")
                break  # 早停
            
            elif callable(callback_result):
                # 如果返回的是一個可調用對象 (我們的 niching_selection_func)
                # 將其註冊為下一代使用的選擇器
                print(f"下一代將使用 Niching 選擇器。")
                toolbox.register("select", callback_result)
            else:
                # 如果返回 False 或 None，恢復默認選擇器
                toolbox.register("select", ranked_selection)

    return pop, logbook, hof
```

---

## 6. 預期效果

實施**方案 A**後，您將獲得：

| 指標 | 修改前 | 修改後 | 改善 |
| :--- | :--- | :--- | :--- |
| **代碼重複** | ~150-200 行 | 0 行 | ✅ **-100%** |
| **深度超限 Bug** | 存在 (76% 違規率) | **已修復** | ✅ **-100%** |
| **維護成本** | 需同步修改 2 個文件 | 只需維護 `engine.py` | ✅ **-50%** |
| **專案一致性** | 低 | 高 | ✅ **提升** |

最重要的是，**所有實驗都將在一個統一、正確的框架下運行**，確保了結果的有效性和可比性。

建議您從方案 A 開始，這將立即為您的專案帶來最大的價值。
