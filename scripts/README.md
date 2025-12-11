
# GP Quant Scripts Guide

本目錄包含 `gp_quant` 專案演化算法的各種輔助腳本，主要用於**實驗比較**、**多樣性分析**與**結果視覺化**。

我們的核心研究目標是：
1. **維持 GP 演化過程的多樣性**：避免種群過早收斂，保持基因與行為的多樣。
2. **超越 Baseline**：確保演化出的策略在樣本外測試中優於各個 Baseline。
3. **驗證多樣性優勢**：通過數據證明多樣性維持機制（如 Niching, soft_min）確實比 Baseline 更好地保留了多樣性。

---

## 📂 目錄結構

- **Root (`scripts/`)**: 針對特定研究主題的一次性或核心分析腳本。
- **`analysis/`**: 核心分析工具，用於批量處理實驗記錄，計算多樣性指標和 PnL。
- **`utils/`**: 實用工具，如繼續演化、尋找最佳個體、更新數據等。
- **`visualization/`**: 專門的繪圖腳本，用於生成高品質的論文級圖表。
- **`verify/`**: 開發階段的驗證腳本，用於測試新功能是否正常運作。

---

## 🚀 核心腳本 (Core Scripts)

這些是日常研究中最常用的腳本。

| 腳本路徑 | 用途 | 使用時機 | 備註 |
|:---|:---|:---|:---|
| `scripts/compare_signal_niche.py` | **比較 Niching 策略**<br>比較不同 Niche 方法 (TED vs PnL vs Signal) 的 PnL 表現。 | 當你想看不同 Niching 方法對最終獲利的影響時。 | 支援 Boxplot 和 PnL Curve 繪製。 |
| `scripts/compare_softmin_vs_mean.py` | **比較 Aggregation 方法**<br>比較 `soft_min` 與 `mean` 聚合對 PnL 分布的影響。 | 驗證 `soft_min` 是否能有效減少極端失敗案例時。 | 核心結論腳本之一。 |
| `scripts/analysis/compare_experiments.py` | **多樣性演化比較**<br>比較多個實驗的「多樣性分數」隨世代的變化。 | **(核心目標)** 證明你的方法比 Baseline 更好地維持了多樣性。 | 生成多樣性趨勢圖。 |
| `scripts/analysis/analyze_effective_breadth.py` | **有效廣度分析**<br>計算策略池的「有效廣度」(Effective Breadth)，衡量策略間的低相關性。 | 分析演化出的策略池是否具有真正的分散風險能力。 | 這是衡量 Portfolio 多樣性的關鍵指標。 |
| `scripts/visualization/plot_overlay_pnl.py` | **PnL 疊圖比較**<br>將多個實驗的 PnL 曲線畫在同一張圖上比較。 | 直觀展示策略優劣，用於報告或論文插圖。 | 簡單明瞭的視覺化。 |

---

## 📊 分析腳本 (`scripts/analysis/`)

用於深入挖掘實驗數據。

| 腳本 | 功能描述 | 狀態 |
|:---|:---|:---|
| `analyze_experiments.py` | **批量實驗分析**<br>掃描 `experiments_results` 資料夾，自動生成勝率、回測摘要報告。 | ⚠️ 硬編碼路徑，需注意 |
| `analyze_best_generation.py` | **最佳世代分析**<br>統計最佳個體通常出現在第幾代，判斷是否過早收斂。 | ⚠️ 使用舊目錄結構 |
| `compute_diversity_metrics.py` | **計算多樣性指標**<br>對指定實驗的所有世代計算 TED 距離和 PnL 相關性矩陣。 | **核心工具**<br>跑完實驗後必跑 |
| `visualize_diversity.py` | **多樣性視覺化**<br>繪製單一實驗的多樣性熱圖 (Heatmap) 和分佈圖。 | 配合 `compute_...` 使用 |
| `compare_rsfgp_rolling.py` | **RSFGP vs Rolling**<br>比較這兩種不同評估架構的實驗結果。 | 專項分析 |

---

## 🛠 實用工具 (`scripts/utils/`)

| 腳本 | 功能描述 |
|:---|:---|
| `continue_evolution.py` | **繼續演化**<br>從中斷的 checkpoint 繼續執行演化，或延長演化世代。 |
| `find_best_test_individual.py` | **尋找最佳測試個體**<br>在庫外 (Test) 數據上評估並找出表現最好的個體（注意：僅供分析，不可用於訓練選擇）。 |
| `update_data.py` | **更新數據**<br>從 yfinance 下載最新的 SPY 等數據到 `history/`。 |
| `debug_tree.py` | **除錯工具**<br>快速檢查一個表達式樹的結構和輸出。 |

---

## 🎨 視覺化 (`scripts/visualization/`)

| 腳本 | 功能描述 |
|:---|:---|
| `visualize_best_individual_pnl.py` | **最佳個體 PnL**<br>畫出單一最佳個體在 Train/Test 期間的詳細 PnL 曲線和進出場點。 |
| `plot_overlay_pnl.py` | **PnL 疊圖**<br>見核心腳本區。 |

---

## 🗑 過時或待清理 (Deprecated)

以下腳本可能使用舊的目錄結構或已被新腳本取代，使用前請確認：

* `scripts/plot_overfitting_gap.py`: 用於繪製 Train-Test 差距的 Boxplot（臨時腳本，功能可能已整合）。
* `scripts/visualize_three_experiments_v2.py`: **已建議刪除**，硬編碼嚴重。
* `scripts/analysis/analyze_all_perfect_niches.py`: 針對特定舊實驗的分析。

---

## 💡 使用建議

1. **跑完新實驗後**：
   - 先跑 `scripts/analysis/compute_diversity_metrics.py` 生成多樣性數據。
   - 再跑 `scripts/analysis/compare_experiments.py` 看看多樣性是否如預期維持住。

2. **要看獲利表現**：
   - 用 `scripts/visualization/plot_overlay_pnl.py` 快速比較 PnL。
   - 用 `scripts/visualization/visualize_best_individual_pnl.py` 深入看單一策略行為。

3. **要寫論文/報告**：
   - 使用 `scripts/compare_signal_niche.py` 或 `scripts/compare_softmin_vs_mean.py` 生成統計圖表。
