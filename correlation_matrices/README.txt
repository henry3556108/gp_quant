# PnL Correlation Matrices - 欄位說明

## 檔案結構

correlation_matrices/
├── correlation_summary.csv          # 所有世代的統計總結
├── gen###_correlation_matrix.csv    # 各世代的完整相關係數矩陣
└── gen###_individual_info.csv       # 各世代的個體資訊

## correlation_summary.csv 欄位說明

1. generation
   - 世代編號（0-50）

2. n_valid_individuals
   - 有效個體數量（能產生有變異的 PnL 曲線）
   - 範圍：採樣 50 個中的有效數量
   - 越多越好，表示策略更穩定

3. n_pairs
   - 計算的相關係數配對數
   - 公式：n × (n-1) / 2
   - 例：50 個體 → 1225 個配對

4. corr_mean
   - 平均相關係數
   - 範圍：-1.0 到 1.0
   - 接近 1.0：策略高度相似（趨同）
   - 接近 0.0：策略多樣化
   - 接近 -1.0：策略相反

5. corr_std
   - 相關係數的標準差
   - 高：族群中既有相似也有差異的個體
   - 低：相關性一致

6. corr_min
   - 最小相關係數
   - 最不相似（或最相反）的兩個個體

7. corr_max
   - 最大相關係數
   - 最相似的兩個個體

8. corr_median
   - 中位數相關係數
   - 比平均值更穩健（不受極端值影響）

9. corr_q25
   - 第 25 百分位數（下四分位數）
   - 25% 的配對低於此值

10. corr_q75
    - 第 75 百分位數（上四分位數）
    - 75% 的配對低於此值

## gen###_individual_info.csv 欄位說明

1. ind_id
   - 個體編號（在採樣中的索引）

2. fitness
   - 適應度值（excess return）

3. height
   - GP 樹的高度

4. length
   - GP 樹的節點數

5. pnl_final
   - 最終累積 PnL

6. pnl_std
   - PnL 曲線的標準差

## 演化趨勢觀察

Generation 0 → 50:
- 有效個體：29 → 50 (+72%)
- 平均相關係數：0.377 → 0.676 (+79%)
- 標準差：0.406 → 0.362 (-11%)

結論：族群隨演化逐漸趨同，策略相似度提高
