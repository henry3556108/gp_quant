GP 框架升級計畫書：雙重生態位演算法 (Dual-Niche GP)

版本: 1.0

日期: 2025/11/25

目標: 改良現有 GP 框架，引入基於「表現型」與「基因型」的雙重生態位機制，以維持族群多樣性並避免早熟收斂。

1. 系統參數設定 (Hyperparameters)

1.1 基礎演算法參數

Population Size (POP_SIZE): 5,000

Initialization Method: Ramped Half-and-Half

Generational Strategy: 100% Replacement (無菁英保留，全由子代取代)

1.2 演化操作機率 (總和 1.0)

Crossover Rate: 0.75 (產生 7,500 個體)

Mutation Rate: 0.20 (產生 2,000 個體)

Reproduction Rate: 0.05 (產生 500 個體)

註：操作採平行互斥模式，一個個體僅會經歷一種操作。

1.3 生態位與選擇參數 (新增)

Target Niches (K): 預計分群的數量 (例如：10)

Min Niche Size (N): 每個生態位保留的最佳個體數下限 (例如：50)

Selection Logic: Tournament Selection (Size t)

Mating Strategy (CROSS_GROUP_RATIO): * 設定跨群配對 (Cross-Group) 與同群配對 (In-Group) 的比例。

例如：0.3 代表 30% 的 Parents 來自不同群的組合，70% 來自同群。

2. 核心流程架構 (Workflow)

2.1 初始化 (Initialization)

使用 Ramped Half-and-Half 方法生成 POP_SIZE (5,000) 個初始個體。

確保樹的深度在設定範圍內（例如 depth 2-6）。

2.2 評估與特徵提取 (Evaluation & Extraction)

對當前代 (Generation) 的所有個體進行平行運算：

表現型評估 (Phenotypic): * 對多個標的進行 Backtest。

計算綜合績效曲線 (Aggregate Performance Curve)。

建議實作：將各標的績效 Normalize 後加權平均，避免單一高波動標的主導。

基因型特徵 (Genotypic):

提取樹結構特徵以利後續計算距離。

效能優化點： 若 POP_SIZE 為 5,000，強烈建議同時計算「結構指紋 (Structural Fingerprint)」作為 TED 的替代或初篩方案。

2.3 雙重生態位分群 (Dual-Niche Clustering)

系統需維護兩套並行的生態位邏輯，擇一或混合使用（需在程式中設定權重或輪替機制，或本計畫預設為兩者並存但用於不同目的？需確認：是兩者同時運算取交集，還是分開兩次實驗？以下假設為「單一世代內同時存在兩種分群觀點，供 Parent Selection 混合使用」或「擇一主要依據」。本計畫書預設以「表現型」為主，「基因型」為輔，或兩者獨立運算後合併 Pool。）

A. 表現型生態位 (Phenotypic Niche)

輸入: 所有個體的綜合績效曲線。

距離計算: 計算個體間的相關係數 (Correlation Coefficient)，轉化為距離矩陣 (1 - Corr)。

分群演算法: * 使用階層式分群 (Hierarchical Clustering, Ward's method)。

切割成 K 個初始群。

篩選與平衡 (Top N Enforcement):

對每個群內的個體按 Fitness 排序。

若群大小 < N，強制從鄰近群合併或從全域替補（依設計決定）。

保留每個群的 Top N 個體。

B. 基因型生態位 (Genotypic Niche)

輸入: 所有個體的樹結構。

距離計算: * 標準版: Tree Edit Distance (TED) Matrix (需標準化)。

效能版: 基於結構特徵向量的 Euclidean Distance。

分群演算法: 階層式分群。

篩選與平衡: 同上，保留每個群的 Top N 個體。

3. 親代選擇與演化操作 (Selection & Reproduction)

從上述生態位篩選出的 Top N Pool (可能是表現型與基因型 Top N 的聯集，或是單一模式的結果) 進行選擇。

3.1 準備配對池 (Parent Pool)

假設經過篩選後，我們有 $K$ 個群，每個群有 $N$ 個菁英。

3.2 Crossover (75%)

目標：產生 $POP\_SIZE \times 0.75$ 個新個體。
需形成 Pairs (Parent 1, Parent 2)：

決定配對類型: 根據 CROSS_GROUP_RATIO 決定是「同群」還是「跨群」。

選擇 Parent 1: * 隨機選擇一個群 $G_i$。

在 $G_i$ 的 Top $N$ 中使用 Tournament Selection 選出 Parent 1。

選擇 Parent 2:

若是同群 (In-Group): 在 同一個 $G_i$ 中使用 Tournament 選出 Parent 2。

若是跨群 (Cross-Group): 隨機選擇另一個群 $G_j (i \neq j)$，在 $G_j$ 中使用 Tournament 選出 Parent 2。

執行: 進行 Subtree Crossover。

3.3 Mutation (20%)

目標：產生 $POP\_SIZE \times 0.20$ 個新個體。

來源: 從所有生態位的 Top $N$ 集合中，依群聚權重隨機挑選群，再經 Tournament 選出 Parent。

執行: 進行 Point/Subtree Mutation。

3.4 Reproduction (5%)

目標：產生 $POP\_SIZE \times 0.05$ 個新個體。

來源: 同 Mutation，從 Top $N$ 集合中選出。

執行: 直接複製進入下一代。