# GP Quant 專案上手指南 (Onboarding Guide)

歡迎加入！本文件旨在幫助你快速了解 `GP Quant` 專案的目標、架構與開發流程。

---

## 1. 專案目標：我們在做什麼？

本專案的核心目標是利用 **基因程式設計 (Genetic Programming, GP)** 自動演化出能夠獲利的 **量化交易策略**。

簡單來說，我們讓電腦透過「物競天擇」的方式，從一堆基本的數學運算子和技術指標（如移動平均、RSI）中，自行組合出複雜的交易規則（例如 `如果 RSI(14) > 70 且 5日均線 > 20日均線，則賣出`），並從中找出表現最好的策略。

---

## 2. 核心技術棧

- **語言**: Python 3.10+
- **核心框架**: `DEAP` (一個強大的演化計算框架，我們主要用它的 `gp` 模組)
- **數據處理**: `pandas`, `numpy`
- **環境管理**: `Conda`

---

## 3. 快速開始：設定你的開發環境

我們提供了一鍵設定腳本，你只需要執行一個指令就能完成所有環境設定與測試。

1.  **確認你有 Conda**: 如果沒有，請先安裝 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda。

2.  **執行設定腳本**: 在專案根目錄下打開終端機，執行：

    ```bash
    chmod +x setup_and_test.sh
    ./setup_and_test.sh
    ```

    這個腳本會自動完成以下工作：
    - 建立名為 `gp_quant` 的 Conda 環境。
    - 在該環境中安裝 `requirements.txt` 裡的所有相依套件。
    - 執行所有單元測試，確保環境設定正確無誤。

    看到 `--- Setup and test script finished successfully! ---` 的訊息，就代表你的環境已經準備就緒！

3.  **啟動 Conda 環境**: 未來開發時，請務必先啟動這個環境：
    ```bash
    conda activate gp_quant
    ```

---

## 4. 專案架構導覽

理解每個檔案的作用是最高效的學習方式。

```
├── gp_quant/                 # 專案核心程式碼
│   ├── gp/
│   │   ├── operators.py      # 【開發熱點】定義 GP 的「基因庫」(Primitive Set)
│   │   └── primitives.py     # 【開發熱點】所有技術指標的具體數學實作
│   ├── backtesting/
│   │   └── engine.py         # 演化主流程，控制 DEAP 如何演化
│   └── data/
│       └── loader.py         # 資料載入與清理
├── tests/                    # 所有測試案例
│   ├── gp/
{{ ... }}
│   └── backtesting/
│       └── test_engine.py    # 【必看】回測引擎的測試，了解交易邏輯
├── main.py                   # 專案主執行入口
├── samples/                  # 範例程式碼
│   └── evolution/
│       └── sample_run.py     # 一個獨立的範例，展示完整的演化與回測流程
├── setup_and_test.sh         # 一鍵安裝與測試腳本
└── ONBOARDING.md             # 就是你正在閱讀的這份文件
```

---

## 5. 開發流程範例：如何新增一個技術指標？

這是最常見的開發需求。得益於我們全新的向量化架構，現在新增一個指標只需要 **簡單的兩步**！

假設我們要新增一個 `MACD` 指標：

**第 1 步：在 `primitives.py` 中實作向量化邏輯**

在 `gp_quant/gp/primitives.py` 中，新增 `macd` 函數。它的輸入和輸出都必須是 `np.ndarray`。

```python
# gp_quant/gp/primitives.py
import pandas as pd

def macd(series: np.ndarray, fast: int, slow: int, signal: int) -> np.ndarray:
    """Calculates the vectorized MACD."""
    s = pd.Series(series)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    # signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # 我們可以直接回傳 MACD 線，或者 MACD 線與訊號線的差值
    return macd_line.to_numpy()
```

**第 2 步：在 `operators.py` 中註冊新原語**

在 `gp_quant/gp/operators.py` 中，將你的 `macd` 函數加入到 `pset` 中，並定義它的輸入輸出型別。

```python
# gp_quant/gp/operators.py

# ... 在 "Vectorized financial primitives" 區塊新增
pset.addPrimitive(prim.macd, [Vector, int, int, int], Vector, name="MACD")
```

**就是這麼簡單！** 你不需要再關心 `_price` vs `_volume` 的區別，也不需要處理任何資料注入的邏輯。框架會自動將 `Price` 或 `Volume` 向量傳遞給你的函數。

最後，別忘了在 `tests/gp/test_primitives.py` 中為你的新指標新增單元測試，以確保計算的正確性。

---

## 6. 提高效率的關鍵：注意事項與設計決策

了解這些「坑」能讓你少走很多彎路。

1.  **完全向量化 (Full Vectorization)**
    - **核心設計**: 專案的效能關鍵在於「完全向量化」。我們不使用 Python 迴圈逐日計算，而是利用 `NumPy` 和 `Pandas` 的能力，**一次性地** 計算出整個時間序列的所有指標和交易訊號。
    - **你的工作**: 當你新增一個指標時，必須確保它是向量化的——輸入是 `np.ndarray`，輸出也是 `np.ndarray`。

2.  **`is True` vs `== True` 的陷阱**
    - **問題**: `numpy` 的布林型別 (`numpy.bool_`) 和 Python 原生的 `bool` 是不同的物件。在回測引擎中，GP 規則產生的訊號可能是 `numpy.bool_`，使用 `signal is True` 會判斷失敗。
    - **解決方案**: **永遠使用 `signal == True` 或 `signal == False`** 來判斷交易訊號，確保型別相容性。

3.  **Numba JIT 加速**
    - **機制**: 雖然訊號生成已經被向量化，但最終的「交易買賣模擬」仍然是一個迴圈。為了加速這個迴圈，我們使用了 `@numba.jit` 裝飾器 (`_numba_simulation_loop` 函數)。
    - **結果**: 這個 JIT 編譯器將交易迴圈從慢速的 Python 轉化為高效的機器碼，將原本需要數分鐘的回測縮短到秒級。

---

## 7. 接下來可以做什麼？

- **擴充指標庫**: 嘗試新增你熟悉的技術指標，如布林通道 (Bollinger Bands)、KDJ 等。
- **優化適應度函數**: 目前的適應度函數 (Fitness) 已被修正為有意義的 **「超額回報」 (Excess Return)**。

  `Fitness = (GP 策略總回報) - (買入持有策略總回報)`

  這是一個「蘋果對蘋果」的公平比較，因為兩個回報都是基於相同的初始資本計算的。在演化過程中，日誌中的 `max` 值現在會顯示真實的超額回報金額，一個正數代表當代最好的策略跑贏了市場。
- **完善 `main.py`**: 將演化和回測流程封裝成一個可透過命令列執行的完整應用。

如果你有任何問題，隨時提出！
