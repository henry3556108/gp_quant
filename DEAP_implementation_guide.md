# DEAP 實作技術指南：基於《利用遺傳編程生成交易規則》

本文件旨在將 `PRD.md` 中定義的技術規格，轉化為使用 Python 的 `DEAP` (Distributed Evolutionary Algorithms in Python) 函式庫的具體實作指南。內容涵蓋了從環境設定、型別定義到演化流程的各個環節，以利於後續的程式開發。

---

## **1. 核心 DEAP 模組與物件設定**

在開始之前，我們需要設定 DEAP 的基礎結構，包括定義適應度（Fitness）和個體（Individual）的類別。

- **主要模組**：`deap.creator`、`deap.base`、`deap.tools`、`deap.gp`
- **設定程式碼範例**：

```python
from deap import base, creator, tools, gp
import random
import operator

# 1. 定義適應度：單一目標，最大化超額回報
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 2. 定義個體：以樹狀結構（PrimitiveTree）為基礎，並繼承適應度
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
```

---

## **2. 強型別基因編程（Strongly Typed GP）**

對應 `PRD.md` 第 2 節的層次化樹結構，我們必須使用 `gp.PrimitiveSetTyped` 來確保函式和終端的型別正確性。

### **2.1. 建立 PrimitiveSetTyped**

樹的根節點必須回傳布林值（買/賣信號），並接受當日價格 `p` 和成交量 `v` 作為輸入。

```python
# 假設輸入為兩個浮點數 (p, v)，回傳一個布林值
pset = gp.PrimitiveSetTyped("MAIN", in_types=[float, float], ret_type=bool)

# 重新命名輸入參數以增加可讀性
pset.renameArguments(ARG0='p', ARG1='v')
```

### **2.2. 新增原語（Primitives）與終端（Terminals）**

根據 `PRD.md` 的函式集與終端集，定義每個元素的型別簽章。

```python
# --- 保護性運算 ---
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

# --- 布林運算子 (回傳 bool) ---
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# --- 關係運算子 (float -> bool) ---
pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)

# --- 算術運算子 (回傳 float) ---
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protected_div, [float, float], float)

# --- 實數函式 (回傳 float) ---
# 注意：avg, max, min 等函式需要自行實現，其簽名可能更複雜
# 這裡以簡化版為例，假設它們接受一個浮點數和一個整數
# pset.addPrimitive(avg, [float, int], float)
# pset.addPrimitive(max, [float, int], float)
# ... 其他實數函式 ...

# --- 終端集 ---
# 布林常數
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# 實數常數 (使用 Ephemeral Constant)
# 每次生成個體時，會呼叫 lambda 函式產生一個新的隨機數
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float)

# 整數常數 (用於 avg(s, n) 的 n)
pset.addEphemeralConstant("n_days", lambda: random.randint(5, 50), int)
```

**注意事項**：
- **保護性運算**：如 `protected_div`，是防止執行期間出現 `ZeroDivisionError` 的關鍵。
- **自訂函式**：如 `avg`, `max`, `RSI` 等，需要你自行編寫。這些函式通常需要存取歷史數據（例如一個 Pandas DataFrame），這可以透過閉包或傳遞一個包含數據的 context 物件來實現。
- **型別檢查**：如果在生成樹時 DEAP 找不到某個型別可用的節點，會拋出 `IndexError`。請確保每個型別（`float`, `bool`, `int`）都有至少一個可以生成的終端或原語。

---

## **3. 演化流程設定（Toolbox）**

`Toolbox` 是 DEAP 的核心，用於註冊所有演化操作（生成、評估、選擇、交配、突變）。

```python
# 建立 Toolbox
toolbox = base.Toolbox()

# --- 註冊生成操作 (對應 PRD 5.1) ---
# 使用 ramped half-and-half 方法生成樹，初始深度在 2 到 6 之間
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 註冊評估函式 (對應 PRD 4) ---
# `evaluate` 函式需要你自行實現
# 它接收一個個體，執行回測，並返回超額回報
def evaluate(individual, data):
    # 1. 將樹編譯成可執行的 Python 函式
    rule = gp.compile(expr=individual, pset=pset)
    # 2. 執行交易模擬回測...
    # 3. 計算 Return_GP 和 Return_B&H
    # 4. 計算超額回報
    excess_return = 0.0 # 替換為你的計算結果
    return (excess_return,)

toolbox.register("evaluate", evaluate, data=my_historical_data)

# --- 註冊演化運算子 ---
# 選擇 (對應 PRD 5.2)
# DEAP 內建了 SUS，但排名轉換需要手動實現
toolbox.register("select", tools.selStochasticUniversalSampling)

# 交配 (對應 PRD 5.3)
# cxOnePoint 在強型別下會自動確保型別相容性
# cxOnePointLeafBiased 可控制在內部節點或葉節點交配的機率
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1) # 10% 機率選葉節點

# 突變 (對應 PRD 5.4)
# mutUniform 會隨機選擇一個點，並用一個新的、型別相容的子樹替換它
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# --- 樹深度限制 (膨脹控制) ---
# 在交配和突變後，限制最大深度為 17
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
```

### **關於排名選擇的特別說明**

`PRD.md` 5.2 節描述的「排名 + SUS」選擇機制，在 DEAP 中需要一個額外的步驟：

1.  在每一代，首先根據原始適應度（超額回報）對族群進行排序。
2.  為每個個體計算其排名後的新適應度分數 `f_i`。
3.  將這個新分數暫時賦予每個個體。
4.  在這些新分數上執行 `tools.selStochasticUniversalSampling`。

**注意**：`SUS` 要求適應度值不能為負。你需要確保計算出的 `f_i` 始終大於等於 0。

---

## **4. 主演化迴圈**

設定好 `Toolbox` 後，就可以編寫標準的遺傳演算法迴圈。

```python
def main():
    pop = toolbox.population(n=500) # 族群大小
    hof = tools.HallOfFame(1) # 名人堂，只保存最佳個體
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # 執行演化
    # algorithms.eaSimple 是 DEAP 提供的標準流程
    # 你也可以自訂迴圈以實現更複雜的邏輯（如排名選擇）
    pop, log = algorithms.eaSimple(pop, toolbox, 
                                   cxpb=0.6, mutpb=0.05, ngen=50, # 參數對應 PRD 6
                                   stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

if __name__ == "__main__":
    main()
```

本文件提供了將 `PRD.md` 對應到 `DEAP` 的框架。下一步是填充 `evaluate` 函式中的交易邏輯和 `main` 迴圈中的自訂選擇機制。
