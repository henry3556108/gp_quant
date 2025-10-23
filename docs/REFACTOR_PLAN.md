# Refactor Plan: 統一演化引擎

## 🎯 目標

將 `run_portfolio_experiment.py` 重構為使用 `gp_quant/evolution/engine.py`，消除代碼重複並修復深度超限問題。

---

## 📋 Branch 資訊

- **Branch 名稱**: `refactor/unify-evolution-engine`
- **從**: `master`
- **目的**: 實作方案 C（快速修復）

---

## 🔧 實作步驟

### Phase 1: 準備工作 ✅

- [x] 將 `feature/save-cluster-labels` merge 到 `master`
- [x] 從 `master` 創建新 branch `refactor/unify-evolution-engine`
- [x] 創建實作計劃文檔

### Phase 2: 修改 `run_portfolio_experiment.py`

#### 2.1 修改 imports
- [ ] 添加 `from gp_quant.evolution.engine import run_evolution`
- [ ] 移除不需要的 DEAP imports（toolbox 設置相關）

#### 2.2 移除重複的 DEAP 設置代碼
- [ ] 移除 `toolbox = base.Toolbox()` 及相關設置（約第 212-224 行）
- [ ] 移除演化循環代碼（約第 580-620 行）

#### 2.3 實作 `generation_callback` 函數
```python
def create_generation_callback(CONFIG, niching_selector, early_stopping, 
                               generations_dir, gen_log_list, ...):
    """
    創建 generation callback 來處理：
    1. Niching 策略（相似度計算、聚類、跨群選擇）
    2. 早停檢查
    3. 日誌記錄
    4. 族群儲存（已由 engine.py 處理，但需要額外資訊）
    """
    def callback(gen, pop, hof, logbook):
        # Niching 邏輯
        if CONFIG['niching_enabled'] and gen % CONFIG['niching_update_frequency'] == 0:
            # 計算相似度矩陣
            # 聚類
            # 更新 niche_labels
            pass
        
        # 早停檢查
        if CONFIG['early_stopping_enabled']:
            if early_stopping.should_stop(hof[0].fitness.values[0]):
                return True  # 停止演化
        
        # 記錄日誌
        # ...
        
        return False  # 繼續演化
    
    return callback
```

#### 2.4 修改主演化邏輯
```python
# 創建 callback
generation_callback = create_generation_callback(
    CONFIG, niching_selector, early_stopping, generations_dir, gen_log_list, ...
)

# 調用 engine.py
pop, log, hof = run_evolution(
    data=train_data,
    population_size=CONFIG['population_size'],
    n_generations=CONFIG['generations'],
    crossover_prob=CONFIG['crossover_prob'],
    mutation_prob=CONFIG['mutation_prob'],
    individual_records_dir=str(generations_dir),
    generation_callback=generation_callback
)
```

### Phase 3: 處理 Niching 邏輯

#### 3.1 問題：Niching 需要修改 population
當前 `engine.py` 的演化循環：
```python
offspring = toolbox.select(pop, len(pop))
```

但 Niching 需要使用 `CrossNicheSelector`：
```python
offspring = niching_selector.select(population, niche_labels, len(population))
```

#### 3.2 解決方案選項

**選項 A：在 callback 中修改 population（不推薦）**
- ❌ Callback 在演化循環之外
- ❌ 無法直接修改 selection 行為

**選項 B：擴展 engine.py 支援自定義 selector（推薦）**
```python
def run_evolution(..., custom_selector=None):
    if custom_selector:
        offspring = custom_selector(pop)
    else:
        offspring = toolbox.select(pop, len(pop))
```

**選項 C：在 callback 中返回新的 selector（創新）**
```python
def callback(gen, pop, hof, logbook):
    if niching_enabled:
        # 返回一個 selector 函數
        return {
            'selector': lambda p: niching_selector.select(p, niche_labels, len(p))
        }
    return None
```

**決定：採用選項 B**

### Phase 4: 修改 `engine.py`

#### 4.1 添加 `custom_selector` 參數
```python
def run_evolution(
    data, 
    population_size=500, 
    n_generations=50, 
    crossover_prob=0.6, 
    mutation_prob=0.05,
    individual_records_dir=None,
    generation_callback=None,
    custom_selector=None  # 新增
):
```

#### 4.2 修改演化循環
```python
for gen in range(1, n_generations + 1):
    # 使用自定義 selector 或默認 selector
    if custom_selector:
        offspring = custom_selector(pop, gen)  # 傳入 gen 以便動態決策
    else:
        offspring = toolbox.select(pop, len(pop))
    
    offspring = list(map(toolbox.clone, offspring))
    # ... 其餘邏輯不變 ...
```

### Phase 5: 整合 Niching

#### 5.1 創建 Niching wrapper
```python
class NichingSelector:
    def __init__(self, CONFIG, niching_selector):
        self.CONFIG = CONFIG
        self.niching_selector = niching_selector
        self.niche_labels = None
        self.sim_matrix = None
    
    def __call__(self, pop, gen):
        """自定義 selector 接口"""
        # 每 N 代更新相似度矩陣
        if gen % self.CONFIG['niching_update_frequency'] == 0:
            self._update_niching(pop, gen)
        
        # 使用 niching selector 或 fallback
        if self.niche_labels is not None:
            return self.niching_selector.select(pop, self.niche_labels, len(pop))
        else:
            # Fallback to tournament selection
            return tools.selTournament(pop, len(pop), tournsize=3)
    
    def _update_niching(self, pop, gen):
        """更新相似度矩陣和聚類"""
        # 計算相似度
        # 聚類
        # 更新 self.niche_labels
        pass
```

### Phase 6: 測試與驗證

#### 6.1 單元測試
- [ ] 測試 `generation_callback` 正確觸發
- [ ] 測試 Niching 邏輯正確執行
- [ ] 測試早停機制正常工作

#### 6.2 集成測試
- [ ] 運行小規模實驗（population=100, generations=10）
- [ ] 檢查深度限制（應該 0% 違規）
- [ ] 檢查 Niching 統計（silhouette score 等）
- [ ] 檢查早停是否正常觸發

#### 6.3 完整實驗
- [ ] 運行完整實驗（population=500, generations=50）
- [ ] 使用 `check_portfolio_depth.py` 驗證深度
- [ ] 對比修改前後的結果

### Phase 7: 代碼清理

#### 7.1 移除死代碼
- [ ] 移除 `run_portfolio_experiment.py` 中未使用的 imports
- [ ] 移除註釋掉的舊代碼

#### 7.2 更新文檔
- [ ] 更新 `docs/CLUSTER_LABELS_USAGE.md`
- [ ] 更新 `README.md`（如果有）

#### 7.3 代碼審查
- [ ] 檢查所有修改
- [ ] 確保沒有遺留的 TODO
- [ ] 確保代碼風格一致

---

## 📊 預期結果

### 代碼質量

| 指標 | 修改前 | 修改後 | 改善 |
|------|--------|--------|------|
| `run_portfolio_experiment.py` 行數 | ~600 | ~450 | -25% |
| 重複代碼 | 150 行 | 0 行 | -100% |
| 深度違規率 | 76% | 0% | -100% |
| 最大深度 | 69 | ≤17 | -75% |

### 功能驗證

- ✅ 深度限制正常工作
- ✅ Niching 策略正常工作
- ✅ 早停機制正常工作
- ✅ 族群儲存包含 cluster_labels
- ✅ 日誌記錄完整

---

## ⚠️ 風險與緩解

### 風險 1：Niching 邏輯複雜

**風險**：Niching 需要在 selection 階段介入，可能難以整合

**緩解**：
- 使用 `custom_selector` 參數
- 創建 `NichingSelector` wrapper 封裝邏輯
- 保持 callback 簡單

### 風險 2：向後相容性

**風險**：修改 `engine.py` 可能影響現有代碼

**緩解**：
- `custom_selector` 是可選參數（默認 None）
- 保持現有接口不變
- 添加單元測試

### 風險 3：性能影響

**風險**：額外的 callback 調用可能影響性能

**緩解**：
- Callback 只在必要時執行邏輯
- 相似度計算已經是瓶頸，callback 開銷可忽略

---

## 🎯 成功標準

1. ✅ 所有測試通過
2. ✅ 深度違規率 = 0%
3. ✅ Niching 統計與修改前一致
4. ✅ 代碼減少 150+ 行
5. ✅ 沒有功能退化

---

## 📅 時間估計

| Phase | 預計時間 | 說明 |
|-------|---------|------|
| Phase 1 | ✅ 完成 | 準備工作 |
| Phase 2 | 30 分鐘 | 修改 run_portfolio_experiment.py |
| Phase 3 | 30 分鐘 | 設計 Niching 整合方案 |
| Phase 4 | 30 分鐘 | 修改 engine.py |
| Phase 5 | 45 分鐘 | 實作 NichingSelector |
| Phase 6 | 45 分鐘 | 測試與驗證 |
| Phase 7 | 30 分鐘 | 代碼清理 |
| **總計** | **3.5 小時** | |

---

## 📝 Commit 策略

### Commit 1: 準備工作
```
docs: Add refactor plan for unifying evolution engine
```

### Commit 2: 修改 engine.py
```
refactor(engine): Add custom_selector parameter for flexible selection

- Add optional custom_selector parameter to run_evolution()
- Allow custom selection logic while maintaining backward compatibility
- Prepare for Niching integration
```

### Commit 3: 重構 run_portfolio_experiment.py
```
refactor(portfolio): Use engine.py instead of duplicated code

- Remove 150+ lines of duplicated evolution logic
- Use run_evolution() from engine.py
- Implement generation_callback for Niching and early stopping
- Fix depth violation issue (76% -> 0%)

BREAKING CHANGE: None (functionality preserved)
```

### Commit 4: 測試與驗證
```
test: Verify refactored portfolio experiment

- Add tests for generation_callback
- Verify depth limits (0% violation)
- Verify Niching statistics
- Verify early stopping
```

### Commit 5: 文檔更新
```
docs: Update documentation for refactored code

- Update CLUSTER_LABELS_USAGE.md
- Add refactor notes to CODE_DUPLICATION_ANALYSIS.md
```

---

## 🚀 開始實作

準備好開始了嗎？

**下一步：Phase 2 - 修改 `run_portfolio_experiment.py`**

請確認是否開始實作！
