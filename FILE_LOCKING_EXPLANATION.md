# 檔案鎖定機制說明

## 問題背景

在並行執行多個實驗時（8 個進程），發現 `save_population()` 函數會靜默失敗，導致只有部分 generation 被儲存。

### 原因分析

1. **I/O 競爭**: 8 個進程同時寫入磁碟，造成 I/O 瓶頸
2. **`dill.dump()` 靜默失敗**: 在高並發下，序列化可能失敗但不拋出異常
3. **檔案系統限制**: 同時開啟過多檔案可能觸發系統限制

## 解決方案：全局檔案鎖定

### 核心概念

使用 **全局鎖定檔案**（`.population_save.lock`）來確保：
- **同一時間只有一個進程**可以執行 `dill.dump()`
- 其他進程必須等待，避免 I/O 競爭
- 使用 `fcntl.flock()` 實現進程間互斥鎖

### 工作流程

```
進程 1: 嘗試獲取鎖 → 成功 → 寫入 population.pkl → 釋放鎖
進程 2: 嘗試獲取鎖 → 等待（進程 1 持有鎖）→ 獲取鎖 → 寫入 → 釋放鎖
進程 3: 嘗試獲取鎖 → 等待 → 獲取鎖 → 寫入 → 釋放鎖
...
```

### 實作細節

#### 1. 全局鎖定檔案位置

```python
global_lock_file = os.path.join(
    os.path.dirname(individual_records_dir), 
    ".population_save.lock"
)
```

例如：`experiments_results/ABX_TO/.population_save.lock`

#### 2. 非阻塞鎖定 + 超時機制

```python
fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
```

- `LOCK_EX`: 排他鎖（exclusive lock）
- `LOCK_NB`: 非阻塞（non-blocking）
- 如果鎖被佔用，立即返回錯誤而不是阻塞

#### 3. 超時保護

```python
if elapsed > lock_timeout:
    raise TimeoutError(f"Failed to acquire lock after {lock_timeout:.1f}s")
```

- 預設 60 秒超時
- 避免無限等待

#### 4. 重試機制

- 最多重試 3 次
- 指數退避（exponential backoff）
- 每次重試等待時間遞增

#### 5. 寫入驗證

```python
file_size = os.path.getsize(pickle_file)
if file_size > 0:
    return  # Success
```

確保檔案確實被寫入且不為空

### 優點

✅ **避免 Race Condition**: 同一時間只有一個進程寫入  
✅ **防止 I/O 瓶頸**: 序列化寫入，避免磁碟過載  
✅ **可靠性高**: 重試機制 + 寫入驗證  
✅ **跨進程安全**: `fcntl.flock()` 是作業系統級別的鎖  

### 缺點

⚠️ **寫入速度變慢**: 從並行變成序列化，總時間可能增加  
⚠️ **可能出現等待**: 如果某個進程寫入很慢，其他進程需要等待  

### 性能影響估算

**假設**:
- 每個 generation 寫入時間: ~0.1 秒
- 8 個並行進程
- 50 個 generation

**無鎖定（有失敗風險）**:
- 理論時間: 50 × 0.1 = 5 秒/實驗

**有鎖定（可靠）**:
- 最壞情況: 8 個進程輪流寫入 50 個 generation
- 實際上不會這麼糟，因為不同進程的 generation 進度不同
- 預估增加時間: 10-20%

### 替代方案

如果性能影響太大，可以考慮：

1. **降低並行數量**: 從 8 降到 4
2. **使用信號量**: 允許最多 N 個進程同時寫入（N=2 或 3）
3. **批次寫入**: 每 5 個 generation 寫入一次
4. **異步寫入**: 使用後台線程處理寫入

## 使用方式

無需修改調用方式，`save_population()` 會自動處理鎖定：

```python
save_population(pop, generation, individual_records_dir)
```

## 監控與除錯

如果出現問題，檢查：

1. **鎖定檔案**: `experiments_results/*/. population_save.lock`
2. **警告訊息**: 查看 log 中的 "Warning" 或 "ERROR"
3. **檔案大小**: 確認 `population.pkl` 不為 0 bytes

## 測試建議

重新執行實驗後，檢查：

```bash
# 檢查每個實驗的 generation 數量
find experiments_results -type d -name "individual_records_*" -exec sh -c 'echo "{}:" && ls {} | wc -l' \;

# 應該都是 51（generation 0-50）
```
