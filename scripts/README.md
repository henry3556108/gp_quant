# Scripts 目錄

這個目錄包含各種輔助腳本和驗證工具。

## 📁 目錄結構

```
scripts/
├── verify/          # 功能驗證腳本
│   ├── verify_early_stopping.py
│   ├── verify_norm_operator.py
│   ├── verify_sharpe_fitness.py
│   ├── verify_portfolio_experiment.py
│   └── verify_portfolio_train_test.py
└── README.md
```

## 🔍 verify/ - 功能驗證腳本

這些腳本用於快速驗證特定功能是否正常運作，**不依賴測試框架**（unittest/pytest）。

### 特點
- ✅ 可直接執行：`python scripts/verify/verify_xxx.py`
- ✅ 包含詳細的輸出和說明
- ✅ 適合快速檢查和除錯
- ✅ 用於開發時的功能驗證

### 與 `tests/` 的區別

| 特性 | `scripts/verify/` | `tests/` |
|------|------------------|----------|
| **測試框架** | 無（直接執行） | unittest/pytest |
| **用途** | 功能驗證、演示 | 正式單元測試 |
| **輸出** | 詳細的說明和結果 | 簡潔的 pass/fail |
| **運行方式** | `python scripts/verify/xxx.py` | `pytest tests/` |
| **適用場景** | 開發、除錯、演示 | CI/CD、回歸測試 |

### 使用範例

```bash
# 驗證 Early Stopping 功能
python scripts/verify/verify_early_stopping.py

# 驗證 Norm Operator
python scripts/verify/verify_norm_operator.py

# 驗證 Sharpe Ratio Fitness
python scripts/verify/verify_sharpe_fitness.py

# 驗證 Portfolio 實驗
python scripts/verify/verify_portfolio_experiment.py

# 驗證訓練/測試分割
python scripts/verify/verify_portfolio_train_test.py
```

## 📝 何時使用

### 使用 `scripts/verify/`
- 開發新功能後快速驗證
- 除錯特定功能
- 向他人演示功能
- 需要詳細輸出時

### 使用 `tests/`
- 運行完整測試套件
- CI/CD 自動化測試
- 回歸測試
- 確保代碼品質

---

**注意**: 這些驗證腳本不會被 pytest 自動發現，需要手動執行。
