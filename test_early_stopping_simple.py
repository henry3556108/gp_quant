"""
Simple tests for EarlyStopping class (without pytest)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gp_quant.evolution.early_stopping import EarlyStopping


def test_basic_early_stopping():
    """測試基本早停功能"""
    print("Test 1: 基本早停功能")
    es = EarlyStopping(patience=3, min_delta=0.0, mode='max')
    
    # 前 3 代有進步
    assert not es.step(1.0), "第 1 代不應停止"
    assert es.counter == 0
    assert es.best_fitness == 1.0
    
    assert not es.step(1.5), "第 2 代不應停止"
    assert es.counter == 0
    assert es.best_fitness == 1.5
    
    assert not es.step(2.0), "第 3 代不應停止"
    assert es.counter == 0
    assert es.best_fitness == 2.0
    
    # 後 3 代無進步
    assert not es.step(2.0), "第 4 代不應停止"
    assert es.counter == 1
    
    assert not es.step(2.0), "第 5 代不應停止"
    assert es.counter == 2
    
    assert es.step(2.0), "第 6 代應該停止"
    assert es.counter == 3
    assert es.should_stop is True
    
    print("   ✓ 通過")


def test_early_stopping_with_min_delta():
    """測試帶閾值的早停"""
    print("\nTest 2: 帶閾值的早停")
    es = EarlyStopping(patience=2, min_delta=0.1, mode='max')
    
    assert not es.step(1.0)
    assert es.best_fitness == 1.0
    
    # 改進 0.05 < 0.1，計數 +1
    assert not es.step(1.05)
    assert es.counter == 1
    assert es.best_fitness == 1.0  # 未更新
    
    # 改進 0.03 < 0.1，計數 +1，觸發早停
    assert es.step(1.08)
    assert es.counter == 2
    assert es.should_stop is True
    
    print("   ✓ 通過")


def test_early_stopping_reset_on_improvement():
    """測試有進步時重置計數器"""
    print("\nTest 3: 有進步時重置計數器")
    es = EarlyStopping(patience=3, min_delta=0.0, mode='max')
    
    assert not es.step(1.0)
    assert not es.step(1.0)  # 無進步，counter = 1
    assert es.counter == 1
    
    assert not es.step(1.0)  # 無進步，counter = 2
    assert es.counter == 2
    
    assert not es.step(1.5)  # 有進步，counter 重置
    assert es.counter == 0
    assert es.best_fitness == 1.5
    
    # 再次無進步
    assert not es.step(1.5)
    assert es.counter == 1
    
    print("   ✓ 通過")


def test_mode_min():
    """測試 mode='min' 的情況"""
    print("\nTest 4: mode='min' 的情況")
    es = EarlyStopping(patience=2, min_delta=0.0, mode='min')
    
    # fitness 越小越好
    assert not es.step(10.0)
    assert es.best_fitness == 10.0
    
    assert not es.step(5.0)  # 有進步（減少）
    assert es.counter == 0
    assert es.best_fitness == 5.0
    
    assert not es.step(5.0)  # 無進步
    assert es.counter == 1
    
    assert es.step(5.0)  # 無進步，觸發早停
    assert es.should_stop is True
    
    print("   ✓ 通過")


def test_get_status():
    """測試獲取狀態"""
    print("\nTest 5: 獲取狀態")
    es = EarlyStopping(patience=5, min_delta=0.01, mode='max')
    
    es.step(1.0)
    es.step(1.0)
    
    status = es.get_status()
    
    assert status['counter'] == 1
    assert status['best_fitness'] == 1.0
    assert status['should_stop'] is False
    assert status['generation'] == 2
    assert status['patience'] == 5
    assert status['min_delta'] == 0.01
    assert status['mode'] == 'max'
    
    print("   ✓ 通過")


def test_reset():
    """測試重置功能"""
    print("\nTest 6: 重置功能")
    es = EarlyStopping(patience=3, min_delta=0.0, mode='max')
    
    es.step(1.0)
    es.step(1.0)
    es.step(1.0)
    
    assert es.counter == 2
    assert es.best_fitness == 1.0
    assert es.generation == 3
    
    es.reset()
    
    assert es.counter == 0
    assert es.best_fitness is None
    assert es.should_stop is False
    assert es.generation == 0
    
    print("   ✓ 通過")


def test_sharpe_ratio_scenario():
    """測試 Sharpe Ratio fitness 場景"""
    print("\nTest 7: Sharpe Ratio fitness 場景")
    # Sharpe Ratio 通常在 -3 到 5 之間
    es = EarlyStopping(patience=10, min_delta=0.001, mode='max')
    
    # 模擬 Sharpe Ratio 演化（持續小幅改進）
    sharpe_values = [0.5, 0.8, 1.2, 1.5, 1.52, 1.53, 1.535, 1.538, 1.540, 1.541]
    
    stopped = False
    for i, sharpe in enumerate(sharpe_values):
        should_stop = es.step(sharpe)
        if should_stop:
            stopped = True
            break
    
    # 因為持續有小幅改進，不應該停止
    assert not stopped, "持續改進不應觸發早停"
    
    print("   ✓ 通過")


def test_excess_return_scenario():
    """測試 Excess Return fitness 場景"""
    print("\nTest 8: Excess Return fitness 場景")
    # Excess Return 通常在幾千到幾萬
    es = EarlyStopping(patience=5, min_delta=100.0, mode='max')
    
    # 模擬 Excess Return 演化（改進逐漸變小）
    returns = [5000, 8000, 12000, 12050, 12080, 12090, 12095, 12096]
    
    stopped_at = None
    for i, ret in enumerate(returns):
        should_stop = es.step(ret)
        if should_stop:
            stopped_at = i
            break
    
    # 應該在連續 5 代改進 < 100 時停止
    assert stopped_at is not None, "應該觸發早停"
    assert stopped_at >= 5, f"應該至少在第 6 代後停止，實際在第 {stopped_at + 1} 代"
    assert es.should_stop is True
    
    print("   ✓ 通過")


def test_invalid_inputs():
    """測試無效輸入"""
    print("\nTest 9: 無效輸入")
    
    # 測試無效 patience
    try:
        EarlyStopping(patience=0)
        assert False, "應該拋出 ValueError"
    except ValueError as e:
        assert "patience must be >= 1" in str(e)
    
    # 測試無效 mode
    try:
        EarlyStopping(mode='invalid')
        assert False, "應該拋出 ValueError"
    except ValueError as e:
        assert "mode must be 'max' or 'min'" in str(e)
    
    print("   ✓ 通過")


if __name__ == '__main__':
    print("="*80)
    print("🧪 EarlyStopping 單元測試")
    print("="*80)
    print()
    
    try:
        test_basic_early_stopping()
        test_early_stopping_with_min_delta()
        test_early_stopping_reset_on_improvement()
        test_mode_min()
        test_get_status()
        test_reset()
        test_sharpe_ratio_scenario()
        test_excess_return_scenario()
        test_invalid_inputs()
        
        print()
        print("="*80)
        print("✅ 所有測試通過！")
        print("="*80)
        
    except AssertionError as e:
        print()
        print("="*80)
        print(f"❌ 測試失敗: {e}")
        print("="*80)
        sys.exit(1)
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ 錯誤: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
