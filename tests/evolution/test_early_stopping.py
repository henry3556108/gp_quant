"""
Unit tests for EarlyStopping class
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gp_quant.evolution.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Test cases for EarlyStopping class"""
    
    def test_initialization(self):
        """測試初始化"""
        es = EarlyStopping(patience=5, min_delta=0.01, mode='max')
        
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == 'max'
        assert es.counter == 0
        assert es.best_fitness is None
        assert es.should_stop is False
        assert es.generation == 0
    
    def test_invalid_patience(self):
        """測試無效的 patience"""
        with pytest.raises(ValueError, match="patience must be >= 1"):
            EarlyStopping(patience=0)
    
    def test_invalid_mode(self):
        """測試無效的 mode"""
        with pytest.raises(ValueError, match="mode must be 'max' or 'min'"):
            EarlyStopping(mode='invalid')
    
    def test_basic_early_stopping(self):
        """測試基本早停功能"""
        es = EarlyStopping(patience=3, min_delta=0.0, mode='max')
        
        # 前 3 代有進步
        assert not es.step(1.0)
        assert es.counter == 0
        assert es.best_fitness == 1.0
        
        assert not es.step(1.5)
        assert es.counter == 0
        assert es.best_fitness == 1.5
        
        assert not es.step(2.0)
        assert es.counter == 0
        assert es.best_fitness == 2.0
        
        # 後 3 代無進步
        assert not es.step(2.0)
        assert es.counter == 1
        
        assert not es.step(2.0)
        assert es.counter == 2
        
        assert es.step(2.0)  # 第 3 代無進步，應該停止
        assert es.counter == 3
        assert es.should_stop is True
    
    def test_early_stopping_with_min_delta(self):
        """測試帶閾值的早停"""
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
    
    def test_early_stopping_reset_on_improvement(self):
        """測試有進步時重置計數器"""
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
    
    def test_mode_min(self):
        """測試 mode='min' 的情況"""
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
    
    def test_get_status(self):
        """測試獲取狀態"""
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
    
    def test_reset(self):
        """測試重置功能"""
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
    
    def test_repr(self):
        """測試字符串表示"""
        es = EarlyStopping(patience=10, min_delta=0.001, mode='max')
        es.step(1.0)
        
        repr_str = repr(es)
        
        assert 'EarlyStopping' in repr_str
        assert 'patience=10' in repr_str
        assert 'min_delta=0.001' in repr_str
        assert "mode='max'" in repr_str
        assert 'counter=0' in repr_str
        assert 'generation=1' in repr_str
    
    def test_sharpe_ratio_scenario(self):
        """測試 Sharpe Ratio fitness 場景"""
        # Sharpe Ratio 通常在 -3 到 5 之間
        es = EarlyStopping(patience=10, min_delta=0.001, mode='max')
        
        # 模擬 Sharpe Ratio 演化
        sharpe_values = [0.5, 0.8, 1.2, 1.5, 1.52, 1.53, 1.53, 1.53, 1.53, 1.53]
        
        for i, sharpe in enumerate(sharpe_values):
            should_stop = es.step(sharpe)
            if should_stop:
                # 應該在第 10 代停止（連續 10 代改進 < 0.001）
                assert i == 9
                break
        
        assert es.should_stop is False  # 因為改進雖小但仍在進步
    
    def test_excess_return_scenario(self):
        """測試 Excess Return fitness 場景"""
        # Excess Return 通常在幾千到幾萬
        es = EarlyStopping(patience=5, min_delta=100.0, mode='max')
        
        # 模擬 Excess Return 演化
        returns = [5000, 8000, 12000, 12050, 12080, 12090, 12095, 12096]
        
        for i, ret in enumerate(returns):
            should_stop = es.step(ret)
            if should_stop:
                # 應該在連續 5 代改進 < 100 時停止
                assert i >= 5
                break
        
        assert es.should_stop is True


if __name__ == '__main__':
    # 運行測試
    pytest.main([__file__, '-v'])
