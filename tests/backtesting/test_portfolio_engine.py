"""
Unit tests for Portfolio Backtesting Engine
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.backtesting.rebalancing import EventDrivenRebalancer, CapitalAllocation
from gp_quant.backtesting.metrics import PortfolioMetrics


class TestCapitalAllocation:
    """Test CapitalAllocation dataclass"""
    
    def test_total_value(self):
        """Test total value calculation"""
        alloc = CapitalAllocation(
            stock_ticker='TEST',
            initial_capital=10000,
            available_cash=5000,
            position_value=6000,
            shares_held=100
        )
        assert alloc.total_value == 11000  # 5000 + 6000


class TestEventDrivenRebalancer:
    """Test EventDrivenRebalancer class"""
    
    def test_initialization(self):
        """Test rebalancer initialization"""
        tickers = ['STOCK1', 'STOCK2']
        rebalancer = EventDrivenRebalancer(
            tickers=tickers,
            initial_capital=100000,
            equal_weight=True
        )
        
        assert len(rebalancer.allocations) == 2
        assert rebalancer.allocations['STOCK1'].initial_capital == 50000
        assert rebalancer.allocations['STOCK2'].initial_capital == 50000
    
    def test_buy_signal(self):
        """Test buy signal handling"""
        rebalancer = EventDrivenRebalancer(
            tickers=['TEST'],
            initial_capital=10000
        )
        
        # Buy at $100 per share
        transaction = rebalancer.handle_buy_signal(
            'TEST',
            datetime(2020, 1, 1),
            100.0
        )
        
        assert transaction is not None
        assert transaction['action'] == 'BUY'
        assert transaction['shares'] == 100  # 10000 / 100
        assert rebalancer.allocations['TEST'].shares_held == 100
        assert rebalancer.allocations['TEST'].available_cash == 0
    
    def test_sell_signal(self):
        """Test sell signal handling"""
        rebalancer = EventDrivenRebalancer(
            tickers=['TEST'],
            initial_capital=10000
        )
        
        # Buy first
        rebalancer.handle_buy_signal('TEST', datetime(2020, 1, 1), 100.0)
        
        # Sell at $120 per share
        transaction = rebalancer.handle_sell_signal(
            'TEST',
            datetime(2020, 1, 2),
            120.0
        )
        
        assert transaction is not None
        assert transaction['action'] == 'SELL'
        assert transaction['shares'] == 100
        assert rebalancer.allocations['TEST'].shares_held == 0
        assert rebalancer.allocations['TEST'].available_cash == 12000  # 100 * 120
    
    def test_no_cash_for_buy(self):
        """Test buy signal with insufficient cash"""
        rebalancer = EventDrivenRebalancer(
            tickers=['TEST'],
            initial_capital=50  # Only $50
        )
        
        # Try to buy at $100 per share
        transaction = rebalancer.handle_buy_signal(
            'TEST',
            datetime(2020, 1, 1),
            100.0
        )
        
        assert transaction is None  # Not enough cash
    
    def test_no_shares_for_sell(self):
        """Test sell signal with no position"""
        rebalancer = EventDrivenRebalancer(
            tickers=['TEST'],
            initial_capital=10000
        )
        
        # Try to sell without buying first
        transaction = rebalancer.handle_sell_signal(
            'TEST',
            datetime(2020, 1, 1),
            100.0
        )
        
        assert transaction is None  # No position to sell


class TestPortfolioMetrics:
    """Test PortfolioMetrics class"""
    
    def test_calculate_return(self):
        """Test return calculation"""
        ret = PortfolioMetrics.calculate_return(100000, 120000)
        assert ret == 0.2  # 20% return
    
    def test_calculate_excess_return(self):
        """Test excess return calculation"""
        excess = PortfolioMetrics.calculate_excess_return(0.15, 0.05)
        assert excess == 0.10
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = PortfolioMetrics.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation"""
        equity_curve = pd.Series([100, 110, 105, 95, 100, 120])
        max_dd = PortfolioMetrics.calculate_max_drawdown(equity_curve)
        assert max_dd < 0  # Drawdown should be negative
        assert max_dd >= -1  # Should be between -1 and 0
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        vol = PortfolioMetrics.calculate_volatility(returns)
        assert vol > 0
    
    def test_calculate_win_rate(self):
        """Test win rate calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        win_rate = PortfolioMetrics.calculate_win_rate(returns)
        assert win_rate == 0.6  # 3 out of 5 positive


class TestPortfolioBacktestingEngine:
    """Test PortfolioBacktestingEngine class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        data = {}
        for ticker in ['STOCK1', 'STOCK2']:
            df = pd.DataFrame({
                'Open': np.random.uniform(90, 110, 100),
                'High': np.random.uniform(95, 115, 100),
                'Low': np.random.uniform(85, 105, 100),
                'Close': np.random.uniform(90, 110, 100),
                'Volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
            data[ticker] = df
        
        return data
    
    def test_initialization(self, sample_data):
        """Test engine initialization"""
        engine = PortfolioBacktestingEngine(
            data=sample_data,
            backtest_start='2020-01-01',
            backtest_end='2020-04-09',
            initial_capital=100000
        )
        
        assert len(engine.tickers) == 2
        assert engine.initial_capital == 100000
        assert len(engine.common_dates) > 0
    
    def test_invalid_data(self):
        """Test with invalid data"""
        with pytest.raises(ValueError):
            PortfolioBacktestingEngine(
                data={},  # Empty data
                backtest_start='2020-01-01',
                backtest_end='2020-04-09'
            )
    
    def test_missing_columns(self):
        """Test with missing columns"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Close': np.random.uniform(90, 110, 10)
            # Missing other required columns
        }, index=dates)
        
        with pytest.raises(ValueError):
            PortfolioBacktestingEngine(
                data={'TEST': df},
                backtest_start='2020-01-01',
                backtest_end='2020-01-10'
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
