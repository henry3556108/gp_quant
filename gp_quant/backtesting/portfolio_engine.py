"""
Portfolio Backtesting Engine

This module implements multi-stock portfolio backtesting with event-driven rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from .rebalancing import EventDrivenRebalancer
from .metrics import PortfolioMetrics


class PortfolioBacktestingEngine:
    """
    Multi-stock portfolio backtesting engine.
    
    Features:
    - Event-driven capital rebalancing
    - Independent capital pools per stock
    - Comprehensive performance metrics
    
    Thread Safety:
    - This class is NOT thread-safe
    - Each worker should create its own instance
    - Do NOT share instances across threads/processes
    """
    
    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 backtest_start: str,
                 backtest_end: str,
                 initial_capital: float = 100000.0):
        """
        Initialize portfolio backtesting engine.
        
        Args:
            data: Dict mapping ticker to price DataFrame
                  Each DataFrame should have DatetimeIndex and OHLCV columns
            backtest_start: Start date for backtesting (YYYY-MM-DD)
            backtest_end: End date for backtesting (YYYY-MM-DD)
            initial_capital: Initial portfolio capital
            
        Example:
            data = {
                'ABX.TO': df_abx,
                'RY.TO': df_ry,
                ...
            }
        """
        self.data = data
        self.tickers = list(data.keys())
        self.backtest_start = pd.to_datetime(backtest_start)
        self.backtest_end = pd.to_datetime(backtest_end)
        self.initial_capital = initial_capital
        
        # Validate and prepare data
        self._validate_data()
        self._prepare_data()
        
        # Initialize rebalancer
        self.rebalancer = EventDrivenRebalancer(
            tickers=self.tickers,
            initial_capital=initial_capital,
            equal_weight=True
        )
    
    def _validate_data(self):
        """Validate input data"""
        if not self.data:
            raise ValueError("Data dictionary is empty")
        
        for ticker, df in self.data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{ticker}: DataFrame must have DatetimeIndex")
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"{ticker}: Missing columns {missing_cols}")
    
    def _prepare_data(self):
        """Prepare data for backtesting"""
        # Filter data to backtest period
        self.backtest_data = {}
        for ticker, df in self.data.items():
            # Filter by date range
            mask = (df.index >= self.backtest_start) & (df.index <= self.backtest_end)
            filtered_df = df[mask].copy()
            
            if len(filtered_df) == 0:
                raise ValueError(f"{ticker}: No data in backtest period")
            
            self.backtest_data[ticker] = filtered_df
        
        # Get common trading dates (intersection of all tickers)
        date_sets = [set(df.index) for df in self.backtest_data.values()]
        self.common_dates = sorted(set.intersection(*date_sets))
        
        if len(self.common_dates) == 0:
            raise ValueError("No common trading dates across all stocks")
    
    def backtest(self, individual: Any) -> Dict[str, Any]:
        """
        Run backtest for a GP individual across all stocks.
        
        Args:
            individual: DEAP individual (GP tree)
            
        Returns:
            Dictionary containing:
            - equity_curve: Portfolio value over time
            - metrics: Performance metrics
            - per_stock_pnl: PnL contribution per stock
            - transactions: Transaction history
        """
        # Reset rebalancer
        self.rebalancer.reset()
        
        # Initialize equity curve
        equity_curve = []
        dates = []
        
        # Generate signals for each stock
        signals = self._generate_signals_for_all_stocks(individual)
        
        # Run backtest day by day
        for date in self.common_dates:
            # Get current prices
            prices = {ticker: self.backtest_data[ticker].loc[date, 'Close']
                     for ticker in self.tickers}
            
            # Process signals for each stock
            for ticker in self.tickers:
                signal = signals[ticker].get(date, 0)
                
                if signal > 0:  # Buy signal
                    self.rebalancer.handle_buy_signal(ticker, date, prices[ticker])
                elif signal < 0:  # Sell signal
                    self.rebalancer.handle_sell_signal(ticker, date, prices[ticker])
            
            # Update position values
            self.rebalancer.update_position_values(prices)
            
            # Record portfolio value
            portfolio_value = self.rebalancer.get_portfolio_value()
            equity_curve.append(portfolio_value)
            dates.append(date)
        
        # Create equity curve series
        equity_series = pd.Series(equity_curve, index=dates)
        
        # Calculate metrics
        metrics = PortfolioMetrics.calculate_portfolio_metrics(
            equity_series,
            self.initial_capital
        )
        
        # Calculate per-stock contribution
        per_stock_pnl = self._calculate_per_stock_pnl()
        
        return {
            'equity_curve': equity_series,
            'metrics': metrics,
            'per_stock_pnl': per_stock_pnl,
            'transactions': self.rebalancer.get_transaction_history(),
            'allocation_summary': self.rebalancer.get_allocation_summary()
        }
    
    def _generate_signals_for_all_stocks(self, individual: Any) -> Dict[str, Dict]:
        """
        Generate trading signals for all stocks.
        
        Args:
            individual: DEAP individual
            
        Returns:
            Dict mapping ticker to {date: signal} dict
        """
        from deap import gp
        
        signals = {}
        
        for ticker in self.tickers:
            df = self.backtest_data[ticker]
            ticker_signals = {}
            
            # Compile the individual into a callable function
            func = gp.compile(expr=individual, pset=self._get_pset())
            
            for date in self.common_dates:
                if date not in df.index:
                    ticker_signals[date] = 0
                    continue
                
                row = df.loc[date]
                
                try:
                    # Evaluate the GP tree with current market data
                    # Note: This assumes the GP tree uses specific terminal names
                    # You may need to adjust based on your primitive set
                    signal = func(
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Volume']
                    )
                    
                    # Convert to trading signal (-1, 0, 1)
                    if signal > 0:
                        ticker_signals[date] = 1  # Buy
                    elif signal < 0:
                        ticker_signals[date] = -1  # Sell
                    else:
                        ticker_signals[date] = 0  # Hold
                        
                except Exception as e:
                    # If evaluation fails, hold
                    ticker_signals[date] = 0
            
            signals[ticker] = ticker_signals
        
        return signals
    
    def _get_pset(self):
        """
        Get primitive set for GP compilation.
        This should match the primitive set used in evolution.
        
        Note: This is a placeholder. You should import the actual pset
        from your GP configuration.
        """
        # TODO: Import from gp_quant.gp.primitives
        # For now, return None and handle in _generate_signals_for_all_stocks
        return None
    
    def _calculate_per_stock_pnl(self) -> Dict[str, float]:
        """Calculate PnL contribution per stock"""
        per_stock_pnl = {}
        
        for ticker, allocation in self.rebalancer.allocations.items():
            pnl = allocation.total_value - allocation.initial_capital
            per_stock_pnl[ticker] = pnl
        
        return per_stock_pnl
    
    def get_fitness(self, individual: Any) -> float:
        """
        Calculate fitness score for an individual.
        
        Args:
            individual: DEAP individual
            
        Returns:
            Fitness score (excess return)
        """
        result = self.backtest(individual)
        return result['metrics']['excess_return']
    
    def get_pnl_curve(self, individual: Any) -> pd.Series:
        """
        Get PnL curve for an individual.
        
        Args:
            individual: DEAP individual
            
        Returns:
            Series of PnL values over time
        """
        result = self.backtest(individual)
        equity_curve = result['equity_curve']
        pnl_curve = equity_curve - self.initial_capital
        return pnl_curve
