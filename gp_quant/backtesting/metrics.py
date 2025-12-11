"""
Portfolio Performance Metrics

This module calculates various performance metrics for portfolio backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class PortfolioMetrics:
    """
    Calculate portfolio performance metrics.
    
    Thread Safety:
    - All methods are stateless and thread-safe
    - Can be used concurrently by multiple workers
    """
    
    @staticmethod
    def calculate_pnl(initial_value: float, final_value: float) -> float:
        """Calculate profit and loss"""
        return final_value - initial_value
    
    @staticmethod
    def calculate_return(initial_value: float, final_value: float) -> float:
        """Calculate simple return"""
        if initial_value == 0:
            return 0.0
        return (final_value - initial_value) / initial_value
    
    @staticmethod
    def calculate_excess_return(portfolio_return: float, 
                               benchmark_return: float = 0.0) -> float:
        """
        Calculate excess return over benchmark.
        
        Args:
            portfolio_return: Portfolio return
            benchmark_return: Benchmark return (default: 0 for risk-free rate)
            
        Returns:
            Excess return
        """
        return portfolio_return - benchmark_return
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of portfolio values over time
            
        Returns:
            Maximum drawdown (negative value)
        """
        if len(equity_curve) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, 
                            periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Annualized volatility
        """
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_calmar_ratio(total_return: float, 
                              max_drawdown: float,
                              years: float) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            total_return: Total return over period
            max_drawdown: Maximum drawdown (negative value)
            years: Number of years
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return 0.0
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return / abs(max_drawdown)
    
    @staticmethod
    def calculate_win_rate(returns: pd.Series) -> float:
        """
        Calculate win rate (percentage of positive returns).
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate (0 to 1)
        """
        if len(returns) == 0:
            return 0.0
        
        return (returns > 0).sum() / len(returns)
    
    @staticmethod
    def calculate_portfolio_metrics(equity_curve: pd.Series,
                                   initial_capital: float,
                                   risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            equity_curve: Series of portfolio values over time
            initial_capital: Initial portfolio capital
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary of metrics
        """
        if len(equity_curve) == 0:
            return {
                'total_return': 0.0,
                'excess_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'volatility': 0.0,
                'calmar_ratio': 0.0,
                'win_rate': 0.0,
                'final_value': initial_capital
            }
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Total return
        final_value = equity_curve.iloc[-1]
        total_return = PortfolioMetrics.calculate_return(initial_capital, final_value)
        
        # Excess return
        excess_return = PortfolioMetrics.calculate_excess_return(
            total_return, risk_free_rate
        )
        
        # Sharpe ratio
        sharpe = PortfolioMetrics.calculate_sharpe_ratio(
            returns, risk_free_rate
        )
        
        # Max drawdown
        max_dd = PortfolioMetrics.calculate_max_drawdown(equity_curve)
        
        # Volatility
        vol = PortfolioMetrics.calculate_volatility(returns)
        
        # Calmar ratio
        years = len(equity_curve) / 252  # Assuming daily data
        calmar = PortfolioMetrics.calculate_calmar_ratio(
            total_return, max_dd, years
        )
        
        # Win rate
        win_rate = PortfolioMetrics.calculate_win_rate(returns)
        
        # Calculate Sterling Ratio
        sterling = PortfolioMetrics.calculate_sterling_ratio(total_return, max_dd)

        return {
            'total_return': total_return,
            'excess_return': excess_return,
            'sharpe_ratio': sharpe,
            'sterling_ratio': sterling,
            'max_drawdown': max_dd,
            'volatility': vol,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'final_value': final_value,
            'pnl': final_value - initial_capital
        }

    @staticmethod
    def calculate_sterling_ratio(total_return: float, 
                                max_drawdown: float,
                                risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sterling Ratio (Total Return / |Max Drawdown|).
        
        Note: Some definitions use Annualized Return / |Max Drawdown|.
        Here we use Total Return as per user specification, but it's easily adaptable.
        If Max Drawdown is 0, returns a large number or 0 depending on return.
        
        Args:
            total_return: Total return over period
            max_drawdown: Maximum drawdown (negative value)
            risk_free_rate: Risk free rate (subtracted from return if needed, usually 0 for Sterling)
            
        Returns:
            Sterling ratio
        """
        if max_drawdown == 0:
            return 0.0 if total_return <= 0 else 100.0  # Avoid division by zero
            
        return (total_return - risk_free_rate) / abs(max_drawdown)

    @staticmethod
    def calculate_shrinkage(train_metric: float, test_metric: float) -> float:
        """
        Calculate Shrinkage ((Train - Test) / Train).
        
        Args:
            train_metric: Metric value in training period
            test_metric: Metric value in testing period
            
        Returns:
            Shrinkage percentage (e.g., 0.20 for 20% drop)
        """
        if train_metric == 0:
            return 0.0
            
        return (train_metric - test_metric) / train_metric


