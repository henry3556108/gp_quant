"""
Event-Driven Rebalancing Strategy for Portfolio Backtesting

This module implements event-driven capital rebalancing for multi-stock portfolios.
Capital is allocated independently to each stock and only rebalanced when trading signals occur.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class CapitalAllocation:
    """Tracks capital allocation for a single stock"""
    stock_ticker: str
    initial_capital: float
    available_cash: float
    position_value: float
    shares_held: int
    
    @property
    def total_value(self) -> float:
        """Total value = cash + position value"""
        return self.available_cash + self.position_value
    
    def __repr__(self):
        return (f"CapitalAllocation({self.stock_ticker}: "
                f"cash=${self.available_cash:.2f}, "
                f"position=${self.position_value:.2f}, "
                f"shares={self.shares_held})")


class EventDrivenRebalancer:
    """
    Event-driven rebalancing strategy for portfolio backtesting.
    
    Key Features:
    - Equal initial capital allocation to each stock
    - Independent capital pools (no cross-stock transfers)
    - Event-driven: only rebalance on buy/sell signals
    - Track all transactions for analysis
    
    Thread Safety:
    - This class is NOT thread-safe
    - Each worker should have its own instance
    """
    
    def __init__(self, 
                 tickers: List[str],
                 initial_capital: float = 100000.0,
                 equal_weight: bool = True):
        """
        Initialize rebalancer with capital allocation.
        
        Args:
            tickers: List of stock tickers
            initial_capital: Total initial capital
            equal_weight: If True, allocate capital equally to each stock
        """
        self.tickers = tickers
        self.total_initial_capital = initial_capital
        self.equal_weight = equal_weight
        
        # Initialize capital allocation for each stock
        self.allocations: Dict[str, CapitalAllocation] = {}
        self._initialize_allocations()
        
        # Transaction history
        self.transactions: List[Dict] = []
        
    def _initialize_allocations(self):
        """Initialize capital allocation for each stock"""
        if self.equal_weight:
            capital_per_stock = self.total_initial_capital / len(self.tickers)
        else:
            # Future: support custom weights
            capital_per_stock = self.total_initial_capital / len(self.tickers)
        
        for ticker in self.tickers:
            self.allocations[ticker] = CapitalAllocation(
                stock_ticker=ticker,
                initial_capital=capital_per_stock,
                available_cash=capital_per_stock,
                position_value=0.0,
                shares_held=0
            )
    
    def handle_buy_signal(self, 
                         ticker: str, 
                         date: datetime, 
                         price: float) -> Optional[Dict]:
        """
        Handle buy signal for a specific stock.
        
        Args:
            ticker: Stock ticker
            date: Trading date
            price: Current stock price
            
        Returns:
            Transaction record if trade executed, None otherwise
        """
        allocation = self.allocations[ticker]
        
        # Check if we have cash to buy
        if allocation.available_cash < price:
            return None  # Not enough cash
        
        # Calculate shares to buy (use all available cash)
        shares_to_buy = int(allocation.available_cash / price)
        
        if shares_to_buy == 0:
            return None  # Can't afford even 1 share
        
        # Execute trade
        cost = shares_to_buy * price
        allocation.available_cash -= cost
        allocation.shares_held += shares_to_buy
        allocation.position_value = allocation.shares_held * price
        
        # Record transaction
        transaction = {
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares_to_buy,
            'price': price,
            'cost': cost,
            'cash_after': allocation.available_cash,
            'position_value_after': allocation.position_value
        }
        self.transactions.append(transaction)
        
        return transaction
    
    def handle_sell_signal(self, 
                          ticker: str, 
                          date: datetime, 
                          price: float) -> Optional[Dict]:
        """
        Handle sell signal for a specific stock.
        
        Args:
            ticker: Stock ticker
            date: Trading date
            price: Current stock price
            
        Returns:
            Transaction record if trade executed, None otherwise
        """
        allocation = self.allocations[ticker]
        
        # Check if we have shares to sell
        if allocation.shares_held == 0:
            return None  # No position to sell
        
        # Sell all shares
        shares_to_sell = allocation.shares_held
        proceeds = shares_to_sell * price
        
        allocation.available_cash += proceeds
        allocation.shares_held = 0
        allocation.position_value = 0.0
        
        # Record transaction
        transaction = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares_to_sell,
            'price': price,
            'proceeds': proceeds,
            'cash_after': allocation.available_cash,
            'position_value_after': allocation.position_value
        }
        self.transactions.append(transaction)
        
        return transaction
    
    def update_position_values(self, prices: Dict[str, float]):
        """
        Update position values based on current prices.
        
        Args:
            prices: Dict mapping ticker to current price
        """
        for ticker, allocation in self.allocations.items():
            if ticker in prices and allocation.shares_held > 0:
                allocation.position_value = allocation.shares_held * prices[ticker]
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value across all stocks"""
        return sum(alloc.total_value for alloc in self.allocations.values())
    
    def get_allocation_summary(self) -> pd.DataFrame:
        """Get summary of current allocations"""
        data = []
        for ticker, alloc in self.allocations.items():
            data.append({
                'ticker': ticker,
                'cash': alloc.available_cash,
                'position_value': alloc.position_value,
                'total_value': alloc.total_value,
                'shares': alloc.shares_held,
                'allocation_pct': alloc.total_value / self.get_portfolio_value() * 100
            })
        return pd.DataFrame(data)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        if not self.transactions:
            return pd.DataFrame()
        return pd.DataFrame(self.transactions)
    
    def reset(self):
        """Reset rebalancer to initial state"""
        self._initialize_allocations()
        self.transactions = []
