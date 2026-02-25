"""
Portfolio Backtesting Engine

This module implements multi-stock portfolio backtesting with event-driven rebalancing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from gp_quant.backtesting.rebalancing import EventDrivenRebalancer
from gp_quant.backtesting.metrics import PortfolioMetrics


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
                 initial_capital: float = 100000.0,
                 pset=None):
        """
        Initialize portfolio backtesting engine.
        
        Args:
            data: Dict mapping ticker to price DataFrame
                  Each DataFrame should have DatetimeIndex and OHLCV columns
            backtest_start: Start date for backtesting (YYYY-MM-DD)
            backtest_end: End date for backtesting (YYYY-MM-DD)
            initial_capital: Initial portfolio capital
            pset: DEAP primitive set (optional, will import default if None)
            
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
        
        # Store or import pset and make a deep copy
        # This is CRITICAL: we need our own copy to avoid race conditions
        # and to properly set terminal values
        import copy
        if pset is None:
            from gp_quant.evolution.components.gp.operators import pset as default_pset
            self.pset = copy.deepcopy(default_pset)
        else:
            self.pset = copy.deepcopy(pset)
        
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
            
            # Drop rows with NaN values (holidays, missing data)
            filtered_df = filtered_df.dropna()
            
            if len(filtered_df) == 0:
                raise ValueError(f"{ticker}: No valid data after removing NaN")
            
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
        for i, date in enumerate(self.common_dates):
            # Get current prices
            prices = {ticker: self.backtest_data[ticker].loc[date, 'Close']
                     for ticker in self.tickers}
            
            # Determine signal date (T-1) for Next Day Execution
            if i > 0:
                prev_date = self.common_dates[i-1]
            else:
                prev_date = None

            # Process signals for each stock
            for ticker in self.tickers:
                # Get signal from PREVIOUS trading day (T-1)
                # This implements "Signal on Close -> Trade on Next Open/Close"
                if prev_date:
                    signal = signals[ticker].get(prev_date, 0)
                else:
                    signal = 0 # No signal on first day
                
                allocation = self.rebalancer.allocations[ticker]
                
                # 買入邏輯：訊號為 1 且目前沒有持股
                if signal > 0 and allocation.shares_held == 0:
                    self.rebalancer.handle_buy_signal(ticker, date, prices[ticker])
                # 賣出邏輯：訊號為 0 且目前有持股
                elif signal == 0 and allocation.shares_held > 0:
                    self.rebalancer.handle_sell_signal(ticker, date, prices[ticker])
            
            # Update position values
            self.rebalancer.update_position_values(prices)
            
            # Record portfolio value
            portfolio_value = self.rebalancer.get_portfolio_value()
            equity_curve.append(portfolio_value)
            dates.append(date)

        # At the end of backtest, force liquidation of any remaining positions
        # using the final day's close price so that entry_exit_points will have
        # matching SELL transactions for long positions that are held to expiry.
        if self.common_dates:
            final_date = self.common_dates[-1]
            final_prices = {ticker: self.backtest_data[ticker].loc[final_date, 'Close']
                            for ticker in self.tickers}
            for ticker in self.tickers:
                allocation = self.rebalancer.allocations[ticker]
                if allocation.shares_held > 0:
                    # This SELL will not change total portfolio value because we
                    # are using the same close price already used in
                    # update_position_values, but it will create an explicit
                    # closing transaction.
                    self.rebalancer.handle_sell_signal(ticker, final_date, final_prices[ticker])

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
            'allocation_summary': self.rebalancer.get_allocation_summary(),
            'signals': signals
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
        
        # Get NumVector type from pset's terminals keys
        # This ensures we use the SAME NumVector class that was used to create pset
        NumVector = None
        for key in self.pset.terminals.keys():
            if hasattr(key, '__name__') and 'NumVector' in key.__name__:
                NumVector = key
                break
        
        if NumVector is None:
            raise ValueError("NumVector type not found in pset.terminals")
        
        signals = {}
        
        for ticker in self.tickers:
            # Use FULL data (self.data) for technical indicator calculation
            # This ensures we have enough historical data for indicators like ROC(170)
            df = self.data[ticker]
            
            # Prepare price and volume vectors from FULL data
            price_vec = df['Close'].values.astype(float)
            volume_vec = df['Volume'].values.astype(float)
            
            # Convert to NumVector type
            price_vec = price_vec.view(NumVector)
            volume_vec = volume_vec.view(NumVector)
            
            try:
                # Set terminal values BEFORE compiling (like BacktestingEngine does)
                self.pset.terminals[NumVector][0].value = price_vec
                self.pset.terminals[NumVector][1].value = volume_vec
                
                # Compile the individual
                # Use str(individual) to avoid potential pickling issues with Ephemeral constants
                # or mismatched terminal objects.
                expr = str(individual)
                func = gp.compile(expr=expr, pset=self.pset)
                
                # Execute - try without arguments first
                try:
                    signal_vector = func()
                except TypeError as e:
                    # If it needs arguments, provide them
                    if "missing" in str(e) and "required positional arguments" in str(e):
                        signal_vector = func(price_vec, volume_vec)
                    else:
                        raise
                
                # Handle single boolean return (scalar broadcasting)
                # Handle scalar outputs (boolean or numpy scalar)
                if isinstance(signal_vector, (bool, np.bool_)):
                    signal_vector = np.full(len(price_vec), signal_vector, dtype=bool)
                elif isinstance(signal_vector, np.ndarray):
                    # Handle 0-d array or 1-element array
                    if signal_vector.ndim == 0 or signal_vector.size == 1:
                        # Extract scalar value
                        scalar_val = signal_vector.item() if signal_vector.ndim == 0 else signal_vector[0]
                        signal_vector = np.full(len(price_vec), bool(scalar_val), dtype=bool)
                else:
                    # Fallback for other scalar types (float, int, etc.)
                    signal_vector = np.full(len(price_vec), bool(signal_vector), dtype=bool)
                
                # Convert boolean vector to trading signals
                # 使用持續持有邏輯：True = 持有多頭 (1), False = 空倉 (0)
                ticker_signals = {}
                common_dates_set = set(self.common_dates)
                
                for i, date in enumerate(df.index):
                    if date in common_dates_set:
                        # True = 持有多頭，False = 空倉
                        ticker_signals[date] = 1 if signal_vector[i] else 0
                
                signals[ticker] = ticker_signals
                
            except Exception as e:
                # If evaluation fails, no signals
                import traceback
                print(f"❌ Error evaluating individual for {ticker}: {e}")
                traceback.print_exc()
                ticker_signals = {date: 0 for date in self.common_dates}
                signals[ticker] = ticker_signals
        
        return signals
    
    def _get_pset(self):
        """
        Get primitive set for GP compilation.
        This should match the primitive set used in evolution.
        """
        return self.pset
    
    def _calculate_per_stock_pnl(self) -> Dict[str, float]:
        """Calculate PnL contribution per stock"""
        per_stock_pnl = {}
        
        for ticker, allocation in self.rebalancer.allocations.items():
            pnl = allocation.total_value - allocation.initial_capital
            per_stock_pnl[ticker] = pnl
        
        return per_stock_pnl
    
    def calculate_consistent_sharpe(self, equity_curve: pd.Series, **kwargs) -> float:
        """
        Calculate Consistent Sharpe Ratio: Mean(Annual Sharpes) - lambda * Std(Annual Sharpes)
        
        Args:
            equity_curve: Portfolio equity curve
            **kwargs: Parameters (e.g., 'lambda')
            
        Returns:
            Consistent Sharpe score
        """
        # Get parameters
        lambda_param = kwargs.get('lambda', 1.0)
        
        # Edge case: insufficient data
        if len(equity_curve) < 252: # Require at least ~1 year of data
            return 0.0
            
        # Calculate daily returns
        returns = equity_curve.pct_change().dropna()
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0.0
            
        # Group by year
        annual_sharpes = []
        years = returns.index.year.unique()
        
        for year in years:
            year_returns = returns[returns.index.year == year]
            
            # Skip years with too few data points (e.g. < 100 days)
            if len(year_returns) < 100:
                continue
                
            mean_ret = year_returns.mean()
            std_ret = year_returns.std()
            
            if std_ret == 0 or not np.isfinite(std_ret):
                sharpe = 0.0
            else:
                sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252))
                
            # Clip extreme values
            sharpe = max(min(sharpe, 10.0), -10.0)
            annual_sharpes.append(sharpe)
        
        if not annual_sharpes:
            return 0.0
            
        # Calculate fitness
        mean_sharpe = np.mean(annual_sharpes)
        std_sharpe = np.std(annual_sharpes)
        
        fitness = mean_sharpe - (lambda_param * std_sharpe)
        
        # Sanity check
        if not np.isfinite(fitness):
            return -100000.0
            
        return fitness
    
    def get_fitness(self, individual: Any, fitness_metric: str = 'excess_return', fitness_params: Dict[str, Any] = None) -> float:
        """
        Calculate fitness score for an individual.
        
        Args:
            individual: DEAP individual
            fitness_metric: 'excess_return', 'sharpe_ratio', or 'consistent_sharpe'
            fitness_params: Optional parameters for fitness calculation
            
        Returns:
            Fitness score
        """
        if fitness_metric == 'sharpe_ratio':
            # Calculate Sharpe Ratio from equity curve
            result = self.backtest(individual)
            equity_curve = result['equity_curve']
            
            # Edge case: insufficient data
            if len(equity_curve) < 2:
                return 0.0
            
            # Calculate returns
            returns = equity_curve.pct_change().dropna()
            
            if len(returns) == 0:
                return 0.0
            
            # Filter NaN and Inf
            returns = returns[np.isfinite(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            # Calculate Sharpe
            mean_return = returns.mean()
            std_return = returns.std()
            
            if std_return == 0 or not np.isfinite(std_return):
                return 0.0
            
            sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
            
            # Sanity check
            if not np.isfinite(sharpe) or sharpe > 10 or sharpe < -10:
                return -100000.0
            
            return sharpe
            
        elif fitness_metric == 'consistent_sharpe':
            result = self.backtest(individual)
            equity_curve = result['equity_curve']
            params = fitness_params or {}
            return self.calculate_consistent_sharpe(equity_curve, **params)
            
        else:
            # Default: excess return
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
    
    def get_signals(self, individual: Any) -> np.ndarray:
        """
        Get trading signals for an individual as a flat boolean array.
        
        Generates signals for all tickers and concatenates them into
        a single array for comparison purposes.
        
        Args:
            individual: DEAP individual
            
        Returns:
            np.ndarray: Boolean array of shape (n_tickers * n_dates,)
        """
        signals_dict = self._generate_signals_for_all_stocks(individual)
        
        # Flatten signals from all tickers into a single array
        all_signals = []
        for ticker in sorted(self.tickers):  # Sort for consistent ordering
            ticker_signals = signals_dict.get(ticker, {})
            for date in self.common_dates:
                signal = ticker_signals.get(date, 0)
                all_signals.append(bool(signal))
        
        return np.array(all_signals, dtype=bool)

