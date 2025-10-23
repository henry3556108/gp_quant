"""
Backtesting Engine for Fitness Evaluation

This module provides the core engine for simulating a trading strategy defined
by a GP-evolved rule over historical data. It calculates the performance of the
rule and returns a fitness score (excess return over buy-and-hold).
"""
import pandas as pd
import numpy as np
import numba
from deap import gp
from typing import Callable, Dict, List

from gp_quant.gp.operators import pset, NumVector


@numba.jit(nopython=True)
def _numba_simulation_loop(signals, open_prices, close_prices, initial_capital):
    """
    A Numba-JIT compiled function to run the trading simulation at high speed.
    This function only works with NumPy arrays.
    """
    position = 0  # 0 for None, 1 for LONG
    capital = initial_capital
    shares = 0.0

    for i in range(len(signals)):
        signal = signals[i]
        next_day_open_price = open_prices[i + 1]

        if position == 0 and signal == True:  # Buy signal
            if next_day_open_price > 0:
                shares = capital / next_day_open_price
                capital = 0.0
                position = 1
        elif position == 1 and signal == False:  # Sell signal
            capital = shares * next_day_open_price
            shares = 0.0
            position = 0

    # Calculate final capital
    if position == 1:
        capital = shares * close_prices[-1]

    return capital - initial_capital


class BacktestingEngine:
    """
    A class to simulate a trading strategy and evaluate its fitness.
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0,
                 backtest_start: str = None, backtest_end: str = None):
        """
        Initializes the backtesting engine.

        Args:
            data: A DataFrame with historical market data (OHLCV). Should include
                  initial period for technical indicator calculation.
            initial_capital: The starting capital for the simulation.
            backtest_start: Start date for return calculation (optional).
                           If None, uses entire data period.
            backtest_end: End date for return calculation (optional).
                         If None, uses entire data period.
        """
        self.data = data
        self.initial_capital = initial_capital
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end
        
        # Determine the backtest data range
        if self.backtest_start or self.backtest_end:
            # Adjust backtest_start if it's before data starts
            actual_backtest_start = self.backtest_start
            if self.backtest_start and pd.Timestamp(self.backtest_start) < self.data.index[0]:
                actual_backtest_start = self.data.index[0].strftime('%Y-%m-%d')
                print(f"  Warning: backtest_start ({self.backtest_start}) is before data start ({self.data.index[0].date()})")
                print(f"  Using actual data start: {actual_backtest_start}")
                self.backtest_start = actual_backtest_start
            
            # Create mask for backtest period
            mask = pd.Series(True, index=self.data.index)
            if self.backtest_start:
                mask &= (self.data.index >= self.backtest_start)
            if self.backtest_end:
                mask &= (self.data.index <= self.backtest_end)
            self.backtest_data = self.data[mask]
        else:
            # Use entire data for backward compatibility
            self.backtest_data = self.data
        
        # Create a copy of pset to avoid modifying the global one
        import copy
        self.pset = copy.deepcopy(pset)

    def evaluate(self, individual: gp.PrimitiveTree, fitness_metric: str = 'excess_return') -> tuple[float]:
        """
        Evaluates the fitness of a single GP individual using vectorization.
        
        Args:
            individual: GP tree to evaluate
            fitness_metric: 'excess_return' or 'sharpe_ratio'
        
        Returns:
            Tuple containing fitness score
        """
        # Inject the full data vectors into the pset terminals for NumVector type
        price_vec = self.data['Close'].to_numpy()
        volume_vec = self.data['Volume'].to_numpy()
        self.pset.terminals[NumVector][0].value = price_vec
        self.pset.terminals[NumVector][1].value = volume_vec

        try:
            # --- Step 1: Compile and execute the rule ---
            rule: Callable = gp.compile(expr=individual, pset=self.pset)
            signals = rule()

            # --- Step 2: Sanitize the output ---
            # A simple rule (e.g., a V_TRUE terminal) might return a single boolean.
            if not isinstance(signals, np.ndarray):
                signals = np.full(self.data.shape[0], signals, dtype=np.bool_)
            
            # Check for and handle numerical instability (inf, nan)
            if not np.all(np.isfinite(signals)):
                # print(f"\n[DIAGNOSTIC] Sanitizing non-finite signals for: {individual}") # Uncomment for debug
                signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        except TypeError as e:
            if "missing" in str(e) and "required positional arguments" in str(e):
                # This means the individual expects arguments (loaded from file)
                try:
                    rule: Callable = gp.compile(expr=individual, pset=self.pset)
                    signals = rule(price_vec, volume_vec)
                    # Handle single boolean return from simple individuals
                    if not isinstance(signals, np.ndarray):
                        signals = np.full(self.data.shape[0], signals, dtype=np.bool_)
                    
                    # Check for and handle numerical instability (inf, nan)
                    if not np.all(np.isfinite(signals)):
                        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
                        
                except Exception as e2:
                    # print(f"[ERROR] Could not evaluate individual with arguments {individual}: {e2}") # Uncomment for debug
                    return -100000.0,
            else:
                # print(f"[ERROR] TypeError in individual {individual}: {e}") # Uncomment for debug
                return -100000.0,
        except (OverflowError, ValueError, FloatingPointError, Exception) as e:
            # If ANY other error occurs during compilation or signal generation
            # we assign a penalty fitness and stop processing this individual.
            # print(f"[ERROR] Could not evaluate individual {individual}: {e}") # Uncomment for debug
            return -100000.0,

        # Ensure signals is a boolean array for the simulation loop
        signals = signals.astype(np.bool_)

        # --- Run Simulation with the pre-calculated signal vector ---
        # Only use signals from backtest period
        if self.backtest_start or self.backtest_end:
            # Get the mask for backtest period
            mask = pd.Series(False, index=self.data.index)
            if self.backtest_start:
                mask |= (self.data.index >= self.backtest_start)
            if self.backtest_end:
                mask &= (self.data.index <= self.backtest_end)
            backtest_signals = signals[mask.values]
        else:
            backtest_signals = signals
        
        gp_return = self._run_vectorized_simulation(backtest_signals, self.backtest_data)

        # --- Calculate Buy-and-Hold Return ---
        # Calculate B&H return only for backtest period
        start_price = self.backtest_data['Close'].iloc[0]
        end_price = self.backtest_data['Close'].iloc[-1]
        if start_price > 0:
            buy_and_hold_return = (end_price / start_price - 1) * self.initial_capital
        else:
            buy_and_hold_return = 0

        # --- Calculate Fitness based on metric ---
        if fitness_metric == 'sharpe_ratio':
            # Use Sharpe Ratio as fitness
            sharpe = self._calculate_sharpe_ratio(backtest_signals, self.backtest_data)
            return sharpe,
        else:
            # Default: Excess Return
            excess_return = gp_return - buy_and_hold_return

            # --- Final Fitness Sanity Check ---
            # Fitness should be within economically reasonable bounds
            # Based on PRD: training period average return was 131.17%, so 1000x is very generous
            MAX_REASONABLE_FITNESS = self.initial_capital * 1000
            MIN_REASONABLE_FITNESS = -self.initial_capital * 2
            
            if not np.isfinite(excess_return) or \
               excess_return > MAX_REASONABLE_FITNESS or \
               excess_return < MIN_REASONABLE_FITNESS:
                # print(f"[WARNING] Unreasonable fitness detected: {excess_return:.2e}, assigning penalty") # Uncomment for debug
                return -100000.0,

            # print(f"[DIAGNOSTIC] Final fitness for {individual}: {excess_return:.2f}") # Uncomment for extreme verbosity
            
            return excess_return,

    def _calculate_sharpe_ratio(self, signals: np.ndarray, data: pd.DataFrame, 
                                risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe Ratio for the trading strategy.
        
        Args:
            signals: Boolean array of trading signals
            data: DataFrame with price data
            risk_free_rate: Annual risk-free rate (default: 0.0)
        
        Returns:
            Annualized Sharpe Ratio, or 0.0 for edge cases, or -100000.0 for penalties
        """
        # Get equity curve
        equity_curve = self._run_simulation_with_equity_curve(signals, data)
        
        # Edge case: No trades or insufficient data
        if len(equity_curve) < 2:
            return 0.0  # Neutral fitness for no-trade strategies
        
        # Calculate daily returns
        returns = equity_curve.pct_change().dropna()
        
        # Edge case: No valid returns
        if len(returns) == 0:
            return 0.0
        
        # Filter out NaN and Inf
        returns = returns[np.isfinite(returns)]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate mean and std
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Edge case: Zero volatility (no trading or constant value)
        if std_return == 0 or not np.isfinite(std_return):
            return 0.0
        
        # Calculate annualized Sharpe Ratio (assuming 252 trading days)
        sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
        
        # Sanity check: Sharpe should be reasonable
        if not np.isfinite(sharpe):
            return -100000.0  # Penalty for numerical errors
        
        # Extreme values are likely errors
        if sharpe > 10 or sharpe < -10:
            return -100000.0  # Penalty for unrealistic Sharpe
        
        return sharpe
    
    def _run_simulation_with_equity_curve(self, signals: np.ndarray, 
                                         data: pd.DataFrame) -> pd.Series:
        """
        Run simulation and return daily equity curve.
        
        Args:
            signals: Boolean array of trading signals
            data: DataFrame with OHLC data
        
        Returns:
            Series of daily portfolio values indexed by date
        """
        position = 0  # 0 for no position, 1 for long
        capital = self.initial_capital
        shares = 0.0
        equity_values = []
        
        open_prices = data['Open'].to_numpy()
        close_prices = data['Close'].to_numpy()
        
        for i in range(len(signals)):
            signal = signals[i]
            
            # Execute trades at next day's open (if not last day)
            if i < len(signals) - 1:
                next_open = open_prices[i + 1]
                
                if position == 0 and signal == True:  # Buy signal
                    if next_open > 0:
                        shares = capital / next_open
                        capital = 0.0
                        position = 1
                elif position == 1 and signal == False:  # Sell signal
                    capital = shares * next_open
                    shares = 0.0
                    position = 0
            
            # Record equity at end of day (close price)
            current_equity = capital + shares * close_prices[i]
            equity_values.append(current_equity)
        
        return pd.Series(equity_values, index=data.index)
    
    def _run_vectorized_simulation(self, signals: np.ndarray, data: pd.DataFrame = None) -> float:
        """
        Runs the simulation using the fast Numba JIT-compiled loop.
        
        Args:
            signals: Boolean array of trading signals
            data: DataFrame to use for simulation. If None, uses self.data
        """
        if data is None:
            data = self.data
            
        # Ensure signals is always an array to prevent Numba typing errors
        if not hasattr(signals, '__len__'):
            # If signal is a single boolean, broadcast it to the shape of the data
            signals = np.full(data.shape[0], signals, dtype=np.bool_)

        open_prices_np = data['Open'].to_numpy()
        close_prices_np = data['Close'].to_numpy()

        # The first call to a JIT function is slow due to compilation.
        # Subsequent calls are fast.
        gp_return = _numba_simulation_loop(
            signals,
            open_prices_np,
            close_prices_np,
            self.initial_capital
        )
        
        # Check if return is reasonable (within 1000x of initial capital)
        # This catches numerical overflow in the simulation loop
        MAX_REASONABLE_RETURN = self.initial_capital * 1000
        MIN_REASONABLE_RETURN = -self.initial_capital * 2  # Allow some leverage losses
        
        if not np.isfinite(gp_return) or \
           gp_return > MAX_REASONABLE_RETURN or \
           gp_return < MIN_REASONABLE_RETURN:
            # Return worst case scenario (lose all capital)
            return -self.initial_capital
        
        return gp_return

    def run_detailed_simulation(self, individual: gp.PrimitiveTree) -> dict:
        """
        Runs a full simulation and returns detailed trade logs and performance metrics.
        Only records trades within the backtest period.
        """
        signals = self.get_signals(individual)
        if len(signals) == 0:
            return {
                'gp_return': 0,
                'buy_and_hold_return': 0,
                'trades': [],
                'error': 'Could not generate signals from individual.'
            }

        # Use backtest_data for simulation
        data_to_use = self.backtest_data
        
        # Get signals for backtest period only
        if self.backtest_start or self.backtest_end:
            mask = pd.Series(False, index=self.data.index)
            if self.backtest_start:
                mask |= (self.data.index >= self.backtest_start)
            if self.backtest_end:
                mask &= (self.data.index <= self.backtest_end)
            backtest_signals = signals[mask.values]
        else:
            backtest_signals = signals

        trades = []
        position = 0
        capital = self.initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date = None

        open_prices = data_to_use['Open'].to_numpy()
        dates = data_to_use.index

        for i in range(len(backtest_signals) - 1):
            signal = backtest_signals[i]
            next_day_open_price = open_prices[i + 1]

            if position == 0 and signal == True and capital > 0:
                shares = capital / next_day_open_price
                capital = 0.0
                position = 1
                entry_price = next_day_open_price
                entry_date = dates[i + 1]

            elif position == 1 and signal == False:
                capital = shares * next_day_open_price
                pnl = (next_day_open_price - entry_price) * shares
                trades.append({
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': dates[i + 1].strftime('%Y-%m-%d'),
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(next_day_open_price, 2),
                    'shares': round(shares, 2),
                    'pnl': round(pnl, 2)
                })
                shares = 0.0
                position = 0

        if position == 1:
            last_close_price = data_to_use['Close'].iloc[-1]
            capital = shares * last_close_price
            pnl = (last_close_price - entry_price) * shares
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': data_to_use.index[-1].strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 2),
                'exit_price': round(last_close_price, 2),
                'shares': round(shares, 2),
                'pnl': round(pnl, 2)
            })

        gp_total_return = capital - self.initial_capital

        # Calculate B&H for backtest period
        start_price = data_to_use['Close'].iloc[0]
        end_price = data_to_use['Close'].iloc[-1]
        buy_and_hold_return = (end_price / start_price - 1) * self.initial_capital if start_price > 0 else 0

        return {
            'gp_return': gp_total_return,
            'buy_and_hold_return': buy_and_hold_return,
            'trades': trades
        }

    def get_signals(self, individual: gp.PrimitiveTree) -> np.ndarray:
        """
        Extract trading signals from a GP individual without running full evaluation.
        Returns the boolean signal array.
        """
        price_vec = self.data['Close'].to_numpy()
        volume_vec = self.data['Volume'].to_numpy()

        try:
            # Try the pset approach first (for individuals created during evolution)
            self.pset.terminals[NumVector][0].value = price_vec
            self.pset.terminals[NumVector][1].value = volume_vec
            rule: Callable = gp.compile(expr=individual, pset=self.pset)
            signals = rule()
            # Handle single boolean return from simple individuals (e.g., V_TRUE)
            if not isinstance(signals, np.ndarray):
                signals = np.full(self.data.shape[0], signals, dtype=np.bool_)
            
            signals = signals.astype(np.bool_)
            return signals
        except TypeError as e:
            if "missing" in str(e) and "required positional arguments" in str(e):
                # This means the individual expects arguments (loaded from file)
                try:
                    print("Trying with arguments...")
                    rule: Callable = gp.compile(expr=individual, pset=self.pset)
                    signals = rule(price_vec, volume_vec)
                    # Handle single boolean return from simple individuals
                    if not isinstance(signals, np.ndarray):
                        signals = np.full(self.data.shape[0], signals, dtype=np.bool_)

                    signals = signals.astype(np.bool_)
                    return signals
                except Exception as e2:
                    print(f"Error with arguments: {e2}")
                    return np.array([], dtype=bool)
            else:
                print(f"Other TypeError: {e}")
                return np.array([], dtype=bool)
        except Exception as e:
            print(f"Error getting signals from individual: {e}")
            return np.array([], dtype=bool)
    
    def get_pnl_curve(self, individual: gp.PrimitiveTree) -> pd.Series:
        """
        Generate the cumulative PnL curve for an individual over the backtest period.
        
        Args:
            individual: A GP tree representing the trading rule
            
        Returns:
            pd.Series: Cumulative PnL indexed by date (backtest period only)
                      Returns empty Series if evaluation fails
        """
        # Get signals for the individual
        signals = self.get_signals(individual)
        if len(signals) == 0:
            return pd.Series(dtype=float)
        
        # Use backtest_data for simulation
        data_to_use = self.backtest_data
        
        # Get signals for backtest period only
        if self.backtest_start or self.backtest_end:
            mask = pd.Series(False, index=self.data.index)
            if self.backtest_start:
                mask |= (self.data.index >= self.backtest_start)
            if self.backtest_end:
                mask &= (self.data.index <= self.backtest_end)
            backtest_signals = signals[mask.values]
        else:
            backtest_signals = signals
        
        # Simulate trading and track daily PnL
        position = 0
        capital = self.initial_capital
        shares = 0.0
        
        open_prices = data_to_use['Open'].to_numpy()
        close_prices = data_to_use['Close'].to_numpy()
        dates = data_to_use.index
        
        # Track cumulative PnL at each day
        pnl_curve = np.zeros(len(backtest_signals))
        
        for i in range(len(backtest_signals) - 1):
            signal = backtest_signals[i]
            next_day_open_price = open_prices[i + 1]
            
            # Execute trades
            if position == 0 and signal == True and capital > 0:
                # Buy
                shares = capital / next_day_open_price
                capital = 0.0
                position = 1
            elif position == 1 and signal == False:
                # Sell
                capital = shares * next_day_open_price
                shares = 0.0
                position = 0
            
            # Calculate current portfolio value
            if position == 1:
                current_value = shares * close_prices[i + 1]
            else:
                current_value = capital
            
            # Cumulative PnL = current value - initial capital
            pnl_curve[i + 1] = current_value - self.initial_capital
        
        # Handle final position
        if position == 1:
            final_value = shares * close_prices[-1]
            pnl_curve[-1] = final_value - self.initial_capital
        
        # Return as pandas Series with dates
        pnl_series = pd.Series(pnl_curve, index=dates)
        
        # Handle NaN values (replace with forward fill, then 0)
        if pnl_series.isna().any():
            pnl_series = pnl_series.ffill().fillna(0)
        
        return pnl_series


class PortfolioBacktestingEngine:
    """
    A class to simulate a portfolio trading strategy across multiple tickers.
    
    This engine evaluates a single GP individual on multiple stocks simultaneously,
    with equal capital allocation to each ticker. The fitness is the sum of excess
    returns across all tickers.
    """
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], total_capital: float = 100000.0,
                 backtest_config: Dict[str, Dict] = None):
        """
        Initializes the portfolio backtesting engine.
        
        Args:
            data_dict: Dictionary mapping ticker symbols to their DataFrames
            total_capital: Total initial capital to be split equally among tickers
            backtest_config: Optional dictionary mapping ticker to backtest configuration:
                            {ticker: {'backtest_start': str, 'backtest_end': str}}
        """
        self.data_dict = data_dict
        self.total_capital = total_capital
        self.tickers = list(data_dict.keys())
        self.n_tickers = len(self.tickers)
        self.backtest_config = backtest_config or {}
        
        # Equal weight allocation
        self.capital_per_ticker = total_capital / self.n_tickers
        
        # Create a BacktestingEngine for each ticker
        self.engines = {}
        for ticker, data in data_dict.items():
            # Get backtest configuration for this ticker
            config = self.backtest_config.get(ticker, {})
            backtest_start = config.get('backtest_start', None)
            backtest_end = config.get('backtest_end', None)
            
            self.engines[ticker] = BacktestingEngine(
                data, 
                self.capital_per_ticker,
                backtest_start=backtest_start,
                backtest_end=backtest_end
            )
        
        print(f"Portfolio initialized with {self.n_tickers} tickers")
        print(f"Capital per ticker: ${self.capital_per_ticker:,.2f}")
    
    def evaluate(self, individual: gp.PrimitiveTree, fitness_metric: str = 'excess_return') -> tuple[float]:
        """
        Evaluates the fitness of a GP individual across all tickers in the portfolio.
        
        Args:
            individual: A GP tree representing the trading rule
            fitness_metric: Fitness calculation method:
                - 'excess_return': Sum of excess returns (default)
                - 'sharpe_ratio': Portfolio-level Sharpe Ratio
                - 'avg_sharpe': Average of individual Sharpe Ratios
            
        Returns:
            A tuple containing the portfolio fitness
        """
        if fitness_metric == 'sharpe_ratio':
            # Portfolio-level Sharpe: combine all equity curves
            return self._calculate_portfolio_sharpe(individual),
        
        elif fitness_metric == 'avg_sharpe':
            # Average of individual Sharpe Ratios
            total_sharpe = 0.0
            valid_count = 0
            
            for ticker in self.tickers:
                engine = self.engines[ticker]
                sharpe = engine.evaluate(individual, fitness_metric='sharpe_ratio')[0]
                
                # Only count valid Sharpe values (not penalties)
                if sharpe > -10000:  # Not a penalty value
                    total_sharpe += sharpe
                    valid_count += 1
            
            if valid_count == 0:
                return 0.0,  # No valid strategies
            
            avg_sharpe = total_sharpe / valid_count
            return avg_sharpe,
        
        else:
            # Default: Sum of excess returns
            total_excess_return = 0.0
            ticker_results = {}
            
            for ticker in self.tickers:
                # Evaluate the individual on this ticker
                engine = self.engines[ticker]
                excess_return = engine.evaluate(individual, fitness_metric='excess_return')[0]
                
                ticker_results[ticker] = excess_return
                total_excess_return += excess_return
            
            # Optional: Print detailed results for debugging
            # print(f"Portfolio evaluation: {ticker_results}, Total: {total_excess_return:.2f}")
            
            return total_excess_return,
    
    def _calculate_portfolio_sharpe(self, individual: gp.PrimitiveTree) -> float:
        """
        Calculate portfolio-level Sharpe Ratio by combining equity curves.
        
        Args:
            individual: GP tree to evaluate
        
        Returns:
            Portfolio Sharpe Ratio
        """
        # Get equity curves from all tickers
        equity_curves = []
        
        # Compile the GP tree once (not per ticker) for performance
        # Use the pset from the first engine (all engines share the same pset)
        first_engine = self.engines[self.tickers[0]]
        try:
            rule = gp.compile(expr=individual, pset=first_engine.pset)
        except Exception as e:
            return -100000.0,  # Penalty for compilation errors (return tuple!)
        
        for ticker in self.tickers:
            engine = self.engines[ticker]
            
            # Inject ticker-specific data into terminals
            price_vec = engine.data['Close'].to_numpy()
            volume_vec = engine.data['Volume'].to_numpy()
            engine.pset.terminals[NumVector][0].value = price_vec
            engine.pset.terminals[NumVector][1].value = volume_vec
            
            # Execute the pre-compiled rule
            try:
                signals = rule()
                
                # Handle single boolean return
                if not isinstance(signals, np.ndarray):
                    signals = np.full(engine.data.shape[0], signals, dtype=np.bool_)
                
                # Sanitize signals
                if not np.all(np.isfinite(signals)):
                    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
                
                signals = signals.astype(np.bool_)
                
            except TypeError as e:
                # Try with arguments (for loaded individuals)
                if "missing" in str(e) and "required positional arguments" in str(e):
                    try:
                        signals = rule(price_vec, volume_vec)
                        if not isinstance(signals, np.ndarray):
                            signals = np.full(engine.data.shape[0], signals, dtype=np.bool_)
                        if not np.all(np.isfinite(signals)):
                            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
                        signals = signals.astype(np.bool_)
                    except Exception:
                        return -100000.0,
                else:
                    return -100000.0,
            except Exception:
                return -100000.0,
            
            # Slice signals to match backtest_data period using mask
            if engine.backtest_start or engine.backtest_end:
                mask = pd.Series(True, index=engine.data.index)
                if engine.backtest_start:
                    mask &= (engine.data.index >= engine.backtest_start)
                if engine.backtest_end:
                    mask &= (engine.data.index <= engine.backtest_end)
                backtest_signals = signals[mask.values]
            else:
                backtest_signals = signals
            
            equity_curve = engine._run_simulation_with_equity_curve(
                backtest_signals,
                engine.backtest_data
            )
            equity_curves.append(equity_curve)
        
        # Combine equity curves (sum across tickers)
        # Align by date index
        combined_equity = pd.concat(equity_curves, axis=1).sum(axis=1)
        
        # Edge case: insufficient data
        if len(combined_equity) < 2:
            return 0.0
        
        # Calculate returns
        returns = combined_equity.pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        # Filter out NaN and Inf
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
    
    def run_detailed_simulation(self, individual: gp.PrimitiveTree) -> Dict:
        """
        Runs detailed simulation for all tickers and returns comprehensive results.
        
        Returns:
            Dictionary containing results for each ticker and portfolio summary
        """
        results = {
            'tickers': {},
            'portfolio_summary': {}
        }
        
        total_gp_return = 0.0
        total_bh_return = 0.0
        
        for ticker in self.tickers:
            engine = self.engines[ticker]
            ticker_result = engine.run_detailed_simulation(individual)
            
            results['tickers'][ticker] = ticker_result
            total_gp_return += ticker_result['gp_return']
            total_bh_return += ticker_result['buy_and_hold_return']
        
        results['portfolio_summary'] = {
            'total_gp_return': total_gp_return,
            'total_bh_return': total_bh_return,
            'total_excess_return': total_gp_return - total_bh_return,
            'capital_per_ticker': self.capital_per_ticker,
            'total_capital': self.total_capital
        }
        
        return results

