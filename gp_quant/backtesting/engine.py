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
from typing import Callable

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

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000.0):
        """
        Initializes the backtesting engine.

        Args:
            data: A DataFrame with historical market data (OHLCV).
            initial_capital: The starting capital for the simulation.
        """
        self.data = data
        self.initial_capital = initial_capital
        # Create a copy of pset to avoid modifying the global one
        import copy
        self.pset = copy.deepcopy(pset)

    def evaluate(self, individual: gp.PrimitiveTree) -> tuple[float]:
        """
        Evaluates the fitness of a single GP individual using vectorization.
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
        gp_return = self._run_vectorized_simulation(signals)

        # --- Calculate Buy-and-Hold Return ---
        # Correctly calculate B&H return based on initial capital
        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
        if start_price > 0:
            buy_and_hold_return = (end_price / start_price - 1) * self.initial_capital
        else:
            buy_and_hold_return = 0

        # --- Fitness is the Excess Return ---
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

    def _run_vectorized_simulation(self, signals: np.ndarray) -> float:
        """
        Runs the simulation using the fast Numba JIT-compiled loop.
        """
        # Ensure signals is always an array to prevent Numba typing errors
        if not hasattr(signals, '__len__'):
            # If signal is a single boolean, broadcast it to the shape of the data
            signals = np.full(self.data.shape[0], signals, dtype=np.bool_)

        open_prices_np = self.data['Open'].to_numpy()
        close_prices_np = self.data['Close'].to_numpy()

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
        """
        signals = self.get_signals(individual)
        if len(signals) == 0:
            return {
                'gp_return': 0,
                'buy_and_hold_return': 0,
                'trades': [],
                'error': 'Could not generate signals from individual.'
            }

        trades = []
        position = 0
        capital = self.initial_capital
        shares = 0.0
        entry_price = 0.0
        entry_date = None

        open_prices = self.data['Open'].to_numpy()
        dates = self.data.index

        for i in range(len(signals) - 1):
            signal = signals[i]
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
            last_close_price = self.data['Close'].iloc[-1]
            capital = shares * last_close_price
            pnl = (last_close_price - entry_price) * shares
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': self.data.index[-1].strftime('%Y-%m-%d'),
                'entry_price': round(entry_price, 2),
                'exit_price': round(last_close_price, 2),
                'shares': round(shares, 2),
                'pnl': round(pnl, 2)
            })

        gp_total_return = capital - self.initial_capital

        start_price = self.data['Close'].iloc[0]
        end_price = self.data['Close'].iloc[-1]
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

