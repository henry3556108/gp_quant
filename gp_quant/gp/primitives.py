"""
Custom Primitives for Genetic Programming

This module contains the functions that will be used as primitives in the
GP-evolved trading rules. These functions include safe arithmetic operations
and various financial technical indicators.

The functions are designed to work with Pandas Series, which is efficient for
time series operations.
"""
import numpy as np
import pandas as pd  # Keep pandas for its efficient rolling window and diff operations


def identity_int(x: int) -> int:
    """Returns the integer unchanged. Used to satisfy DEAP's generator."""
    return x

def protected_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized protected division that returns 1.0 in case of division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(left, right)
    # Where the divisor is close to zero, the result is 1.0
    # Also handles cases where the result might be inf or NaN
    result[np.abs(right) < 1e-6] = 1.0
    result[np.isinf(result)] = 1.0
    result = np.nan_to_num(result, nan=1.0)
    return result

# --- Financial Indicator Primitives ---

def moving_average(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving average."""
    if n == 0:
        return series
    s = pd.Series(series)
    return s.rolling(window=n, min_periods=1).mean().to_numpy()

def moving_max(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving maximum."""
    if n == 0:
        return series
    s = pd.Series(series)
    return s.rolling(window=n, min_periods=1).max().to_numpy()

def moving_min(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving minimum."""
    if n == 0:
        return series
    s = pd.Series(series)
    return s.rolling(window=n, min_periods=1).min().to_numpy()

def lag(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized lag."""
    if n <= 0:
        return series
    result = np.full_like(series, np.nan)
    result[n:] = series[:-n]
    return result

def volatility(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized volatility."""
    if n < 2:
        return np.zeros_like(series)
    s = pd.Series(series)
    returns = s.pct_change(fill_method=None)
    result = returns.rolling(window=n, min_periods=1).std().to_numpy()
    # Only handle inf/nan, don't restrict normal values
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

def rate_of_change(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized Rate of Change (ROC)."""
    if n < 1:
        return np.zeros_like(series)
    
    lagged_series = lag(series, n)
    # Use np.divide to handle division by zero safely
    with np.errstate(divide='ignore', invalid='ignore'):
        roc = np.divide(series - lagged_series, lagged_series) * 100
    # Handle inf/nan but allow large legitimate values
    return np.nan_to_num(roc, nan=0.0, posinf=1e6, neginf=-1e6)

def relative_strength_index(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized Relative Strength Index (RSI)."""
    if n < 1:
        return np.full_like(series, 50.0)
    s = pd.Series(series)
    delta = s.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n, min_periods=1).mean()

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], 0).fillna(0)

    rsi = 100 - (100 / (1 + rs))
    result = rsi.to_numpy()
    # RSI is mathematically bounded to 0-100
    return np.clip(result, 0, 100)

# --- Safe Arithmetic Primitives ---

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized addition."""
    return np.add(a, b)

def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized subtraction."""
    return np.subtract(a, b)

def logical_not(a: np.ndarray) -> np.ndarray:
    """Vectorized logical NOT."""
    return np.logical_not(a)

def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized multiplication with overflow protection."""
    finfo = np.finfo(np.float64)
    # Ignore warnings for this check, we handle it manually
    with np.errstate(over='ignore'):
        # Check where an overflow would occur: abs(a) * abs(b) > max_float
        # This is equivalent to: abs(a) > max_float / abs(b)
        # Add a small epsilon to b to avoid division by zero in the check itself
        abs_b_safe = np.abs(b) + 1e-9
        problematic_indices = np.where(np.abs(a) > finfo.max / abs_b_safe)
    
    if len(problematic_indices[0]) > 0:
        idx = problematic_indices[0][0]
        error_msg = (
            f"Overflow detected in mul primitive!\n"
            f"Index: {idx}\n"
            f"Value a[{idx}]: {a[idx]:.2e}\n"
            f"Value b[{idx}]: {b[idx]:.2e}\n"
            f"Result would exceed: {finfo.max:.2e}"
        )
        # Print the error for visibility before the evolution halts
        print("\n--- ASSERTION TRIGGERED ---")
        print(error_msg)
        print("--- END ASSERTION ---\n")
        raise AssertionError(error_msg)
        
    return np.multiply(a, b)

