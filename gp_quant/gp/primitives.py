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
import warnings

# Suppress numpy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


def identity_int(x: int) -> int:
    """Returns the integer unchanged. Used to satisfy DEAP's generator."""
    return x

def protected_div(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized protected division that returns 1.0 in case of division by zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(left, right)
        
    # Handle scalar/0-d array case
    if np.ndim(result) == 0:
        if np.abs(right) < 1e-6 or np.isinf(result) or np.isnan(result):
            return np.array(1.0) if isinstance(result, np.ndarray) else 1.0
        return result

    # Where the divisor is close to zero, the result is 1.0
    # Also handles cases where the result might be inf or NaN
    result[np.abs(right) < 1e-6] = 1.0
    result[np.isinf(result)] = 1.0
    result = np.nan_to_num(result, nan=1.0)
    return result

# --- Financial Indicator Primitives ---

def moving_average(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving average."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return float(series)
    if n == 0:
        return series
    try:
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).mean().to_numpy()
    except Exception:
        return np.full_like(series, np.nan)

def moving_max(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving maximum."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return float(series)
    if n == 0:
        return series
    try:
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).max().to_numpy()
    except Exception:
        return np.full_like(series, np.nan)

def moving_min(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized moving minimum."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return float(series)
    if n == 0:
        return series
    try:
        s = pd.Series(series, dtype=np.float64)
        return s.rolling(window=n, min_periods=1).min().to_numpy()
    except Exception:
        return np.full_like(series, np.nan)

def lag(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized lag."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return float(series)
    if n <= 0:
        return series
    result = np.full_like(series, np.nan)
    result[n:] = series[:-n]
    return result

def volatility(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized volatility."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return 0.0
    if n < 2:
        return np.zeros_like(series)
    
    try:
        # Use numpy for pct_change to avoid pandas type issues
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(series) / series[:-1]
        returns = np.concatenate([[0.0], returns])  # Pad first value
        
        # Calculate rolling std using pandas
        returns_series = pd.Series(returns, dtype=np.float64)
        result = returns_series.rolling(window=n, min_periods=1).std().to_numpy()
        
        # Handle inf/nan
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)
        return result
    except Exception:
        # Fallback: return zeros if any error occurs
        return np.zeros_like(series)

def rate_of_change(series: np.ndarray, n: int) -> np.ndarray:
    """Calculates the vectorized Rate of Change (ROC)."""
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return 0.0
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
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return 50.0
    if n < 1:
        return np.full_like(series, 50.0)
    
    try:
        s = pd.Series(series, dtype=np.float64)
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
    except Exception:
        # Fallback: return neutral RSI
        return np.full_like(series, 50.0)

# --- New Primitives (Alpha101 inspired) ---

def ts_rank(series: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the rolling rank (normalized to [0, 1]).
    ts_rank(x, d) = (rank(x) - 1) / (d - 1)
    """
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return 0.5
    if n < 2:
        return np.zeros_like(series)
        
    try:
        s = pd.Series(series, dtype=np.float64)
        # rank() returns 1 to N
        ranks = s.rolling(window=n, min_periods=1).rank()
        counts = s.rolling(window=n, min_periods=1).count()
        
        # Normalize: (rank - 1) / (count - 1)
        # Avoid division by zero when count <= 1
        with np.errstate(divide='ignore', invalid='ignore'):
            result = (ranks - 1) / (counts - 1)
            
        result = result.fillna(0.5) # Default to middle rank
        result[counts <= 1] = 0.5
        
        return result.to_numpy()
    except Exception:
        return np.full_like(series, 0.5)

def correlation(series1: np.ndarray, series2: np.ndarray, n: int) -> np.ndarray:
    """Calculates the rolling correlation between two series."""
    if isinstance(series1, (float, int, np.float64, np.int64)) or np.ndim(series1) == 0 or \
       isinstance(series2, (float, int, np.float64, np.int64)) or np.ndim(series2) == 0:
        return np.zeros_like(series1) if isinstance(series1, np.ndarray) and np.ndim(series1) > 0 else 0.0
        
    if n < 2:
        return np.zeros_like(series1)
        
    try:
        s1 = pd.Series(series1, dtype=np.float64)
        s2 = pd.Series(series2, dtype=np.float64)
        return s1.rolling(window=n, min_periods=2).corr(s2).fillna(0.0).to_numpy()
    except Exception:
        return np.zeros_like(series1)

def decay_linear(series: np.ndarray, n: int) -> np.ndarray:
    """
    Calculates the linear weighted moving average.
    Weights are 1, 2, ..., n (most recent has weight n).
    """
    if isinstance(series, (float, int, np.float64, np.int64)) or np.ndim(series) == 0:
        return float(series)
    if n < 1:
        return series
        
    try:
        # Efficient implementation using pandas rolling apply is slow.
        # We can use a trick: sum of moving sums?
        # LWMA = (Sum(Price * Weight)) / Sum(Weights)
        # This is hard to vectorize efficiently without convolution.
        # Let's use numpy convolve for full series? No, it's rolling.
        # For small n, we can iterate? No, slow.
        # Let's use a simplified approach or pandas apply for now, assuming n is small (<200).
        
        # Optimization:
        # LWMA[t] = (P[t]*n + P[t-1]*(n-1) + ... + P[t-n+1]*1) / Sum(1..n)
        
        weights = np.arange(1, n + 1)
        sum_weights = np.sum(weights)
        
        # Use strided view for efficiency if possible, but pandas apply is safer for NaN handling
        # Given the constraints, let's use a simple convolution if no NaNs, but we have NaNs.
        # Let's stick to pandas rolling apply for correctness first.
        
        def weighted_mean(x):
            return np.dot(x, weights) / sum_weights
            
        s = pd.Series(series, dtype=np.float64)
        # raw=True speeds up apply significantly
        return s.rolling(window=n).apply(weighted_mean, raw=True).bfill().to_numpy()
        
    except Exception:
        return series

def log(series: np.ndarray) -> np.ndarray:
    """Calculates natural logarithm (safe)."""
    if isinstance(series, (float, int, np.float64, np.int64)):
        return np.log(abs(series)) if series != 0 else 0.0
        
    # Use np.log on absolute value, handle 0
    s_abs = np.abs(series)
    
    # Handle scalar/0-d array case
    if np.ndim(s_abs) == 0:
        if s_abs < 1e-9:
            s_abs = 1e-9
    else:
        # Replace 0 with small epsilon to avoid -inf
        s_abs[s_abs < 1e-9] = 1e-9
        
    return np.log(s_abs)

def sign(series: np.ndarray) -> np.ndarray:
    """Calculates sign of the series (1, 0, -1)."""
    return np.sign(series)

def abs_val(series: np.ndarray) -> np.ndarray:
    """Calculates absolute value."""
    return np.abs(series)

# --- Safe Arithmetic Primitives ---

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized addition with overflow protection."""
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.add(a, b)
    result = np.nan_to_num(result, nan=0.0, posinf=1e300, neginf=-1e300)
    return result

def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized subtraction with overflow protection."""
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.subtract(a, b)
    result = np.nan_to_num(result, nan=0.0, posinf=1e300, neginf=-1e300)
    return result

def logical_not(a: np.ndarray) -> np.ndarray:
    """Vectorized logical NOT."""
    return np.logical_not(a)

def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized multiplication with overflow protection (clamping)."""
    # Use numpy's error handling to catch overflows and invalid operations
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.multiply(a, b)
        
    # Clamp infinite values to a large finite number
    # using 1e300 as a safe upper bound for float64
    result = np.nan_to_num(result, nan=0.0, posinf=1e300, neginf=-1e300)
    return result

def norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorized Norm operator: calculates the absolute difference between two vectors.
    
    Norm(r1, r2) = |r1 - r2|
    
    This operator is commonly used in technical analysis to measure the distance
    between two price series or indicators.
    
    Args:
        a: First numerical vector
        b: Second numerical vector
        
    Returns:
        Absolute difference between the two vectors
    """
    return np.abs(np.subtract(a, b))

