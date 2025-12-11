"""
Data Loading and Preprocessing Module

This module is responsible for loading financial data from CSV files,
cleaning it, and preparing it for use in the genetic programming
and backtesting engines.
"""
import pandas as pd
from typing import Dict, List, Tuple
import os

def load_and_process_data(data_dir: str, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads and processes stock data for a list of tickers from a specified directory.

    For each ticker, this function reads the corresponding CSV file, converts the 'Date'
    column to datetime objects, sets it as the index, and handles missing values.

    Args:
        data_dir: The directory where the CSV files are located.
        tickers: A list of stock tickers (e.g., ['ry.TO', 'ABX.TO']). The CSV
                 filenames are expected to match these tickers (e.g., 'ry.TO.csv').

    Returns:
        A dictionary where keys are ticker symbols and values are the processed
        Pandas DataFrames.
    """
    data = {}
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Data file for {ticker} not found at {file_path}. Skipping.")
            continue

        df = pd.read_csv(file_path)

        # --- Data Cleaning and Processing ---
        # 1. Convert 'Date' column to datetime and set as index
        # Use utc=True to handle mixed timezones and convert to timezone-naive
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df.set_index('Date', inplace=True)

        # 2. Handle missing values. The data shows entire rows can be empty.
        # We can drop rows where all values are NaN, then forward-fill others.
        df.dropna(how='all', inplace=True)
        df.ffill(inplace=True)

        # 3. Ensure correct data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # 4. Sort by date to ensure chronological order
        df.sort_index(inplace=True)

        data[ticker] = df
        print(f"Successfully loaded and processed data for {ticker}.")

    return data


def split_train_test_data(
    data: Dict[str, pd.DataFrame],
    train_data_start: str,
    train_backtest_start: str,
    train_backtest_end: str,
    test_data_start: str,
    test_backtest_start: str,
    test_backtest_end: str,
    # Optional validate parameters for backward compatibility
    validate_data_start: str = None,
    validate_backtest_start: str = None,
    validate_backtest_end: str = None
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Splits data into training (in-sample), validation, and testing (out-of-sample) periods
    with separate initial periods for technical indicator calculation.
    
    Validate parameters are optional for backward compatibility.
    If not provided, returns None for validate_data.
    
    Args:
        data: Dictionary of ticker -> DataFrame
        train_data_start: Training initial period start
        train_backtest_start: Training backtest period start
        train_backtest_end: Training backtest period end
        test_data_start: Testing initial period start
        test_backtest_start: Testing backtest period start
        test_backtest_end: Testing backtest period end
        validate_data_start: (Optional) Validation initial period start
        validate_backtest_start: (Optional) Validation backtest period start
        validate_backtest_end: (Optional) Validation backtest period end
    
    Returns:
        Tuple of (train_data, test_data, validate_data) dictionaries.
        validate_data is None if validate parameters are not provided.
    """
    train_data = {}
    test_data = {}
    validate_data = {} if validate_data_start else None
    
    has_validate = all([validate_data_start, validate_backtest_start, validate_backtest_end])
    
    for ticker, df in data.items():
        # Check if data covers the required periods
        data_start_date = df.index[0]
        data_end_date = df.index[-1]
        
        # Training data: from data_start to backtest_end (includes initial period)
        actual_train_start = max(pd.Timestamp(train_data_start), data_start_date)
        train_df = df.loc[actual_train_start:train_backtest_end].copy()
        
        train_data[ticker] = {
            'data': train_df,
            'backtest_start': train_backtest_start,
            'backtest_end': train_backtest_end
        }
        
        # Testing data: from data_start to backtest_end (includes initial period)
        actual_test_start = max(pd.Timestamp(test_data_start), data_start_date)
        test_df = df.loc[actual_test_start:test_backtest_end].copy()
        
        test_data[ticker] = {
            'data': test_df,
            'backtest_start': test_backtest_start,
            'backtest_end': test_backtest_end
        }
        
        # Validation data (optional)
        if has_validate:
            actual_validate_start = max(pd.Timestamp(validate_data_start), data_start_date)
            validate_df = df.loc[actual_validate_start:validate_backtest_end].copy()
            
            validate_data[ticker] = {
                'data': validate_df,
                'backtest_start': validate_backtest_start,
                'backtest_end': validate_backtest_end
            }
        
        # Calculate period lengths
        train_initial_days = len(df.loc[actual_train_start:train_backtest_start]) - 1
        train_backtest_days = len(df.loc[train_backtest_start:train_backtest_end])
        test_initial_days = len(df.loc[actual_test_start:test_backtest_start]) - 1
        test_backtest_days = len(df.loc[test_backtest_start:test_backtest_end])
        
        print(f"{ticker} - Train: {len(train_df)} days total")
        print(f"  Data available from: {data_start_date.date()}")
        print(f"  Initial period: {train_initial_days} days ({actual_train_start.date()} to {train_backtest_start})")
        print(f"  Backtest period: {train_backtest_days} days ({train_backtest_start} to {train_backtest_end})")
        
        if has_validate:
            validate_initial_days = len(df.loc[actual_validate_start:validate_backtest_start]) - 1
            validate_backtest_days = len(df.loc[validate_backtest_start:validate_backtest_end])
            print(f"{ticker} - Validate: {len(validate_df)} days total")
            print(f"  Initial period: {validate_initial_days} days ({actual_validate_start.date()} to {validate_backtest_start})")
            print(f"  Backtest period: {validate_backtest_days} days ({validate_backtest_start} to {validate_backtest_end})")
        
        print(f"{ticker} - Test: {len(test_df)} days total")
        print(f"  Initial period: {test_initial_days} days ({actual_test_start.date()} to {test_backtest_start})")
        print(f"  Backtest period: {test_backtest_days} days ({test_backtest_start} to {test_backtest_end})")
    
    return train_data, test_data, validate_data

