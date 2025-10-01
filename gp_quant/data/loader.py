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
        df['Date'] = pd.to_datetime(df['Date'])
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
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Splits data into training (in-sample) and testing (out-of-sample) periods.
    
    Based on PRD Section 7.2 (Long Training Period):
    - Training Initial Period: 1992-06-30 to 1993-07-02 (250 days)
    - Training Period: 1993-07-02 to 1999-06-25 (1498 days)
    - Testing Initial Period: 1998-07-07 to 1999-06-25 (250 days)
    - Testing Period: 1999-06-28 to 2000-06-30 (256 days)
    
    Args:
        data: Dictionary of ticker -> DataFrame
        train_start: Training period start date (e.g., '1992-06-30')
        train_end: Training period end date (e.g., '1999-06-25')
        test_start: Testing period start date (e.g., '1999-06-28')
        test_end: Testing period end date (e.g., '2000-06-30')
    
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    train_data = {}
    test_data = {}
    
    for ticker, df in data.items():
        # Split training data
        train_df = df.loc[train_start:train_end].copy()
        train_data[ticker] = train_df
        
        # Split testing data
        test_df = df.loc[test_start:test_end].copy()
        test_data[ticker] = test_df
        
        print(f"{ticker} - Train: {len(train_df)} days ({train_df.index[0].date()} to {train_df.index[-1].date()})")
        print(f"{ticker} - Test: {len(test_df)} days ({test_df.index[0].date()} to {test_df.index[-1].date()})")
    
    return train_data, test_data
