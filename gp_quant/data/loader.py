"""
Data Loading and Preprocessing Module

This module is responsible for loading financial data from CSV files,
cleaning it, and preparing it for use in the genetic programming
and backtesting engines.
"""
import pandas as pd
from typing import Dict, List
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
