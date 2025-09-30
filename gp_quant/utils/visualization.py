"""
Visualization Utilities Module

This module provides helper functions for creating various plots and charts
to visualize financial data and the results of the genetic programming algorithm.
"""
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_stock_price(df: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
    """
    Plots the closing price of a stock over time.

    Args:
        df: The DataFrame containing the stock data, with the date as the index.
        ticker: The stock ticker symbol, used for the plot title.
        save_path: If provided, the plot will be saved to this file path.
                   Otherwise, the plot will be displayed interactively.
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(df.index, df['Close'], label='Close Price', color='dodgerblue')
    ax.set_title(f'Historical Closing Price for {ticker}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (CAD)', fontsize=12)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
