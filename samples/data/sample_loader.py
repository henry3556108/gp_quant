"""
Sample script to demonstrate the data loading and visualization components.

This script performs the following actions:
1. Defines the path to the data directory and the tickers to be loaded.
2. Calls the `load_and_process_data` function from the data loader module.
3. Selects the data for a single stock ('RY.TO').
4. Calls the `plot_stock_price` function from the visualization utility to create
   and save a plot of the stock's closing price.
"""
import os
import sys

# Add the project root to the Python path to allow for absolute imports
# This is a common practice for making scripts runnable from any directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from gp_quant.data.loader import load_and_process_data
from gp_quant.utils.visualization import plot_stock_price

def main():
    """Main function to run the demonstration."""
    print("--- Running Data Loader and Visualization Sample ---")

    # Define the data directory and the tickers to load
    # Assumes this script is in 'samples/data/' and data is in 'TSE300_selected/'
    data_dir = os.path.join(project_root, 'TSE300_selected')
    tickers = ['RY.TO', 'ABX.TO']

    # Load the data
    all_stock_data = load_and_process_data(data_dir, tickers)

    # Check if data for 'RY.TO' was loaded successfully
    if 'RY.TO' in all_stock_data:
        ry_data = all_stock_data['RY.TO']
        print("\nData for RY.TO loaded successfully.")
        print("First 5 rows:")
        print(ry_data.head())
        print("\nLast 5 rows:")
        print(ry_data.tail())

        # Define a path to save the plot
        output_dir = os.path.join(project_root, 'samples', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'ry_closing_price.png')

        # Plot the closing price
        print(f"\nGenerating plot for RY.TO closing price...")
        plot_stock_price(ry_data, 'RY.TO', save_path=save_path)
    else:
        print("\nCould not find data for RY.TO. Please check the data directory and ticker list.")

    print("\n--- Sample script finished. ---")

if __name__ == "__main__":
    main()
