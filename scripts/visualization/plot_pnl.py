"""
Plot PnL charts for all trading strategies
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read all CSV files
tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
data = {}

for ticker in tickers:
    df = pd.read_csv(f'trade_details_{ticker}.csv')
    data[ticker] = df

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Trading Strategy PnL Analysis', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

for idx, ticker in enumerate(tickers):
    df = data[ticker]
    ax = axes[idx]
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    
    # Create trade numbers
    trade_numbers = np.arange(1, len(df) + 1)
    
    # Plot individual trade PnL as bars
    colors = ['green' if x > 0 else 'red' for x in df['pnl']]
    ax.bar(trade_numbers, df['pnl'], color=colors, alpha=0.6, label='Individual Trade PnL')
    
    # Plot cumulative PnL as line
    ax2 = ax.twinx()
    ax2.plot(trade_numbers, df['cumulative_pnl'], color='blue', linewidth=2, 
             marker='o', markersize=3, label='Cumulative PnL')
    
    # Formatting
    ax.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
    ax.set_ylabel('Individual Trade PnL ($)', fontsize=11, fontweight='bold', color='black')
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=11, fontweight='bold', color='blue')
    ax.set_title(f'{ticker} - Total Trades: {len(df)}', fontsize=13, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Color the y-axis labels
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add statistics text box
    total_pnl = df['pnl'].sum()
    winning_trades = (df['pnl'] > 0).sum()
    losing_trades = (df['pnl'] < 0).sum()
    win_rate = winning_trades / len(df) * 100
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
    
    stats_text = f'Total PnL: ${total_pnl:,.2f}\n'
    stats_text += f'Win Rate: {win_rate:.1f}%\n'
    stats_text += f'Avg Win: ${avg_win:,.2f}\n'
    stats_text += f'Avg Loss: ${avg_loss:,.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
              bbox_to_anchor=(0.02, 0.75), fontsize=9)

plt.tight_layout()
plt.savefig('pnl_analysis.png', dpi=300, bbox_inches='tight')
print("PnL chart saved as 'pnl_analysis.png'")

# Print summary statistics
print("\n" + "="*80)
print("TRADING STRATEGY SUMMARY STATISTICS")
print("="*80)

for ticker in tickers:
    df = data[ticker]
    print(f"\n{ticker}:")
    print(f"  Total Trades: {len(df)}")
    print(f"  Total PnL: ${df['pnl'].sum():,.2f}")
    print(f"  Winning Trades: {(df['pnl'] > 0).sum()} ({(df['pnl'] > 0).sum()/len(df)*100:.1f}%)")
    print(f"  Losing Trades: {(df['pnl'] < 0).sum()} ({(df['pnl'] < 0).sum()/len(df)*100:.1f}%)")
    print(f"  Average Win: ${df[df['pnl'] > 0]['pnl'].mean():,.2f}")
    print(f"  Average Loss: ${df[df['pnl'] < 0]['pnl'].mean():,.2f}")
    print(f"  Largest Win: ${df['pnl'].max():,.2f}")
    print(f"  Largest Loss: ${df['pnl'].min():,.2f}")
    print(f"  Final Cumulative PnL: ${df['pnl'].sum():,.2f}")

print("\n" + "="*80)
