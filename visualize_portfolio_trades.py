"""
視覺化 Portfolio 交易記錄

為訓練期和測試期分別繪製：
- 4 個股票的個別績效曲線
- 1 條總和績效曲線
- 1 條 Buy-and-Hold 基準線
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_buy_and_hold(trades_file, initial_capital_per_stock=25000.0):
    """
    計算 Buy-and-Hold 基準績效
    
    使用實際股價數據：第一天買入，持有到最後一天
    
    Args:
        trades_file: 交易記錄 CSV 文件
        initial_capital_per_stock: 每個股票的初始資金
        
    Returns:
        dates: 日期列表
        bh_pnl: Buy-and-Hold PnL 列表
    """
    # 讀取交易記錄以獲取股票列表和日期範圍
    trades = pd.read_csv(trades_file)
    trades['date'] = pd.to_datetime(trades['date'])
    
    tickers = trades['ticker'].unique()
    
    # 載入股價數據
    project_root = Path(__file__).parent
    stock_data = {}
    
    for ticker in tickers:
        file_path = project_root / f"TSE300_selected/{ticker}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            stock_data[ticker] = df
    
    if not stock_data:
        return [], []
    
    # 找到交易日期範圍
    all_dates = sorted(trades['date'].unique())
    start_date = all_dates[0]
    end_date = all_dates[-1]
    
    # 為每個股票計算 Buy-and-Hold PnL
    bh_pnl_by_ticker = {}
    
    for ticker in tickers:
        if ticker not in stock_data:
            continue
        
        df = stock_data[ticker]
        
        # 過濾到交易期間
        mask = (df.index >= start_date) & (df.index <= end_date)
        period_data = df[mask].copy()
        
        if len(period_data) == 0:
            continue
        
        # 第一天的收盤價（買入價）
        first_close = period_data['Close'].iloc[0]
        
        # 計算每天的 PnL
        # PnL = (當前價格 - 買入價) / 買入價 * 初始資金
        shares = initial_capital_per_stock / first_close
        period_data['pnl'] = (period_data['Close'] - first_close) * shares
        
        bh_pnl_by_ticker[ticker] = period_data[['pnl']]
    
    if not bh_pnl_by_ticker:
        return [], []
    
    # 合併所有股票的 PnL
    # 找到所有共同的交易日
    common_dates = None
    for ticker, data in bh_pnl_by_ticker.items():
        if common_dates is None:
            common_dates = set(data.index)
        else:
            common_dates = common_dates.intersection(set(data.index))
    
    common_dates = sorted(list(common_dates))
    
    # 計算總 PnL
    total_pnl = []
    for date in common_dates:
        daily_total = sum(
            bh_pnl_by_ticker[ticker].loc[date, 'pnl']
            for ticker in bh_pnl_by_ticker.keys()
            if date in bh_pnl_by_ticker[ticker].index
        )
        total_pnl.append(daily_total)
    
    return common_dates, total_pnl

def calculate_sharpe_ratio(pnl_series, dates, risk_free_rate=0.0, initial_capital=100000.0):
    """
    計算 Sharpe Ratio
    
    Args:
        pnl_series: PnL 序列
        dates: 日期序列
        risk_free_rate: 無風險利率（年化）
        initial_capital: 初始資金
    
    Returns:
        Sharpe Ratio
    """
    if len(pnl_series) < 2:
        return 0.0
    
    # 計算每日回報率
    daily_returns = []
    for i in range(1, len(pnl_series)):
        # 計算資產價值 = 初始資金 + PnL
        prev_value = initial_capital + pnl_series[i-1]
        curr_value = initial_capital + pnl_series[i]
        
        if prev_value > 0:
            ret = (curr_value - prev_value) / prev_value
        else:
            ret = 0.0
        daily_returns.append(ret)
    
    if len(daily_returns) == 0:
        return 0.0
    
    # 過濾掉 nan 和 inf
    daily_returns = [r for r in daily_returns if np.isfinite(r)]
    
    if len(daily_returns) == 0:
        return 0.0
    
    # 計算平均回報和標準差
    mean_return = np.mean(daily_returns)
    std_return = np.std(daily_returns, ddof=1)
    
    if std_return == 0 or not np.isfinite(std_return):
        return 0.0
    
    # 年化 Sharpe Ratio (假設 252 個交易日)
    sharpe = (mean_return * 252 - risk_free_rate) / (std_return * np.sqrt(252))
    
    return sharpe if np.isfinite(sharpe) else 0.0

def plot_portfolio_performance(ax, trades_file, title):
    """
    繪製組合績效圖
    
    Args:
        trades_file: 交易記錄 CSV 文件路徑
        title: 圖表標題
        output_file: 輸出圖片路徑
    """
    # 讀取交易記錄
    trades = pd.read_csv(trades_file)
    trades['date'] = pd.to_datetime(trades['date'])
    
    # 獲取所有股票
    tickers = trades['ticker'].unique()
    
    # 初始化每個股票的資金
    initial_capital_per_stock = 25000.0  # 100000 / 4
    
    # 為每個股票計算累積 PnL
    stock_pnl = {}
    
    for ticker in tickers:
        ticker_trades = trades[trades['ticker'] == ticker].sort_values('date')
        
        dates = []
        pnl_curve = []
        current_pnl = 0.0
        
        for _, trade in ticker_trades.iterrows():
            dates.append(trade['date'])
            
            if trade['action'] == 'BUY':
                # 買入時記錄當前 PnL（通常是 0 或之前的 PnL）
                pnl_curve.append(current_pnl)
            elif trade['action'] == 'SELL':
                # 賣出時更新 PnL
                if pd.notna(trade['proceeds']):
                    # PnL = 賣出收益 - 初始資金
                    current_pnl = trade['proceeds'] - initial_capital_per_stock
                    pnl_curve.append(current_pnl)
        
        stock_pnl[ticker] = {
            'dates': dates,
            'pnl': pnl_curve
        }
    
    # 繪製每個股票的 PnL 曲線
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, ticker in enumerate(sorted(tickers)):
        if ticker in stock_pnl and len(stock_pnl[ticker]['dates']) > 0:
            ax.plot(stock_pnl[ticker]['dates'], 
                   stock_pnl[ticker]['pnl'],
                   label=ticker,
                   linewidth=2,
                   alpha=0.7,
                   color=colors[i % len(colors)])
    
    # 計算總和 PnL
    # 找到所有日期
    all_dates = sorted(set(date for ticker_data in stock_pnl.values() 
                          for date in ticker_data['dates']))
    
    total_pnl = []
    for date in all_dates:
        daily_total = 0.0
        for ticker in tickers:
            if ticker in stock_pnl:
                # 找到該日期或之前最近的 PnL
                ticker_dates = stock_pnl[ticker]['dates']
                ticker_pnls = stock_pnl[ticker]['pnl']
                
                # 找到小於等於當前日期的最後一個 PnL
                valid_pnls = [pnl for d, pnl in zip(ticker_dates, ticker_pnls) if d <= date]
                if valid_pnls:
                    daily_total += valid_pnls[-1]
        
        total_pnl.append(daily_total)
    
    # 繪製總和曲線（加粗）
    ax.plot(all_dates, total_pnl,
           label='GP 策略 (Portfolio)',
           linewidth=3,
           color='black',
           linestyle='-',
           alpha=0.9,
           zorder=10)
    
    # 計算並繪製 Buy-and-Hold 基準線
    bh_dates, bh_pnl = calculate_buy_and_hold(trades_file)
    if len(bh_dates) > 0:
        ax.plot(bh_dates, bh_pnl,
               label='Buy-and-Hold',
               linewidth=2.5,
               color='red',
               linestyle='--',
               alpha=0.8,
               zorder=9)
    
    # 添加零線
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # 設置標題和標籤
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel('累積 PnL ($)', fontsize=11)
    
    # 設置圖例
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    
    # 設置網格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 格式化 y 軸為貨幣格式
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 旋轉 x 軸標籤
    ax.tick_params(axis='x', rotation=45)
    
    # 計算 Sharpe Ratio
    gp_sharpe = calculate_sharpe_ratio(total_pnl, all_dates)
    bh_sharpe = calculate_sharpe_ratio(bh_pnl, bh_dates) if len(bh_pnl) > 0 else 0.0
    
    # 返回統計信息
    stats = {
        'gp_final_pnl': total_pnl[-1] if total_pnl else 0,
        'bh_final_pnl': bh_pnl[-1] if bh_pnl else 0,
        'excess_return': (total_pnl[-1] - bh_pnl[-1]) if (total_pnl and bh_pnl) else 0,
        'gp_sharpe': gp_sharpe,
        'bh_sharpe': bh_sharpe,
        'stock_pnl': {ticker: stock_pnl[ticker]['pnl'][-1] 
                     for ticker in sorted(tickers) 
                     if ticker in stock_pnl and len(stock_pnl[ticker]['pnl']) > 0}
    }
    
    return stats

def main():
    # 設置路徑
    exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251013_144624')
    
    train_trades = exp_dir / 'best_individual_train_trades.csv'
    test_trades = exp_dir / 'best_individual_test_trades.csv'
    
    # 檢查文件是否存在
    if not train_trades.exists():
        print(f"✗ 找不到訓練期交易記錄: {train_trades}")
        return
    
    if not test_trades.exists():
        print(f"✗ 找不到測試期交易記錄: {test_trades}")
        return
    
    print("="*80)
    print("📊 視覺化 Portfolio 交易績效")
    print("="*80)
    print()
    
    # 創建上下子圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 繪製訓練期績效
    print("1️⃣  繪製訓練期績效...")
    train_stats = plot_portfolio_performance(
        ax1,
        train_trades,
        '訓練期（樣本內）Portfolio 績效'
    )
    
    print(f"\n訓練期統計:")
    print(f"  GP 策略最終 PnL: ${train_stats['gp_final_pnl']:,.2f}")
    print(f"  GP 策略 Sharpe Ratio: {train_stats['gp_sharpe']:.4f}")
    print(f"  Buy-and-Hold PnL: ${train_stats['bh_final_pnl']:,.2f}")
    print(f"  Buy-and-Hold Sharpe Ratio: {train_stats['bh_sharpe']:.4f}")
    print(f"  超額回報: ${train_stats['excess_return']:,.2f}")
    for ticker, pnl in train_stats['stock_pnl'].items():
        print(f"  {ticker} PnL: ${pnl:,.2f}")
    print()
    
    # 繪製測試期績效
    print("2️⃣  繪製測試期績效...")
    test_stats = plot_portfolio_performance(
        ax2,
        test_trades,
        '測試期（樣本外）Portfolio 績效'
    )
    
    print(f"\n測試期統計:")
    print(f"  GP 策略最終 PnL: ${test_stats['gp_final_pnl']:,.2f}")
    print(f"  GP 策略 Sharpe Ratio: {test_stats['gp_sharpe']:.4f}")
    print(f"  Buy-and-Hold PnL: ${test_stats['bh_final_pnl']:,.2f}")
    print(f"  Buy-and-Hold Sharpe Ratio: {test_stats['bh_sharpe']:.4f}")
    print(f"  超額回報: ${test_stats['excess_return']:,.2f}")
    for ticker, pnl in test_stats['stock_pnl'].items():
        print(f"  {ticker} PnL: ${pnl:,.2f}")
    print()
    
    # 調整子圖間距
    plt.tight_layout(pad=3.0)
    
    # 儲存合併圖片
    combined_output = exp_dir / 'portfolio_performance_combined.png'
    plt.savefig(combined_output, dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存合併圖: {combined_output}")
    
    plt.close()
    
    print()
    print("="*80)
    print("✅ 視覺化完成！")
    print("="*80)
    print()
    print(f"輸出文件:")
    print(f"  📈 合併圖: {combined_output}")
    print()
    
    # 顯示比較
    print(f"\n📊 訓練期 vs 測試期比較:")
    print(f"  {'指標':<30} {'訓練期':>15} {'測試期':>15} {'差異':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'GP 策略 PnL':<30} ${train_stats['gp_final_pnl']:>14,.2f} ${test_stats['gp_final_pnl']:>14,.2f} ${test_stats['gp_final_pnl'] - train_stats['gp_final_pnl']:>+14,.2f}")
    print(f"  {'GP 策略 Sharpe Ratio':<30} {train_stats['gp_sharpe']:>15.4f} {test_stats['gp_sharpe']:>15.4f} {test_stats['gp_sharpe'] - train_stats['gp_sharpe']:>+15.4f}")
    print(f"  {'Buy-and-Hold PnL':<30} ${train_stats['bh_final_pnl']:>14,.2f} ${test_stats['bh_final_pnl']:>14,.2f} ${test_stats['bh_final_pnl'] - train_stats['bh_final_pnl']:>+14,.2f}")
    print(f"  {'Buy-and-Hold Sharpe Ratio':<30} {train_stats['bh_sharpe']:>15.4f} {test_stats['bh_sharpe']:>15.4f} {test_stats['bh_sharpe'] - train_stats['bh_sharpe']:>+15.4f}")
    print(f"  {'超額回報':<30} ${train_stats['excess_return']:>14,.2f} ${test_stats['excess_return']:>14,.2f} ${test_stats['excess_return'] - train_stats['excess_return']:>+14,.2f}")
    print()

if __name__ == '__main__':
    main()
