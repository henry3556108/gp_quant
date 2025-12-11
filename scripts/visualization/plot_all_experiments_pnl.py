"""
視覺化所有實驗的最佳個體 PnL 曲線（Train & Test）並與 Buy and Hold 比較
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import sys
import os
sys.path.append('/Users/hongyicheng/Downloads/gp_quant')

from deap import gp
from gp_quant.evolution.components.gp.operators import pset
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data


def load_best_individual(exp_dir: str) -> Tuple:
    """載入實驗的最佳個體"""
    exp_path = Path(exp_dir)
    
    # 載入最終結果
    with open(exp_path / 'final_result.json', 'r') as f:
        final_result = json.load(f)
    
    final_gen = final_result.get('final_generation', final_result.get('generation', 0))
    
    # 載入最終世代族群
    pop_file = exp_path / 'populations' / f'generation_{final_gen:03d}.pkl'
    with open(pop_file, 'rb') as f:
        population = pickle.load(f)
    
    # 找到最佳個體
    best_ind = max(population, key=lambda ind: ind.fitness.values[0] if ind.fitness.values else -float('inf'))
    
    return best_ind, final_result['best_fitness']


def backtest_individual(individual, data: Dict, config: Dict, period: str) -> Dict:
    """回測個體並返回結果"""
    
    # 處理數據格式
    processed_data = {}
    for ticker, ticker_data in data.items():
        if isinstance(ticker_data, dict) and 'data' in ticker_data:
            processed_data[ticker] = ticker_data['data']
        else:
            processed_data[ticker] = ticker_data
    
    # 根據期間選擇日期
    if period == 'train':
        start_date = config['data']['train_backtest_start']
        end_date = config['data']['train_backtest_end']
    else:  # test
        start_date = config['data']['test_backtest_start']
        end_date = config['data']['test_backtest_end']
    
    # 創建回測引擎
    engine = PortfolioBacktestingEngine(
        data=processed_data,
        backtest_start=start_date,
        backtest_end=end_date,
        initial_capital=100000.0,
        pset=pset
    )
    
    # 執行回測
    result = engine.backtest(individual)
    
    return result


def calculate_buy_and_hold(data: Dict, config: Dict, period: str, initial_capital: float = 100000.0) -> pd.Series:
    """計算 Buy and Hold 策略的 PnL"""
    
    # 處理數據格式
    processed_data = {}
    for ticker, ticker_data in data.items():
        if isinstance(ticker_data, dict) and 'data' in ticker_data:
            processed_data[ticker] = ticker_data['data']
        else:
            processed_data[ticker] = ticker_data
    
    # 根據期間選擇日期
    if period == 'train':
        start_date = config['data']['train_backtest_start']
        end_date = config['data']['train_backtest_end']
    else:  # test
        start_date = config['data']['test_backtest_start']
        end_date = config['data']['test_backtest_end']
    
    # 計算等權重投資組合
    n_assets = len(processed_data)
    capital_per_asset = initial_capital / n_assets
    
    portfolio_values = []
    dates = None
    
    for ticker, df in processed_data.items():
        # 過濾日期範圍
        mask = (df.index >= start_date) & (df.index <= end_date)
        df_period = df[mask]
        
        if dates is None:
            dates = df_period.index
        
        # 計算該資產的價值 - 使用 Close 而不是 close
        close_col = 'Close' if 'Close' in df_period.columns else 'close'
        initial_price = df_period[close_col].iloc[0]
        shares = capital_per_asset / initial_price
        asset_values = df_period[close_col] * shares
        
        portfolio_values.append(asset_values.values)
    
    # 總投資組合價值
    total_portfolio = np.sum(portfolio_values, axis=0)
    
    # 計算 PnL
    pnl = total_portfolio - initial_capital
    
    return pd.Series(pnl, index=dates)


def calculate_metrics(equity_curve: pd.Series) -> Tuple[float, float]:
    """Calculate Sharpe Ratio and Consistent Sharpe Ratio"""
    if len(equity_curve) < 2:
        return 0.0, 0.0
        
    returns = equity_curve.pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return 0.0, 0.0
        
    # 1. Sharpe Ratio
    mean_return = returns.mean()
    std_return = returns.std()
    
    if std_return == 0 or not np.isfinite(std_return):
        sharpe = 0.0
    else:
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        
    # 2. Consistent Sharpe Ratio
    if len(equity_curve) < 252:
        consistent_sharpe = 0.0
    else:
        annual_sharpes = []
        years = returns.index.year.unique()
        
        for year in years:
            year_returns = returns[returns.index.year == year]
            if len(year_returns) < 100:
                continue
            
            mean_ret = year_returns.mean()
            std_ret = year_returns.std()
            
            if std_ret == 0 or not np.isfinite(std_ret):
                yr_sharpe = 0.0
            else:
                yr_sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252))
            
            yr_sharpe = max(min(yr_sharpe, 10.0), -10.0)
            annual_sharpes.append(yr_sharpe)
            
        if annual_sharpes:
            consistent_sharpe = np.mean(annual_sharpes) - np.std(annual_sharpes)
        else:
            consistent_sharpe = 0.0
            
    return sharpe, consistent_sharpe

def plot_all_experiments_pnl(experiments: List[Dict], timestamp: str):
    """繪製所有實驗的 PnL 曲線 (Overlay Comparison)"""
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # 定義顏色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # matplotlib default cycle
    bh_color = '#7f7f7f' # Gray for Buy & Hold
    
    # 1. Train Period (Top Subplot)
    ax_train = axes[0]
    ax_train.set_title('In-Sample (Train) PnL Comparison', fontsize=14, fontweight='bold')
    
    # Plot Buy & Hold (only once, from the first experiment)
    if experiments:
        train_bh = experiments[0]['train_bh']
        sharpe, cons_sharpe = calculate_metrics(train_bh)
        ax_train.plot(train_bh.index, train_bh.values, 
                     label=f'Buy & Hold (Sharpe: {sharpe:.2f}, CS: {cons_sharpe:.2f})', 
                     color=bh_color, linewidth=2, linestyle='--', alpha=0.7)
    
    # Plot each experiment
    for idx, exp_info in enumerate(experiments):
        exp_name = exp_info['name']
        train_result = exp_info['train_result']
        train_pnl = train_result['equity_curve'] - 100000.0
        
        # Calculate metrics
        sharpe, cons_sharpe = calculate_metrics(train_result['equity_curve'])
        
        ax_train.plot(train_pnl.index, train_pnl.values, 
                     label=f'{exp_name} (Sharpe: {sharpe:.2f}, CS: {cons_sharpe:.2f})', 
                     color=colors[idx % len(colors)], linewidth=2, alpha=0.9)

    ax_train.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax_train.set_ylabel('PnL ($)', fontsize=12)
    ax_train.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_train.grid(True, alpha=0.3)
    
    # 2. Test Period (Bottom Subplot)
    ax_test = axes[1]
    ax_test.set_title('Out-of-Sample (Test) PnL Comparison', fontsize=14, fontweight='bold')
    
    # Plot Buy & Hold (only once)
    if experiments:
        test_bh = experiments[0]['test_bh']
        if not test_bh.empty:
            sharpe, cons_sharpe = calculate_metrics(test_bh)
            ax_test.plot(test_bh.index, test_bh.values, 
                        label=f'Buy & Hold (Sharpe: {sharpe:.2f}, CS: {cons_sharpe:.2f})', 
                        color=bh_color, linewidth=2, linestyle='--', alpha=0.7)
    
    # Plot each experiment
    for idx, exp_info in enumerate(experiments):
        exp_name = exp_info['name']
        test_result = exp_info['test_result']
        
        if test_result and 'equity_curve' in test_result and not test_result['equity_curve'].empty:
            test_pnl = test_result['equity_curve'] - 100000.0
            
            # Calculate metrics
            sharpe, cons_sharpe = calculate_metrics(test_result['equity_curve'])
            
            ax_test.plot(test_pnl.index, test_pnl.values, 
                        label=f'{exp_name} (Sharpe: {sharpe:.2f}, CS: {cons_sharpe:.2f})', 
                        color=colors[idx % len(colors)], linewidth=2, alpha=0.9)
    
    ax_test.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax_test.set_ylabel('PnL ($)', fontsize=12)
    ax_test.set_xlabel('Date', fontsize=12)
    ax_test.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_test.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存圖表
    output_file = f'all_experiments_pnl_comparison_{timestamp}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 PnL 曲線圖已儲存: {output_file}")
    
    # plt.show() # Commented out for non-interactive execution


def create_pnl_summary_table(experiments: List[Dict], timestamp: str):
    """創建 PnL 彙總表格"""
    data = []
    
    for exp_info in experiments:
        exp_name = exp_info['name']
        
        # Train 數據
        train_pnl = exp_info['train_result']['equity_curve'] - 100000.0
        train_bh = exp_info['train_bh']
        train_final_pnl = train_pnl.iloc[-1]
        train_bh_final = train_bh.iloc[-1]
        train_outperformance = train_final_pnl - train_bh_final
        
        # Test 數據
        if exp_info['test_result'] and 'equity_curve' in exp_info['test_result'] and not exp_info['test_result']['equity_curve'].empty:
            test_pnl = exp_info['test_result']['equity_curve'] - 100000.0
            test_bh = exp_info['test_bh']
            test_final_pnl = test_pnl.iloc[-1]
            test_bh_final = test_bh.iloc[-1] if not test_bh.empty else 0
            test_outperformance = test_final_pnl - test_bh_final
            
            test_gp_pnl_str = f'${test_final_pnl:,.0f}'
            test_bh_pnl_str = f'${test_bh_final:,.0f}'
            test_out_str = f'${test_outperformance:,.0f}'
            test_fit_str = f'{exp_info["test_fitness"]:.6f}'
        else:
            test_gp_pnl_str = 'N/A'
            test_bh_pnl_str = 'N/A'
            test_out_str = 'N/A'
            test_fit_str = 'N/A'
        
        data.append({
            'Experiment': exp_name,
            'Train GP PnL': f'${train_final_pnl:,.0f}',
            'Train B&H PnL': f'${train_bh_final:,.0f}',
            'Train Outperformance': f'${train_outperformance:,.0f}',
            'Test GP PnL': test_gp_pnl_str,
            'Test B&H PnL': test_bh_pnl_str,
            'Test Outperformance': test_out_str,
            'Train Fitness': f'{exp_info["train_fitness"]:.6f}',
            'Test Fitness': test_fit_str
        })
    
    df = pd.DataFrame(data)
    
    # 儲存表格
    output_file = f'all_experiments_pnl_summary_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"📄 PnL 彙總表格已儲存: {output_file}")
    
    return df


def main():
    """主函式"""
    from datetime import datetime
    
    # 生成時間戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    base_dir = '/Users/hongyicheng/Downloads/gp_quant'
    
    print("=" * 80)
    print("所有實驗的最佳個體 PnL 曲線分析")
    print("=" * 80)
    print(f"分析時間戳: {timestamp}\n")
    
    # 定義所有實驗
    experiment_dirs = [
        ('Baseline (No Niche)', f'{base_dir}/auto_20251126_1259'),
        ('PnL Niche (dynamic_k)', f'{base_dir}/pnl_niche_test_records_dynamic_k_20251126_0354'),
        ('PnL Niche (fixed_k=3)', f'{base_dir}/pnl_niche_test_records_fixed_k_3_20251126_0134'),
        ('PnL Niche (fixed_k=5)', f'{base_dir}/pnl_niche_test_records_fixed_k_5_20251126_0152')
    ]
    
    experiments_data = []
    
    for exp_name, exp_dir in experiment_dirs:
        print(f"\n處理實驗: {exp_name}")
        print(f"  目錄: {exp_dir}")
        
        exp_path = Path(exp_dir)
        
        # 載入配置
        with open(exp_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # 載入最佳個體
        print("  載入最佳個體...")
        best_ind, train_fitness = load_best_individual(exp_dir)
        
        # 載入數據
        print("  載入數據...")
        tickers_dir = Path(config['data']['tickers_dir'])
        csv_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
        tickers = [f.replace('.csv', '') for f in csv_files]
        
        raw_data = load_and_process_data(str(tickers_dir), tickers)
        
        train_data, test_data = split_train_test_data(
            raw_data,
            train_data_start=config['data']['train_data_start'],
            train_backtest_start=config['data']['train_backtest_start'],
            train_backtest_end=config['data']['train_backtest_end'],
            test_data_start=config['data']['test_data_start'],
            test_backtest_start=config['data']['test_backtest_start'],
            test_backtest_end=config['data']['test_backtest_end']
        )
        
        # Train Period 回測
        print("  回測 Train Period...")
        train_result = backtest_individual(best_ind, train_data, config, 'train')
        train_bh = calculate_buy_and_hold(train_data, config, 'train')
        
        # Test Period 回測
        print("  回測 Test Period...")
        test_result = backtest_individual(best_ind, test_data, config, 'test')
        test_bh = calculate_buy_and_hold(test_data, config, 'test')
        test_fitness = test_result['metrics']['excess_return']
        
        # 儲存結果
        experiments_data.append({
            'name': exp_name,
            'train_result': train_result,
            'test_result': test_result,
            'train_bh': train_bh,
            'test_bh': test_bh,
            'train_fitness': train_fitness,
            'test_fitness': test_fitness
        })
        
        print(f"  ✅ 完成")
        print(f"     Train Fitness: {train_fitness:.6f}")
        print(f"     Test Fitness: {test_fitness:.6f}")
    
    # 創建 PnL 彙總表格
    print("\n" + "=" * 80)
    print("PnL 彙總表格")
    print("=" * 80)
    summary_df = create_pnl_summary_table(experiments_data, timestamp)
    print("\n" + summary_df.to_string(index=False))
    
    # 繪製所有實驗的 PnL 曲線
    print("\n" + "=" * 80)
    print("繪製 PnL 曲線...")
    print("=" * 80)
    plot_all_experiments_pnl(experiments_data, timestamp)
    
    print(f"\n✅ 分析完成！所有結果文件都標記了時間戳: {timestamp}")


if __name__ == '__main__':
    main()
