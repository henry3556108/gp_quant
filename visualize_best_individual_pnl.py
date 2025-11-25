#!/usr/bin/env python3
"""
Visualize Best Individual PnL Comparison

從實驗結果中找出 global best individual，計算其在樣本內外的 PnL curve，
並與 buy-and-hold 策略對比視覺化。
"""

import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
from deap import creator, base, gp

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators  # 導入以配置 primitive set
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_all_populations(records_dir: Path) -> Dict[int, list]:
    """
    載入所有世代的族群
    
    Args:
        records_dir: 記錄目錄路徑
        
    Returns:
        字典，鍵為世代號，值為族群列表
    """
    populations = {}
    populations_dir = records_dir / 'populations'
    
    if not populations_dir.exists():
        raise ValueError(f"Populations directory not found: {populations_dir}")
    
    gen_files = sorted(populations_dir.glob('generation_*.pkl'))
    
    print(f"📂 找到 {len(gen_files)} 個世代文件")
    
    for gen_file in gen_files:
        gen_num = int(gen_file.stem.split('_')[1])
        
        try:
            with open(gen_file, 'rb') as f:
                population = pickle.load(f)
            populations[gen_num] = population
            print(f"   ✅ 載入世代 {gen_num}: {len(population)} 個個體")
        except Exception as e:
            print(f"   ⚠️  載入世代 {gen_num} 失敗: {e}")
    
    return populations


def find_global_best_individual(populations: Dict[int, list]) -> Tuple[Any, int]:
    """
    從所有世代中找出 fitness 最高的個體
    
    Args:
        populations: 所有世代的族群字典
        
    Returns:
        (best_individual, generation_number)
    """
    best_individual = None
    best_fitness = float('-inf')
    best_generation = -1
    
    for gen_num, population in populations.items():
        for individual in population:
            fitness = individual.fitness.values[0]
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual
                best_generation = gen_num
    
    print(f"\n🏆 Global Best Individual:")
    print(f"   Generation: {best_generation}")
    print(f"   Fitness: {best_fitness:.6f}")
    print(f"   Tree Size: {len(best_individual)}")
    print(f"   Tree Depth: {best_individual.height}")
    print(f"   Tree Structure: {str(best_individual)[:200]}...")
    
    return best_individual, best_generation


def calculate_buy_and_hold_pnl(data_dict: Dict[str, pd.DataFrame], 
                                backtest_start: str, 
                                backtest_end: str,
                                initial_capital: float = 100000.0) -> pd.Series:
    """
    計算 buy-and-hold 策略的 PnL curve
    
    Args:
        data_dict: 股票數據字典
        backtest_start: 回測開始日期
        backtest_end: 回測結束日期
        initial_capital: 初始資金
        
    Returns:
        PnL curve (pd.Series)
    """
    n_tickers = len(data_dict)
    capital_per_ticker = initial_capital / n_tickers
    
    # 對每個 ticker 計算 buy-and-hold 收益
    ticker_pnls = []
    
    for ticker, df in data_dict.items():
        # 篩選回測期間
        mask = (df.index >= backtest_start) & (df.index <= backtest_end)
        df_period = df[mask].copy()
        
        if len(df_period) == 0:
            continue
        
        # 計算收益率
        initial_price = df_period['Close'].iloc[0]
        shares = capital_per_ticker / initial_price
        
        # 計算每日價值
        portfolio_value = shares * df_period['Close']
        pnl = portfolio_value - capital_per_ticker
        
        ticker_pnls.append(pnl)
    
    # 合併所有 ticker 的 PnL
    if len(ticker_pnls) == 0:
        return pd.Series(dtype=float)
    
    # 對齊日期並求和
    combined_pnl = pd.concat(ticker_pnls, axis=1).sum(axis=1)
    
    return combined_pnl


def visualize_pnl_comparison(best_individual: Any,
                             train_data: Dict,
                             test_data: Dict,
                             config: Dict,
                             output_path: Path):
    """
    視覺化最佳個體與 buy-and-hold 的 PnL 對比
    
    Args:
        best_individual: 最佳個體
        train_data: 訓練數據
        test_data: 測試數據
        config: 配置字典
        output_path: 輸出圖表路徑
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # ========== 上子圖：Train Period ==========
    ax_train = axes[0]
    
    print("\n💰 計算訓練期 PnL curves...")
    
    # 提取訓練數據
    train_data_dict = {ticker: info['data'] for ticker, info in train_data.items()}
    
    # 創建訓練期回測引擎
    train_engine = PortfolioBacktestingEngine(
        data=train_data_dict,
        backtest_start=config['data']['train_backtest_start'],
        backtest_end=config['data']['train_backtest_end'],
        initial_capital=100000.0
    )
    
    # 計算最佳個體的 PnL 和交易記錄
    try:
        backtest_result_train = train_engine.backtest(best_individual)
        equity_curve_train = backtest_result_train['equity_curve']
        best_pnl_train = equity_curve_train - train_engine.initial_capital
        transactions_train = backtest_result_train['transactions']
        print(f"   ✅ Best Individual PnL (Train): {len(best_pnl_train)} 個交易日")
        print(f"   ✅ 交易次數 (Train): {len(transactions_train)} 筆")
    except Exception as e:
        print(f"   ❌ 計算 Best Individual PnL (Train) 失敗: {e}")
        import traceback
        traceback.print_exc()
        best_pnl_train = pd.Series(dtype=float)
        transactions_train = []
    
    # 計算 buy-and-hold 的 PnL
    try:
        bh_pnl_train = calculate_buy_and_hold_pnl(
            train_data_dict,
            config['data']['train_backtest_start'],
            config['data']['train_backtest_end'],
            initial_capital=100000.0
        )
        print(f"   ✅ Buy-and-Hold PnL (Train): {len(bh_pnl_train)} 個交易日")
    except Exception as e:
        print(f"   ❌ 計算 Buy-and-Hold PnL (Train) 失敗: {e}")
        bh_pnl_train = pd.Series(dtype=float)
    
    # 繪製訓練期
    if len(best_pnl_train) > 0:
        ax_train.plot(best_pnl_train.index, best_pnl_train.values, 
                     label=f'Best Individual (Fitness: {best_individual.fitness.values[0]:.4f})', 
                     linewidth=2.5, color='#2E86AB', alpha=0.9)
    
    if len(bh_pnl_train) > 0:
        ax_train.plot(bh_pnl_train.index, bh_pnl_train.values, 
                     label='Buy-and-Hold', 
                     linewidth=2.5, color='#A23B72', alpha=0.9, linestyle='--')
    
    # 標註進出場點
    if len(transactions_train) > 0 and len(best_pnl_train) > 0:
        buy_dates = []
        sell_dates = []
        for _, txn in transactions_train.iterrows():
            if txn['date'] in best_pnl_train.index:
                if txn['action'] == 'BUY':
                    buy_dates.append(txn['date'])
                elif txn['action'] == 'SELL':
                    sell_dates.append(txn['date'])
        
        # 繪製買入點（綠色向上三角形）
        if buy_dates:
            buy_pnls = [best_pnl_train.loc[d] for d in buy_dates]
            ax_train.scatter(buy_dates, buy_pnls, marker='^', s=100, 
                           color='green', alpha=0.6, zorder=5, label='Entry')
        
        # 繪製賣出點（紅色向下三角形）
        if sell_dates:
            sell_pnls = [best_pnl_train.loc[d] for d in sell_dates]
            ax_train.scatter(sell_dates, sell_pnls, marker='v', s=100, 
                           color='red', alpha=0.6, zorder=5, label='Exit')
    
    ax_train.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_train.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax_train.set_ylabel('Cumulative PnL ($)', fontsize=13, fontweight='bold')
    ax_train.set_title('Training Period: Best Individual vs Buy-and-Hold', 
                      fontsize=15, fontweight='bold', pad=15)
    ax_train.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_train.grid(True, alpha=0.3, linestyle='--')
    
    # 添加統計信息
    if len(best_pnl_train) > 0 and len(bh_pnl_train) > 0:
        best_final = best_pnl_train.iloc[-1]
        bh_final = bh_pnl_train.iloc[-1]
        outperformance = best_final - bh_final
        
        stats_text = f'Best Final PnL: ${best_final:,.2f} | BH Final PnL: ${bh_final:,.2f} | Outperformance: ${outperformance:,.2f}'
        ax_train.text(0.5, 0.02, stats_text, transform=ax_train.transAxes,
                     ha='center', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== 下子圖：Test Period ==========
    ax_test = axes[1]
    
    print("\n💰 計算測試期 PnL curves...")
    
    # 提取測試數據
    test_data_dict = {ticker: info['data'] for ticker, info in test_data.items()}
    
    # 創建測試期回測引擎
    test_engine = PortfolioBacktestingEngine(
        data=test_data_dict,
        backtest_start=config['data']['test_backtest_start'],
        backtest_end=config['data']['test_backtest_end'],
        initial_capital=100000.0
    )
    
    # 計算最佳個體的 PnL 和交易記錄
    try:
        backtest_result_test = test_engine.backtest(best_individual)
        equity_curve_test = backtest_result_test['equity_curve']
        best_pnl_test = equity_curve_test - test_engine.initial_capital
        transactions_test = backtest_result_test['transactions']
        print(f"   ✅ Best Individual PnL (Test): {len(best_pnl_test)} 個交易日")
        print(f"   ✅ 交易次數 (Test): {len(transactions_test)} 筆")
    except Exception as e:
        print(f"   ❌ 計算 Best Individual PnL (Test) 失敗: {e}")
        import traceback
        traceback.print_exc()
        best_pnl_test = pd.Series(dtype=float)
        transactions_test = []
    
    # 計算 buy-and-hold 的 PnL
    try:
        bh_pnl_test = calculate_buy_and_hold_pnl(
            test_data_dict,
            config['data']['test_backtest_start'],
            config['data']['test_backtest_end'],
            initial_capital=100000.0
        )
        print(f"   ✅ Buy-and-Hold PnL (Test): {len(bh_pnl_test)} 個交易日")
    except Exception as e:
        print(f"   ❌ 計算 Buy-and-Hold PnL (Test) 失敗: {e}")
        bh_pnl_test = pd.Series(dtype=float)
    
    # 繪製測試期
    if len(best_pnl_test) > 0:
        ax_test.plot(best_pnl_test.index, best_pnl_test.values, 
                    label=f'Best Individual (Fitness: {best_individual.fitness.values[0]:.4f})', 
                    linewidth=2.5, color='#2E86AB', alpha=0.9)
    
    if len(bh_pnl_test) > 0:
        ax_test.plot(bh_pnl_test.index, bh_pnl_test.values, 
                    label='Buy-and-Hold', 
                    linewidth=2.5, color='#A23B72', alpha=0.9, linestyle='--')
    
    # 標註進出場點
    if len(transactions_test) > 0 and len(best_pnl_test) > 0:
        buy_dates = []
        sell_dates = []
        for _, txn in transactions_test.iterrows():
            if txn['date'] in best_pnl_test.index:
                if txn['action'] == 'BUY':
                    buy_dates.append(txn['date'])
                elif txn['action'] == 'SELL':
                    sell_dates.append(txn['date'])
        
        # 繪製買入點（綠色向上三角形）
        if buy_dates:
            buy_pnls = [best_pnl_test.loc[d] for d in buy_dates]
            ax_test.scatter(buy_dates, buy_pnls, marker='^', s=100, 
                          color='green', alpha=0.6, zorder=5, label='Entry')
        
        # 繪製賣出點（紅色向下三角形）
        if sell_dates:
            sell_pnls = [best_pnl_test.loc[d] for d in sell_dates]
            ax_test.scatter(sell_dates, sell_pnls, marker='v', s=100, 
                          color='red', alpha=0.6, zorder=5, label='Exit')
    
    ax_test.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_test.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax_test.set_ylabel('Cumulative PnL ($)', fontsize=13, fontweight='bold')
    ax_test.set_title('Test Period (Out-of-Sample): Best Individual vs Buy-and-Hold', 
                     fontsize=15, fontweight='bold', pad=15)
    ax_test.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_test.grid(True, alpha=0.3, linestyle='--')
    
    # 添加統計信息
    if len(best_pnl_test) > 0 and len(bh_pnl_test) > 0:
        best_final = best_pnl_test.iloc[-1]
        bh_final = bh_pnl_test.iloc[-1]
        outperformance = best_final - bh_final
        
        stats_text = f'Best Final PnL: ${best_final:,.2f} | BH Final PnL: ${bh_final:,.2f} | Outperformance: ${outperformance:,.2f}'
        ax_test.text(0.5, 0.02, stats_text, transform=ax_test.transAxes,
                    ha='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="視覺化最佳個體與 buy-and-hold 策略的 PnL 對比"
    )
    parser.add_argument(
        '--records',
        type=str,
        required=True,
        help='實驗記錄目錄路徑'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路徑'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='輸出圖表路徑（默認保存在記錄目錄中）'
    )
    
    args = parser.parse_args()
    
    records_dir = Path(args.records)
    config_file = Path(args.config)
    
    if not records_dir.exists():
        print(f"❌ 記錄目錄不存在: {records_dir}")
        return
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    print("=" * 80)
    print("🎯 Best Individual PnL Comparison")
    print("=" * 80)
    print(f"Records directory: {records_dir}")
    print(f"Config file: {config_file}\n")
    
    # 1. 設置 DEAP
    setup_deap_creator()
    
    # 2. 載入配置
    print("📋 載入配置...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 3. 載入所有世代的族群
    print("\n📦 載入所有世代的族群...")
    populations = load_all_populations(records_dir)
    
    if len(populations) == 0:
        print("❌ 沒有找到任何世代數據")
        return
    
    # 4. 找出 global best individual
    best_individual, best_generation = find_global_best_individual(populations)
    
    if best_individual is None:
        print("❌ 無法找到最佳個體")
        return
    
    # 5. 載入數據
    print("\n📊 載入數據...")
    tickers_dir = config['data']['tickers_dir']
    
    # 自動發現 tickers
    import os
    ticker_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in ticker_files]
    print(f"   發現 {len(tickers)} 個 ticker: {tickers[:5]}...")
    
    data = load_and_process_data(tickers_dir, tickers)
    train_data, test_data = split_train_test_data(
        data,
        train_data_start=config['data']['train_data_start'],
        train_backtest_start=config['data']['train_backtest_start'],
        train_backtest_end=config['data']['train_backtest_end'],
        test_data_start=config['data']['test_data_start'],
        test_backtest_start=config['data']['test_backtest_start'],
        test_backtest_end=config['data']['test_backtest_end']
    )
    print(f"✅ 載入 {len(train_data)} 個股票的數據")
    
    # 6. 視覺化
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = records_dir / 'best_individual_pnl_comparison.png'
    
    visualize_pnl_comparison(
        best_individual,
        train_data,
        test_data,
        config,
        output_path
    )
    
    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
