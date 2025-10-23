"""
從 generation.pkl 中提取 Top3 個體並繪製它們的表現曲線

使用方法:
python visualize_top3_individuals.py <generation.pkl 路徑>
"""

import sys
from pathlib import Path
import dill
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

# 添加項目根目錄到 path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from deap import base, creator, gp, tools
from gp_quant.gp.operators import pset
from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine

# 初始化 DEAP creator
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_generation(pkl_path):
    """載入 generation.pkl"""
    print(f"📂 載入文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    
    generation = data['generation']
    population = data['population']
    hall_of_fame = data.get('hall_of_fame', [])
    
    print(f"   ✓ Generation: {generation}")
    print(f"   ✓ Population size: {len(population)}")
    print(f"   ✓ Hall of Fame size: {len(hall_of_fame)}")
    
    return data


def get_top3_individuals(population):
    """從 population 中獲取 top3 個體"""
    # 按 fitness 排序
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    
    top3 = sorted_pop[:3]
    
    print(f"\n🏆 Top 3 個體:")
    for i, ind in enumerate(top3, 1):
        fitness = ind.fitness.values[0]
        print(f"   {i}. Fitness: {fitness:.4f}")
        print(f"      深度: {ind.height}, 節點數: {len(ind)}")
        print(f"      規則: {str(ind)[:80]}{'...' if len(str(ind)) > 80 else ''}")
    
    return top3


def backtest_individual(individual, engine, ticker_list):
    """回測單個個體並返回交易記錄"""
    # 執行回測
    result = engine.backtest(individual)
    
    # 提取 equity curve
    equity_curve = result['equity_curve']
    per_stock_pnl = result['per_stock_pnl']
    
    # 轉換為 DataFrame 格式
    equity_df = pd.DataFrame({
        'date': equity_curve.index,
        'portfolio_value': equity_curve.values,
        'cumulative_pnl': equity_curve.values - engine.initial_capital
    })
    
    # 每個股票的 PnL（簡化版，只有最終值）
    pnl_by_ticker = {}
    for ticker in ticker_list:
        pnl = per_stock_pnl.get(ticker, 0)
        pnl_by_ticker[ticker] = pnl
    
    return equity_df, pnl_by_ticker, result['metrics']


def calculate_buy_and_hold(data, ticker_list, backtest_start, backtest_end, initial_capital_per_stock=25000.0):
    """計算 Buy-and-Hold 基準"""
    bh_data = {}
    
    for ticker in ticker_list:
        df = data[ticker]
        df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
        
        if len(df) == 0:
            continue
        
        # 第一天買入
        first_price = df['Close'].iloc[0]
        shares = initial_capital_per_stock / first_price
        
        # 計算每日 PnL
        df = df.copy()
        df['portfolio_value'] = df['Close'] * shares
        df['pnl'] = df['portfolio_value'] - initial_capital_per_stock
        
        bh_data[ticker] = df[['pnl']].reset_index()
        bh_data[ticker].columns = ['date', 'pnl']
    
    # 合併所有股票的 PnL
    all_dates = set()
    for ticker_df in bh_data.values():
        all_dates.update(ticker_df['date'])
    
    all_dates = sorted(all_dates)
    
    # 計算總 PnL
    total_pnl = []
    for date in all_dates:
        daily_pnl = 0
        for ticker_df in bh_data.values():
            ticker_pnl = ticker_df[ticker_df['date'] == date]['pnl']
            if len(ticker_pnl) > 0:
                daily_pnl += ticker_pnl.iloc[0]
        total_pnl.append(daily_pnl)
    
    bh_df = pd.DataFrame({'date': all_dates, 'pnl': total_pnl})
    
    return bh_df


def plot_top3_performance(top3_results, bh_df, output_path, period_name):
    """繪製 Top3 個體的表現曲線"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Top 3 Individuals Performance - {period_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 藍、紫、橙
    
    # 為每個 Top 個體繪製圖表
    for idx, (individual, equity_df, pnl_by_ticker, fitness, metrics) in enumerate(top3_results):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # 繪製總 PnL
        if len(equity_df) > 0:
            ax.plot(equity_df['date'], equity_df['cumulative_pnl'], 
                   color=colors[idx], linewidth=2.5, label=f'GP Strategy', alpha=0.9)
        
        # 繪製 Buy-and-Hold
        if len(bh_df) > 0:
            ax.plot(bh_df['date'], bh_df['pnl'], 
                   color='gray', linewidth=2, linestyle='--', 
                   label='Buy-and-Hold', alpha=0.7)
        
        # 添加零線
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 設置標題和標籤
        ax.set_title(f'Top {idx + 1} - Fitness: {fitness:.4f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative PnL ($)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 旋轉 x 軸標籤
        ax.tick_params(axis='x', rotation=45)
        
        # 添加最終 PnL 標註
        if len(equity_df) > 0:
            final_pnl = equity_df['cumulative_pnl'].iloc[-1]
            sharpe = metrics.get('sharpe_ratio', 0)
            ax.text(0.02, 0.98, f'Final PnL: ${final_pnl:,.0f}\nSharpe: {sharpe:.4f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 如果只有 2 個或 1 個，隱藏多餘的子圖
    for idx in range(len(top3_results), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path}")
    plt.close()


def plot_top3_comparison(top3_results, bh_df, output_path, period_name):
    """繪製 Top3 個體的對比圖（單張圖）"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 藍、紫、橙
    
    # 繪製每個 Top 個體
    for idx, (individual, equity_df, pnl_by_ticker, fitness, metrics) in enumerate(top3_results):
        if len(equity_df) > 0:
            ax.plot(equity_df['date'], equity_df['cumulative_pnl'], 
                   color=colors[idx], linewidth=2.5, 
                   label=f'Top {idx + 1} (Fitness: {fitness:.4f})', 
                   alpha=0.8)
    
    # 繪製 Buy-and-Hold
    if len(bh_df) > 0:
        ax.plot(bh_df['date'], bh_df['pnl'], 
               color='gray', linewidth=2.5, linestyle='--', 
               label='Buy-and-Hold', alpha=0.7)
    
    # 添加零線
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 設置標題和標籤
    ax.set_title(f'Top 3 Individuals Performance Comparison - {period_name}', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative PnL ($)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 旋轉 x 軸標籤
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("使用方法: python visualize_top3_individuals.py <generation.pkl 路徑>")
        print("範例: python visualize_top3_individuals.py portfolio_experiment_results/.../generations/generation_006_final.pkl")
        sys.exit(1)
    
    pkl_path = Path(sys.argv[1])
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("🎨 Top 3 個體表現可視化")
    print("="*100 + "\n")
    
    # 1. 載入 generation
    data = load_generation(pkl_path)
    population = data['population']
    generation = data['generation']
    
    # 2. 獲取 top3
    top3 = get_top3_individuals(population)
    
    # 3. 讀取實驗配置
    exp_dir = pkl_path.parent.parent
    config_file = exp_dir / "config.json"
    
    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"\n📋 實驗配置:")
    print(f"   股票: {', '.join(config['tickers'])}")
    print(f"   訓練期: {config['train_backtest_start']} ~ {config['train_backtest_end']}")
    print(f"   測試期: {config['test_backtest_start']} ~ {config['test_backtest_end']}")
    
    # 4. 載入股價數據
    print(f"\n📊 載入股價數據...")
    stock_data = {}
    for ticker in config['tickers']:
        file_path = project_root / f"TSE300_selected/{ticker}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            stock_data[ticker] = df
            print(f"   ✓ {ticker}: {len(df)} 天")
        else:
            print(f"   ✗ {ticker}: 文件不存在")
    
    # 5. 初始化回測引擎（訓練期和測試期）
    print(f"\n🔧 初始化回測引擎...")
    
    train_engine = PortfolioBacktestingEngine(
        data=stock_data,
        backtest_start=config['train_backtest_start'],
        backtest_end=config['train_backtest_end'],
        initial_capital=config['initial_capital'],
        pset=pset
    )
    print(f"   ✓ 訓練期引擎: {len(train_engine.common_dates)} 天")
    
    test_engine = PortfolioBacktestingEngine(
        data=stock_data,
        backtest_start=config['test_backtest_start'],
        backtest_end=config['test_backtest_end'],
        initial_capital=config['initial_capital'],
        pset=pset
    )
    print(f"   ✓ 測試期引擎: {len(test_engine.common_dates)} 天")
    
    # 6. 回測 Top3（訓練期）
    print(f"\n🔄 回測 Top 3 個體（訓練期）...")
    train_results = []
    for i, ind in enumerate(top3, 1):
        print(f"   處理 Top {i}...")
        equity_df, pnl_by_ticker, metrics = backtest_individual(ind, train_engine, config['tickers'])
        fitness = ind.fitness.values[0]
        train_results.append((ind, equity_df, pnl_by_ticker, fitness, metrics))
    
    # 7. 回測 Top3（測試期）
    print(f"\n🔄 回測 Top 3 個體（測試期）...")
    test_results = []
    for i, ind in enumerate(top3, 1):
        print(f"   處理 Top {i}...")
        equity_df, pnl_by_ticker, metrics = backtest_individual(ind, test_engine, config['tickers'])
        fitness = ind.fitness.values[0]  # 使用訓練期的 fitness
        test_results.append((ind, equity_df, pnl_by_ticker, fitness, metrics))
    
    # 8. 計算 Buy-and-Hold 基準
    print(f"\n📈 計算 Buy-and-Hold 基準...")
    train_bh = calculate_buy_and_hold(
        stock_data, config['tickers'],
        config['train_backtest_start'], config['train_backtest_end'],
        config['initial_capital'] / len(config['tickers'])
    )
    print(f"   ✓ 訓練期 B&H")
    
    test_bh = calculate_buy_and_hold(
        stock_data, config['tickers'],
        config['test_backtest_start'], config['test_backtest_end'],
        config['initial_capital'] / len(config['tickers'])
    )
    print(f"   ✓ 測試期 B&H")
    
    # 9. 創建輸出目錄
    output_dir = exp_dir / "top3_visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 輸出目錄: {output_dir}")
    
    # 10. 繪製圖表
    print(f"\n🎨 繪製圖表...")
    
    # 訓練期 - 分開的圖
    plot_top3_performance(
        train_results, train_bh,
        output_dir / f"generation_{generation:03d}_top3_train.png",
        "Training Period"
    )
    
    # 訓練期 - 對比圖
    plot_top3_comparison(
        train_results, train_bh,
        output_dir / f"generation_{generation:03d}_top3_train_comparison.png",
        "Training Period"
    )
    
    # 測試期 - 分開的圖
    plot_top3_performance(
        test_results, test_bh,
        output_dir / f"generation_{generation:03d}_top3_test.png",
        "Testing Period"
    )
    
    # 測試期 - 對比圖
    plot_top3_comparison(
        test_results, test_bh,
        output_dir / f"generation_{generation:03d}_top3_test_comparison.png",
        "Testing Period"
    )
    
    # 11. 儲存 Top3 的 equity curve
    print(f"\n💾 儲存 Equity Curve...")
    for i, (ind, equity_df, _, fitness, metrics) in enumerate(train_results, 1):
        equity_df.to_csv(output_dir / f"top{i}_train_equity.csv", index=False)
        print(f"   ✓ Top {i} 訓練期 equity curve")
    
    for i, (ind, equity_df, _, fitness, metrics) in enumerate(test_results, 1):
        equity_df.to_csv(output_dir / f"top{i}_test_equity.csv", index=False)
        print(f"   ✓ Top {i} 測試期 equity curve")
    
    # 12. 儲存 Top3 的規則和指標
    print(f"\n📝 儲存交易規則和指標...")
    with open(output_dir / "top3_rules_and_metrics.txt", 'w') as f:
        for i, (ind, _, _, fitness, metrics) in enumerate(train_results, 1):
            f.write(f"Top {i} (Training Fitness: {fitness:.4f})\n")
            f.write(f"深度: {ind.height}, 節點數: {len(ind)}\n")
            f.write(f"規則: {str(ind)}\n")
            f.write(f"\n訓練期指標:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "="*80 + "\n\n")
    print(f"   ✓ 已儲存: top3_rules_and_metrics.txt")
    
    # 13. 完成
    print("\n" + "="*100)
    print("✅ 可視化完成！")
    print("="*100)
    print(f"\n📊 生成的文件:")
    print(f"  訓練期:")
    print(f"    - generation_{generation:03d}_top3_train.png (分開顯示)")
    print(f"    - generation_{generation:03d}_top3_train_comparison.png (對比圖)")
    print(f"  測試期:")
    print(f"    - generation_{generation:03d}_top3_test.png (分開顯示)")
    print(f"    - generation_{generation:03d}_top3_test_comparison.png (對比圖)")
    print(f"  Equity Curves:")
    print(f"    - top1/2/3_train_equity.csv")
    print(f"    - top1/2/3_test_equity.csv")
    print(f"  規則和指標:")
    print(f"    - top3_rules_and_metrics.txt")
    print(f"\n📁 保存位置: {output_dir}")
    print()


if __name__ == "__main__":
    main()
