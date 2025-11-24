"""
Test PnL Diversity Calculation

從實驗結果中選擇 5 個個體，計算他們的 PnL 相關性矩陣並視覺化
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from deap import creator, base, gp
from typing import List, Dict, Any

from gp_quant.evolution.components.gp import operators  # 導入以配置 primitive set
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_individuals_from_generation(records_dir: Path, generation: int) -> List:
    """從指定世代載入個體"""
    pkl_file = records_dir / "populations" / f"generation_{generation:03d}.pkl"
    
    if not pkl_file.exists():
        raise FileNotFoundError(f"Generation file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        population = pickle.load(f)
    
    print(f"✅ 載入世代 {generation}: {len(population)} 個個體")
    return population


def select_diverse_individuals(population: List, n: int = 5) -> List:
    """
    選擇多樣化的個體
    
    策略：選擇不同適應度範圍的個體
    """
    # 按適應度排序
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
    
    # 選擇分佈在不同適應度區間的個體
    indices = np.linspace(0, len(sorted_pop) - 1, n, dtype=int)
    selected = [sorted_pop[i] for i in indices]
    
    print(f"\n📊 選擇的 {n} 個個體:")
    for i, ind in enumerate(selected):
        print(f"   {i+1}. Fitness: {ind.fitness.values[0]:.4f}, Size: {len(ind)}, Height: {ind.height}")
    
    return selected


def calculate_pnl_curves(individuals: List, train_data: Dict, backtest_config: Dict) -> tuple:
    """計算個體的 PnL 曲線"""
    # PortfolioBacktestingEngine 需要每個 ticker 的數據字典
    # 從 train_data 提取實際數據
    data_dict = {ticker: info['data'] for ticker, info in train_data.items()}
    
    # 初始化回測引擎
    engine = PortfolioBacktestingEngine(
        data=data_dict,
        backtest_start=backtest_config['backtest_start'],
        backtest_end=backtest_config['backtest_end'],
        initial_capital=100000.0
    )
    
    pnl_curves = []
    valid_individuals = []
    
    print(f"\n💰 計算 PnL 曲線...")
    
    for i, individual in enumerate(individuals):
        try:
            pnl_curve = engine.get_pnl_curve(individual)
            
            # 檢查是否有效（不是全零）
            if len(pnl_curve) > 0 and not np.allclose(pnl_curve.values, 0):
                pnl_curves.append(pnl_curve)
                valid_individuals.append(individual)
                print(f"   ✅ 個體 {i+1} (Fitness: {individual.fitness.values[0]:.4f}): Valid PnL curve")
            else:
                print(f"   ⚠️  個體 {i+1} (Fitness: {individual.fitness.values[0]:.4f}): Invalid PnL curve (all zeros)")
                
        except Exception as e:
            print(f"   ❌ 個體 {i+1}: Error - {e}")
    
    return pnl_curves, valid_individuals


def calculate_correlation_matrix(pnl_curves: List[pd.Series]) -> np.ndarray:
    """計算 PnL 曲線的相關性矩陣"""
    n = len(pnl_curves)
    corr_matrix = np.zeros((n, n))
    
    print(f"\n📈 計算相關性矩陣 ({n} x {n})...")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                # 計算 Pearson 相關係數
                corr = np.corrcoef(pnl_curves[i].values, pnl_curves[j].values)[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                print(f"   Corr({i+1}, {j+1}) = {corr:.4f}")
    
    return corr_matrix


def visualize_pnl_curves(pnl_curves: List[pd.Series], individuals: List, output_path: Path):
    """視覺化 PnL 曲線"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子圖 1: 累積 PnL
    ax1 = axes[0]
    for i, pnl in enumerate(pnl_curves):
        fitness = individuals[i].fitness.values[0]
        ax1.plot(pnl.index, pnl.values, label=f'Individual {i+1} (Fitness: {fitness:.4f})', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative PnL ($)', fontsize=12)
    ax1.set_title('PnL Curves of Selected Individuals', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 子圖 2: 標準化 PnL (方便比較形狀)
    ax2 = axes[1]
    for i, pnl in enumerate(pnl_curves):
        # 標準化: (x - mean) / std
        normalized = (pnl - pnl.mean()) / pnl.std()
        fitness = individuals[i].fitness.values[0]
        ax2.plot(normalized.index, normalized.values, label=f'Individual {i+1} (Fitness: {fitness:.4f})', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Normalized PnL (z-score)', fontsize=12)
    ax2.set_title('Normalized PnL Curves (for Shape Comparison)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ PnL curves saved: {output_path}")
    plt.close()


def visualize_correlation_matrix(corr_matrix: np.ndarray, individuals: List, output_path: Path):
    """視覺化相關性矩陣"""
    n = len(individuals)
    
    # 創建標籤
    labels = [f'Ind {i+1}\n(F:{ind.fitness.values[0]:.3f})' for i, ind in enumerate(individuals)]
    
    # 繪製熱圖
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 seaborn 繪製熱圖
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title('PnL Correlation Matrix Between Individuals', fontsize=14, fontweight='bold', pad=20)
    
    # 添加統計信息
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper_tri]
    
    stats_text = f'Mean: {np.mean(correlations):.3f} | Std: {np.std(correlations):.3f} | Min: {np.min(correlations):.3f} | Max: {np.max(correlations):.3f}'
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Correlation matrix saved: {output_path}")
    plt.close()


def main():
    records_dir = Path("test_evolution_records_20251125_0000")
    
    print("=" * 80)
    print("🧪 PnL Diversity Test")
    print("=" * 80)
    print(f"Records directory: {records_dir}\n")
    
    # 1. 設置 DEAP
    setup_deap_creator()
    
    # 2. 從實驗結果目錄載入配置（使用實驗時的配置）
    config_file = records_dir / "config.json"
    print(f"📋 載入實驗配置: {config_file}")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"   Tickers directory: {config['data']['tickers_dir']}")
    print(f"   Train backtest: {config['data']['train_backtest_start']} to {config['data']['train_backtest_end']}")
    
    # 3. 載入數據（從實驗配置中的 tickers_dir）
    print(f"\n📊 載入數據...")
    tickers_dir = config['data']['tickers_dir']
    
    # 自動發現 tickers_dir 中的所有 CSV 文件
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
    print(f"✅ 載入 {len(train_data)} 個股票的訓練數據\n")
    
    # 4. 載入初始世代的個體（與 best_signals 對應）
    generation = 0  # 使用初始世代，與 best_signals/generation_000 對應
    population = load_individuals_from_generation(records_dir, generation)
    
    # 5. 選擇 5 個多樣化的個體
    selected_individuals = select_diverse_individuals(population, n=5)
    
    # 6. 計算 PnL 曲線
    backtest_config = {
        'backtest_start': config['data']['train_backtest_start'],
        'backtest_end': config['data']['train_backtest_end']
    }
    pnl_curves, valid_individuals = calculate_pnl_curves(selected_individuals, train_data, backtest_config)
    
    if len(pnl_curves) < 2:
        print("\n❌ 沒有足夠的有效 PnL 曲線進行分析")
        return
    
    # 7. 計算相關性矩陣
    corr_matrix = calculate_correlation_matrix(pnl_curves)
    
    # 8. 視覺化
    print("\n🎨 生成視覺化...")
    visualize_pnl_curves(pnl_curves, valid_individuals, records_dir / "pnl_curves_comparison.png")
    visualize_correlation_matrix(corr_matrix, valid_individuals, records_dir / "pnl_correlation_matrix.png")
    
    # 9. 輸出統計摘要
    print("\n" + "=" * 80)
    print("📊 PnL Correlation Statistics")
    print("=" * 80)
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper_tri]
    
    print(f"Number of individuals: {len(valid_individuals)}")
    print(f"Number of correlation pairs: {len(correlations)}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Std correlation: {np.std(correlations):.4f}")
    print(f"Min correlation: {np.min(correlations):.4f}")
    print(f"Max correlation: {np.max(correlations):.4f}")
    print(f"Median correlation: {np.median(correlations):.4f}")
    print("=" * 80)
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
