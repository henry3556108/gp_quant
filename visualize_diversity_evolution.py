#!/usr/bin/env python3
"""
Visualize Diversity Evolution

計算所有世代的多樣性指標（平均 PnL correlation 和平均標準化 TED distance），
並繪製演化趨勢折線圖。
"""

import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple
from deap import creator, base, gp
from joblib import Parallel, delayed
from tqdm import tqdm

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.data.loader import load_and_process_data, split_train_test_data
from gp_quant.similarity.tree_edit_distance import compute_ted


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_all_generations(records_dir: Path) -> Dict[int, List]:
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
    
    if len(gen_files) == 0:
        raise ValueError("No generation files found")
    
    print(f"📂 找到 {len(gen_files)} 個世代文件")
    
    for gen_file in gen_files:
        gen_num = int(gen_file.stem.split('_')[1])
        
        try:
            with open(gen_file, 'rb') as f:
                population = pickle.load(f)
            populations[gen_num] = population
        except Exception as e:
            print(f"   ⚠️  載入世代 {gen_num} 失敗: {e}")
    
    print(f"✅ 成功載入 {len(populations)} 個世代\n")
    
    return populations


def sample_population(population: List, sample_size: int, strategy: str = 'stratified') -> List:
    """
    從族群中採樣
    
    Args:
        population: 完整族群
        sample_size: 採樣大小
        strategy: 採樣策略
        
    Returns:
        採樣後的族群
    """
    n = len(population)
    
    if sample_size >= n:
        return population
    
    if strategy == 'stratified':
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        indices = np.linspace(0, n - 1, sample_size, dtype=int)
        sampled = [sorted_pop[i] for i in indices]
    else:
        indices = np.random.choice(n, sample_size, replace=False)
        sampled = [population[i] for i in indices]
    
    return sampled


def calculate_pnl_for_individual_worker(individual: Any, 
                                         train_data_dict: Dict,
                                         backtest_start: str,
                                         backtest_end: str) -> np.ndarray:
    """
    計算單個個體的 PnL curve（worker 函數，在子進程中執行）
    
    每個子進程創建自己的 engine，避免序列化問題
    """
    try:
        # 在子進程中創建 engine
        engine = PortfolioBacktestingEngine(
            data=train_data_dict,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            initial_capital=100000.0
        )
        pnl_curve = engine.get_pnl_curve(individual)
        return pnl_curve.values
    except Exception as e:
        # 如果失敗，返回全零（需要知道長度，使用固定值）
        return np.zeros(504)  # 默認訓練期長度


def calculate_pnl_diversity(population: List,
                            train_data_dict: Dict,
                            backtest_start: str,
                            backtest_end: str,
                            n_jobs: int = 4) -> Tuple[float, int, int]:
    """
    計算 PnL diversity（平均相關性）
    
    Args:
        population: 族群列表
        train_data_dict: 訓練數據字典
        backtest_start: 回測開始日期
        backtest_end: 回測結束日期
        n_jobs: 平行處理器數量
        
    Returns:
        (mean_correlation, valid_count, total_count)
    """
    n = len(population)
    
    # 平行計算所有個體的 PnL curves
    # 使用 threading backend 避免 DEAP creator 的 pickle 問題
    pnl_curves = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(calculate_pnl_for_individual_worker)(
            ind, train_data_dict, backtest_start, backtest_end
        )
        for ind in population
    )
    
    # 轉換為矩陣
    pnl_matrix = np.array(pnl_curves)
    
    # 檢查有效性
    valid_mask = ~np.all(pnl_matrix == 0, axis=1)
    invalid_indices = np.where(~valid_mask)[0].tolist()
    valid_count = np.sum(valid_mask)
    
    # 計算相關性矩陣
    corr_matrix = np.corrcoef(pnl_matrix)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 對於無效個體，設為 0
    for idx in invalid_indices:
        corr_matrix[idx, :] = 0.0
        corr_matrix[:, idx] = 0.0
        corr_matrix[idx, idx] = 1.0
    
    np.fill_diagonal(corr_matrix, 1.0)
    
    # 計算有效個體之間的平均相關性
    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if i not in invalid_indices and j not in invalid_indices:
                valid_pairs.append(corr_matrix[i, j])
    
    if len(valid_pairs) > 0:
        mean_corr = np.mean(valid_pairs)
    else:
        mean_corr = 0.0
    
    return mean_corr, valid_count, n


def calculate_ted_for_pair(i: int, j: int, ind_i: Any, ind_j: Any) -> Tuple[int, int, float]:
    """計算一對個體的標準化 TED"""
    try:
        ted = compute_ted(ind_i, ind_j)
        max_size = max(len(ind_i), len(ind_j))
        norm_ted = ted / max_size if max_size > 0 else 0.0
        return i, j, norm_ted
    except Exception as e:
        return i, j, 1.0


def calculate_ted_diversity(population: List, n_jobs: int = 4) -> float:
    """
    計算 TED diversity（平均距離）
    
    Args:
        population: 族群列表
        n_jobs: 平行處理器數量
        
    Returns:
        mean_ted_distance
    """
    n = len(population)
    
    # 初始化矩陣
    ted_matrix = np.zeros((n, n))
    
    # 生成所有需要計算的配對
    pairs = [(i, j, population[i], population[j]) 
             for i in range(n) for j in range(i + 1, n)]
    
    # 平行計算
    # 使用 threading backend 避免 DEAP creator 的 pickle 問題
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(calculate_ted_for_pair)(i, j, ind_i, ind_j)
        for i, j, ind_i, ind_j in pairs
    )
    
    # 填充矩陣
    for i, j, ted in results:
        ted_matrix[i, j] = ted
        ted_matrix[j, i] = ted
    
    # 計算平均距離（非對角線元素）
    upper_tri = np.triu_indices(n, k=1)
    mean_ted = np.mean(ted_matrix[upper_tri])
    
    return mean_ted


def analyze_generation_diversity(generation: int,
                                 population: List,
                                 train_data_dict: Dict,
                                 backtest_start: str,
                                 backtest_end: str,
                                 sample_size: int = None,
                                 n_jobs: int = 4) -> Dict:
    """
    分析單個世代的多樣性
    
    Args:
        generation: 世代號
        population: 族群
        train_data_dict: 訓練數據字典
        backtest_start: 回測開始日期
        backtest_end: 回測結束日期
        sample_size: 採樣大小
        n_jobs: 平行處理器數量
        
    Returns:
        包含多樣性指標的字典
    """
    # 採樣
    if sample_size and sample_size < len(population):
        population = sample_population(population, sample_size, 'stratified')
    
    # 計算 PnL diversity
    mean_pnl_corr, valid_count, total_count = calculate_pnl_diversity(
        population, train_data_dict, backtest_start, backtest_end, n_jobs
    )
    
    # 計算 TED diversity
    mean_ted_dist = calculate_ted_diversity(population, n_jobs)
    
    return {
        'generation': generation,
        'population_size': len(population),
        'valid_individuals': valid_count,
        'mean_pnl_correlation': mean_pnl_corr,
        'mean_ted_distance': mean_ted_dist
    }


def visualize_diversity_evolution(diversity_stats: List[Dict],
                                  output_path: Path):
    """
    視覺化多樣性演化趨勢
    
    Args:
        diversity_stats: 每個世代的多樣性統計
        output_path: 輸出圖表路徑
    """
    # 提取數據
    generations = [s['generation'] for s in diversity_stats]
    pnl_corrs = [s['mean_pnl_correlation'] for s in diversity_stats]
    ted_dists = [s['mean_ted_distance'] for s in diversity_stats]
    
    # 創建圖表
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ========== 上子圖：PnL Correlation ==========
    ax_pnl = axes[0]
    
    ax_pnl.plot(generations, pnl_corrs, 
               marker='o', linewidth=2.5, markersize=8,
               color='#2E86AB', alpha=0.9, label='Mean PnL Correlation')
    
    ax_pnl.set_xlabel('Generation', fontsize=13, fontweight='bold')
    ax_pnl.set_ylabel('Mean PnL Correlation', fontsize=13, fontweight='bold')
    ax_pnl.set_title('Evolution of PnL Correlation (Phenotypic Diversity)', 
                    fontsize=15, fontweight='bold', pad=15)
    ax_pnl.grid(True, alpha=0.3, linestyle='--')
    ax_pnl.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 添加趨勢線
    z = np.polyfit(generations, pnl_corrs, 2)
    p = np.poly1d(z)
    ax_pnl.plot(generations, p(generations), 
               linestyle='--', color='red', alpha=0.5, linewidth=2,
               label='Trend (2nd order)')
    ax_pnl.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 統計信息
    stats_text = f'Range: [{min(pnl_corrs):.4f}, {max(pnl_corrs):.4f}] | Mean: {np.mean(pnl_corrs):.4f} | Std: {np.std(pnl_corrs):.4f}'
    ax_pnl.text(0.5, 0.02, stats_text, transform=ax_pnl.transAxes,
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== 下子圖：TED Distance ==========
    ax_ted = axes[1]
    
    ax_ted.plot(generations, ted_dists,
               marker='s', linewidth=2.5, markersize=8,
               color='#A23B72', alpha=0.9, label='Mean TED Distance')
    
    ax_ted.set_xlabel('Generation', fontsize=13, fontweight='bold')
    ax_ted.set_ylabel('Mean Normalized TED Distance', fontsize=13, fontweight='bold')
    ax_ted.set_title('Evolution of TED Distance (Genotypic Diversity)', 
                    fontsize=15, fontweight='bold', pad=15)
    ax_ted.grid(True, alpha=0.3, linestyle='--')
    ax_ted.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 添加趨勢線
    z = np.polyfit(generations, ted_dists, 2)
    p = np.poly1d(z)
    ax_ted.plot(generations, p(generations),
               linestyle='--', color='red', alpha=0.5, linewidth=2,
               label='Trend (2nd order)')
    ax_ted.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 統計信息
    stats_text = f'Range: [{min(ted_dists):.4f}, {max(ted_dists):.4f}] | Mean: {np.mean(ted_dists):.4f} | Std: {np.std(ted_dists):.4f}'
    ax_ted.text(0.5, 0.02, stats_text, transform=ax_ted.transAxes,
               ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="視覺化多樣性演化趨勢"
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
        '--sample-size',
        type=int,
        default=None,
        help='採樣大小（默認使用全部個體）'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='平行處理器數量（默認：4）'
    )
    parser.add_argument(
        '--save-matrices',
        action='store_true',
        help='保存每個世代的矩陣到 CSV'
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
    print("🎯 Diversity Evolution Analysis")
    print("=" * 80)
    print(f"Records directory: {records_dir}")
    print(f"Config file: {config_file}")
    print(f"Sample size: {args.sample_size if args.sample_size else 'All'}")
    print(f"N jobs: {args.n_jobs}\n")
    
    # 1. 設置 DEAP
    setup_deap_creator()
    
    # 2. 載入配置
    print("📋 載入配置...")
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # 3. 載入所有世代
    print("\n📦 載入所有世代...")
    populations = load_all_generations(records_dir)
    
    if len(populations) == 0:
        print("❌ 沒有找到任何世代數據")
        return
    
    # 4. 載入數據
    print("📊 載入數據...")
    import os
    tickers_dir = config['data']['tickers_dir']
    ticker_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in ticker_files]
    
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
    print(f"✅ 載入 {len(train_data)} 個股票的數據\n")
    
    # 5. 準備回測參數（傳遞給子進程）
    print("🏗️  準備回測參數...")
    train_data_dict = {ticker: info['data'] for ticker, info in train_data.items()}
    backtest_start = config['data']['train_backtest_start']
    backtest_end = config['data']['train_backtest_end']
    print("✅ 參數準備完成\n")
    
    # 6. 分析每個世代的多樣性（順序處理世代，但每個世代內部平行化）
    print("🔬 分析每個世代的多樣性...")
    print("=" * 80)
    
    diversity_stats = []
    
    for gen_num in tqdm(sorted(populations.keys()), desc="處理世代", ncols=80):
        population = populations[gen_num]
        
        stats = analyze_generation_diversity(
            gen_num,
            population,
            train_data_dict,
            backtest_start,
            backtest_end,
            args.sample_size,
            args.n_jobs
        )
        
        diversity_stats.append(stats)
        
        print(f"Gen {gen_num:3d}: PnL Corr={stats['mean_pnl_correlation']:.4f}, "
              f"TED Dist={stats['mean_ted_distance']:.4f}, "
              f"Valid={stats['valid_individuals']}/{stats['population_size']}")
    
    # 7. 保存統計摘要
    print("\n💾 保存統計摘要...")
    summary_df = pd.DataFrame(diversity_stats)
    summary_path = records_dir / 'diversity_evolution_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ 摘要已保存: {summary_path}")
    
    # 8. 視覺化
    print("\n📊 生成視覺化圖表...")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = records_dir / 'diversity_evolution.png'
    
    visualize_diversity_evolution(diversity_stats, output_path)
    
    # 9. 輸出總結
    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)
    print(f"分析世代數: {len(diversity_stats)}")
    print(f"\nPnL Correlation 趨勢:")
    print(f"  起始值: {diversity_stats[0]['mean_pnl_correlation']:.4f}")
    print(f"  最終值: {diversity_stats[-1]['mean_pnl_correlation']:.4f}")
    print(f"  變化量: {diversity_stats[-1]['mean_pnl_correlation'] - diversity_stats[0]['mean_pnl_correlation']:.4f}")
    print(f"\nTED Distance 趨勢:")
    print(f"  起始值: {diversity_stats[0]['mean_ted_distance']:.4f}")
    print(f"  最終值: {diversity_stats[-1]['mean_ted_distance']:.4f}")
    print(f"  變化量: {diversity_stats[-1]['mean_ted_distance'] - diversity_stats[0]['mean_ted_distance']:.4f}")
    print(f"\n輸出文件:")
    print(f"  - {output_path}")
    print(f"  - {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
