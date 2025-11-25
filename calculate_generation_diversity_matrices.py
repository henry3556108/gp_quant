#!/usr/bin/env python3
"""
Calculate Generation Diversity Matrices

計算指定世代的 PnL correlation matrix 和標準化 TED distance matrix。
支援採樣和平行化計算。
"""

import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
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


def load_generation_population(records_dir: Path, generation: int = None) -> Tuple[List, int]:
    """
    載入指定世代的族群
    
    Args:
        records_dir: 記錄目錄路徑
        generation: 世代號（None 表示最新一代）
        
    Returns:
        (population, generation_number)
    """
    populations_dir = records_dir / 'populations'
    
    if not populations_dir.exists():
        raise ValueError(f"Populations directory not found: {populations_dir}")
    
    # 找到所有 generation_XXX.pkl 文件
    gen_files = sorted(populations_dir.glob('generation_*.pkl'))
    
    if len(gen_files) == 0:
        raise ValueError("No generation files found")
    
    # 如果未指定世代，使用最新一代
    if generation is None:
        gen_file = gen_files[-1]
        generation = int(gen_file.stem.split('_')[1])
    else:
        gen_file = populations_dir / f'generation_{generation:03d}.pkl'
        if not gen_file.exists():
            raise ValueError(f"Generation {generation} not found")
    
    print(f"📂 載入世代 {generation}: {gen_file.name}")
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    print(f"   ✅ 載入 {len(population)} 個個體")
    
    return population, generation


def sample_population(population: List, sample_size: int, strategy: str = 'stratified') -> List:
    """
    從族群中採樣
    
    Args:
        population: 完整族群
        sample_size: 採樣大小
        strategy: 採樣策略 ('stratified' 或 'random')
        
    Returns:
        採樣後的族群
    """
    n = len(population)
    
    if sample_size >= n:
        print(f"   ℹ️  採樣大小 ({sample_size}) >= 族群大小 ({n})，使用全部個體")
        return population
    
    print(f"   🎲 採樣策略: {strategy}, 從 {n} 個個體中採樣 {sample_size} 個")
    
    if strategy == 'stratified':
        # 按 fitness 排序後均勻採樣
        sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        indices = np.linspace(0, n - 1, sample_size, dtype=int)
        sampled = [sorted_pop[i] for i in indices]
        print(f"   ✅ 分層採樣完成（包含高、中、低 fitness 個體）")
    else:
        # 隨機採樣
        indices = np.random.choice(n, sample_size, replace=False)
        sampled = [population[i] for i in indices]
        print(f"   ✅ 隨機採樣完成")
    
    return sampled


def calculate_pnl_for_individual(individual: Any, 
                                  engine: PortfolioBacktestingEngine) -> np.ndarray:
    """
    計算單個個體的 PnL curve
    
    Args:
        individual: DEAP individual
        engine: 回測引擎
        
    Returns:
        PnL curve as numpy array
    """
    try:
        pnl_curve = engine.get_pnl_curve(individual)
        return pnl_curve.values
    except Exception as e:
        # 如果計算失敗，返回全零
        return np.zeros(len(engine.common_dates))


def calculate_pnl_correlation_matrix(population: List,
                                     train_data: Dict,
                                     config: Dict,
                                     n_jobs: int = 4) -> Tuple[np.ndarray, List[int]]:
    """
    計算 PnL correlation matrix（平行化）
    
    Args:
        population: 族群列表
        train_data: 訓練數據
        config: 配置字典
        n_jobs: 平行處理器數量
        
    Returns:
        (Correlation matrix (n x n), list of invalid indices)
    """
    n = len(population)
    print(f"\n💰 計算 PnL Correlation Matrix ({n} x {n})...")
    
    # 提取訓練數據
    train_data_dict = {ticker: info['data'] for ticker, info in train_data.items()}
    
    # 創建回測引擎
    engine = PortfolioBacktestingEngine(
        data=train_data_dict,
        backtest_start=config['data']['train_backtest_start'],
        backtest_end=config['data']['train_backtest_end'],
        initial_capital=100000.0
    )
    
    # 平行計算所有個體的 PnL curves
    print(f"   🔄 平行計算 PnL curves (n_jobs={n_jobs})...")
    pnl_curves = Parallel(n_jobs=n_jobs)(
        delayed(calculate_pnl_for_individual)(ind, engine)
        for ind in tqdm(population, desc="   計算 PnL", ncols=80)
    )
    
    # 轉換為矩陣
    pnl_matrix = np.array(pnl_curves)
    
    # 檢查有效性（識別全零的 PnL curves）
    valid_mask = ~np.all(pnl_matrix == 0, axis=1)
    invalid_indices = np.where(~valid_mask)[0].tolist()
    valid_count = np.sum(valid_mask)
    
    print(f"   ✅ 有效 PnL curves: {valid_count}/{n}")
    if len(invalid_indices) > 0:
        print(f"   ⚠️  無效個體（全零 PnL）: {invalid_indices}")
        print(f"      這些個體的 fitness: {[population[i].fitness.values[0] for i in invalid_indices]}")
    
    # 計算相關性矩陣
    print(f"   📊 計算相關性矩陣...")
    corr_matrix = np.corrcoef(pnl_matrix)
    
    # 處理 NaN（由於全零 PnL curve 導致的標準差為 0）
    # 將 NaN 替換為 0（表示無相關性）
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 對於無效個體，明確標記
    # 將其與其他個體的相關性設為 0（表示無效）
    for idx in invalid_indices:
        corr_matrix[idx, :] = 0.0
        corr_matrix[:, idx] = 0.0
        corr_matrix[idx, idx] = 1.0  # 對角線保持為 1
    
    # 確保對角線為 1（對於所有個體）
    np.fill_diagonal(corr_matrix, 1.0)
    
    # 統計信息（只計算有效個體之間的相關性）
    valid_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if i not in invalid_indices and j not in invalid_indices:
                valid_pairs.append(corr_matrix[i, j])
    
    if len(valid_pairs) > 0:
        mean_corr = np.mean(valid_pairs)
        std_corr = np.std(valid_pairs)
        print(f"   ✅ 平均相關性（有效個體）: {mean_corr:.4f} ± {std_corr:.4f}")
        print(f"   ✅ 相關性範圍: [{np.min(valid_pairs):.4f}, {np.max(valid_pairs):.4f}]")
    else:
        print(f"   ⚠️  沒有足夠的有效個體來計算相關性")
    
    return corr_matrix, invalid_indices


def calculate_ted_for_pair(i: int, j: int, ind_i: Any, ind_j: Any) -> Tuple[int, int, float]:
    """
    計算一對個體的標準化 TED
    
    Args:
        i, j: 個體索引
        ind_i, ind_j: 個體
        
    Returns:
        (i, j, normalized_ted)
    """
    try:
        ted = compute_ted(ind_i, ind_j)
        max_size = max(len(ind_i), len(ind_j))
        norm_ted = ted / max_size if max_size > 0 else 0.0
        return i, j, norm_ted
    except Exception as e:
        # 如果計算失敗，返回最大距離
        return i, j, 1.0


def calculate_ted_distance_matrix(population: List, n_jobs: int = 4) -> np.ndarray:
    """
    計算標準化 TED distance matrix（平行化）
    
    Args:
        population: 族群列表
        n_jobs: 平行處理器數量
        
    Returns:
        Normalized TED distance matrix (n x n)
    """
    n = len(population)
    print(f"\n🌳 計算標準化 TED Distance Matrix ({n} x {n})...")
    
    # 初始化矩陣
    ted_matrix = np.zeros((n, n))
    
    # 生成所有需要計算的配對（上三角）
    pairs = [(i, j, population[i], population[j]) 
             for i in range(n) for j in range(i + 1, n)]
    
    total_pairs = len(pairs)
    print(f"   🔄 平行計算 {total_pairs} 對 TED (n_jobs={n_jobs})...")
    
    # 平行計算
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_ted_for_pair)(i, j, ind_i, ind_j)
        for i, j, ind_i, ind_j in tqdm(pairs, desc="   計算 TED", ncols=80)
    )
    
    # 填充矩陣（對稱）
    for i, j, ted in results:
        ted_matrix[i, j] = ted
        ted_matrix[j, i] = ted
    
    # 對角線為 0
    np.fill_diagonal(ted_matrix, 0.0)
    
    # 統計信息
    upper_tri = np.triu_indices(n, k=1)
    mean_ted = np.mean(ted_matrix[upper_tri])
    std_ted = np.std(ted_matrix[upper_tri])
    
    print(f"   ✅ 平均 TED 距離: {mean_ted:.4f} ± {std_ted:.4f}")
    print(f"   ✅ TED 範圍: [{np.min(ted_matrix[upper_tri]):.4f}, {np.max(ted_matrix[upper_tri]):.4f}]")
    
    return ted_matrix


def save_matrix_to_csv(matrix: np.ndarray, output_path: Path, generation: int, matrix_type: str):
    """
    保存矩陣到 CSV
    
    Args:
        matrix: 矩陣
        output_path: 輸出路徑
        generation: 世代號
        matrix_type: 矩陣類型（'pnl_corr' 或 'ted_dist'）
    """
    df = pd.DataFrame(matrix)
    df.to_csv(output_path, index=False, header=False)
    print(f"   💾 {matrix_type} 矩陣已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="計算世代多樣性矩陣（PnL correlation 和 TED distance）"
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
        '--generation',
        type=int,
        default=None,
        help='世代號（默認為最新一代）'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='採樣大小（默認使用全部個體）'
    )
    parser.add_argument(
        '--sample-strategy',
        type=str,
        default='stratified',
        choices=['stratified', 'random'],
        help='採樣策略（默認：stratified）'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=4,
        help='平行處理器數量（默認：4）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='輸出目錄（默認保存在記錄目錄中）'
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
    print("🎯 Generation Diversity Matrices Calculation")
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
    
    # 3. 載入世代族群
    print("\n📦 載入世代族群...")
    population, generation = load_generation_population(records_dir, args.generation)
    
    # 4. 採樣（如果需要）
    if args.sample_size:
        population = sample_population(population, args.sample_size, args.sample_strategy)
    
    # 5. 載入數據
    print("\n📊 載入數據...")
    import os
    tickers_dir = config['data']['tickers_dir']
    ticker_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in ticker_files]
    print(f"   發現 {len(tickers)} 個 ticker")
    
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
    
    # 6. 計算 PnL correlation matrix
    pnl_corr_matrix, invalid_indices = calculate_pnl_correlation_matrix(
        population, train_data, config, args.n_jobs
    )
    
    # 7. 計算 TED distance matrix
    ted_dist_matrix = calculate_ted_distance_matrix(population, args.n_jobs)
    
    # 8. 保存矩陣
    print("\n💾 保存矩陣...")
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = records_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pnl_corr_path = output_dir / f'pnl_correlation_matrix_gen{generation:03d}.csv'
    ted_dist_path = output_dir / f'ted_distance_matrix_normalized_gen{generation:03d}.csv'
    
    save_matrix_to_csv(pnl_corr_matrix, pnl_corr_path, generation, 'PnL Correlation')
    save_matrix_to_csv(ted_dist_matrix, ted_dist_path, generation, 'TED Distance')
    
    # 9. 輸出摘要
    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)
    print(f"世代: {generation}")
    print(f"個體數量: {len(population)}")
    print(f"PnL 平均相關性: {np.mean(pnl_corr_matrix[np.triu_indices(len(population), k=1)]):.4f}")
    print(f"TED 平均距離: {np.mean(ted_dist_matrix[np.triu_indices(len(population), k=1)]):.4f}")
    print(f"\n輸出文件:")
    print(f"  - {pnl_corr_path}")
    print(f"  - {ted_dist_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
