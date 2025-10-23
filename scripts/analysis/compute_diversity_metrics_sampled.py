#!/usr/bin/env python3
"""
計算實驗中所有世代的多樣性指標（採樣版本）

這個腳本會：
1. 載入所有 generation_*.pkl 文件
2. 使用採樣方法計算多樣性（而不是計算所有配對）
3. 計算多樣性指標（平均相似度、多樣性分數等）
4. 儲存結果到 diversity_metrics.json

使用方式：
    python scripts/analysis/compute_diversity_metrics_sampled.py \
        --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
        --n_workers 4 \
        --sample_size 1000
"""

import argparse
import json
import pickle
import time
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加項目根目錄到路徑
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.similarity import ParallelSimilarityMatrix, SimilarityMatrix

# 初始化 DEAP creator（用於 pickle 反序列化）
from deap import base, creator, gp, tools
import operator

# 設置 DEAP creator（如果尚未設置）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def compute_single_generation_sampled(pkl_file: Path, sample_size: int = 1000, n_workers: int = 4):
    """
    計算單一世代的多樣性指標（使用採樣）
    
    Args:
        pkl_file: generation_XXX.pkl 文件路徑
        sample_size: 採樣大小（從族群中隨機選擇的個體數）
        n_workers: 並行工作進程數
        
    Returns:
        dict: 多樣性指標
    """
    try:
        # 載入族群
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 提取族群（處理字典格式）
        if isinstance(data, dict) and 'population' in data:
            population = data['population']
        else:
            population = data
        
        original_size = len(population)
        
        # 如果族群小於採樣大小，使用全部
        if original_size <= sample_size:
            sampled_population = population
        else:
            # 隨機採樣
            sampled_population = random.sample(population, sample_size)
        
        # 提取世代編號
        gen_num = int(pkl_file.stem.split('_')[1])
        
        # 計算相似度矩陣
        start_time = time.time()
        
        if len(sampled_population) >= 200:
            sim_matrix = ParallelSimilarityMatrix(sampled_population, n_workers=n_workers)
        else:
            sim_matrix = SimilarityMatrix(sampled_population)
        
        similarity_matrix = sim_matrix.compute(show_progress=False)
        
        computation_time = time.time() - start_time
        
        # 計算統計指標
        # 排除對角線（自己與自己的相似度）
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        metrics = {
            'generation': gen_num,
            'population_size': original_size,
            'sample_size': len(sampled_population),
            'sampled': original_size > sample_size,
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'diversity_score': 1.0 - float(np.mean(similarities)),
            'computation_time': computation_time
        }
        
        return metrics
        
    except Exception as e:
        print(f"✗ 處理 {pkl_file.name} 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_diversity_metrics(exp_dir: Path, sample_size: int = 1000, n_workers: int = 4, 
                              batch_parallel: bool = False, cooldown: int = 0):
    """
    計算實驗中所有世代的多樣性指標（採樣版本）
    
    Args:
        exp_dir: 實驗目錄路徑
        sample_size: 每個世代的採樣大小
        n_workers: 並行工作進程數
        batch_parallel: 是否批次並行處理多個世代
        cooldown: 每個世代之間的冷卻時間（秒）
        
    Returns:
        dict: 包含所有世代多樣性指標的字典
    """
    # 獲取所有 generation pkl 文件
    gen_dir = exp_dir / "generations"
    pkl_files = sorted(gen_dir.glob("generation_*.pkl"))
    
    if not pkl_files:
        raise ValueError(f"在 {gen_dir} 中找不到 generation_*.pkl 文件")
    
    print(f"找到 {len(pkl_files)} 個世代文件")
    print(f"採樣大小: {sample_size}")
    print(f"Workers: {n_workers}")
    print()
    
    all_metrics = []
    
    if batch_parallel:
        # 批次並行：同時處理多個世代
        print("使用批次並行計算...")
        print()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(compute_single_generation_sampled, pkl_file, sample_size, 2): pkl_file
                for pkl_file in pkl_files
            }
            
            with tqdm(total=len(pkl_files), desc="計算多樣性") as pbar:
                for future in as_completed(futures):
                    pkl_file = futures[future]
                    try:
                        metrics = future.result()
                        if metrics:
                            all_metrics.append(metrics)
                        pbar.update(1)
                    except Exception as e:
                        print(f"✗ {pkl_file.name} 計算失敗: {e}")
                        pbar.update(1)
    else:
        # 序列計算：一個一個處理
        print("使用序列計算...")
        print()
        
        for pkl_file in tqdm(pkl_files, desc="計算多樣性"):
            metrics = compute_single_generation_sampled(pkl_file, sample_size, n_workers)
            if metrics:
                all_metrics.append(metrics)
            
            # 冷卻時間
            if cooldown > 0:
                time.sleep(cooldown)
    
    # 按世代編號排序
    all_metrics.sort(key=lambda x: x['generation'])
    
    # 構建完整數據
    result = {
        'experiment': exp_dir.name,
        'experiment_path': str(exp_dir),
        'total_generations': len(all_metrics),
        'population_size': all_metrics[0]['population_size'] if all_metrics else 0,
        'sample_size': sample_size,
        'computation_date': datetime.now().isoformat(),
        'n_workers': n_workers,
        'batch_parallel': batch_parallel,
        'total_computation_time': sum(m['computation_time'] for m in all_metrics),
        'metrics': all_metrics
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='計算實驗的多樣性指標（採樣版本）')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='實驗目錄路徑')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='每個世代的採樣大小（默認: 1000）')
    parser.add_argument('--n_workers', type=int, default=4,
                       help='並行工作進程數（默認: 4）')
    parser.add_argument('--batch_parallel', action='store_true',
                       help='是否批次並行處理多個世代')
    parser.add_argument('--no_batch_parallel', action='store_true',
                       help='不使用批次並行（序列處理）')
    parser.add_argument('--cooldown', type=int, default=0,
                       help='每個世代之間的冷卻時間（秒，默認: 0）')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出文件路徑（默認: exp_dir/diversity_metrics.json）')
    
    args = parser.parse_args()
    
    # 處理路徑
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise ValueError(f"實驗目錄不存在: {exp_dir}")
    
    # 決定是否批次並行
    batch_parallel = args.batch_parallel and not args.no_batch_parallel
    
    # 計算多樣性指標
    print("=" * 70)
    print("計算多樣性指標（採樣版本）")
    print("=" * 70)
    print(f"實驗目錄: {exp_dir}")
    print(f"採樣大小: {args.sample_size}")
    print(f"Workers: {args.n_workers}")
    print(f"批次並行: {batch_parallel}")
    print("=" * 70)
    print()
    
    start_time = time.time()
    result = compute_diversity_metrics(
        exp_dir=exp_dir,
        sample_size=args.sample_size,
        n_workers=args.n_workers,
        batch_parallel=batch_parallel,
        cooldown=args.cooldown
    )
    total_time = time.time() - start_time
    
    # 輸出結果
    output_file = args.output if args.output else exp_dir / "diversity_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print()
    print("=" * 70)
    print("✅ 計算完成！")
    print("=" * 70)
    print(f"總世代數: {result['total_generations']}")
    print(f"族群大小: {result['population_size']}")
    print(f"採樣大小: {result['sample_size']}")
    print(f"總計算時間: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
    print(f"平均每世代: {total_time/result['total_generations']:.1f} 秒")
    print(f"結果已儲存到: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
