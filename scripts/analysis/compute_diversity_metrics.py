#!/usr/bin/env python3
"""
計算實驗中所有世代的多樣性指標

這個腳本會：
1. 載入所有 generation_*.pkl 文件
2. 使用並行計算相似度矩陣
3. 計算多樣性指標（平均相似度、多樣性分數等）
4. 儲存結果到 diversity_metrics.json

使用方式：
    python scripts/analysis/compute_diversity_metrics.py \
        --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
        --n_workers 8
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 添加項目根目錄到路徑
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.similarity import ParallelSimilarityMatrix, SimilarityMatrix


def compute_single_generation(pkl_file: Path, use_parallel: bool = True, n_workers: int = 8):
    """
    計算單一世代的多樣性指標
    
    Args:
        pkl_file: generation_XXX.pkl 文件路徑
        use_parallel: 是否使用並行計算
        n_workers: 並行工作進程數
        
    Returns:
        dict: 多樣性指標
    """
    try:
        # 載入族群
        with open(pkl_file, 'rb') as f:
            population = pickle.load(f)
        
        # 提取世代編號
        gen_num = int(pkl_file.stem.split('_')[1])
        
        # 計算相似度矩陣
        start_time = time.time()
        
        if use_parallel and len(population) >= 200:
            sim_matrix = ParallelSimilarityMatrix(population, n_workers=n_workers)
        else:
            sim_matrix = SimilarityMatrix(population)
        
        similarity_matrix = sim_matrix.compute(show_progress=False)
        
        computation_time = time.time() - start_time
        
        # 計算統計指標
        # 排除對角線（自己與自己的相似度）
        mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        similarities = similarity_matrix[mask]
        
        metrics = {
            'generation': gen_num,
            'population_size': len(population),
            'avg_similarity': float(np.mean(similarities)),
            'diversity_score': float(1.0 - np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'computation_time': computation_time
        }
        
        return metrics
        
    except Exception as e:
        print(f"✗ 處理 {pkl_file.name} 時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_all_diversity_metrics(exp_dir: Path, n_workers: int = 8, batch_parallel: bool = True):
    """
    計算所有世代的多樣性指標
    
    Args:
        exp_dir: 實驗目錄
        n_workers: 並行工作進程數
        batch_parallel: 是否使用批次並行（同時處理多個世代）
        
    Returns:
        dict: 完整的多樣性指標數據
    """
    generations_dir = exp_dir / 'generations'
    
    if not generations_dir.exists():
        raise FileNotFoundError(f"找不到 generations 目錄: {generations_dir}")
    
    # 獲取所有 pkl 文件並排序
    pkl_files = sorted(generations_dir.glob('generation_*.pkl'))
    
    if not pkl_files:
        raise FileNotFoundError(f"在 {generations_dir} 中找不到 generation_*.pkl 文件")
    
    print(f"找到 {len(pkl_files)} 個世代文件")
    print()
    
    # 計算多樣性指標
    all_metrics = []
    
    if batch_parallel:
        # 批次並行：同時處理多個世代
        print(f"使用批次並行計算（{n_workers} workers）...")
        print()
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # 提交所有任務
            future_to_file = {
                executor.submit(compute_single_generation, pkl_file, True, 8): pkl_file
                for pkl_file in pkl_files
            }
            
            # 收集結果（使用進度條）
            with tqdm(total=len(pkl_files), desc="計算多樣性") as pbar:
                for future in as_completed(future_to_file):
                    pkl_file = future_to_file[future]
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
            metrics = compute_single_generation(pkl_file, True, n_workers)
            if metrics:
                all_metrics.append(metrics)
    
    # 按世代編號排序
    all_metrics.sort(key=lambda x: x['generation'])
    
    # 構建完整數據
    result = {
        'experiment': exp_dir.name,
        'experiment_path': str(exp_dir),
        'total_generations': len(all_metrics),
        'population_size': all_metrics[0]['population_size'] if all_metrics else 0,
        'computation_date': datetime.now().isoformat(),
        'n_workers': n_workers,
        'batch_parallel': batch_parallel,
        'total_computation_time': sum(m['computation_time'] for m in all_metrics),
        'metrics': all_metrics
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='計算實驗中所有世代的多樣性指標',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本使用
  python scripts/analysis/compute_diversity_metrics.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353
  
  # 指定並行數
  python scripts/analysis/compute_diversity_metrics.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \\
      --n_workers 8
  
  # 使用序列計算（不使用批次並行）
  python scripts/analysis/compute_diversity_metrics.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \\
      --no_batch_parallel
        """
    )
    
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help='實驗目錄路徑'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=8,
        help='並行工作進程數（預設: 8）'
    )
    
    parser.add_argument(
        '--no_batch_parallel',
        action='store_true',
        help='不使用批次並行（一次只處理一個世代）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='輸出文件路徑（預設: exp_dir/diversity_metrics.json）'
    )
    
    args = parser.parse_args()
    
    # 解析路徑
    exp_dir = Path(args.exp_dir)
    
    if not exp_dir.exists():
        print(f"✗ 實驗目錄不存在: {exp_dir}")
        return 1
    
    # 設置輸出路徑
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = exp_dir / 'diversity_metrics.json'
    
    print("="*80)
    print("📊 計算多樣性指標")
    print("="*80)
    print()
    print(f"實驗目錄: {exp_dir}")
    print(f"並行數: {args.n_workers}")
    print(f"批次並行: {'是' if not args.no_batch_parallel else '否'}")
    print(f"輸出文件: {output_file}")
    print()
    
    # 計算多樣性指標
    try:
        start_time = time.time()
        
        result = compute_all_diversity_metrics(
            exp_dir,
            n_workers=args.n_workers,
            batch_parallel=not args.no_batch_parallel
        )
        
        total_time = time.time() - start_time
        
        print()
        print("="*80)
        print("✅ 計算完成")
        print("="*80)
        print()
        print(f"總世代數: {result['total_generations']}")
        print(f"族群大小: {result['population_size']}")
        print(f"總計算時間: {total_time:.1f}s ({total_time/60:.1f} 分鐘)")
        print(f"平均每代: {total_time/result['total_generations']:.1f}s")
        print()
        
        # 顯示多樣性趨勢
        print("📈 多樣性趨勢:")
        metrics = result['metrics']
        first_gen = metrics[0]
        last_gen = metrics[-1]
        
        print(f"  第 {first_gen['generation']} 代:")
        print(f"    平均相似度: {first_gen['avg_similarity']:.4f}")
        print(f"    多樣性分數: {first_gen['diversity_score']:.4f}")
        print()
        print(f"  第 {last_gen['generation']} 代:")
        print(f"    平均相似度: {last_gen['avg_similarity']:.4f}")
        print(f"    多樣性分數: {last_gen['diversity_score']:.4f}")
        print()
        
        diversity_change = last_gen['diversity_score'] - first_gen['diversity_score']
        print(f"  多樣性變化: {diversity_change:+.4f} ({diversity_change/first_gen['diversity_score']*100:+.1f}%)")
        print()
        
        # 儲存結果
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ 結果已儲存: {output_file}")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print(f"✗ 計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
