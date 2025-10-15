#!/usr/bin/env python3
"""
一鍵分析實驗的多樣性

這個腳本會：
1. 檢查是否已有 diversity_metrics.json，沒有則計算
2. 繪製多樣性演化曲線
3. 選擇關鍵世代繪製詳細分析（熱圖、分佈圖、t-SNE）
4. 生成完整的分析報告

使用方式：
    python scripts/analysis/analyze_experiment.py \
        --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \
        --key_generations 1 10 25 50
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.similarity import (
    plot_diversity_evolution,
    plot_similarity_heatmap,
    plot_similarity_distribution,
    plot_population_tsne
)


def analyze_experiment(
    exp_dir: Path,
    key_generations: list = None,
    force_recompute: bool = False,
    n_workers: int = 8
):
    """
    完整分析實驗的多樣性
    
    Args:
        exp_dir: 實驗目錄
        key_generations: 關鍵世代列表（用於詳細分析）
        force_recompute: 是否強制重新計算多樣性指標
        n_workers: 並行工作進程數
    """
    print("="*80)
    print("📊 實驗多樣性分析")
    print("="*80)
    print()
    print(f"實驗目錄: {exp_dir}")
    print()
    
    # 檢查目錄
    if not exp_dir.exists():
        print(f"✗ 實驗目錄不存在: {exp_dir}")
        return 1
    
    generations_dir = exp_dir / 'generations'
    if not generations_dir.exists():
        print(f"✗ 找不到 generations 目錄: {generations_dir}")
        return 1
    
    diversity_file = exp_dir / 'diversity_metrics.json'
    
    # 步驟 1: 計算多樣性指標（如果需要）
    if not diversity_file.exists() or force_recompute:
        print("📈 步驟 1: 計算多樣性指標")
        print("-" * 80)
        
        if force_recompute and diversity_file.exists():
            print("⚠️  強制重新計算（已存在的文件將被覆蓋）")
        
        # 調用 compute_diversity_metrics.py
        import subprocess
        cmd = [
            sys.executable,
            str(Path(__file__).parent / 'compute_diversity_metrics.py'),
            '--exp_dir', str(exp_dir),
            '--n_workers', str(n_workers)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print("✗ 計算多樣性指標失敗")
            return 1
        
        print()
    else:
        print("✓ 多樣性指標已存在，跳過計算")
        print(f"  文件: {diversity_file}")
        print()
    
    # 步驟 2: 繪製多樣性演化曲線
    print("📈 步驟 2: 繪製多樣性演化曲線")
    print("-" * 80)
    
    evolution_plot = exp_dir / 'diversity_evolution.png'
    plot_diversity_evolution(diversity_file, save_path=evolution_plot)
    print()
    
    # 步驟 3: 關鍵世代詳細分析
    if key_generations:
        print("📈 步驟 3: 關鍵世代詳細分析")
        print("-" * 80)
        print(f"關鍵世代: {key_generations}")
        print()
        
        for gen in key_generations:
            pkl_file = generations_dir / f'generation_{gen:03d}.pkl'
            
            if not pkl_file.exists():
                print(f"⚠️  跳過 Generation {gen}（文件不存在）")
                continue
            
            print(f"分析 Generation {gen}...")
            
            # 熱圖
            heatmap_file = exp_dir / f'similarity_heatmap_gen{gen:03d}.png'
            print(f"  - 繪製相似度矩陣熱圖...")
            plot_similarity_heatmap(pkl_file, generation=gen, save_path=heatmap_file)
            
            # 分佈圖
            dist_file = exp_dir / f'similarity_distribution_gen{gen:03d}.png'
            print(f"  - 繪製相似度分佈...")
            plot_similarity_distribution(pkl_file, generation=gen, save_path=dist_file)
            
            # t-SNE
            tsne_file = exp_dir / f'population_tsne_gen{gen:03d}.png'
            print(f"  - 繪製 t-SNE 降維圖...")
            plot_population_tsne(pkl_file, generation=gen, save_path=tsne_file, method='tsne')
            
            print()
    
    # 完成
    print("="*80)
    print("✅ 分析完成")
    print("="*80)
    print()
    print("輸出文件:")
    print(f"  📈 多樣性演化: {evolution_plot}")
    
    if key_generations:
        print(f"  📊 關鍵世代分析:")
        for gen in key_generations:
            print(f"    - Generation {gen}:")
            print(f"      • 熱圖: similarity_heatmap_gen{gen:03d}.png")
            print(f"      • 分佈: similarity_distribution_gen{gen:03d}.png")
            print(f"      • t-SNE: population_tsne_gen{gen:03d}.png")
    
    print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='一鍵分析實驗的多樣性',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本分析（只繪製演化曲線）
  python scripts/analysis/analyze_experiment.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353
  
  # 完整分析（包含關鍵世代）
  python scripts/analysis/analyze_experiment.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \\
      --key_generations 1 10 25 50
  
  # 強制重新計算
  python scripts/analysis/analyze_experiment.py \\
      --exp_dir portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353 \\
      --key_generations 1 10 25 50 \\
      --force_recompute
        """
    )
    
    parser.add_argument(
        '--exp_dir',
        type=str,
        required=True,
        help='實驗目錄路徑'
    )
    
    parser.add_argument(
        '--key_generations',
        type=int,
        nargs='+',
        default=None,
        help='關鍵世代列表（用於詳細分析）'
    )
    
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='強制重新計算多樣性指標'
    )
    
    parser.add_argument(
        '--n_workers',
        type=int,
        default=8,
        help='並行工作進程數（預設: 8）'
    )
    
    args = parser.parse_args()
    
    # 解析路徑
    exp_dir = Path(args.exp_dir)
    
    # 執行分析
    return analyze_experiment(
        exp_dir,
        key_generations=args.key_generations,
        force_recompute=args.force_recompute,
        n_workers=args.n_workers
    )


if __name__ == '__main__':
    exit(main())
