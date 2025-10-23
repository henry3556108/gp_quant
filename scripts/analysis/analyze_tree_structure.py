#!/usr/bin/env python3
"""
分析實驗中每個世代的樹結構統計

輸出每個世代的：
- 樹深度（平均、最小、最大）
- 節點數（平均、最小、最大）
"""

import argparse
import pickle
import json
from pathlib import Path
import sys

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deap import base, creator, gp

# 設置 DEAP creator（用於 pickle 反序列化）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def analyze_generation(pkl_file: Path) -> dict:
    """
    分析單一世代的樹結構
    
    Args:
        pkl_file: generation_XXX.pkl 文件路徑
        
    Returns:
        dict: 統計資訊
    """
    # 載入族群
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 提取族群
    if isinstance(data, dict) and 'population' in data:
        population = data['population']
    else:
        population = data
    
    # 提取世代編號
    gen_num = int(pkl_file.stem.split('_')[1])
    
    # 計算統計資訊
    depths = [ind.height for ind in population]
    sizes = [len(ind) for ind in population]
    
    stats = {
        'generation': gen_num,
        'population_size': len(population),
        'depth': {
            'mean': sum(depths) / len(depths),
            'min': min(depths),
            'max': max(depths)
        },
        'nodes': {
            'mean': sum(sizes) / len(sizes),
            'min': min(sizes),
            'max': max(sizes)
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='分析實驗中每個世代的樹結構統計')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='實驗目錄路徑')
    parser.add_argument('--output', type=str, default=None,
                       help='輸出 JSON 文件路徑（可選）')
    
    args = parser.parse_args()
    
    # 解析路徑
    exp_dir = Path(args.exp_dir)
    generations_dir = exp_dir / 'generations'
    
    if not generations_dir.exists():
        print(f"✗ 找不到 generations 目錄: {generations_dir}")
        return 1
    
    # 獲取所有 pkl 文件
    pkl_files = sorted(generations_dir.glob('generation_*.pkl'))
    
    if not pkl_files:
        print(f"✗ 在 {generations_dir} 中找不到 generation_*.pkl 文件")
        return 1
    
    print("=" * 80)
    print("🌲 樹結構統計分析")
    print("=" * 80)
    print()
    print(f"實驗目錄: {exp_dir}")
    print(f"世代數: {len(pkl_files)}")
    print()
    
    # 分析每個世代
    all_stats = []
    
    print("正在分析...")
    print()
    
    for pkl_file in pkl_files:
        stats = analyze_generation(pkl_file)
        all_stats.append(stats)
        
        # 輸出到終端
        print(f"Generation {stats['generation']:3d}:")
        print(f"  族群大小: {stats['population_size']}")
        print(f"  樹深度:   平均={stats['depth']['mean']:6.2f}  "
              f"最小={stats['depth']['min']:3d}  最大={stats['depth']['max']:3d}")
        print(f"  節點數:   平均={stats['nodes']['mean']:6.2f}  "
              f"最小={stats['nodes']['min']:3d}  最大={stats['nodes']['max']:3d}")
        print()
    
    # 計算總體統計
    all_depths_mean = [s['depth']['mean'] for s in all_stats]
    all_nodes_mean = [s['nodes']['mean'] for s in all_stats]
    
    print("=" * 80)
    print("📊 總體統計")
    print("=" * 80)
    print()
    print(f"平均樹深度範圍: {min(all_depths_mean):.2f} - {max(all_depths_mean):.2f}")
    print(f"平均節點數範圍: {min(all_nodes_mean):.2f} - {max(all_nodes_mean):.2f}")
    print()
    
    # 儲存到 JSON（如果指定）
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = exp_dir / 'tree_structure_stats.json'
    
    result = {
        'experiment': exp_dir.name,
        'experiment_path': str(exp_dir),
        'total_generations': len(all_stats),
        'statistics': all_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 結果已儲存到: {output_file}")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
