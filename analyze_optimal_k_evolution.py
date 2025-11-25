#!/usr/bin/env python3
"""
分析最佳 K 值在演化過程中的變化

讀取 generation_stats.json，提取每個世代的 optimal_k 和 elite_pool_size，
並生成視覺化圖表。
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_generation_stats(records_dir: Path):
    """載入世代統計數據"""
    stats_file = records_dir / 'generation_stats.json'
    
    if not stats_file.exists():
        raise FileNotFoundError(f"找不到統計文件: {stats_file}")
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    return stats


def extract_selection_strategy_info(stats):
    """提取選擇策略信息"""
    generations = []
    optimal_ks = []
    elite_pool_sizes = []
    best_fitness = []
    avg_fitness = []
    
    for gen_stat in stats:
        generation = gen_stat['generation']
        generations.append(generation)
        
        # 提取選擇策略信息
        if 'selection_strategy' in gen_stat:
            strategy = gen_stat['selection_strategy']
            optimal_k = strategy.get('optimal_k')
            elite_size = strategy.get('elite_pool_size', 0)
        else:
            optimal_k = None
            elite_size = 0
        
        optimal_ks.append(optimal_k)
        elite_pool_sizes.append(elite_size)
        
        # 提取 fitness 信息
        best_fitness.append(gen_stat.get('best_fitness', 0))
        avg_fitness.append(gen_stat.get('avg_fitness', 0))
    
    return {
        'generations': generations,
        'optimal_ks': optimal_ks,
        'elite_pool_sizes': elite_pool_sizes,
        'best_fitness': best_fitness,
        'avg_fitness': avg_fitness
    }


def visualize_optimal_k_evolution(data, output_dir: Path):
    """視覺化最佳 K 值的演化"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generations = data['generations']
    optimal_ks = data['optimal_ks']
    elite_pool_sizes = data['elite_pool_sizes']
    best_fitness = data['best_fitness']
    avg_fitness = data['avg_fitness']
    
    # 創建 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 最佳 K 值隨世代變化
    ax1 = axes[0, 0]
    # 過濾掉 None 值
    valid_gens = [g for g, k in zip(generations, optimal_ks) if k is not None]
    valid_ks = [k for k in optimal_ks if k is not None]
    
    if valid_gens:
        ax1.plot(valid_gens, valid_ks, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Optimal K', fontsize=12)
        ax1.set_title('最佳 K 值隨世代變化', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks(range(int(min(valid_ks)), int(max(valid_ks)) + 1))
        
        # 標註平均 K 值
        mean_k = np.mean(valid_ks)
        ax1.axhline(y=mean_k, color='red', linestyle='--', linewidth=2, 
                   label=f'平均 K={mean_k:.2f}')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No optimal K data', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14)
    
    # 2. Elite Pool 大小隨世代變化
    ax2 = axes[0, 1]
    ax2.plot(generations, elite_pool_sizes, marker='s', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Elite Pool Size', fontsize=12)
    ax2.set_title('Elite Pool 大小隨世代變化', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 標註平均大小
    if elite_pool_sizes:
        mean_size = np.mean([s for s in elite_pool_sizes if s > 0])
        ax2.axhline(y=mean_size, color='red', linestyle='--', linewidth=2, 
                   label=f'平均大小={mean_size:.1f}')
        ax2.legend()
    
    # 3. K 值分布（直方圖）
    ax3 = axes[1, 0]
    if valid_ks:
        unique_ks, counts = np.unique(valid_ks, return_counts=True)
        ax3.bar(unique_ks, counts, color='lightgreen', alpha=0.7, width=0.6)
        ax3.set_xlabel('K Value', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('K 值分布', fontsize=13, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_xticks(unique_ks)
        
        # 標註每個 bar 的數值
        for k, count in zip(unique_ks, counts):
            ax3.text(k, count, f'{count}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No K distribution data', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=14)
    
    # 4. Fitness 演化（與 K 值對比）
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # 繪製 fitness
    line1 = ax4.plot(generations, best_fitness, marker='o', linewidth=2, 
                     markersize=6, color='green', label='Best Fitness')
    line2 = ax4.plot(generations, avg_fitness, marker='s', linewidth=2, 
                     markersize=6, color='orange', label='Avg Fitness')
    
    # 繪製 K 值（右軸）
    if valid_gens:
        line3 = ax4_twin.plot(valid_gens, valid_ks, marker='^', linewidth=2, 
                             markersize=8, color='purple', alpha=0.6, label='Optimal K')
    
    ax4.set_xlabel('Generation', fontsize=12)
    ax4.set_ylabel('Fitness', fontsize=12, color='green')
    ax4_twin.set_ylabel('Optimal K', fontsize=12, color='purple')
    ax4.set_title('Fitness 演化 vs K 值', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='green')
    ax4_twin.tick_params(axis='y', labelcolor='purple')
    
    # 合併圖例
    lines = line1 + line2
    if valid_gens:
        lines += line3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_k_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_dir / 'optimal_k_evolution.png'}")
    plt.close()


def print_summary(data):
    """打印統計摘要"""
    optimal_ks = [k for k in data['optimal_ks'] if k is not None]
    elite_pool_sizes = [s for s in data['elite_pool_sizes'] if s > 0]
    
    print("\n" + "="*80)
    print("📊 統計摘要")
    print("="*80)
    
    if optimal_ks:
        print(f"\n最佳 K 值:")
        print(f"  平均: {np.mean(optimal_ks):.2f}")
        print(f"  中位數: {np.median(optimal_ks):.0f}")
        print(f"  最小: {min(optimal_ks)}")
        print(f"  最大: {max(optimal_ks)}")
        print(f"  標準差: {np.std(optimal_ks):.2f}")
        
        # K 值分布
        unique_ks, counts = np.unique(optimal_ks, return_counts=True)
        print(f"\nK 值分布:")
        for k, count in zip(unique_ks, counts):
            percentage = count / len(optimal_ks) * 100
            print(f"  K={k}: {count} 次 ({percentage:.1f}%)")
    
    if elite_pool_sizes:
        print(f"\nElite Pool 大小:")
        print(f"  平均: {np.mean(elite_pool_sizes):.1f}")
        print(f"  中位數: {np.median(elite_pool_sizes):.0f}")
        print(f"  最小: {min(elite_pool_sizes)}")
        print(f"  最大: {max(elite_pool_sizes)}")
        print(f"  標準差: {np.std(elite_pool_sizes):.2f}")
    
    # Fitness 統計
    best_fitness = data['best_fitness']
    print(f"\nBest Fitness:")
    print(f"  初始: {best_fitness[0]:.4f}")
    print(f"  最終: {best_fitness[-1]:.4f}")
    print(f"  改善: {best_fitness[-1] - best_fitness[0]:.4f} ({(best_fitness[-1] - best_fitness[0]) / best_fitness[0] * 100:.1f}%)")
    print(f"  最大: {max(best_fitness):.4f}")


def main():
    """主函數"""
    if len(sys.argv) < 2:
        print("使用方式: python analyze_optimal_k_evolution.py <records_dir>")
        print("範例: python analyze_optimal_k_evolution.py test_evolution_records_20251125_2236")
        sys.exit(1)
    
    records_dir = Path(sys.argv[1])
    
    if not records_dir.exists():
        print(f"❌ 記錄目錄不存在: {records_dir}")
        sys.exit(1)
    
    print("="*80)
    print("🔍 分析最佳 K 值演化")
    print("="*80)
    print(f"📁 記錄目錄: {records_dir}")
    
    # 載入數據
    print(f"\n📊 載入世代統計數據...")
    stats = load_generation_stats(records_dir)
    print(f"   ✅ 載入 {len(stats)} 個世代的數據")
    
    # 提取信息
    print(f"\n🔍 提取選擇策略信息...")
    data = extract_selection_strategy_info(stats)
    print(f"   ✅ 提取完成")
    
    # 打印摘要
    print_summary(data)
    
    # 生成視覺化
    print(f"\n📊 生成視覺化...")
    output_dir = records_dir / 'analysis'
    visualize_optimal_k_evolution(data, output_dir)
    
    print(f"\n✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
