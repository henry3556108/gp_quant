#!/usr/bin/env python3
"""
尋找最佳 K 值（Cluster 數量）

使用一次階層式聚類，然後在不同的 K 值切割，比較 CV（變異係數）
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deap import creator, base, gp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.evolution.components.strategies import TEDNicheSelectionStrategy


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_test_population(records_dir: Path, generation: int = 0):
    """載入測試族群"""
    populations_dir = records_dir / 'populations'
    gen_file = populations_dir / f'generation_{generation:03d}.pkl'
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    return population


def analyze_k_clusters(distance_matrix, k_values, population, M=50):
    """
    分析不同 K 值的聚類效果
    
    Args:
        distance_matrix: 距離矩陣
        k_values: K 值列表（例如 [2, 3, 4, 5]）
        population: 族群列表
        M: 每個 cluster 保留的 Top M 個體
    
    Returns:
        results: 每個 K 值的分析結果
    """
    print(f"\n{'='*80}")
    print(f"📊 執行一次階層式聚類（Complete Linkage）")
    print(f"{'='*80}")
    
    # 將距離矩陣轉換為壓縮形式（上三角）
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # 執行一次階層式聚類，生成樹狀結構
    print(f"計算階層式聚類樹...")
    linkage_matrix = linkage(condensed_dist, method='complete')
    print(f"✅ 聚類樹計算完成")
    
    results = []
    
    print(f"\n{'='*80}")
    print(f"📊 從聚類樹中提取不同 K 值的切割")
    print(f"{'='*80}")
    
    for k in k_values:
        print(f"\n{'─'*80}")
        print(f"K = {k}")
        print(f"{'─'*80}")
        
        # 從聚類樹中切割出 K 個 clusters
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1  # 轉為 0-based
        
        # 統計每個 cluster 的大小
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        print(f"\nCluster 分布:")
        print(f"{'Cluster':<10} {'大小':<10} {'比例':<10} {'視覺化'}")
        print("-" * 60)
        
        cluster_info = []
        for label, count in zip(unique_labels, counts):
            ratio = count / len(population) * 100
            bar = '█' * int(ratio / 2)  # 每個 █ 代表 2%
            print(f"{label:<10} {count:<10} {ratio:>6.2f}%    {bar}")
            cluster_info.append({
                'cluster': int(label),
                'size': int(count),
                'ratio': float(ratio)
            })
        
        # 統計指標
        mean_size = np.mean(counts)
        std_size = np.std(counts)
        min_size = np.min(counts)
        max_size = np.max(counts)
        cv = std_size / mean_size  # 變異係數（越小越平衡）
        
        print(f"\n統計指標:")
        print(f"  平均大小: {mean_size:.1f}")
        print(f"  標準差: {std_size:.1f}")
        print(f"  變異係數 (CV): {cv:.4f}")
        print(f"  最小大小: {min_size}")
        print(f"  最大大小: {max_size}")
        print(f"  大小範圍: {max_size - min_size}")
        
        # 計算 Elite Pool 達成率
        elite_pool_size = sum(min(count, M) for count in counts)
        expected_size = k * M
        achievement_rate = elite_pool_size / expected_size * 100
        
        print(f"\nElite Pool (Top {M} per cluster):")
        print(f"  實際大小: {elite_pool_size}")
        print(f"  預期大小: {expected_size}")
        print(f"  達成率: {achievement_rate:.1f}%")
        
        # 計算每個 cluster 的 fitness 統計
        print(f"\nCluster Fitness 統計:")
        print(f"{'Cluster':<10} {'大小':<10} {'平均':<10} {'最大':<10} {'最小':<10}")
        print("-" * 60)
        
        cluster_fitness_stats = []
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_individuals = [population[i] for i in cluster_indices]
            
            fitnesses = [ind.fitness.values[0] for ind in cluster_individuals]
            mean_fit = np.mean(fitnesses)
            max_fit = np.max(fitnesses)
            min_fit = np.min(fitnesses)
            
            print(f"{label:<10} {len(cluster_individuals):<10} {mean_fit:<10.4f} {max_fit:<10.4f} {min_fit:<10.4f}")
            
            cluster_fitness_stats.append({
                'cluster': int(label),
                'size': len(cluster_individuals),
                'mean_fitness': float(mean_fit),
                'max_fitness': float(max_fit),
                'min_fitness': float(min_fit)
            })
        
        # 保存結果
        results.append({
            'k': k,
            'cluster_labels': cluster_labels,
            'cluster_info': cluster_info,
            'stats': {
                'mean_size': float(mean_size),
                'std_size': float(std_size),
                'cv': float(cv),
                'min_size': int(min_size),
                'max_size': int(max_size),
                'range': int(max_size - min_size)
            },
            'elite_pool': {
                'actual_size': int(elite_pool_size),
                'expected_size': int(expected_size),
                'achievement_rate': float(achievement_rate)
            },
            'fitness_stats': cluster_fitness_stats
        })
    
    return results


def visualize_results(results, output_dir):
    """視覺化不同 K 值的比較結果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = [r['k'] for r in results]
    
    # 創建 2x2 子圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 變異係數 (CV)
    cvs = [r['stats']['cv'] for r in results]
    axes[0, 0].plot(k_values, cvs, marker='o', linewidth=2, markersize=10, color='coral')
    axes[0, 0].set_xlabel('K (Cluster 數量)', fontsize=12)
    axes[0, 0].set_ylabel('變異係數 (CV)', fontsize=12)
    axes[0, 0].set_title('變異係數 vs K\n(越小越平衡)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(k_values)
    
    # 標註最小值
    min_cv_idx = cvs.index(min(cvs))
    axes[0, 0].scatter([k_values[min_cv_idx]], [cvs[min_cv_idx]], 
                       color='red', s=200, zorder=5, marker='*', label='最佳')
    axes[0, 0].legend()
    
    for i, (k, cv) in enumerate(zip(k_values, cvs)):
        axes[0, 0].text(k, cv, f'{cv:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Elite Pool 達成率
    rates = [r['elite_pool']['achievement_rate'] for r in results]
    axes[0, 1].plot(k_values, rates, marker='s', linewidth=2, markersize=10, color='lightgreen')
    axes[0, 1].axhline(y=100, color='red', linestyle='--', linewidth=2, label='目標 100%')
    axes[0, 1].set_xlabel('K (Cluster 數量)', fontsize=12)
    axes[0, 1].set_ylabel('達成率 (%)', fontsize=12)
    axes[0, 1].set_title('Elite Pool 達成率 vs K', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(k_values)
    axes[0, 1].legend()
    
    for i, (k, rate) in enumerate(zip(k_values, rates)):
        axes[0, 1].text(k, rate, f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. Cluster 大小範圍
    ranges = [r['stats']['range'] for r in results]
    axes[1, 0].bar(k_values, ranges, color='skyblue', alpha=0.7, width=0.6)
    axes[1, 0].set_xlabel('K (Cluster 數量)', fontsize=12)
    axes[1, 0].set_ylabel('大小範圍 (Max - Min)', fontsize=12)
    axes[1, 0].set_title('Cluster 大小範圍 vs K\n(越小越平衡)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_xticks(k_values)
    
    for i, (k, r) in enumerate(zip(k_values, ranges)):
        axes[1, 0].text(k, r, f'{r}', ha='center', va='bottom', fontsize=10)
    
    # 4. 標準差
    stds = [r['stats']['std_size'] for r in results]
    axes[1, 1].bar(k_values, stds, color='plum', alpha=0.7, width=0.6)
    axes[1, 1].set_xlabel('K (Cluster 數量)', fontsize=12)
    axes[1, 1].set_ylabel('標準差', fontsize=12)
    axes[1, 1].set_title('Cluster 大小標準差 vs K\n(越小越平衡)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].set_xticks(k_values)
    
    for i, (k, std) in enumerate(zip(k_values, stds)):
        axes[1, 1].text(k, std, f'{std:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_k_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_dir / 'optimal_k_analysis.png'}")
    plt.close()
    
    # 創建 Cluster 大小分布的詳細圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        k = result['k']
        clusters = [info['cluster'] for info in result['cluster_info']]
        sizes = [info['size'] for info in result['cluster_info']]
        
        axes[idx].bar(clusters, sizes, color='steelblue', alpha=0.7)
        axes[idx].axhline(y=50, color='red', linestyle='--', linewidth=2, label='M=50')
        axes[idx].set_xlabel('Cluster ID', fontsize=11)
        axes[idx].set_ylabel('Cluster Size', fontsize=11)
        axes[idx].set_title(f"K={k}, CV={result['stats']['cv']:.4f}", 
                           fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        
        for cluster, size in zip(clusters, sizes):
            axes[idx].text(cluster, size, f'{int(size)}', 
                          ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_dir / 'cluster_distributions.png'}")
    plt.close()


def main():
    """主函數"""
    print("="*80)
    print("🔍 尋找最佳 K 值（Cluster 數量）")
    print("="*80)
    
    # 設置
    setup_deap_creator()
    
    # 載入測試族群
    records_dir = Path("/Users/hongyicheng/Downloads/gp_quant/test_evolution_11241221_records_20251125_1335")
    population = load_test_population(records_dir, generation=0)
    
    print(f"\n📦 載入族群: {len(population)} 個個體")
    
    # 創建策略並計算 TED matrix（只計算一次）
    print(f"\n📊 計算 TED Distance Matrix（只計算一次）...")
    strategy = TEDNicheSelectionStrategy(
        n_clusters=5,  # 這裡的值不重要，只是用來計算 TED matrix
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        n_jobs=6
    )
    
    ted_matrix = strategy._calculate_ted_distance_matrix(population)
    print(f"✅ TED Matrix 計算完成: {ted_matrix.shape}")
    
    # 分析不同的 K 值（2, 3, 4, 5）
    k_values = [2, 3, 4, 5]
    results = analyze_k_clusters(ted_matrix, k_values, population, M=50)
    
    # 比較總結
    print(f"\n{'='*80}")
    print("📊 比較總結")
    print(f"{'='*80}")
    
    print(f"\n{'K':<5} {'CV':<12} {'達成率':<12} {'範圍':<12} {'標準差':<12} {'推薦'}")
    print("-" * 70)
    
    best_cv_idx = min(range(len(results)), key=lambda i: results[i]['stats']['cv'])
    best_rate_idx = max(range(len(results)), key=lambda i: results[i]['elite_pool']['achievement_rate'])
    
    for idx, result in enumerate(results):
        k = result['k']
        cv = result['stats']['cv']
        rate = result['elite_pool']['achievement_rate']
        range_val = result['stats']['range']
        std = result['stats']['std_size']
        
        markers = []
        if idx == best_cv_idx:
            markers.append('✅ CV最佳')
        if idx == best_rate_idx:
            markers.append('✅ 達成率最高')
        
        marker_str = ', '.join(markers) if markers else ''
        
        print(f"{k:<5} {cv:<12.4f} {rate:<12.1f} {range_val:<12} {std:<12.1f} {marker_str}")
    
    # 推薦
    print(f"\n{'='*80}")
    print("💡 推薦")
    print(f"{'='*80}")
    
    best_k = results[best_cv_idx]['k']
    best_cv = results[best_cv_idx]['stats']['cv']
    best_rate = results[best_cv_idx]['elite_pool']['achievement_rate']
    
    print(f"\n✅ 推薦使用 K = {best_k}")
    print(f"   - 變異係數 (CV): {best_cv:.4f} (最小，最平衡)")
    print(f"   - Elite Pool 達成率: {best_rate:.1f}%")
    print(f"   - 在平衡性和達成率之間取得最佳權衡")
    
    # 生成視覺化
    print(f"\n{'='*80}")
    print("📊 生成視覺化")
    print(f"{'='*80}")
    
    output_dir = Path("optimal_k_results")
    visualize_results(results, output_dir)
    
    print(f"\n✅ 分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
