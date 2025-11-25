#!/usr/bin/env python3
"""
比較不同 Linkage 方法的分群效果

測試 Complete vs Average Linkage 對 cluster 分布的影響
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deap import creator, base, gp
from sklearn.cluster import AgglomerativeClustering

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


def test_linkage_method(ted_matrix, n_clusters, linkage_method, population):
    """測試特定 linkage 方法"""
    print(f"\n{'='*80}")
    print(f"📊 測試 {linkage_method.upper()} Linkage")
    print(f"{'='*80}")
    
    # 執行分群
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage_method
    )
    
    cluster_labels = clustering.fit_predict(ted_matrix)
    
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
    
    # 統計
    mean_size = np.mean(counts)
    std_size = np.std(counts)
    min_size = np.min(counts)
    max_size = np.max(counts)
    cv = std_size / mean_size  # 變異係數（越小越平衡）
    
    print(f"\n統計:")
    print(f"  平均大小: {mean_size:.1f}")
    print(f"  標準差: {std_size:.1f}")
    print(f"  變異係數 (CV): {cv:.4f}")
    print(f"  最小大小: {min_size}")
    print(f"  最大大小: {max_size}")
    print(f"  大小範圍: {max_size - min_size}")
    
    # 計算 Elite Pool 達成率
    M = 50
    elite_pool_size = sum(min(count, M) for count in counts)
    expected_size = n_clusters * M
    achievement_rate = elite_pool_size / expected_size * 100
    
    print(f"\nElite Pool (Top {M} per cluster):")
    print(f"  實際大小: {elite_pool_size}")
    print(f"  預期大小: {expected_size}")
    print(f"  達成率: {achievement_rate:.1f}%")
    
    # 計算每個 cluster 的 fitness 統計
    print(f"\nCluster Fitness 統計:")
    print(f"{'Cluster':<10} {'平均':<10} {'最大':<10} {'最小':<10}")
    print("-" * 50)
    
    for label in unique_labels:
        cluster_mask = cluster_labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_individuals = [population[i] for i in cluster_indices]
        
        fitnesses = [ind.fitness.values[0] for ind in cluster_individuals]
        mean_fit = np.mean(fitnesses)
        max_fit = np.max(fitnesses)
        min_fit = np.min(fitnesses)
        
        print(f"{label:<10} {mean_fit:<10.4f} {max_fit:<10.4f} {min_fit:<10.4f}")
    
    return {
        'linkage': linkage_method,
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
        }
    }


def visualize_comparison(results_list, output_dir):
    """視覺化比較結果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_methods = len(results_list)
    
    # 1. Cluster 大小分布比較
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        ax = axes[idx]
        
        clusters = [info['cluster'] for info in results['cluster_info']]
        sizes = [info['size'] for info in results['cluster_info']]
        
        bars = ax.bar(clusters, sizes, color='steelblue', alpha=0.7)
        
        # 標記 M=50 的線
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='M=50')
        
        # 在每個 bar 上標註數值
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(size)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Cluster Size', fontsize=12)
        ax.set_title(f"{results['linkage'].upper()} Linkage\nCV={results['stats']['cv']:.4f}", 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_size_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_dir / 'cluster_size_comparison.png'}")
    plt.close()
    
    # 2. 統計指標比較
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    linkage_names = [r['linkage'].upper() for r in results_list]
    
    # CV (變異係數)
    cvs = [r['stats']['cv'] for r in results_list]
    axes[0, 0].bar(linkage_names, cvs, color='coral', alpha=0.7)
    axes[0, 0].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[0, 0].set_title('變異係數 (越小越平衡)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(cvs):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Elite Pool 達成率
    rates = [r['elite_pool']['achievement_rate'] for r in results_list]
    axes[0, 1].bar(linkage_names, rates, color='lightgreen', alpha=0.7)
    axes[0, 1].set_ylabel('Achievement Rate (%)', fontsize=11)
    axes[0, 1].set_title('Elite Pool 達成率', fontsize=12, fontweight='bold')
    axes[0, 1].axhline(y=100, color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(rates):
        axes[0, 1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Cluster 大小範圍
    ranges = [r['stats']['range'] for r in results_list]
    axes[1, 0].bar(linkage_names, ranges, color='skyblue', alpha=0.7)
    axes[1, 0].set_ylabel('Size Range', fontsize=11)
    axes[1, 0].set_title('Cluster 大小範圍 (Max - Min)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(ranges):
        axes[1, 0].text(i, v, f'{v}', ha='center', va='bottom', fontsize=10)
    
    # 標準差
    stds = [r['stats']['std_size'] for r in results_list]
    axes[1, 1].bar(linkage_names, stds, color='plum', alpha=0.7)
    axes[1, 1].set_ylabel('Standard Deviation', fontsize=11)
    axes[1, 1].set_title('Cluster 大小標準差', fontsize=12, fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(stds):
        axes[1, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_dir / 'statistics_comparison.png'}")
    plt.close()


def main():
    """主函數"""
    print("="*80)
    print("🔍 比較不同 Linkage 方法的分群效果")
    print("="*80)
    
    # 設置
    setup_deap_creator()
    
    # 載入測試族群
    records_dir = Path("/Users/hongyicheng/Downloads/gp_quant/test_evolution_11241221_records_20251125_1335")
    population = load_test_population(records_dir, generation=0)
    
    print(f"\n📦 載入族群: {len(population)} 個個體")
    
    # 創建策略並計算 TED matrix
    print(f"\n📊 計算 TED Distance Matrix...")
    strategy = TEDNicheSelectionStrategy(
        n_clusters=5,
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        n_jobs=6
    )
    
    ted_matrix = strategy._calculate_ted_distance_matrix(population)
    print(f"✅ TED Matrix 計算完成: {ted_matrix.shape}")
    
    # 測試不同的 linkage 方法
    linkage_methods = ['complete', 'average']
    results_list = []
    
    for method in linkage_methods:
        results = test_linkage_method(ted_matrix, n_clusters=5, 
                                     linkage_method=method, population=population)
        results_list.append(results)
    
    # 比較總結
    print(f"\n{'='*80}")
    print("📊 比較總結")
    print(f"{'='*80}")
    
    print(f"\n{'指標':<25} {'Complete':<15} {'Average':<15} {'差異'}")
    print("-" * 70)
    
    complete_results = results_list[0]
    average_results = results_list[1]
    
    # 變異係數
    cv_complete = complete_results['stats']['cv']
    cv_average = average_results['stats']['cv']
    cv_diff = cv_average - cv_complete
    cv_better = "Average 更好" if cv_average < cv_complete else "Complete 更好"
    print(f"{'變異係數 (CV)':<25} {cv_complete:<15.4f} {cv_average:<15.4f} {cv_diff:+.4f} ({cv_better})")
    
    # Elite Pool 達成率
    rate_complete = complete_results['elite_pool']['achievement_rate']
    rate_average = average_results['elite_pool']['achievement_rate']
    rate_diff = rate_average - rate_complete
    rate_better = "Average 更好" if rate_average > rate_complete else "Complete 更好"
    print(f"{'Elite Pool 達成率 (%)':<25} {rate_complete:<15.1f} {rate_average:<15.1f} {rate_diff:+.1f} ({rate_better})")
    
    # 大小範圍
    range_complete = complete_results['stats']['range']
    range_average = average_results['stats']['range']
    range_diff = range_average - range_complete
    range_better = "Average 更好" if range_average < range_complete else "Complete 更好"
    print(f"{'Cluster 大小範圍':<25} {range_complete:<15} {range_average:<15} {range_diff:+} ({range_better})")
    
    # 標準差
    std_complete = complete_results['stats']['std_size']
    std_average = average_results['stats']['std_size']
    std_diff = std_average - std_complete
    std_better = "Average 更好" if std_average < std_complete else "Complete 更好"
    print(f"{'標準差':<25} {std_complete:<15.1f} {std_average:<15.1f} {std_diff:+.1f} ({std_better})")
    
    # 推薦
    print(f"\n{'='*80}")
    print("💡 推薦")
    print(f"{'='*80}")
    
    # 計算綜合得分
    complete_score = 0
    average_score = 0
    
    if cv_average < cv_complete:
        average_score += 1
    else:
        complete_score += 1
    
    if rate_average > rate_complete:
        average_score += 1
    else:
        complete_score += 1
    
    if range_average < range_complete:
        average_score += 1
    else:
        complete_score += 1
    
    if std_average < std_complete:
        average_score += 1
    else:
        complete_score += 1
    
    print(f"\n綜合評分:")
    print(f"  Complete Linkage: {complete_score}/4")
    print(f"  Average Linkage: {average_score}/4")
    
    if average_score > complete_score:
        print(f"\n✅ 推薦使用 **Average Linkage**")
        print(f"   - 更平衡的 cluster 分布")
        print(f"   - 更高的 Elite Pool 達成率")
    elif complete_score > average_score:
        print(f"\n✅ 推薦使用 **Complete Linkage**")
        print(f"   - 更緊密的 cluster")
    else:
        print(f"\n⚖️  兩種方法表現相當，可根據具體需求選擇")
    
    # 生成視覺化
    print(f"\n{'='*80}")
    print("📊 生成視覺化比較")
    print(f"{'='*80}")
    
    output_dir = Path("linkage_comparison_results")
    visualize_comparison(results_list, output_dir)
    
    print(f"\n✅ 比較完成！")
    print("="*80)


if __name__ == "__main__":
    main()
