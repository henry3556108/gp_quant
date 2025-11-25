#!/usr/bin/env python3
"""
Analyze TED-based Clustering with Visualization

計算指定世代的 TED distance matrix，進行階層式分群，並使用 PCA 視覺化。
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
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.similarity.tree_edit_distance import compute_ted


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_generation_population(records_dir: Path, generation: int) -> List:
    """
    載入指定世代的族群
    
    Args:
        records_dir: 記錄目錄路徑
        generation: 世代號
        
    Returns:
        族群列表
    """
    populations_dir = records_dir / 'populations'
    
    if not populations_dir.exists():
        raise ValueError(f"Populations directory not found: {populations_dir}")
    
    gen_file = populations_dir / f'generation_{generation:03d}.pkl'
    
    if not gen_file.exists():
        raise ValueError(f"Generation {generation} file not found: {gen_file}")
    
    print(f"📂 載入世代 {generation}: {gen_file.name}")
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    print(f"   ✅ 載入 {len(population)} 個個體")
    
    return population


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


def calculate_ted_distance_matrix(population: List, n_jobs: int = 6) -> np.ndarray:
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
    
    # 平行計算（使用 threading backend 避免 DEAP creator 序列化問題）
    results = Parallel(n_jobs=n_jobs, backend='threading')(
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


def perform_hierarchical_clustering(distance_matrix: np.ndarray, 
                                     n_clusters: int = 3) -> Tuple[np.ndarray, Any]:
    """
    執行階層式分群
    
    Args:
        distance_matrix: 距離矩陣
        n_clusters: 群數
        
    Returns:
        (cluster_labels, linkage_matrix)
    """
    print(f"\n🔬 執行階層式分群 (K={n_clusters})...")
    
    # 使用 AgglomerativeClustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='complete'  # 使用 complete linkage
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    # 計算 linkage matrix 用於 dendrogram
    # 將距離矩陣轉換為壓縮形式
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # 統計每個群的大小
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print(f"   ✅ 分群完成:")
    for label, count in zip(unique_labels, counts):
        print(f"      群 {label}: {count} 個個體")
    
    return cluster_labels, linkage_matrix


def visualize_clustering_pca(distance_matrix: np.ndarray,
                              cluster_labels: np.ndarray,
                              population: List,
                              output_path: Path):
    """
    使用 PCA 將距離矩陣降維到 2D 並視覺化分群結果
    
    Args:
        distance_matrix: 距離矩陣
        cluster_labels: 群標籤
        population: 族群列表
        output_path: 輸出圖表路徑
    """
    print(f"\n📊 使用 PCA 降維並視覺化...")
    
    # 使用 PCA 降維（從距離矩陣）
    # 注意：PCA 通常用於特徵矩陣，但我們可以用距離矩陣的 MDS 效果
    # 這裡使用簡單的 PCA 作為示範
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(distance_matrix)
    
    print(f"   ✅ PCA 解釋變異量: {pca.explained_variance_ratio_}")
    print(f"   ✅ 累積解釋變異量: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 創建圖表
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # ========== 左圖：PCA 散點圖 ==========
    ax_scatter = axes[0]
    
    # 為每個群使用不同顏色
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    markers = ['o', 's', '^', 'D', 'v']
    
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        mask = cluster_labels == label
        ax_scatter.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=colors[label % len(colors)],
            marker=markers[label % len(markers)],
            s=100,
            alpha=0.6,
            label=f'Cluster {label} (n={np.sum(mask)})',
            edgecolors='black',
            linewidth=0.5
        )
    
    ax_scatter.set_xlabel('PC1', fontsize=13, fontweight='bold')
    ax_scatter.set_ylabel('PC2', fontsize=13, fontweight='bold')
    ax_scatter.set_title('TED-based Clustering (PCA Visualization)', 
                         fontsize=15, fontweight='bold', pad=15)
    ax_scatter.legend(loc='best', fontsize=11, framealpha=0.9)
    ax_scatter.grid(True, alpha=0.3, linestyle='--')
    
    # 添加統計信息
    stats_text = f'Total Individuals: {len(population)} | Clusters: {len(unique_labels)}'
    ax_scatter.text(0.5, 0.02, stats_text, transform=ax_scatter.transAxes,
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # ========== 右圖：每個群的 Fitness 分布 ==========
    ax_fitness = axes[1]
    
    fitness_by_cluster = []
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_fitness = [population[i].fitness.values[0] for i in range(len(population)) if mask[i]]
        fitness_by_cluster.append(cluster_fitness)
    
    # 繪製箱型圖
    bp = ax_fitness.boxplot(fitness_by_cluster, 
                            labels=[f'Cluster {i}' for i in unique_labels],
                            patch_artist=True,
                            notch=True,
                            showmeans=True)
    
    # 設置顏色
    for patch, label in zip(bp['boxes'], unique_labels):
        patch.set_facecolor(colors[label % len(colors)])
        patch.set_alpha(0.6)
    
    ax_fitness.set_xlabel('Cluster', fontsize=13, fontweight='bold')
    ax_fitness.set_ylabel('Fitness', fontsize=13, fontweight='bold')
    ax_fitness.set_title('Fitness Distribution by Cluster', 
                        fontsize=15, fontweight='bold', pad=15)
    ax_fitness.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 添加統計信息
    for i, (label, fitness_list) in enumerate(zip(unique_labels, fitness_by_cluster)):
        mean_fitness = np.mean(fitness_list)
        std_fitness = np.std(fitness_list)
        ax_fitness.text(i + 1, ax_fitness.get_ylim()[1] * 0.95, 
                       f'μ={mean_fitness:.4f}\nσ={std_fitness:.4f}',
                       ha='center', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 圖表已保存: {output_path}")
    plt.close()


def save_clustering_results(distance_matrix: np.ndarray,
                            cluster_labels: np.ndarray,
                            population: List,
                            output_dir: Path,
                            generation: int):
    """
    保存分群結果
    
    Args:
        distance_matrix: 距離矩陣
        cluster_labels: 群標籤
        population: 族群列表
        output_dir: 輸出目錄
        generation: 世代號
    """
    print(f"\n💾 保存分群結果...")
    
    # 1. 保存距離矩陣
    dist_path = output_dir / f'ted_distance_matrix_gen{generation:03d}.csv'
    pd.DataFrame(distance_matrix).to_csv(dist_path, index=False, header=False)
    print(f"   ✅ 距離矩陣已保存: {dist_path}")
    
    # 2. 保存群標籤和個體信息
    cluster_info = []
    for i, (individual, label) in enumerate(zip(population, cluster_labels)):
        cluster_info.append({
            'individual_id': i,
            'cluster': int(label),
            'fitness': individual.fitness.values[0],
            'tree_size': len(individual),
            'tree_depth': individual.height
        })
    
    cluster_df = pd.DataFrame(cluster_info)
    cluster_path = output_dir / f'cluster_assignments_gen{generation:03d}.csv'
    cluster_df.to_csv(cluster_path, index=False)
    print(f"   ✅ 群標籤已保存: {cluster_path}")
    
    # 3. 保存每個群的統計摘要
    summary = []
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        cluster_individuals = [population[i] for i in range(len(population)) if mask[i]]
        
        fitnesses = [ind.fitness.values[0] for ind in cluster_individuals]
        tree_sizes = [len(ind) for ind in cluster_individuals]
        tree_depths = [ind.height for ind in cluster_individuals]
        
        summary.append({
            'cluster': int(label),
            'size': int(np.sum(mask)),
            'mean_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'min_fitness': np.min(fitnesses),
            'max_fitness': np.max(fitnesses),
            'mean_tree_size': np.mean(tree_sizes),
            'mean_tree_depth': np.mean(tree_depths)
        })
    
    summary_df = pd.DataFrame(summary)
    summary_path = output_dir / f'cluster_summary_gen{generation:03d}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"   ✅ 群統計摘要已保存: {summary_path}")
    
    return cluster_df, summary_df


def main():
    parser = argparse.ArgumentParser(
        description="分析 TED-based 分群並視覺化"
    )
    parser.add_argument(
        '--records',
        type=str,
        required=True,
        help='實驗記錄目錄路徑'
    )
    parser.add_argument(
        '--generation',
        type=int,
        default=0,
        help='世代號（默認：0）'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=3,
        help='群數（默認：3）'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=6,
        help='平行處理器數量（默認：6）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='輸出目錄（默認保存在記錄目錄中）'
    )
    
    args = parser.parse_args()
    
    records_dir = Path(args.records)
    
    if not records_dir.exists():
        print(f"❌ 記錄目錄不存在: {records_dir}")
        return
    
    print("=" * 80)
    print("🎯 TED-based Clustering Analysis")
    print("=" * 80)
    print(f"Records directory: {records_dir}")
    print(f"Generation: {args.generation}")
    print(f"Number of clusters: {args.n_clusters}")
    print(f"N jobs: {args.n_jobs}\n")
    
    # 1. 設置 DEAP
    setup_deap_creator()
    
    # 2. 載入世代族群
    print("📦 載入世代族群...")
    population = load_generation_population(records_dir, args.generation)
    
    # 3. 計算 TED distance matrix
    ted_matrix = calculate_ted_distance_matrix(population, args.n_jobs)
    
    # 4. 執行階層式分群
    cluster_labels, linkage_matrix = perform_hierarchical_clustering(
        ted_matrix, args.n_clusters
    )
    
    # 5. 設置輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = records_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. 視覺化
    viz_path = output_dir / f'ted_clustering_gen{args.generation:03d}.png'
    visualize_clustering_pca(ted_matrix, cluster_labels, population, viz_path)
    
    # 7. 保存結果
    cluster_df, summary_df = save_clustering_results(
        ted_matrix, cluster_labels, population, output_dir, args.generation
    )
    
    # 8. 輸出摘要
    print("\n" + "=" * 80)
    print("✅ 完成!")
    print("=" * 80)
    print(f"世代: {args.generation}")
    print(f"個體數量: {len(population)}")
    print(f"群數: {args.n_clusters}")
    print(f"\n群統計:")
    print(summary_df.to_string(index=False))
    print(f"\n輸出文件:")
    print(f"  - {viz_path}")
    print(f"  - {output_dir / f'ted_distance_matrix_gen{args.generation:03d}.csv'}")
    print(f"  - {output_dir / f'cluster_assignments_gen{args.generation:03d}.csv'}")
    print(f"  - {output_dir / f'cluster_summary_gen{args.generation:03d}.csv'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
