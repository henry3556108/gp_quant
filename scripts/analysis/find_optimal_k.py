"""
尋找最佳的 Cluster 數量 (k)

使用 Silhouette Score 評估不同 k 值的聚類品質
輸入：generation.pkl 文件路徑
輸出：最佳 k 值和可視化圖表
"""

import sys
from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
import pandas as pd

# 添加項目根目錄到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import base, creator, gp, tools
from gp_quant.gp.operators import pset
from gp_quant.similarity import SimilarityMatrix, ParallelSimilarityMatrix

# 初始化 DEAP creator
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_generation(pkl_path):
    """載入 generation.pkl"""
    print(f"📂 載入文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    
    generation = data['generation']
    population = data['population']
    
    print(f"   ✓ Generation: {generation}")
    print(f"   ✓ Population size: {len(population)}")
    
    return data


def compute_similarity_matrix(population, use_parallel=True):
    """計算相似度矩陣"""
    print(f"\n🔬 計算相似度矩陣...")
    print(f"   Population size: {len(population)}")
    
    if use_parallel and len(population) >= 200:
        print(f"   使用並行計算（2 核心）...")
        sim_matrix = ParallelSimilarityMatrix(population, n_workers=2)
        similarity_matrix = sim_matrix.compute(show_progress=True)
    else:
        print(f"   使用序列計算...")
        sim_matrix = SimilarityMatrix(population)
        similarity_matrix = sim_matrix.compute(show_progress=True)
    
    print(f"   ✓ 相似度矩陣形狀: {similarity_matrix.shape}")
    print(f"   ✓ 平均相似度: {sim_matrix.get_average_similarity():.4f}")
    print(f"   ✓ 多樣性分數: {sim_matrix.get_diversity_score():.4f}")
    
    return similarity_matrix


def evaluate_k_range(similarity_matrix, k_min=2, k_max=50):
    """評估不同 k 值的聚類品質"""
    print(f"\n📊 評估 k 值範圍: [{k_min}, {k_max}]")
    
    # 轉換為距離矩陣
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    
    results = []
    
    for k in range(k_min, k_max + 1):
        if k >= len(similarity_matrix):
            print(f"   ⚠️  k={k} 超過 population size，跳過")
            break
        
        print(f"   測試 k={k}...", end='')
        
        try:
            # K-means 聚類
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(distance_matrix)
            
            # 計算 Silhouette Score
            silhouette_avg = silhouette_score(
                distance_matrix, 
                labels, 
                metric='precomputed'
            )
            
            # 計算每個樣本的 Silhouette Score
            silhouette_vals = silhouette_samples(
                distance_matrix,
                labels,
                metric='precomputed'
            )
            
            # 計算每個 cluster 的統計
            cluster_stats = {}
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_scores = silhouette_vals[cluster_mask]
                cluster_stats[cluster_id] = {
                    'size': int(np.sum(cluster_mask)),
                    'mean': float(np.mean(cluster_scores)),
                    'std': float(np.std(cluster_scores)),
                    'min': float(np.min(cluster_scores)),
                    'max': float(np.max(cluster_scores))
                }
            
            # 計算 cluster 大小的標準差（平衡度指標）
            cluster_sizes = [stats['size'] for stats in cluster_stats.values()]
            size_std = np.std(cluster_sizes)
            size_cv = size_std / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
            
            results.append({
                'k': k,
                'silhouette_score': silhouette_avg,
                'silhouette_std': float(np.std(silhouette_vals)),
                'cluster_stats': cluster_stats,
                'cluster_size_std': size_std,
                'cluster_size_cv': size_cv,
                'min_cluster_size': min(cluster_sizes),
                'max_cluster_size': max(cluster_sizes)
            })
            
            print(f" Silhouette: {silhouette_avg:.4f}")
            
        except Exception as e:
            print(f" ✗ 失敗: {e}")
            continue
    
    return results


def find_optimal_k(results, method='silhouette'):
    """找出最佳 k 值"""
    if method == 'silhouette':
        # 使用 Silhouette Score
        best_result = max(results, key=lambda x: x['silhouette_score'])
        return best_result
    elif method == 'elbow':
        # 使用 Elbow 方法（需要額外計算 inertia）
        pass
    
    return None


def plot_k_analysis(results, output_dir, generation):
    """繪製 k 值分析圖表"""
    print(f"\n🎨 繪製分析圖表...")
    
    # 提取數據
    k_values = [r['k'] for r in results]
    silhouette_scores = [r['silhouette_score'] for r in results]
    silhouette_stds = [r['silhouette_std'] for r in results]
    size_cvs = [r['cluster_size_cv'] for r in results]
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Silhouette Score vs k
    ax = axes[0, 0]
    ax.plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 標記最佳 k
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    best_score = silhouette_scores[best_idx]
    ax.scatter([best_k], [best_score], color='red', s=200, marker='*', 
              zorder=5, label=f'Best k={best_k} (Score={best_score:.4f})')
    ax.legend(fontsize=11)
    
    # 2. Silhouette Score 分布（帶誤差條）
    ax = axes[0, 1]
    ax.errorbar(k_values, silhouette_scores, yerr=silhouette_stds, 
               fmt='o-', linewidth=2, markersize=6, capsize=5, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score (mean ± std)', fontsize=12, fontweight='bold')
    ax.set_title('Silhouette Score Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Cluster Size 變異係數
    ax = axes[1, 0]
    ax.plot(k_values, size_cvs, marker='s', linewidth=2, markersize=6, color='orange')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster Size CV', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Size Variability (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Top 10 k 值的詳細比較
    ax = axes[1, 1]
    top_10_indices = np.argsort(silhouette_scores)[-10:]
    top_10_k = [k_values[i] for i in top_10_indices]
    top_10_scores = [silhouette_scores[i] for i in top_10_indices]
    
    bars = ax.barh(range(len(top_10_k)), top_10_scores, color='skyblue', edgecolor='black')
    ax.set_yticks(range(len(top_10_k)))
    ax.set_yticklabels([f'k={k}' for k in top_10_k])
    ax.set_xlabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 k Values by Silhouette Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 標記最佳
    best_in_top10 = top_10_scores.index(max(top_10_scores))
    bars[best_in_top10].set_color('gold')
    bars[best_in_top10].set_edgecolor('red')
    bars[best_in_top10].set_linewidth(2)
    
    plt.tight_layout()
    output_path = output_dir / f"generation_{generation:03d}_optimal_k_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path.name}")
    plt.close()
    
    # 繪製每個 k 的 cluster 大小分布
    plot_cluster_size_distribution(results, output_dir, generation)


def plot_cluster_size_distribution(results, output_dir, generation):
    """繪製不同 k 值的 cluster 大小分布"""
    print(f"   繪製 cluster 大小分布...")
    
    # 選擇幾個代表性的 k 值
    k_values = [r['k'] for r in results]
    silhouette_scores = [r['silhouette_score'] for r in results]
    
    # 選擇最佳的和幾個代表性的 k
    best_idx = np.argmax(silhouette_scores)
    representative_indices = [
        0,  # 最小 k
        len(results) // 4,  # 1/4
        len(results) // 2,  # 中間
        3 * len(results) // 4,  # 3/4
        best_idx,  # 最佳
        len(results) - 1  # 最大 k
    ]
    representative_indices = sorted(set(representative_indices))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, result_idx in enumerate(representative_indices[:6]):
        result = results[result_idx]
        k = result['k']
        cluster_stats = result['cluster_stats']
        
        ax = axes[idx]
        
        # 提取 cluster 大小
        cluster_ids = sorted(cluster_stats.keys())
        cluster_sizes = [cluster_stats[cid]['size'] for cid in cluster_ids]
        cluster_means = [cluster_stats[cid]['mean'] for cid in cluster_ids]
        
        # 繪製柱狀圖
        bars = ax.bar(cluster_ids, cluster_sizes, color='skyblue', edgecolor='black', alpha=0.7)
        
        # 根據 silhouette score 著色
        for i, (bar, mean_score) in enumerate(zip(bars, cluster_means)):
            if mean_score > 0.5:
                bar.set_color('green')
            elif mean_score > 0.3:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        ax.set_xlabel('Cluster ID', fontsize=10)
        ax.set_ylabel('Cluster Size', fontsize=10)
        
        is_best = (result_idx == best_idx)
        title = f"k={k} (Silhouette={result['silhouette_score']:.4f})"
        if is_best:
            title += " ⭐ BEST"
            ax.set_title(title, fontsize=11, fontweight='bold', color='red')
        else:
            ax.set_title(title, fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隱藏多餘的子圖
    for idx in range(len(representative_indices), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"generation_{generation:03d}_cluster_size_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path.name}")
    plt.close()


def save_results_to_csv(results, output_dir, generation):
    """儲存結果到 CSV"""
    print(f"\n💾 儲存結果到 CSV...")
    
    # 基本統計
    basic_stats = []
    for r in results:
        basic_stats.append({
            'k': r['k'],
            'silhouette_score': r['silhouette_score'],
            'silhouette_std': r['silhouette_std'],
            'cluster_size_std': r['cluster_size_std'],
            'cluster_size_cv': r['cluster_size_cv'],
            'min_cluster_size': r['min_cluster_size'],
            'max_cluster_size': r['max_cluster_size']
        })
    
    df = pd.DataFrame(basic_stats)
    output_path = output_dir / f"generation_{generation:03d}_k_analysis.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✓ 已儲存: {output_path.name}")
    
    # 詳細的 cluster 統計
    detailed_stats = []
    for r in results:
        k = r['k']
        for cluster_id, stats in r['cluster_stats'].items():
            detailed_stats.append({
                'k': k,
                'cluster_id': cluster_id,
                'size': stats['size'],
                'silhouette_mean': stats['mean'],
                'silhouette_std': stats['std'],
                'silhouette_min': stats['min'],
                'silhouette_max': stats['max']
            })
    
    df_detailed = pd.DataFrame(detailed_stats)
    output_path_detailed = output_dir / f"generation_{generation:03d}_cluster_details.csv"
    df_detailed.to_csv(output_path_detailed, index=False)
    print(f"   ✓ 已儲存: {output_path_detailed.name}")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python find_optimal_k.py <generation.pkl 路徑> [k_min] [k_max]")
        print("範例: python find_optimal_k.py portfolio_experiment_results/.../generations/generation_006_final.pkl 2 50")
        sys.exit(1)
    
    pkl_path = Path(sys.argv[1])
    k_min = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    k_max = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("🔍 尋找最佳 Cluster 數量 (k)")
    print("="*100 + "\n")
    
    # 1. 載入 generation
    data = load_generation(pkl_path)
    population = data['population']
    generation = data['generation']
    
    # 2. 計算相似度矩陣
    similarity_matrix = compute_similarity_matrix(population, use_parallel=True)
    
    # 3. 評估不同 k 值
    results = evaluate_k_range(similarity_matrix, k_min, k_max)
    
    if not results:
        print("❌ 沒有成功評估任何 k 值")
        sys.exit(1)
    
    # 4. 找出最佳 k
    print(f"\n🏆 分析結果:")
    best_result = find_optimal_k(results, method='silhouette')
    
    print(f"\n{'='*100}")
    print(f"⭐ 最佳 k 值: {best_result['k']}")
    print(f"{'='*100}")
    print(f"  Silhouette Score: {best_result['silhouette_score']:.4f}")
    print(f"  Silhouette Std: {best_result['silhouette_std']:.4f}")
    print(f"  Cluster Size CV: {best_result['cluster_size_cv']:.4f}")
    print(f"  Cluster Size Range: [{best_result['min_cluster_size']}, {best_result['max_cluster_size']}]")
    
    print(f"\n  各 Cluster 詳細信息:")
    for cluster_id, stats in best_result['cluster_stats'].items():
        print(f"    Cluster {cluster_id}: size={stats['size']}, "
              f"silhouette={stats['mean']:.4f} (±{stats['std']:.4f})")
    
    # 顯示 Top 5
    print(f"\n📊 Top 5 k 值:")
    sorted_results = sorted(results, key=lambda x: x['silhouette_score'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. k={r['k']}: Silhouette={r['silhouette_score']:.4f}, "
              f"Size CV={r['cluster_size_cv']:.4f}")
    
    # 5. 創建輸出目錄
    exp_dir = pkl_path.parent.parent
    output_dir = exp_dir / "optimal_k_analysis"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 輸出目錄: {output_dir}")
    
    # 6. 繪製圖表
    plot_k_analysis(results, output_dir, generation)
    
    # 7. 儲存結果
    save_results_to_csv(results, output_dir, generation)
    
    # 8. 完成
    print("\n" + "="*100)
    print("✅ 分析完成！")
    print("="*100)
    print(f"\n📊 生成的文件:")
    print(f"  - generation_{generation:03d}_optimal_k_analysis.png")
    print(f"  - generation_{generation:03d}_cluster_size_distribution.png")
    print(f"  - generation_{generation:03d}_k_analysis.csv")
    print(f"  - generation_{generation:03d}_cluster_details.csv")
    print(f"\n📁 保存位置: {output_dir}")
    print()


if __name__ == "__main__":
    main()
