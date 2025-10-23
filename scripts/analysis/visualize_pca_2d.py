"""
使用 PCA 降維並可視化 Generation 的 2D 散點圖

輸入：generation.pkl 文件路徑
輸出：2D PCA 散點圖，根據不同維度著色
"""

import sys
from pathlib import Path
import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

# 添加項目根目錄到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import base, creator, gp, tools
from gp_quant.gp.operators import pset
from gp_quant.similarity import ParallelSimilarityMatrix

# 初始化 DEAP creator
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 設置中文字體和樣式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


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


def compute_similarity_matrix(population, n_workers=2):
    """計算相似度矩陣（使用指定數量的處理器）"""
    print(f"\n🔬 計算相似度矩陣...")
    print(f"   Population size: {len(population)}")
    print(f"   使用並行計算（{n_workers} 核心）...")
    
    sim_matrix = ParallelSimilarityMatrix(population, n_workers=n_workers)
    similarity_matrix = sim_matrix.compute(show_progress=True)
    
    print(f"   ✓ 相似度矩陣形狀: {similarity_matrix.shape}")
    print(f"   ✓ 平均相似度: {sim_matrix.get_average_similarity():.4f}")
    print(f"   ✓ 多樣性分數: {sim_matrix.get_diversity_score():.4f}")
    
    return similarity_matrix


def perform_pca(similarity_matrix, n_components=2):
    """執行 PCA 降維"""
    print(f"\n📊 執行 PCA 降維到 {n_components}D...")
    
    # 轉換為距離矩陣
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)
    
    # 標準化
    scaler = StandardScaler()
    distance_scaled = scaler.fit_transform(distance_matrix)
    
    # PCA
    pca = PCA(n_components=n_components)
    coords_2d = pca.fit_transform(distance_scaled)
    
    print(f"   ✓ PCA 完成")
    print(f"   ✓ 解釋方差比: {pca.explained_variance_ratio_}")
    print(f"   ✓ 累積解釋方差: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return coords_2d, pca


def extract_features(population):
    """提取個體特徵"""
    print(f"\n🔍 提取個體特徵...")
    
    features = {
        'fitness': [],
        'height': [],
        'size': [],
        'has_lag': [],
        'has_vol': [],
        'has_ma': [],
        'has_comparison': []
    }
    
    for ind in population:
        # Fitness
        features['fitness'].append(ind.fitness.values[0] if ind.fitness.valid else 0.0)
        
        # 樹的結構特徵
        features['height'].append(ind.height)
        features['size'].append(len(ind))
        
        # 規則特徵
        rule_str = str(ind)
        features['has_lag'].append(1 if 'lag' in rule_str else 0)
        features['has_vol'].append(1 if 'vol' in rule_str else 0)
        features['has_ma'].append(1 if 'ma' in rule_str else 0)
        features['has_comparison'].append(1 if any(op in rule_str for op in ['lt', 'gt', 'le', 'ge']) else 0)
    
    print(f"   ✓ 提取完成")
    return features


def plot_pca_scatter(coords_2d, features, output_dir, generation):
    """繪製 PCA 散點圖（多個視角）"""
    print(f"\n🎨 繪製 PCA 散點圖...")
    
    # 創建大圖
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'PCA 2D Visualization - Generation {generation}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 根據 Fitness 著色
    ax = axes[0, 0]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=features['fitness'], cmap='viridis', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Colored by Fitness', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Fitness')
    ax.grid(True, alpha=0.3)
    
    # 標記 Top 10
    top_10_indices = np.argsort(features['fitness'])[-10:]
    ax.scatter(coords_2d[top_10_indices, 0], coords_2d[top_10_indices, 1],
              s=200, facecolors='none', edgecolors='red', linewidths=2,
              label='Top 10')
    ax.legend()
    
    # 2. 根據樹高度著色
    ax = axes[0, 1]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=features['height'], cmap='plasma', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Colored by Tree Height', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Height')
    ax.grid(True, alpha=0.3)
    
    # 3. 根據樹大小著色
    ax = axes[0, 2]
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=features['size'], cmap='coolwarm', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Colored by Tree Size', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Size (nodes)')
    ax.grid(True, alpha=0.3)
    
    # 4. 根據是否使用 lag 著色
    ax = axes[1, 0]
    colors = ['blue' if x == 1 else 'gray' for x in features['has_lag']]
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
              c=colors, s=20, alpha=0.5, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Has LAG Operator (Blue=Yes, Gray=No)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 5. 根據是否使用 vol 著色
    ax = axes[1, 1]
    colors = ['green' if x == 1 else 'gray' for x in features['has_vol']]
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
              c=colors, s=20, alpha=0.5, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Has VOL Operator (Green=Yes, Gray=No)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 6. K-means 聚類結果（k=5）
    ax = axes[1, 2]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(coords_2d)
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=cluster_labels, cmap='tab10', 
                        s=20, alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('K-Means Clustering (k=5)', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"generation_{generation:03d}_pca_2d_multi_view.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path.name}")
    plt.close()


def plot_pca_density(coords_2d, features, output_dir, generation):
    """繪製 PCA 密度圖"""
    print(f"   繪製密度圖...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. 2D 密度圖
    ax = axes[0]
    from scipy.stats import gaussian_kde
    
    # 計算密度
    xy = coords_2d.T
    z = gaussian_kde(xy)(xy)
    
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                        c=z, s=20, cmap='YlOrRd', alpha=0.6, edgecolors='none')
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Density Heatmap', fontsize=13, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Density')
    ax.grid(True, alpha=0.3)
    
    # 2. Hexbin 圖
    ax = axes[1]
    hexbin = ax.hexbin(coords_2d[:, 0], coords_2d[:, 1], 
                       gridsize=30, cmap='Blues', mincnt=1)
    ax.set_xlabel('PC1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=12, fontweight='bold')
    ax.set_title('Hexbin Density', fontsize=13, fontweight='bold')
    plt.colorbar(hexbin, ax=ax, label='Count')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"generation_{generation:03d}_pca_2d_density.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path.name}")
    plt.close()


def plot_fitness_distribution(coords_2d, features, output_dir, generation):
    """繪製 Fitness 分布的詳細視圖"""
    print(f"   繪製 Fitness 分布...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 根據 fitness 分層
    fitness_array = np.array(features['fitness'])
    
    # 分成 5 個層級
    percentiles = [0, 20, 40, 60, 80, 100]
    colors = ['#d73027', '#fc8d59', '#fee08b', '#91cf60', '#1a9850']
    labels = ['Bottom 20%', '20-40%', '40-60%', '60-80%', 'Top 20%']
    
    for i in range(len(percentiles) - 1):
        lower = np.percentile(fitness_array, percentiles[i])
        upper = np.percentile(fitness_array, percentiles[i + 1])
        
        mask = (fitness_array >= lower) & (fitness_array <= upper)
        
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                  c=colors[i], label=labels[i], s=30, alpha=0.6, edgecolors='none')
    
    # 標記 Top 10
    top_10_indices = np.argsort(fitness_array)[-10:]
    ax.scatter(coords_2d[top_10_indices, 0], coords_2d[top_10_indices, 1],
              s=300, facecolors='none', edgecolors='black', linewidths=3,
              label='Top 10', zorder=10)
    
    # 標記 Top 1
    best_idx = np.argmax(fitness_array)
    ax.scatter(coords_2d[best_idx, 0], coords_2d[best_idx, 1],
              s=500, marker='*', c='gold', edgecolors='black', linewidths=2,
              label=f'Best (Fitness={fitness_array[best_idx]:.4f})', zorder=11)
    
    ax.set_xlabel('PC1', fontsize=14, fontweight='bold')
    ax.set_ylabel('PC2', fontsize=14, fontweight='bold')
    ax.set_title(f'PCA 2D - Fitness Distribution (Generation {generation})', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"generation_{generation:03d}_pca_2d_fitness_layers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path.name}")
    plt.close()


def save_pca_data(coords_2d, features, pca, output_dir, generation):
    """儲存 PCA 數據到 CSV"""
    print(f"\n💾 儲存 PCA 數據...")
    
    # 創建 DataFrame
    df = pd.DataFrame({
        'PC1': coords_2d[:, 0],
        'PC2': coords_2d[:, 1],
        'fitness': features['fitness'],
        'height': features['height'],
        'size': features['size'],
        'has_lag': features['has_lag'],
        'has_vol': features['has_vol'],
        'has_ma': features['has_ma'],
        'has_comparison': features['has_comparison']
    })
    
    # 儲存
    output_path = output_dir / f"generation_{generation:03d}_pca_2d_data.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✓ 已儲存: {output_path.name}")
    
    # 儲存 PCA 統計信息
    stats_path = output_dir / f"generation_{generation:03d}_pca_stats.txt"
    with open(stats_path, 'w') as f:
        f.write(f"PCA 降維統計信息 - Generation {generation}\n")
        f.write("="*80 + "\n\n")
        f.write(f"解釋方差比:\n")
        f.write(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}\n")
        f.write(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}\n")
        f.write(f"  累積: {np.sum(pca.explained_variance_ratio_):.4f}\n\n")
        f.write(f"特徵統計:\n")
        f.write(f"  Fitness - Mean: {np.mean(features['fitness']):.4f}, Std: {np.std(features['fitness']):.4f}\n")
        f.write(f"  Height - Mean: {np.mean(features['height']):.2f}, Std: {np.std(features['height']):.2f}\n")
        f.write(f"  Size - Mean: {np.mean(features['size']):.2f}, Std: {np.std(features['size']):.2f}\n")
        f.write(f"  Has LAG: {np.sum(features['has_lag'])} ({100*np.mean(features['has_lag']):.1f}%)\n")
        f.write(f"  Has VOL: {np.sum(features['has_vol'])} ({100*np.mean(features['has_vol']):.1f}%)\n")
        f.write(f"  Has MA: {np.sum(features['has_ma'])} ({100*np.mean(features['has_ma']):.1f}%)\n")
    
    print(f"   ✓ 已儲存: {stats_path.name}")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python visualize_pca_2d.py <generation.pkl 路徑> [n_workers]")
        print("範例: python visualize_pca_2d.py portfolio_experiment_results/.../generations/generation_006_final.pkl 2")
        sys.exit(1)
    
    pkl_path = Path(sys.argv[1])
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("🎨 PCA 2D 可視化")
    print("="*100 + "\n")
    
    # 1. 載入 generation
    data = load_generation(pkl_path)
    population = data['population']
    generation = data['generation']
    
    # 2. 計算相似度矩陣
    similarity_matrix = compute_similarity_matrix(population, n_workers=n_workers)
    
    # 3. 執行 PCA
    coords_2d, pca = perform_pca(similarity_matrix, n_components=2)
    
    # 4. 提取特徵
    features = extract_features(population)
    
    # 5. 創建輸出目錄
    exp_dir = pkl_path.parent.parent
    output_dir = exp_dir / "pca_2d_visualization"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 輸出目錄: {output_dir}")
    
    # 6. 繪製圖表
    plot_pca_scatter(coords_2d, features, output_dir, generation)
    plot_pca_density(coords_2d, features, output_dir, generation)
    plot_fitness_distribution(coords_2d, features, output_dir, generation)
    
    # 7. 儲存數據
    save_pca_data(coords_2d, features, pca, output_dir, generation)
    
    # 8. 完成
    print("\n" + "="*100)
    print("✅ PCA 可視化完成！")
    print("="*100)
    print(f"\n📊 生成的文件:")
    print(f"  圖表:")
    print(f"    - generation_{generation:03d}_pca_2d_multi_view.png (6 個視角)")
    print(f"    - generation_{generation:03d}_pca_2d_density.png (密度圖)")
    print(f"    - generation_{generation:03d}_pca_2d_fitness_layers.png (Fitness 分層)")
    print(f"  數據:")
    print(f"    - generation_{generation:03d}_pca_2d_data.csv")
    print(f"    - generation_{generation:03d}_pca_stats.txt")
    print(f"\n📁 保存位置: {output_dir}")
    print()


if __name__ == "__main__":
    main()
