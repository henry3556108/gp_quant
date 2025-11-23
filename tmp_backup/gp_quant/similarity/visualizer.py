"""
Similarity Visualization Module

提供多樣性和相似度的視覺化工具
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Union
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .similarity_matrix import SimilarityMatrix
from .parallel_calculator import ParallelSimilarityMatrix


# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_diversity_evolution(
    diversity_metrics_file: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 8),
    dpi: int = 300
):
    """
    繪製多樣性演化曲線
    
    Args:
        diversity_metrics_file: diversity_metrics.json 文件路徑
        save_path: 儲存路徑（如果為 None，則顯示圖表）
        figsize: 圖表大小
        dpi: 圖片解析度
    """
    # 讀取數據
    with open(diversity_metrics_file, 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    df = pd.DataFrame(metrics)
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 子圖 1: 多樣性分數
    ax1.plot(df['generation'], df['diversity_score'], 
             linewidth=2, color='#2E86AB', marker='o', markersize=4,
             label='多樣性分數')
    ax1.fill_between(df['generation'], 
                      df['diversity_score'] - df['std_similarity'],
                      df['diversity_score'] + df['std_similarity'],
                      alpha=0.2, color='#2E86AB')
    
    ax1.set_ylabel('多樣性分數', fontsize=12, fontweight='bold')
    ax1.set_title(f'多樣性演化 - {data["experiment"]}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 標註起始和結束值
    first_val = df['diversity_score'].iloc[0]
    last_val = df['diversity_score'].iloc[-1]
    ax1.annotate(f'{first_val:.4f}', 
                xy=(df['generation'].iloc[0], first_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='#2E86AB')
    ax1.annotate(f'{last_val:.4f}', 
                xy=(df['generation'].iloc[-1], last_val),
                xytext=(-40, 10), textcoords='offset points',
                fontsize=10, color='#2E86AB')
    
    # 子圖 2: 平均相似度
    ax2.plot(df['generation'], df['avg_similarity'], 
             linewidth=2, color='#A23B72', marker='s', markersize=4,
             label='平均相似度')
    ax2.fill_between(df['generation'], 
                      df['avg_similarity'] - df['std_similarity'],
                      df['avg_similarity'] + df['std_similarity'],
                      alpha=0.2, color='#A23B72')
    
    ax2.set_xlabel('世代', fontsize=12, fontweight='bold')
    ax2.set_ylabel('平均相似度', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # 標註起始和結束值
    first_val = df['avg_similarity'].iloc[0]
    last_val = df['avg_similarity'].iloc[-1]
    ax2.annotate(f'{first_val:.4f}', 
                xy=(df['generation'].iloc[0], first_val),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, color='#A23B72')
    ax2.annotate(f'{last_val:.4f}', 
                xy=(df['generation'].iloc[-1], last_val),
                xytext=(-40, -20), textcoords='offset points',
                fontsize=10, color='#A23B72')
    
    plt.tight_layout()
    
    # 儲存或顯示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_heatmap(
    population_file: Union[str, Path],
    generation: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
    dpi: int = 300,
    sample_size: Optional[int] = None
):
    """
    繪製相似度矩陣熱圖
    
    Args:
        population_file: generation_XXX.pkl 文件路徑
        generation: 世代編號（用於標題）
        save_path: 儲存路徑
        figsize: 圖表大小
        dpi: 圖片解析度
        sample_size: 如果族群太大，可以隨機抽樣（None = 不抽樣）
    """
    # 載入族群
    with open(population_file, 'rb') as f:
        population = pickle.load(f)
    
    # 抽樣（如果需要）
    if sample_size and len(population) > sample_size:
        import random
        population = random.sample(population, sample_size)
        print(f"族群抽樣: {len(population)} 個個體")
    
    # 計算相似度矩陣
    print("計算相似度矩陣...")
    if len(population) >= 200:
        sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
    else:
        sim_matrix = SimilarityMatrix(population)
    
    similarity_matrix = sim_matrix.compute(show_progress=True)
    
    # 計算統計
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    avg_sim = np.mean(similarities)
    diversity = 1.0 - avg_sim
    
    # 繪製熱圖
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(similarity_matrix, 
                cmap='YlOrRd', 
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': '相似度'},
                ax=ax)
    
    # 標題
    if generation:
        title = f'相似度矩陣 - Generation {generation}'
    else:
        title = '相似度矩陣'
    
    title += f'\n平均相似度: {avg_sim:.4f} | 多樣性分數: {diversity:.4f}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('個體索引', fontsize=12)
    ax.set_ylabel('個體索引', fontsize=12)
    
    plt.tight_layout()
    
    # 儲存或顯示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_distribution(
    population_file: Union[str, Path],
    generation: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300
):
    """
    繪製相似度分佈直方圖
    
    Args:
        population_file: generation_XXX.pkl 文件路徑
        generation: 世代編號
        save_path: 儲存路徑
        figsize: 圖表大小
        dpi: 圖片解析度
    """
    # 載入族群
    with open(population_file, 'rb') as f:
        population = pickle.load(f)
    
    # 計算相似度矩陣
    print("計算相似度矩陣...")
    if len(population) >= 200:
        sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
    else:
        sim_matrix = SimilarityMatrix(population)
    
    similarity_matrix = sim_matrix.compute(show_progress=True)
    
    # 提取相似度值（排除對角線）
    mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
    similarities = similarity_matrix[mask]
    
    # 計算統計
    mean_sim = np.mean(similarities)
    median_sim = np.median(similarities)
    std_sim = np.std(similarities)
    
    # 繪製分佈圖
    fig, ax = plt.subplots(figsize=figsize)
    
    # 直方圖
    ax.hist(similarities, bins=50, density=True, alpha=0.6, color='#2E86AB', 
            edgecolor='black', label='直方圖')
    
    # KDE 曲線
    from scipy import stats
    kde = stats.gaussian_kde(similarities)
    x_range = np.linspace(similarities.min(), similarities.max(), 200)
    ax.plot(x_range, kde(x_range), linewidth=2, color='#A23B72', label='KDE')
    
    # 標註統計值
    ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_sim:.4f}')
    ax.axvline(median_sim, color='green', linestyle='--', linewidth=2, label=f'中位數: {median_sim:.4f}')
    
    # 標題和標籤
    if generation:
        title = f'相似度分佈 - Generation {generation}'
    else:
        title = '相似度分佈'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('相似度', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加統計文字框
    stats_text = f'Mean: {mean_sim:.4f}\nMedian: {median_sim:.4f}\nStd: {std_sim:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    plt.tight_layout()
    
    # 儲存或顯示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_population_tsne(
    population_file: Union[str, Path],
    generation: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    method: str = 'tsne',
    figsize: tuple = (12, 10),
    dpi: int = 300
):
    """
    使用 t-SNE 或 PCA 降維視覺化族群
    
    Args:
        population_file: generation_XXX.pkl 文件路徑
        generation: 世代編號
        save_path: 儲存路徑
        method: 降維方法 ('tsne' 或 'pca')
        figsize: 圖表大小
        dpi: 圖片解析度
    """
    # 載入族群
    with open(population_file, 'rb') as f:
        population = pickle.load(f)
    
    # 計算相似度矩陣
    print("計算相似度矩陣...")
    if len(population) >= 200:
        sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
    else:
        sim_matrix = SimilarityMatrix(population)
    
    similarity_matrix = sim_matrix.compute(show_progress=True)
    
    # 提取 fitness 值
    fitness_values = np.array([ind.fitness.values[0] for ind in population])
    
    # 降維
    print(f"使用 {method.upper()} 降維...")
    if method.lower() == 'tsne':
        # t-SNE 使用距離矩陣
        distance_matrix = 1.0 - similarity_matrix
        reducer = TSNE(n_components=2, metric='precomputed', random_state=42)
        embedding = reducer.fit_transform(distance_matrix)
    elif method.lower() == 'pca':
        # PCA 使用相似度矩陣
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(similarity_matrix)
    else:
        raise ValueError(f"不支援的降維方法: {method}")
    
    # 繪製散點圖
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用 fitness 作為顏色
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=fitness_values, cmap='viridis',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # 顏色條
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness', fontsize=12)
    
    # 標題
    if generation:
        title = f'{method.upper()} 降維視覺化 - Generation {generation}'
    else:
        title = f'{method.upper()} 降維視覺化'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} 維度 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 維度 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 儲存或顯示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()
