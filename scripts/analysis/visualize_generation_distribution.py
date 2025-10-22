"""
可視化 Generation 的個體分佈

使用 t-SNE 和 PCA 降維到 2D，並繪製散點圖
輸入：generation.pkl 文件路徑
輸出：兩張 PNG 圖片（t-SNE 和 PCA）
"""

import sys
from pathlib import Path

# 添加項目根目錄到 path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import dill
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from deap import base, creator, gp, tools

# 導入項目模塊（確保 pset 可以被載入）
from gp_quant.gp.operators import pset

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化 DEAP creator（載入 pkl 需要）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def tree_to_vector(individual, max_depth=10, max_nodes=100):
    """
    將 GP tree 轉換為固定長度的向量表示
    
    使用多種特徵：
    1. 樹的結構特徵（深度、節點數等）
    2. 各層節點類型的統計
    3. 操作符和終端符的頻率
    """
    features = []
    
    # 1. 基本結構特徵
    features.append(individual.height)  # 樹深度
    features.append(len(individual))    # 節點總數
    
    # 2. 統計各類型節點的數量
    # 操作符類型
    operators = ['add', 'sub', 'mul', 'div', 'neg', 'abs', 'max', 'min', 
                 'gt', 'lt', 'and_', 'or_', 'not_', 'if_then_else']
    for op in operators:
        count = sum(1 for node in individual if hasattr(node, 'name') and node.name == op)
        features.append(count)
    
    # 終端符類型
    terminals = ['open', 'high', 'low', 'close', 'volume', 
                 'returns', 'log_returns', 'volatility',
                 'sma', 'ema', 'rsi', 'macd', 'bbands']
    for term in terminals:
        count = sum(1 for node in individual if hasattr(node, 'name') and node.name == term)
        features.append(count)
    
    # 3. 常數節點統計
    constants = []
    for node in individual:
        if hasattr(node, 'value'):
            try:
                # 嘗試轉換為數值
                val = float(node.value)
                constants.append(val)
            except (ValueError, TypeError):
                # 如果不是數值，跳過
                pass
    
    if constants:
        features.append(np.mean(constants))
        features.append(np.std(constants))
        features.append(np.min(constants))
        features.append(np.max(constants))
        features.append(len(constants))
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # 4. 樹的形狀特徵（平衡度）
    # 計算左右子樹的大小差異
    if len(individual) > 1:
        try:
            # 簡單的平衡度指標：節點數 / 深度
            balance = len(individual) / max(individual.height, 1)
            features.append(balance)
        except:
            features.append(0)
    else:
        features.append(0)
    
    return np.array(features, dtype=np.float32)


def load_generation(pkl_path):
    """載入 generation.pkl 文件"""
    print(f"📂 載入文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    
    generation = data['generation']
    population = data['population']
    hall_of_fame = data.get('hall_of_fame', [])
    statistics = data.get('statistics', {})
    
    print(f"   ✓ Generation: {generation}")
    print(f"   ✓ Population size: {len(population)}")
    print(f"   ✓ Hall of Fame size: {len(hall_of_fame)}")
    
    if statistics:
        print(f"   ✓ Statistics: avg={statistics.get('avg', 'N/A'):.4f}, "
              f"max={statistics.get('max', 'N/A'):.4f}")
    
    return data


def extract_features(population):
    """提取所有個體的特徵向量"""
    print(f"\n🔍 提取特徵向量...")
    
    features = []
    fitnesses = []
    
    for i, ind in enumerate(population):
        if (i + 1) % 500 == 0:
            print(f"   處理中: {i + 1}/{len(population)}")
        
        # 提取特徵
        feature_vec = tree_to_vector(ind)
        features.append(feature_vec)
        
        # 提取 fitness
        if hasattr(ind, 'fitness') and ind.fitness.valid:
            fitnesses.append(ind.fitness.values[0])
        else:
            fitnesses.append(0.0)
    
    features = np.array(features)
    fitnesses = np.array(fitnesses)
    
    print(f"   ✓ 特徵矩陣形狀: {features.shape}")
    print(f"   ✓ Fitness 範圍: [{fitnesses.min():.4f}, {fitnesses.max():.4f}]")
    
    return features, fitnesses


def apply_tsne(features, random_state=42):
    """應用 t-SNE 降維"""
    print(f"\n🔬 應用 t-SNE 降維...")
    print(f"   參數: perplexity=30, max_iter=1000")
    
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) - 1),
        max_iter=1000,
        random_state=random_state,
        verbose=0
    )
    
    embeddings = tsne.fit_transform(features)
    
    print(f"   ✓ t-SNE 完成")
    print(f"   ✓ 嵌入形狀: {embeddings.shape}")
    
    return embeddings


def apply_pca(features):
    """應用 PCA 降維"""
    print(f"\n🔬 應用 PCA 降維...")
    
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(features)
    
    explained_var = pca.explained_variance_ratio_
    print(f"   ✓ PCA 完成")
    print(f"   ✓ 嵌入形狀: {embeddings.shape}")
    print(f"   ✓ 解釋變異: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, "
          f"Total={explained_var.sum():.2%}")
    
    return embeddings, explained_var


def plot_distribution(embeddings, fitnesses, method, output_path, 
                     generation, explained_var=None):
    """繪製分佈圖"""
    print(f"\n📊 繪製 {method} 分佈圖...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 根據 fitness 著色
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=fitnesses,
        cmap='RdYlGn',
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness (Sharpe Ratio)', fontsize=12, fontweight='bold')
    
    # 標記最佳個體
    best_idx = np.argmax(fitnesses)
    ax.scatter(
        embeddings[best_idx, 0],
        embeddings[best_idx, 1],
        c='red',
        s=200,
        marker='*',
        edgecolors='black',
        linewidth=2,
        label=f'Best (Fitness={fitnesses[best_idx]:.4f})',
        zorder=5
    )
    
    # 標記最差個體
    worst_idx = np.argmin(fitnesses)
    ax.scatter(
        embeddings[worst_idx, 0],
        embeddings[worst_idx, 1],
        c='blue',
        s=200,
        marker='v',
        edgecolors='black',
        linewidth=2,
        label=f'Worst (Fitness={fitnesses[worst_idx]:.4f})',
        zorder=5
    )
    
    # 設置標題和標籤
    if method == 't-SNE':
        title = f'{method} Visualization of Generation {generation}'
        ax.set_xlabel(f'{method} Dimension 1', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12, fontweight='bold')
    else:  # PCA
        title = f'{method} Visualization of Generation {generation}'
        var1, var2 = explained_var
        ax.set_xlabel(f'PC1 ({var1:.1%} variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.1%} variance)', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 添加統計信息
    stats_text = (
        f'Population: {len(fitnesses)}\n'
        f'Fitness: μ={np.mean(fitnesses):.4f}, σ={np.std(fitnesses):.4f}\n'
        f'Range: [{np.min(fitnesses):.4f}, {np.max(fitnesses):.4f}]'
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ 已儲存: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("使用方法: python visualize_generation_distribution.py <generation.pkl 路徑>")
        print("範例: python visualize_generation_distribution.py portfolio_experiment_results/.../generations/generation_001.pkl")
        sys.exit(1)
    
    pkl_path = Path(sys.argv[1])
    
    if not pkl_path.exists():
        print(f"❌ 文件不存在: {pkl_path}")
        sys.exit(1)
    
    print("\n" + "="*100)
    print("🎨 Generation 個體分佈可視化")
    print("="*100 + "\n")
    
    # 1. 載入數據
    data = load_generation(pkl_path)
    population = data['population']
    generation = data['generation']
    
    # 2. 提取特徵
    features, fitnesses = extract_features(population)
    
    # 3. 創建輸出目錄
    exp_dir = pkl_path.parent.parent  # 回到實驗目錄
    output_dir = exp_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 輸出目錄: {output_dir}")
    
    # 4. t-SNE 可視化
    tsne_embeddings = apply_tsne(features)
    tsne_output = output_dir / f"generation_{generation:03d}_tsne.png"
    plot_distribution(
        tsne_embeddings, 
        fitnesses, 
        't-SNE', 
        tsne_output, 
        generation
    )
    
    # 5. PCA 可視化
    pca_embeddings, explained_var = apply_pca(features)
    pca_output = output_dir / f"generation_{generation:03d}_pca.png"
    plot_distribution(
        pca_embeddings, 
        fitnesses, 
        'PCA', 
        pca_output, 
        generation,
        explained_var
    )
    
    # 6. 完成
    print("\n" + "="*100)
    print("✅ 可視化完成！")
    print("="*100)
    print(f"\n📊 生成的圖片:")
    print(f"  1. t-SNE: {tsne_output.name}")
    print(f"  2. PCA:   {pca_output.name}")
    print(f"\n📁 保存位置: {output_dir}")
    print()


if __name__ == "__main__":
    main()
