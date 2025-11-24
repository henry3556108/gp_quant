"""
Simple PnL Diversity Test with Mock Data

創建模擬的 PnL 曲線來展示相關性計算和視覺化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def generate_mock_pnl_curves(n_individuals: int = 5, n_days: int = 504) -> tuple:
    """
    生成模擬的 PnL 曲線
    
    Returns:
        (pnl_curves, fitness_values)
    """
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=n_days, freq='D')
    
    pnl_curves = []
    fitness_values = []
    
    # 生成不同特徵的 PnL 曲線
    for i in range(n_individuals):
        # 基礎趨勢
        trend = np.linspace(0, 50000 * (1 - i * 0.3), n_days)
        
        # 添加隨機波動
        noise = np.cumsum(np.random.randn(n_days) * 500)
        
        # 添加週期性成分
        seasonal = 5000 * np.sin(np.linspace(0, 4 * np.pi, n_days) + i)
        
        # 組合
        pnl = trend + noise + seasonal
        
        # 創建 Series
        pnl_series = pd.Series(pnl, index=dates)
        pnl_curves.append(pnl_series)
        
        # 計算 fitness (最終收益率)
        fitness = pnl[-1] / 100000  # 假設初始資金 100000
        fitness_values.append(fitness)
    
    return pnl_curves, fitness_values


def calculate_correlation_matrix(pnl_curves: list) -> np.ndarray:
    """計算 PnL 曲線的相關性矩陣"""
    n = len(pnl_curves)
    corr_matrix = np.zeros((n, n))
    
    print(f"\n📈 計算相關性矩陣 ({n} x {n})...")
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i, j] = 1.0
            elif i < j:
                # 計算 Pearson 相關係數
                corr = np.corrcoef(pnl_curves[i].values, pnl_curves[j].values)[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                print(f"   Corr(Ind {i+1}, Ind {j+1}) = {corr:.4f}")
    
    return corr_matrix


def visualize_pnl_curves(pnl_curves: list, fitness_values: list, output_path: Path):
    """視覺化 PnL 曲線"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 子圖 1: 累積 PnL
    ax1 = axes[0]
    for i, pnl in enumerate(pnl_curves):
        fitness = fitness_values[i]
        ax1.plot(pnl.index, pnl.values, label=f'Individual {i+1} (Fitness: {fitness:.4f})', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative PnL ($)', fontsize=12)
    ax1.set_title('PnL Curves of Selected Individuals', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 子圖 2: 標準化 PnL (方便比較形狀)
    ax2 = axes[1]
    for i, pnl in enumerate(pnl_curves):
        # 標準化: (x - mean) / std
        normalized = (pnl - pnl.mean()) / pnl.std()
        fitness = fitness_values[i]
        ax2.plot(normalized.index, normalized.values, label=f'Individual {i+1} (Fitness: {fitness:.4f})', linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Normalized PnL (z-score)', fontsize=12)
    ax2.set_title('Normalized PnL Curves (for Shape Comparison)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ PnL curves saved: {output_path}")
    plt.close()


def visualize_correlation_matrix(corr_matrix: np.ndarray, fitness_values: list, output_path: Path):
    """視覺化相關性矩陣"""
    n = len(fitness_values)
    
    # 創建標籤
    labels = [f'Ind {i+1}\n(F:{fitness_values[i]:.3f})' for i in range(n)]
    
    # 繪製熱圖
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用 seaborn 繪製熱圖
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation Coefficient'},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title('PnL Correlation Matrix Between Individuals', fontsize=14, fontweight='bold', pad=20)
    
    # 添加統計信息
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper_tri]
    
    stats_text = f'Mean: {np.mean(correlations):.3f} | Std: {np.std(correlations):.3f} | Min: {np.min(correlations):.3f} | Max: {np.max(correlations):.3f}'
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Correlation matrix saved: {output_path}")
    plt.close()


def main():
    output_dir = Path("test_evolution_records_20251125_0000")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🧪 PnL Diversity Test (with Mock Data)")
    print("=" * 80)
    print("Generating mock PnL curves to demonstrate correlation analysis...\n")
    
    # 1. 生成模擬數據
    print("📊 Generating mock PnL curves...")
    pnl_curves, fitness_values = generate_mock_pnl_curves(n_individuals=5, n_days=504)
    
    print(f"\n✅ Generated {len(pnl_curves)} PnL curves")
    for i, (pnl, fitness) in enumerate(zip(pnl_curves, fitness_values)):
        print(f"   Individual {i+1}: Fitness={fitness:.4f}, Final PnL=${pnl.iloc[-1]:.2f}, Std=${pnl.std():.2f}")
    
    # 2. 計算相關性矩陣
    corr_matrix = calculate_correlation_matrix(pnl_curves)
    
    # 3. 視覺化
    print("\n🎨 Generating visualizations...")
    visualize_pnl_curves(pnl_curves, fitness_values, output_dir / "pnl_curves_comparison_mock.png")
    visualize_correlation_matrix(corr_matrix, fitness_values, output_dir / "pnl_correlation_matrix_mock.png")
    
    # 4. 輸出統計摘要
    print("\n" + "=" * 80)
    print("📊 PnL Correlation Statistics")
    print("=" * 80)
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper_tri]
    
    print(f"Number of individuals: {len(pnl_curves)}")
    print(f"Number of correlation pairs: {len(correlations)}")
    print(f"Mean correlation: {np.mean(correlations):.4f}")
    print(f"Std correlation: {np.std(correlations):.4f}")
    print(f"Min correlation: {np.min(correlations):.4f}")
    print(f"Max correlation: {np.max(correlations):.4f}")
    print(f"Median correlation: {np.median(correlations):.4f}")
    print("=" * 80)
    print("\n✅ Test completed!")
    print(f"\n📁 Output files:")
    print(f"   - {output_dir / 'pnl_curves_comparison_mock.png'}")
    print(f"   - {output_dir / 'pnl_correlation_matrix_mock.png'}")


if __name__ == "__main__":
    main()
