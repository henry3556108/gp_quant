#!/usr/bin/env python3
"""
視覺化多樣性指標

這個腳本讀取多樣性指標 JSON 文件，並生成視覺化圖表：
1. 多樣性分數隨世代變化
2. 平均相似度隨世代變化
3. 多樣性與適應度的關係（如果有演化日誌）
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 設置繪圖風格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_diversity_metrics(metrics_file: Path) -> dict:
    """載入多樣性指標"""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def load_evolution_log(exp_dir: Path) -> pd.DataFrame:
    """載入演化日誌"""
    log_file = exp_dir / 'evolution_log.csv'
    if log_file.exists():
        return pd.read_csv(log_file)
    return None


def plot_diversity_trends(metrics: list, output_file: Path):
    """繪製多樣性趨勢圖"""
    generations = [m['generation'] for m in metrics]
    diversity_scores = [m['diversity_score'] for m in metrics]
    avg_similarities = [m['avg_similarity'] for m in metrics]
    std_similarities = [m['std_similarity'] for m in metrics]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子圖 1: 多樣性分數
    ax1 = axes[0]
    ax1.plot(generations, diversity_scores, 'b-o', linewidth=2, markersize=4, label='Diversity Score')
    ax1.fill_between(generations, 
                      [d - s for d, s in zip(diversity_scores, std_similarities)],
                      [d + s for d, s in zip(diversity_scores, std_similarities)],
                      alpha=0.2, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Diversity Score', fontsize=12)
    ax1.set_title('Diversity Score Over Generations', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子圖 2: 平均相似度
    ax2 = axes[1]
    ax2.plot(generations, avg_similarities, 'r-o', linewidth=2, markersize=4, label='Avg Similarity')
    ax2.fill_between(generations,
                      [a - s for a, s in zip(avg_similarities, std_similarities)],
                      [a + s for a, s in zip(avg_similarities, std_similarities)],
                      alpha=0.2, color='red', label='±1 Std Dev')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Average Similarity', fontsize=12)
    ax2.set_title('Average Similarity Over Generations', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 多樣性趨勢圖已儲存: {output_file}")
    plt.close()


def plot_diversity_fitness_correlation(metrics: list, evolution_log: pd.DataFrame, output_file: Path):
    """繪製多樣性與適應度的關係"""
    if evolution_log is None:
        print("⚠ 無演化日誌，跳過多樣性-適應度關聯圖")
        return
    
    # 提取數據
    generations = [m['generation'] for m in metrics]
    diversity_scores = [m['diversity_score'] for m in metrics]
    
    # 從演化日誌提取適應度數據
    max_fitness = []
    avg_fitness = []
    
    for gen in generations:
        gen_data = evolution_log[evolution_log['generation'] == gen]
        if not gen_data.empty:
            max_fitness.append(gen_data['max_fitness'].values[0])
            avg_fitness.append(gen_data['avg_fitness'].values[0])
        else:
            max_fitness.append(np.nan)
            avg_fitness.append(np.nan)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子圖 1: 多樣性與最大適應度隨時間變化
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(generations, diversity_scores, 'b-o', linewidth=2, markersize=4, label='Diversity')
    line2 = ax1_twin.plot(generations, max_fitness, 'r-s', linewidth=2, markersize=4, label='Max Fitness')
    
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Diversity Score', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Max Fitness', fontsize=12, color='red')
    ax1.set_title('Diversity vs Max Fitness Over Time', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 子圖 2: 多樣性與平均適應度隨時間變化
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    line1 = ax2.plot(generations, diversity_scores, 'b-o', linewidth=2, markersize=4, label='Diversity')
    line2 = ax2_twin.plot(generations, avg_fitness, 'g-^', linewidth=2, markersize=4, label='Avg Fitness')
    
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Diversity Score', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Average Fitness', fontsize=12, color='green')
    ax2.set_title('Diversity vs Avg Fitness Over Time', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 子圖 3: 多樣性 vs 最大適應度散點圖
    ax3 = axes[1, 0]
    scatter = ax3.scatter(diversity_scores, max_fitness, c=generations, cmap='viridis', s=100, alpha=0.6)
    
    # 添加趨勢線
    z = np.polyfit(diversity_scores, max_fitness, 1)
    p = np.poly1d(z)
    ax3.plot(diversity_scores, p(diversity_scores), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax3.set_xlabel('Diversity Score', fontsize=12)
    ax3.set_ylabel('Max Fitness', fontsize=12)
    ax3.set_title('Diversity vs Max Fitness Correlation', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Generation', fontsize=10)
    
    # 計算相關係數
    corr = np.corrcoef(diversity_scores, max_fitness)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax3.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 子圖 4: 多樣性 vs 平均適應度散點圖
    ax4 = axes[1, 1]
    scatter = ax4.scatter(diversity_scores, avg_fitness, c=generations, cmap='viridis', s=100, alpha=0.6)
    
    # 添加趨勢線
    z = np.polyfit(diversity_scores, avg_fitness, 1)
    p = np.poly1d(z)
    ax4.plot(diversity_scores, p(diversity_scores), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax4.set_xlabel('Diversity Score', fontsize=12)
    ax4.set_ylabel('Average Fitness', fontsize=12)
    ax4.set_title('Diversity vs Avg Fitness Correlation', fontsize=14, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Generation', fontsize=10)
    
    # 計算相關係數
    corr = np.corrcoef(diversity_scores, avg_fitness)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax4.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 多樣性-適應度關聯圖已儲存: {output_file}")
    plt.close()


def plot_similarity_distribution(metrics: list, output_file: Path):
    """繪製相似度分佈統計"""
    generations = [m['generation'] for m in metrics]
    
    # 選擇幾個代表性世代
    sample_gens = [1, len(metrics)//4, len(metrics)//2, 3*len(metrics)//4, len(metrics)]
    sample_metrics = [metrics[g-1] for g in sample_gens]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (gen, m) in enumerate(zip(sample_gens, sample_metrics)):
        ax = axes[idx]
        
        # 繪製相似度統計
        stats = [m['min_similarity'], m['avg_similarity'], m['max_similarity']]
        labels = ['Min', 'Avg', 'Max']
        colors = ['green', 'blue', 'red']
        
        bars = ax.bar(labels, stats, color=colors, alpha=0.6, edgecolor='black')
        
        # 添加標準差誤差線
        ax.errorbar(['Avg'], [m['avg_similarity']], 
                   yerr=[m['std_similarity']], 
                   fmt='none', color='black', capsize=5, linewidth=2)
        
        ax.set_ylabel('Similarity', fontsize=11)
        ax.set_title(f'Generation {gen}', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加數值標籤
        for bar, stat in zip(bars, stats):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{stat:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    # 移除多餘的子圖
    if len(sample_gens) < len(axes):
        for idx in range(len(sample_gens), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.suptitle('Similarity Distribution Across Generations', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 相似度分佈圖已儲存: {output_file}")
    plt.close()


def generate_summary_report(data: dict, evolution_log: pd.DataFrame, output_file: Path):
    """生成文字摘要報告"""
    metrics = data['metrics']
    first = metrics[0]
    last = metrics[-1]
    
    # 找出極值
    max_div = max(metrics, key=lambda x: x['diversity_score'])
    min_div = min(metrics, key=lambda x: x['diversity_score'])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("多樣性分析報告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"實驗: {data['experiment']}\n")
        f.write(f"實驗路徑: {data['experiment_path']}\n")
        f.write(f"計算日期: {data['computation_date']}\n")
        f.write(f"總世代數: {data['total_generations']}\n")
        f.write(f"族群大小: {data['population_size']}\n")
        f.write(f"並行工作數: {data['n_workers']}\n")
        f.write(f"總計算時間: {data['total_computation_time']:.1f} 秒 ({data['total_computation_time']/60:.1f} 分鐘)\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("多樣性趨勢摘要\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"第 {first['generation']} 代:\n")
        f.write(f"  平均相似度: {first['avg_similarity']:.4f} ± {first['std_similarity']:.4f}\n")
        f.write(f"  多樣性分數: {first['diversity_score']:.4f}\n")
        f.write(f"  相似度範圍: [{first['min_similarity']:.4f}, {first['max_similarity']:.4f}]\n")
        f.write("\n")
        
        f.write(f"第 {last['generation']} 代:\n")
        f.write(f"  平均相似度: {last['avg_similarity']:.4f} ± {last['std_similarity']:.4f}\n")
        f.write(f"  多樣性分數: {last['diversity_score']:.4f}\n")
        f.write(f"  相似度範圍: [{last['min_similarity']:.4f}, {last['max_similarity']:.4f}]\n")
        f.write("\n")
        
        div_change = last['diversity_score'] - first['diversity_score']
        div_pct = (div_change / first['diversity_score']) * 100
        f.write(f"多樣性變化: {div_change:+.4f} ({div_pct:+.1f}%)\n")
        f.write("\n")
        
        f.write(f"最高多樣性: 第 {max_div['generation']} 代 (分數: {max_div['diversity_score']:.4f})\n")
        f.write(f"最低多樣性: 第 {min_div['generation']} 代 (分數: {min_div['diversity_score']:.4f})\n")
        f.write("\n")
        
        if evolution_log is not None:
            f.write("-" * 80 + "\n")
            f.write("多樣性與適應度關聯\n")
            f.write("-" * 80 + "\n\n")
            
            diversity_scores = [m['diversity_score'] for m in metrics]
            generations = [m['generation'] for m in metrics]
            
            max_fitness = []
            avg_fitness = []
            
            for gen in generations:
                gen_data = evolution_log[evolution_log['generation'] == gen]
                if not gen_data.empty:
                    max_fitness.append(gen_data['max_fitness'].values[0])
                    avg_fitness.append(gen_data['avg_fitness'].values[0])
            
            if max_fitness and avg_fitness:
                corr_max = np.corrcoef(diversity_scores, max_fitness)[0, 1]
                corr_avg = np.corrcoef(diversity_scores, avg_fitness)[0, 1]
                
                f.write(f"多樣性 vs 最大適應度相關係數: {corr_max:.4f}\n")
                f.write(f"多樣性 vs 平均適應度相關係數: {corr_avg:.4f}\n")
                f.write("\n")
                
                if abs(corr_max) > 0.7:
                    f.write(f"⚠ 多樣性與最大適應度呈現{'強正相關' if corr_max > 0 else '強負相關'}\n")
                elif abs(corr_max) > 0.4:
                    f.write(f"• 多樣性與最大適應度呈現{'中度正相關' if corr_max > 0 else '中度負相關'}\n")
                else:
                    f.write(f"• 多樣性與最大適應度相關性較弱\n")
                
                f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("觀察與建議\n")
        f.write("-" * 80 + "\n\n")
        
        if div_change < -0.05:
            f.write("⚠ 多樣性顯著下降，可能導致早熟收斂\n")
            f.write("  建議:\n")
            f.write("  - 增加突變率\n")
            f.write("  - 使用小生境技術\n")
            f.write("  - 考慮多樣性維持機制\n")
        elif div_change > 0.05:
            f.write("✓ 多樣性有所提升，探索能力良好\n")
        else:
            f.write("• 多樣性基本穩定\n")
        
        f.write("\n")
    
    print(f"✓ 摘要報告已儲存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='視覺化多樣性指標')
    parser.add_argument('--metrics_file', type=str, required=True,
                       help='多樣性指標 JSON 文件路徑')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='輸出目錄（默認與指標文件同目錄）')
    
    args = parser.parse_args()
    
    # 載入數據
    metrics_file = Path(args.metrics_file)
    if not metrics_file.exists():
        print(f"❌ 錯誤: 找不到指標文件 {metrics_file}")
        return 1
    
    print(f"📊 載入多樣性指標: {metrics_file}")
    data = load_diversity_metrics(metrics_file)
    
    # 確定輸出目錄
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = metrics_file.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入演化日誌
    exp_dir = Path(data['experiment_path'])
    evolution_log = load_evolution_log(exp_dir)
    
    if evolution_log is not None:
        print(f"✓ 載入演化日誌: {exp_dir / 'evolution_log.csv'}")
    else:
        print(f"⚠ 未找到演化日誌，將跳過適應度關聯分析")
    
    print()
    print("=" * 60)
    print("生成視覺化圖表")
    print("=" * 60)
    print()
    
    # 生成圖表
    plot_diversity_trends(
        data['metrics'],
        output_dir / 'diversity_trends.png'
    )
    
    plot_similarity_distribution(
        data['metrics'],
        output_dir / 'similarity_distribution.png'
    )
    
    if evolution_log is not None:
        plot_diversity_fitness_correlation(
            data['metrics'],
            evolution_log,
            output_dir / 'diversity_fitness_correlation.png'
        )
    
    # 生成摘要報告
    generate_summary_report(
        data,
        evolution_log,
        output_dir / 'diversity_analysis_report.txt'
    )
    
    print()
    print("=" * 60)
    print("✓ 所有視覺化圖表已生成")
    print("=" * 60)
    print(f"輸出目錄: {output_dir}")
    print()
    
    return 0


if __name__ == '__main__':
    exit(main())
