"""
分析每個 Niche 的 Silhouette Score

讀取實驗結果，可視化每個 generation 中每個 niche 的 silhouette score
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_results(exp_dir):
    """載入實驗結果"""
    exp_path = Path(exp_dir)
    
    # 讀取 result.json
    with open(exp_path / "result.json", 'r') as f:
        result = json.load(f)
    
    return result

def extract_per_niche_data(niching_log):
    """提取每個 niche 的數據"""
    data = []
    
    for log in niching_log:
        gen = log['generation']
        selected_k = log['selected_k']
        overall_silhouette = log['silhouette_score']
        
        if 'per_niche_silhouette' in log:
            for niche_id, stats in log['per_niche_silhouette'].items():
                data.append({
                    'generation': gen,
                    'selected_k': selected_k,
                    'overall_silhouette': overall_silhouette,
                    'niche_id': int(niche_id),
                    'niche_silhouette_mean': stats['mean'],
                    'niche_silhouette_std': stats['std'],
                    'niche_silhouette_min': stats['min'],
                    'niche_silhouette_max': stats['max'],
                    'niche_size': stats['size']
                })
    
    return pd.DataFrame(data)

def plot_per_niche_silhouette(df, exp_name, output_dir):
    """繪製每個 niche 的 silhouette score"""
    
    # 1. 時間序列圖：每個 niche 的 silhouette score 隨時間變化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 每個 niche 的 silhouette score (折線圖)
    ax = axes[0, 0]
    for niche_id in sorted(df['niche_id'].unique()):
        niche_data = df[df['niche_id'] == niche_id]
        ax.plot(niche_data['generation'], niche_data['niche_silhouette_mean'], 
                marker='o', label=f'Niche {niche_id}', alpha=0.7)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Silhouette Score (Mean)', fontsize=12)
    ax.set_title('Per-Niche Silhouette Score over Generations', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 1.2 整體 vs 各 niche 的 silhouette score
    ax = axes[0, 1]
    ax.plot(df.groupby('generation')['overall_silhouette'].first(), 
            marker='s', linewidth=2, label='Overall', color='black')
    
    for niche_id in sorted(df['niche_id'].unique()):
        niche_data = df[df['niche_id'] == niche_id]
        ax.plot(niche_data['generation'], niche_data['niche_silhouette_mean'], 
                marker='o', alpha=0.5, label=f'Niche {niche_id}')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Overall vs Per-Niche Silhouette Score', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 1.3 Niche size 隨時間變化
    ax = axes[1, 0]
    for niche_id in sorted(df['niche_id'].unique()):
        niche_data = df[df['niche_id'] == niche_id]
        ax.plot(niche_data['generation'], niche_data['niche_size'], 
                marker='o', label=f'Niche {niche_id}', alpha=0.7)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Niche Size', fontsize=12)
    ax.set_title('Niche Size over Generations', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 1.4 K 值變化
    ax = axes[1, 1]
    k_values = df.groupby('generation')['selected_k'].first()
    ax.plot(k_values.index, k_values.values, marker='s', linewidth=2, color='red')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Selected K', fontsize=12)
    ax.set_title('Selected K over Generations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_per_niche_silhouette_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {exp_name}_per_niche_silhouette_overview.png")
    plt.close()
    
    # 2. 熱力圖：每個 generation 的 niche silhouette score
    pivot_data = df.pivot_table(
        values='niche_silhouette_mean',
        index='niche_id',
        columns='generation',
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Silhouette Score'})
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Niche ID', fontsize=12)
    ax.set_title('Per-Niche Silhouette Score Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_per_niche_silhouette_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {exp_name}_per_niche_silhouette_heatmap.png")
    plt.close()
    
    # 3. 箱線圖：每個 generation 的 niche silhouette score 分布
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 準備數據
    box_data = []
    positions = []
    for gen in sorted(df['generation'].unique()):
        gen_data = df[df['generation'] == gen]['niche_silhouette_mean'].values
        box_data.append(gen_data)
        positions.append(gen)
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    # 疊加整體 silhouette score
    overall_scores = df.groupby('generation')['overall_silhouette'].first()
    ax.plot(overall_scores.index, overall_scores.values, 
            marker='s', linewidth=2, color='black', label='Overall Silhouette', markersize=8)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Distribution of Per-Niche Silhouette Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_per_niche_silhouette_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已儲存: {exp_name}_per_niche_silhouette_distribution.png")
    plt.close()

def generate_summary_statistics(df, output_dir, exp_name):
    """生成統計摘要"""
    
    summary = []
    
    for gen in sorted(df['generation'].unique()):
        gen_data = df[df['generation'] == gen]
        
        summary.append({
            'generation': gen,
            'selected_k': gen_data['selected_k'].iloc[0],
            'overall_silhouette': gen_data['overall_silhouette'].iloc[0],
            'mean_niche_silhouette': gen_data['niche_silhouette_mean'].mean(),
            'std_niche_silhouette': gen_data['niche_silhouette_mean'].std(),
            'min_niche_silhouette': gen_data['niche_silhouette_mean'].min(),
            'max_niche_silhouette': gen_data['niche_silhouette_mean'].max(),
            'range_niche_silhouette': gen_data['niche_silhouette_mean'].max() - gen_data['niche_silhouette_mean'].min(),
            'total_population': gen_data['niche_size'].sum(),
            'min_niche_size': gen_data['niche_size'].min(),
            'max_niche_size': gen_data['niche_size'].max(),
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / f'{exp_name}_per_niche_summary.csv', index=False)
    print(f"✓ 已儲存: {exp_name}_per_niche_summary.csv")
    
    return summary_df

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_per_niche_silhouette.py <實驗目錄>")
        print("範例: python analyze_per_niche_silhouette.py k_comparison_experiments/exp_3_dynamic_calibration")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    exp_name = Path(exp_dir).name
    
    print(f"\n{'='*100}")
    print(f"📊 分析實驗: {exp_name}")
    print(f"{'='*100}\n")
    
    # 載入結果
    print("1️⃣  載入實驗結果...")
    result = load_experiment_results(exp_dir)
    print(f"   ✓ 載入完成")
    print(f"   實驗名稱: {result['experiment_name']}")
    print(f"   總世代數: {len(result['evolution_log'])}")
    print(f"   Niching 記錄: {len(result['niching_log'])} 代")
    print()
    
    # 提取數據
    print("2️⃣  提取每個 Niche 的數據...")
    df = extract_per_niche_data(result['niching_log'])
    print(f"   ✓ 提取完成")
    print(f"   總記錄數: {len(df)}")
    print(f"   Generation 範圍: {df['generation'].min()} - {df['generation'].max()}")
    print(f"   Niche 數量範圍: {df['selected_k'].min()} - {df['selected_k'].max()}")
    print()
    
    # 創建輸出目錄
    output_dir = Path(exp_dir) / "per_niche_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # 生成可視化
    print("3️⃣  生成可視化...")
    plot_per_niche_silhouette(df, exp_name, output_dir)
    print()
    
    # 生成統計摘要
    print("4️⃣  生成統計摘要...")
    summary_df = generate_summary_statistics(df, output_dir, exp_name)
    print()
    
    # 顯示摘要
    print("📊 統計摘要:")
    print(summary_df.to_string(index=False))
    print()
    
    print(f"{'='*100}")
    print(f"✅ 分析完成！")
    print(f"📁 結果保存在: {output_dir}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()
