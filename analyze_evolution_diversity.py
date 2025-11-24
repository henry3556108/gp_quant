#!/usr/bin/env python3
"""
分析演化記錄的多樣性
適配新的記錄格式（populations/ 目錄下的 generation_XXX.pkl 文件）
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from deap import creator, base, gp
from gp_quant.diversity.metrics import DiversityMetrics
from gp_quant.diversity.visualizer import DiversityVisualizer
from gp_quant.similarity.tree_edit_distance import compute_ted


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_populations_from_records(records_dir: Path, verbose: bool = True) -> Dict[int, List[Any]]:
    """
    從記錄目錄載入所有世代的族群
    
    Args:
        records_dir: 記錄目錄路徑
        verbose: 是否顯示詳細信息
        
    Returns:
        字典，鍵為世代號，值為族群列表
    """
    populations = {}
    populations_dir = records_dir / 'populations'
    
    if not populations_dir.exists():
        raise ValueError(f"Populations directory not found: {populations_dir}")
    
    # 找到所有 generation_XXX.pkl 文件
    gen_files = sorted(populations_dir.glob('generation_*.pkl'))
    
    if verbose:
        print(f"📂 找到 {len(gen_files)} 個世代文件")
    
    for gen_file in gen_files:
        # 提取世代號
        gen_num = int(gen_file.stem.split('_')[1])
        
        try:
            with open(gen_file, 'rb') as f:
                population = pickle.load(f)
            
            populations[gen_num] = population
            
            if verbose and (gen_num % 5 == 0 or gen_num < 3):
                print(f"   ✅ 載入世代 {gen_num}: {len(population)} 個個體")
                
        except Exception as e:
            print(f"   ⚠️  載入世代 {gen_num} 失敗: {e}")
    
    if verbose:
        print(f"✅ 成功載入 {len(populations)} 個世代\n")
    
    return populations


def calculate_ted_based_genotypic_diversity(population: List[Any], sample_size: int = 50) -> Dict[str, float]:
    """
    基於 TED 計算基因型多樣性
    
    Args:
        population: 族群列表
        sample_size: 採樣大小（用於大族群以節省時間）
        
    Returns:
        包含 TED 多樣性指標的字典
    """
    n = len(population)
    
    # 如果族群太大，進行採樣
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
        sample_pop = [population[i] for i in indices]
    else:
        sample_pop = population
        sample_size = n
    
    # 計算 TED 距離矩陣
    distance_matrix = np.zeros((sample_size, sample_size))
    
    for i in range(sample_size):
        for j in range(i + 1, sample_size):
            try:
                ted_dist = compute_ted(sample_pop[i], sample_pop[j])
                
                # 標準化：除以較大樹的大小
                max_size = max(len(sample_pop[i]), len(sample_pop[j]))
                norm_ted = ted_dist / max_size if max_size > 0 else 0.0
                
                distance_matrix[i][j] = norm_ted
                distance_matrix[j][i] = norm_ted
            except Exception as e:
                # 如果計算失敗，使用最大距離
                distance_matrix[i][j] = 1.0
                distance_matrix[j][i] = 1.0
    
    # 提取上三角（不包括對角線）
    upper_tri_indices = np.triu_indices(sample_size, k=1)
    distances = distance_matrix[upper_tri_indices]
    
    # 計算唯一個體數量（基於閾值）
    threshold = 0.05  # 標準化距離 < 0.05 視為相同
    unique_count = 0
    assigned = [False] * sample_size
    
    for i in range(sample_size):
        if not assigned[i]:
            unique_count += 1
            assigned[i] = True
            # 將所有與 i 相似的個體標記為已分配
            for j in range(i + 1, sample_size):
                if not assigned[j] and distance_matrix[i][j] < threshold:
                    assigned[j] = True
    
    unique_ratio = unique_count / sample_size
    
    return {
        'ted_mean_distance': float(np.mean(distances)),
        'ted_std_distance': float(np.std(distances)),
        'ted_median_distance': float(np.median(distances)),
        'ted_min_distance': float(np.min(distances)),
        'ted_max_distance': float(np.max(distances)),
        'ted_unique_count': unique_count,
        'ted_unique_ratio': unique_ratio,
        'ted_diversity_score': float(np.mean(distances))  # 平均距離作為多樣性分數
    }


def calculate_diversity_metrics(populations: Dict[int, List[Any]], verbose: bool = True, use_ted: bool = True, ted_sample_size: int = 50) -> pd.DataFrame:
    """
    計算所有世代的多樣性指標
    
    Args:
        populations: 世代族群字典
        verbose: 是否顯示詳細信息
        
    Returns:
        包含多樣性指標的 DataFrame
    """
    diversity_data = []
    
    if verbose:
        print("📊 計算多樣性指標...")
        if use_ted:
            print(f"   使用 TED 計算基因型多樣性（採樣大小: {ted_sample_size}）")
    
    for gen_num in sorted(populations.keys()):
        population = populations[gen_num]
        
        # 計算各類指標
        metrics = {}
        metrics['generation'] = gen_num
        
        # 結構多樣性
        structural = DiversityMetrics.structural_diversity(population)
        metrics.update({f'structural_{k}': v for k, v in structural.items()})
        
        # 基因型多樣性 - 使用 TED
        if use_ted:
            try:
                ted_metrics = calculate_ted_based_genotypic_diversity(population, sample_size=ted_sample_size)
                metrics.update(ted_metrics)
                
                # 也保留原始的字符串比較結果作為參考
                genotypic_str = DiversityMetrics.genotypic_diversity(population)
                metrics['genotypic_string_unique_ratio'] = genotypic_str['unique_ratio']
                metrics['genotypic_string_unique_count'] = genotypic_str['unique_count']
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  世代 {gen_num} TED 計算失敗: {e}")
                # 回退到字符串比較
                genotypic = DiversityMetrics.genotypic_diversity(population)
                metrics.update({f'genotypic_{k}': v for k, v in genotypic.items()})
        else:
            # 使用原始的字符串比較
            genotypic = DiversityMetrics.genotypic_diversity(population)
            metrics.update({f'genotypic_{k}': v for k, v in genotypic.items()})
        
        # 適應度多樣性
        fitness = DiversityMetrics.fitness_diversity(population)
        metrics.update({f'fitness_{k}': v for k, v in fitness.items()})
        
        diversity_data.append(metrics)
        
        if verbose and (gen_num % 5 == 0 or gen_num < 3):
            if use_ted and 'ted_unique_ratio' in metrics:
                print(f"   世代 {gen_num}: "
                      f"TED唯一比例={metrics['ted_unique_ratio']:.3f}, "
                      f"TED多樣性={metrics['ted_diversity_score']:.4f}, "
                      f"適應度標準差={fitness['fitness_std']:.4f}")
            else:
                genotypic_key = 'genotypic_unique_ratio' if 'genotypic_unique_ratio' in metrics else 'ted_unique_ratio'
                if genotypic_key in metrics:
                    print(f"   世代 {gen_num}: "
                          f"唯一基因型={metrics[genotypic_key]:.3f}, "
                          f"適應度標準差={fitness['fitness_std']:.4f}")
    
    df = pd.DataFrame(diversity_data)
    
    if verbose:
        print(f"✅ 計算完成：{len(df)} 個世代\n")
    
    return df


def print_summary_statistics(diversity_df: pd.DataFrame):
    """列印摘要統計"""
    print("=" * 80)
    print("📈 多樣性分析摘要")
    print("=" * 80)
    
    print(f"\n總世代數: {len(diversity_df)}")
    print(f"世代範圍: {diversity_df['generation'].min()} - {diversity_df['generation'].max()}")
    
    # 關鍵指標
    key_metrics = {
        'genotypic_unique_ratio': '基因型唯一比例',
        'genotypic_unique_count': '唯一基因型數量',
        'fitness_std': '適應度標準差',
        'fitness_range': '適應度範圍',
        'structural_height_std': '樹高度標準差',
        'structural_size_std': '樹大小標準差'
    }
    
    print("\n📊 關鍵指標趨勢:")
    print("-" * 80)
    
    for metric, name in key_metrics.items():
        if metric in diversity_df.columns:
            initial = diversity_df[metric].iloc[0]
            final = diversity_df[metric].iloc[-1]
            change = final - initial
            change_pct = (change / initial * 100) if initial != 0 else 0
            
            trend = "📈 增加" if change > 0 else "📉 減少" if change < 0 else "➡️  持平"
            
            print(f"\n{name} ({metric}):")
            print(f"   初始值: {initial:.4f}")
            print(f"   最終值: {final:.4f}")
            print(f"   變化: {change:+.4f} ({change_pct:+.1f}%)")
            print(f"   趨勢: {trend}")
    
    print("\n" + "=" * 80)


def create_visualizations(diversity_df: pd.DataFrame, output_dir: Path):
    """創建視覺化圖表"""
    print("\n🎨 生成視覺化圖表...")
    
    # 設置樣式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. 綜合趨勢圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Population Diversity Analysis', fontsize=16, fontweight='bold')
    
    # 根據可用的指標選擇要繪製的內容
    if 'ted_unique_ratio' in diversity_df.columns:
        # 使用 TED 指標
        metrics_to_plot = [
            ('ted_unique_ratio', 'TED-based Unique Ratio', axes[0, 0]),
            ('ted_unique_count', 'TED-based Unique Count', axes[0, 1]),
            ('ted_diversity_score', 'TED Diversity Score', axes[0, 2]),
            ('fitness_fitness_std', 'Fitness Std Dev', axes[1, 0]),
            ('structural_height_std', 'Tree Height Std Dev', axes[1, 1]),
            ('structural_size_std', 'Tree Size Std Dev', axes[1, 2])
        ]
    else:
        # 使用字符串比較指標
        metrics_to_plot = [
            ('genotypic_unique_ratio', 'Genotypic Unique Ratio', axes[0, 0]),
            ('genotypic_unique_count', 'Unique Genotype Count', axes[0, 1]),
            ('fitness_fitness_std', 'Fitness Std Dev', axes[0, 2]),
            ('fitness_fitness_range', 'Fitness Range', axes[1, 0]),
            ('structural_height_std', 'Tree Height Std Dev', axes[1, 1]),
            ('structural_size_std', 'Tree Size Std Dev', axes[1, 2])
        ]
    
    for metric, title, ax in metrics_to_plot:
        if metric in diversity_df.columns:
            # 檢查數據是否有效
            data = diversity_df[metric].dropna()
            if len(data) > 0 and data.std() > 1e-10:  # 有數據且有變化
                ax.plot(diversity_df['generation'], diversity_df[metric], 
                       marker='o', linewidth=2, markersize=4, color='#2E86AB')
                ax.set_xlabel('Generation', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                # 數據無效或無變化，顯示提示
                ax.text(0.5, 0.5, f'No variation in\n{title}', 
                       ha='center', va='center', fontsize=10, color='gray',
                       transform=ax.transAxes)
                ax.set_xlabel('Generation', fontsize=10)
                ax.set_ylabel(title, fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            # 指標不存在
            ax.text(0.5, 0.5, f'{title}\nNot Available', 
                   ha='center', va='center', fontsize=10, color='red',
                   transform=ax.transAxes)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_dir / 'diversity_trends.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存趨勢圖: {output_file}")
    plt.close()
    
    # 2. 適應度 vs 多樣性相關性
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Fitness vs Diversity Correlation', fontsize=14, fontweight='bold')
    
    # 檢查所需的列是否存在
    has_fitness_mean = 'fitness_fitness_mean' in diversity_df.columns
    has_fitness_std = 'fitness_fitness_std' in diversity_df.columns
    
    # 優先使用 TED 指標，否則使用字符串比較指標
    if 'ted_unique_ratio' in diversity_df.columns:
        genotypic_col = 'ted_unique_ratio'
        diversity_col = 'ted_diversity_score'
        has_genotypic = True
    elif 'genotypic_unique_ratio' in diversity_df.columns:
        genotypic_col = 'genotypic_unique_ratio'
        diversity_col = None
        has_genotypic = True
    else:
        has_genotypic = False
    
    if has_fitness_mean and has_genotypic:
        # 適應度均值 vs 基因型多樣性
        x_data = diversity_df[genotypic_col]
        y_data = diversity_df['fitness_fitness_mean']
        
        # 過濾有效數據
        valid_mask = (~x_data.isna()) & (~y_data.isna())
        if valid_mask.sum() > 0:
            scatter = axes[0].scatter(x_data[valid_mask], 
                           y_data[valid_mask],
                           c=diversity_df['generation'][valid_mask], 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            xlabel = 'TED-based Unique Ratio' if genotypic_col == 'ted_unique_ratio' else 'Genotypic Unique Ratio'
            axes[0].set_xlabel(xlabel, fontsize=11)
            axes[0].set_ylabel('Mean Fitness', fontsize=11)
            axes[0].set_title('Mean Fitness vs Genotypic Diversity', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # 添加趨勢線
            if len(x_data[valid_mask]) > 1:
                z = np.polyfit(x_data[valid_mask], y_data[valid_mask], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data[valid_mask].min(), x_data[valid_mask].max(), 100)
                axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
                axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                        fontsize=12, color='red', transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                    fontsize=12, color='red', transform=axes[0].transAxes)
    
    if has_fitness_std and has_genotypic:
        # 適應度標準差 vs 基因型多樣性
        x_data = diversity_df[genotypic_col]
        y_data = diversity_df['fitness_fitness_std']
        
        # 過濾有效數據
        valid_mask = (~x_data.isna()) & (~y_data.isna())
        if valid_mask.sum() > 0:
            scatter = axes[1].scatter(x_data[valid_mask], 
                           y_data[valid_mask],
                           c=diversity_df['generation'][valid_mask], 
                           cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            xlabel = 'TED-based Unique Ratio' if genotypic_col == 'ted_unique_ratio' else 'Genotypic Unique Ratio'
            axes[1].set_xlabel(xlabel, fontsize=11)
            axes[1].set_ylabel('Fitness Std Dev', fontsize=11)
            axes[1].set_title('Fitness Diversity vs Genotypic Diversity', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # 添加趨勢線
            if len(x_data[valid_mask]) > 1:
                z = np.polyfit(x_data[valid_mask], y_data[valid_mask], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x_data[valid_mask].min(), x_data[valid_mask].max(), 100)
                axes[1].plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Trend')
                axes[1].legend()
            
            # 添加 colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                       norm=plt.Normalize(vmin=diversity_df['generation'].min(), 
                                                         vmax=diversity_df['generation'].max()))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, orientation='vertical', pad=0.02)
            cbar.set_label('Generation', fontsize=11)
        else:
            axes[1].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                        fontsize=12, color='red', transform=axes[1].transAxes)
    else:
        axes[1].text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                    fontsize=12, color='red', transform=axes[1].transAxes)
    
    plt.tight_layout()
    
    output_file = output_dir / 'diversity_fitness_correlation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ 保存相關性圖: {output_file}")
    plt.close()
    
    print("✅ 視覺化完成\n")


def save_results(diversity_df: pd.DataFrame, output_dir: Path):
    """保存結果"""
    print("💾 保存結果...")
    
    # 保存 CSV
    csv_file = output_dir / 'diversity_metrics.csv'
    diversity_df.to_csv(csv_file, index=False)
    print(f"   ✅ CSV: {csv_file}")
    
    # 保存文字報告
    report_file = output_dir / 'diversity_analysis_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Population Diversity Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Generations: {len(diversity_df)}\n")
        f.write(f"Generation Range: {diversity_df['generation'].min()} - {diversity_df['generation'].max()}\n\n")
        
        f.write("Key Metrics Summary:\n")
        f.write("-" * 80 + "\n")
        
        for col in diversity_df.columns:
            if col != 'generation':
                f.write(f"\n{col}:\n")
                f.write(f"  Initial: {diversity_df[col].iloc[0]:.6f}\n")
                f.write(f"  Final: {diversity_df[col].iloc[-1]:.6f}\n")
                f.write(f"  Mean: {diversity_df[col].mean():.6f}\n")
                f.write(f"  Std: {diversity_df[col].std():.6f}\n")
                f.write(f"  Min: {diversity_df[col].min():.6f}\n")
                f.write(f"  Max: {diversity_df[col].max():.6f}\n")
    
    print(f"   ✅ 報告: {report_file}")
    print("✅ 保存完成\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析演化記錄的多樣性')
    parser.add_argument('--records_dir', type=str, required=True,
                       help='記錄目錄路徑')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='顯示詳細信息')
    parser.add_argument('--use_ted', action='store_true', default=True,
                       help='使用 TED 計算基因型多樣性（默認: True）')
    parser.add_argument('--no_ted', action='store_true',
                       help='不使用 TED，使用字符串比較')
    parser.add_argument('--ted_sample_size', type=int, default=50,
                       help='TED 計算的採樣大小（默認: 50）')
    
    args = parser.parse_args()
    
    # 處理 TED 選項
    use_ted = args.use_ted and not args.no_ted
    
    print("=" * 80)
    print("🔬 演化族群多樣性分析")
    print("=" * 80)
    print(f"記錄目錄: {args.records_dir}\n")
    
    try:
        # 設置 DEAP
        setup_deap_creator()
        
        # 載入族群
        records_dir = Path(args.records_dir)
        populations = load_populations_from_records(records_dir, verbose=args.verbose)
        
        if not populations:
            print("❌ 沒有載入任何族群數據")
            return
        
        # 計算多樣性指標
        diversity_df = calculate_diversity_metrics(
            populations, 
            verbose=args.verbose,
            use_ted=use_ted,
            ted_sample_size=args.ted_sample_size
        )
        
        # 列印摘要
        print_summary_statistics(diversity_df)
        
        # 創建視覺化
        create_visualizations(diversity_df, records_dir)
        
        # 保存結果
        save_results(diversity_df, records_dir)
        
        print("=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
