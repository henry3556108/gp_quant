"""
多樣性分析範例

展示如何使用多樣性視覺化工具分析實驗結果
"""

from pathlib import Path
import sys

# 添加項目根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.similarity import (
    plot_diversity_evolution,
    plot_similarity_heatmap,
    plot_similarity_distribution,
    plot_population_tsne
)


def example_1_plot_evolution():
    """
    範例 1: 繪製多樣性演化曲線
    
    最簡單的使用方式，只需要 diversity_metrics.json 文件
    """
    print("="*80)
    print("範例 1: 繪製多樣性演化曲線")
    print("="*80)
    print()
    
    # 指定實驗目錄
    exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353')
    diversity_file = exp_dir / 'diversity_metrics.json'
    
    if not diversity_file.exists():
        print(f"⚠️  找不到 {diversity_file}")
        print("請先運行: python scripts/analysis/compute_diversity_metrics.py --exp_dir {exp_dir}")
        return
    
    # 繪製演化曲線
    output_file = exp_dir / 'diversity_evolution_example.png'
    plot_diversity_evolution(diversity_file, save_path=output_file)
    
    print(f"✓ 完成！圖表已儲存: {output_file}")
    print()


def example_2_analyze_single_generation():
    """
    範例 2: 分析單一世代
    
    繪製特定世代的詳細分析（熱圖、分佈圖、t-SNE）
    """
    print("="*80)
    print("範例 2: 分析單一世代")
    print("="*80)
    print()
    
    # 指定世代文件
    exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353')
    gen_file = exp_dir / 'generations' / 'generation_050.pkl'
    
    if not gen_file.exists():
        print(f"⚠️  找不到 {gen_file}")
        return
    
    print("分析 Generation 50...")
    print()
    
    # 1. 相似度矩陣熱圖
    print("1. 繪製相似度矩陣熱圖...")
    heatmap_file = exp_dir / 'similarity_heatmap_gen050_example.png'
    plot_similarity_heatmap(gen_file, generation=50, save_path=heatmap_file)
    print()
    
    # 2. 相似度分佈
    print("2. 繪製相似度分佈...")
    dist_file = exp_dir / 'similarity_distribution_gen050_example.png'
    plot_similarity_distribution(gen_file, generation=50, save_path=dist_file)
    print()
    
    # 3. t-SNE 降維視覺化
    print("3. 繪製 t-SNE 降維圖...")
    tsne_file = exp_dir / 'population_tsne_gen050_example.png'
    plot_population_tsne(gen_file, generation=50, save_path=tsne_file, method='tsne')
    print()
    
    print("✓ 完成！")
    print()


def example_3_compare_generations():
    """
    範例 3: 比較不同世代
    
    比較初始世代和最終世代的多樣性
    """
    print("="*80)
    print("範例 3: 比較不同世代")
    print("="*80)
    print()
    
    exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251014_191353')
    generations_dir = exp_dir / 'generations'
    
    # 比較 Gen 1 和 Gen 50
    gen1_file = generations_dir / 'generation_001.pkl'
    gen50_file = generations_dir / 'generation_050.pkl'
    
    if not gen1_file.exists() or not gen50_file.exists():
        print("⚠️  找不到世代文件")
        return
    
    print("比較 Generation 1 vs Generation 50")
    print()
    
    # Generation 1
    print("分析 Generation 1...")
    plot_similarity_distribution(gen1_file, generation=1, 
                                save_path=exp_dir / 'dist_gen001_example.png')
    print()
    
    # Generation 50
    print("分析 Generation 50...")
    plot_similarity_distribution(gen50_file, generation=50, 
                                save_path=exp_dir / 'dist_gen050_example.png')
    print()
    
    print("✓ 完成！請比較兩張分佈圖")
    print()


def main():
    """
    主函數：運行所有範例
    """
    print()
    print("="*80)
    print("📊 多樣性分析範例")
    print("="*80)
    print()
    print("這個腳本展示如何使用多樣性視覺化工具")
    print()
    
    # 選擇要運行的範例
    print("請選擇範例:")
    print("  1. 繪製多樣性演化曲線")
    print("  2. 分析單一世代")
    print("  3. 比較不同世代")
    print("  4. 運行所有範例")
    print()
    
    choice = input("請輸入選項 (1-4): ").strip()
    print()
    
    if choice == '1':
        example_1_plot_evolution()
    elif choice == '2':
        example_2_analyze_single_generation()
    elif choice == '3':
        example_3_compare_generations()
    elif choice == '4':
        example_1_plot_evolution()
        example_2_analyze_single_generation()
        example_3_compare_generations()
    else:
        print("無效的選項")
    
    print("="*80)
    print("✅ 範例完成")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
