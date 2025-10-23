"""
分析 per_niche_silhouette = 1.0 的 niche
找出完全相同的個體並視覺化
"""
import json
import dill
import numpy as np
from pathlib import Path
from deap import creator, base, gp
from gp_quant.gp.operators import pset
import matplotlib.pyplot as plt
from collections import Counter

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("分析 Niching 實驗中 Silhouette = 1.0 的 Niche")
print("="*100)

# 讀取實驗紀錄
exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251023_125111')
log_file = exp_dir / 'evolution_log.json'

with open(log_file, 'r') as f:
    data = json.load(f)

niching_data = data.get('niching', {}).get('log', [])
print(f"\n總共有 {len(niching_data)} 個 generation 的 niching 紀錄")

# 找出所有 silhouette = 1.0 的 niche
perfect_niches = []

for gen_data in niching_data:
    gen = gen_data['generation']
    per_niche_silhouette = gen_data.get('per_niche_silhouette', {})
    
    for niche_id, niche_stats in per_niche_silhouette.items():
        if niche_stats['mean'] == 1.0 and niche_stats['std'] == 0.0:
            perfect_niches.append({
                'generation': gen,
                'niche_id': int(niche_id),
                'size': niche_stats['size'],
                'niche_sizes': gen_data['niche_sizes']
            })

print(f"\n發現 {len(perfect_niches)} 個 niche 的 silhouette = 1.0 (完全相同)")
print("="*100)

# 按 generation 分組顯示
from collections import defaultdict
niches_by_gen = defaultdict(list)
for info in perfect_niches:
    niches_by_gen[info['generation']].append(info)

print("\n按 Generation 分組:")
print("-"*100)
for gen in sorted(niches_by_gen.keys()):
    niches = niches_by_gen[gen]
    print(f"\nGeneration {gen}: {len(niches)} 個完全相同的 niche")
    for niche_info in niches:
        print(f"  Niche {niche_info['niche_id']}: {niche_info['size']} 個個體")
        print(f"    所有 niche 大小: {niche_info['niche_sizes']}")

# 選擇一個 generation 來詳細分析（選擇第一個有完全相同 niche 的 generation）
if perfect_niches:
    target_info = perfect_niches[0]
    target_gen = target_info['generation']
    target_niche_id = target_info['niche_id']
    
    print("\n" + "="*100)
    print(f"詳細分析 Generation {target_gen}, Niche {target_niche_id}")
    print("="*100)
    
    # 載入該 generation 的族群
    gen_dir = exp_dir / 'generations'
    pop_file = gen_dir / f'generation_{target_gen:03d}.pkl'
    
    if pop_file.exists():
        print(f"\n載入 Generation {target_gen} 的族群...")
        with open(pop_file, 'rb') as f:
            gen_data = dill.load(f)
        
        population = gen_data['population']
        
        # 重新計算 cluster labels (從 niching log 中獲取)
        # 由於沒有儲存 cluster_labels，我們需要重新聚類
        # 但這裡我們只是為了展示，所以先跳過詳細分析
        print(f"⚠️  注意：cluster_labels 未儲存，無法精確定位 niche 中的個體")
        print(f"   將分析整個族群的統計資訊")
        
        cluster_labels = None
        
        print(f"✓ 成功載入族群 (大小: {len(population)})")
        
        # 由於無法精確定位 niche，分析整個族群
        print(f"\n分析整個族群 (共 {len(population)} 個個體)")
        print(f"注意：Niche {target_niche_id} 理論上有 {target_info['size']} 個完全相同的個體")
        
        # 檢查族群中的重複個體
        print("\n" + "="*100)
        print("分析族群中的重複個體:")
        print("-"*100)
        
        from collections import Counter
        expression_counts = Counter(str(ind) for ind in population)
        
        print(f"族群中不同的表達式數量: {len(expression_counts)}")
        print(f"族群總大小: {len(population)}")
        print(f"重複率: {(1 - len(expression_counts)/len(population))*100:.2f}%")
        
        # 找出出現最多次的表達式
        most_common = expression_counts.most_common(10)
        print(f"\n出現最多的 10 種表達式:")
        print("-"*100)
        for i, (expr, count) in enumerate(most_common, 1):
            print(f"\n{i}. 出現 {count} 次 ({count/len(population)*100:.2f}%):")
            print(f"   {expr}")
        
        # 從最常見的表達式中抽樣
        print("\n" + "="*100)
        print("最常見表達式的個體詳細資訊:")
        print("-"*100)
        
        # 取第一個最常見的表達式
        most_common_expr = most_common[0][0]
        sampled_individuals = [ind for ind in population if str(ind) == most_common_expr][:5]
        
        for i, ind in enumerate(sampled_individuals, 1):
            print(f"\n個體 {i}:")
            print(f"  Fitness: {ind.fitness.values[0]:.6f}")
            print(f"  深度: {ind.height}")
            print(f"  大小 (節點數): {len(ind)}")
            print(f"  表達式: {str(ind)}")
        
        # Fitness 統計 (針對最常見的表達式)
        print("\n" + "="*100)
        print(f"最常見表達式的 Fitness 統計 (共 {most_common[0][1]} 個個體):")
        print("-"*100)
        
        common_individuals = [ind for ind in population if str(ind) == most_common_expr]
        fitnesses = [ind.fitness.values[0] for ind in common_individuals]
        print(f"最小值: {min(fitnesses):.6f}")
        print(f"最大值: {max(fitnesses):.6f}")
        print(f"平均值: {np.mean(fitnesses):.6f}")
        print(f"標準差: {np.std(fitnesses):.6f}")
        print(f"不同 fitness 值數量: {len(set(fitnesses))}")
        
        # 深度統計
        print("\n深度統計:")
        print("-"*100)
        
        depths = [ind.height for ind in common_individuals]
        print(f"最小深度: {min(depths)}")
        print(f"最大深度: {max(depths)}")
        print(f"平均深度: {np.mean(depths):.2f}")
        print(f"不同深度數量: {len(set(depths))}")
        
        # 大小統計
        print("\n大小 (節點數) 統計:")
        print("-"*100)
        
        sizes = [len(ind) for ind in common_individuals]
        print(f"最小大小: {min(sizes)}")
        print(f"最大大小: {max(sizes)}")
        print(f"平均大小: {np.mean(sizes):.2f}")
        print(f"不同大小數量: {len(set(sizes))}")
        
        # 視覺化：繪製 fitness 分布
        print("\n" + "="*100)
        print("生成視覺化圖表...")
        print("-"*100)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Generation {target_gen} - 最常見表達式分析 (出現 {most_common[0][1]} 次)', fontsize=16, fontweight='bold')
        
        # 1. Fitness 分布
        ax1 = axes[0, 0]
        ax1.hist(fitnesses, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(fitnesses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(fitnesses):.4f}')
        ax1.set_xlabel('Fitness (Sharpe Ratio)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Fitness Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 深度分布
        ax2 = axes[0, 1]
        depth_counts = Counter(depths)
        ax2.bar(depth_counts.keys(), depth_counts.values(), edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Tree Depth', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Tree Depth Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 大小分布
        ax3 = axes[1, 0]
        ax3.hist(sizes, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax3.axvline(np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.1f}')
        ax3.set_xlabel('Tree Size (Number of Nodes)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Tree Size Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 統計摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        統計摘要
        {'='*40}
        
        該表達式出現次數: {most_common[0][1]}
        佔族群比例: {most_common[0][1]/len(population)*100:.2f}%
        
        Fitness:
          範圍: [{min(fitnesses):.4f}, {max(fitnesses):.4f}]
          平均: {np.mean(fitnesses):.4f}
          標準差: {np.std(fitnesses):.4f}
        
        深度:
          範圍: [{min(depths)}, {max(depths)}]
          平均: {np.mean(depths):.2f}
        
        大小:
          範圍: [{min(sizes)}, {max(sizes)}]
          平均: {np.mean(sizes):.2f}
        
        結論:
        ✅ 這是族群中最常見的表達式
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 儲存圖表
        output_file = exp_dir / f'perfect_niche_analysis_gen{target_gen}_niche{target_niche_id}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {output_file}")
        
        plt.close()
        
    else:
        print(f"\n❌ 找不到 Generation {target_gen} 的族群檔案")
        print(f"   期望路徑: {pop_file}")

print("\n" + "="*100)
print("分析完成")
print("="*100)
