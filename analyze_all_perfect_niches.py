"""
分析所有 generation 中 silhouette = 1.0 的 niche
從每個完全相同的 niche 中隨機抽樣 3 個個體並記錄
"""
import json
import dill
import numpy as np
from pathlib import Path
from deap import creator, base, gp
from gp_quant.gp.operators import pset
from collections import defaultdict

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("分析所有 Generation 中 Silhouette = 1.0 的 Niche")
print("="*100)

# 讀取實驗紀錄
exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251023_125111')
log_file = exp_dir / 'evolution_log.json'

with open(log_file, 'r') as f:
    data = json.load(f)

niching_data = data.get('niching', {}).get('log', [])
print(f"\n總共有 {len(niching_data)} 個 generation 的 niching 紀錄")

# 找出所有 silhouette = 1.0 的 niche
perfect_niches_by_gen = defaultdict(list)

for gen_data in niching_data:
    gen = gen_data['generation']
    per_niche_silhouette = gen_data.get('per_niche_silhouette', {})
    
    for niche_id, niche_stats in per_niche_silhouette.items():
        if niche_stats['mean'] == 1.0 and niche_stats['std'] == 0.0:
            perfect_niches_by_gen[gen].append({
                'niche_id': int(niche_id),
                'size': niche_stats['size'],
                'niche_sizes': gen_data['niche_sizes']
            })

print(f"\n發現 {sum(len(niches) for niches in perfect_niches_by_gen.values())} 個 niche 的 silhouette = 1.0")
print("="*100)

# 準備輸出文件
output_lines = []
output_lines.append("="*100)
output_lines.append("所有 Generation 中完全相同的 Niche 分析")
output_lines.append("="*100)
output_lines.append("")

# 分析每個 generation
for gen in sorted(set(range(1, len(niching_data) + 1))):
    output_lines.append(f"\n{'='*100}")
    output_lines.append(f"Generation {gen}")
    output_lines.append(f"{'='*100}")
    
    if gen not in perfect_niches_by_gen:
        output_lines.append(f"\n❌ 沒有發現完全相同的 niche")
        print(f"Generation {gen}: 沒有完全相同的 niche")
        continue
    
    perfect_niches = perfect_niches_by_gen[gen]
    output_lines.append(f"\n✓ 發現 {len(perfect_niches)} 個完全相同的 niche")
    print(f"\nGeneration {gen}: 發現 {len(perfect_niches)} 個完全相同的 niche")
    
    # 顯示所有 niche 的大小
    if perfect_niches:
        output_lines.append(f"所有 niche 大小: {perfect_niches[0]['niche_sizes']}")
    
    # 載入該 generation 的族群
    gen_dir = exp_dir / 'generations'
    pop_file = gen_dir / f'generation_{gen:03d}.pkl'
    
    if not pop_file.exists():
        output_lines.append(f"\n⚠️  無法載入 Generation {gen} 的族群檔案")
        print(f"  ⚠️  無法載入族群檔案")
        continue
    
    print(f"  載入族群...")
    with open(pop_file, 'rb') as f:
        gen_data_pkl = dill.load(f)
    
    population = gen_data_pkl['population']
    
    # 由於沒有 cluster_labels，我們需要通過表達式來識別重複的個體
    # 先建立表達式到個體的映射
    from collections import Counter
    expression_to_individuals = defaultdict(list)
    for ind in population:
        expression_to_individuals[str(ind)].append(ind)
    
    # 找出最常見的表達式（這些很可能就是完全相同的 niche）
    expression_counts = Counter(str(ind) for ind in population)
    most_common_expressions = expression_counts.most_common(len(perfect_niches) + 5)
    
    # 對每個完全相同的 niche 進行分析
    for i, niche_info in enumerate(perfect_niches, 1):
        niche_id = niche_info['niche_id']
        niche_size = niche_info['size']
        
        output_lines.append(f"\n{'-'*100}")
        output_lines.append(f"Niche {niche_id} (大小: {niche_size} 個個體)")
        output_lines.append(f"{'-'*100}")
        
        print(f"  分析 Niche {niche_id} (大小: {niche_size})...")
        
        # 找出與這個 niche 大小最接近的表達式
        # 假設最常見的幾個表達式就是完全相同的 niche
        candidate_expr = None
        for expr, count in most_common_expressions:
            # 找到大小接近的表達式（允許一些誤差）
            if abs(count - niche_size) <= 10:
                candidate_expr = expr
                actual_count = count
                break
        
        if candidate_expr is None:
            # 如果找不到精確匹配，就用第 i 個最常見的表達式
            if i <= len(most_common_expressions):
                candidate_expr, actual_count = most_common_expressions[i-1]
            else:
                output_lines.append("⚠️  無法找到對應的表達式")
                continue
        
        # 獲取這個表達式的所有個體
        niche_individuals = expression_to_individuals[candidate_expr]
        
        output_lines.append(f"\n推測的表達式 (出現 {len(niche_individuals)} 次):")
        output_lines.append(f"  {candidate_expr}")
        
        # 隨機抽樣 3 個個體
        sample_size = min(3, len(niche_individuals))
        indices = np.random.choice(len(niche_individuals), size=sample_size, replace=False)
        sampled_individuals = [niche_individuals[i] for i in indices]
        
        output_lines.append(f"\n隨機抽樣 {sample_size} 個個體:")
        output_lines.append("")
        
        for j, ind in enumerate(sampled_individuals, 1):
            output_lines.append(f"個體 {j}:")
            output_lines.append(f"  Fitness: {ind.fitness.values[0]:.6f}")
            output_lines.append(f"  深度: {ind.height}")
            output_lines.append(f"  大小: {len(ind)} 個節點")
            output_lines.append(f"  表達式: {str(ind)}")
            output_lines.append("")
        
        # 統計資訊
        fitnesses = [ind.fitness.values[0] for ind in niche_individuals]
        output_lines.append(f"該 niche 的統計資訊:")
        output_lines.append(f"  Fitness 範圍: [{min(fitnesses):.6f}, {max(fitnesses):.6f}]")
        output_lines.append(f"  Fitness 平均: {np.mean(fitnesses):.6f}")
        output_lines.append(f"  Fitness 標準差: {np.std(fitnesses):.6f}")
        output_lines.append(f"  不同 fitness 值數量: {len(set(fitnesses))}")

# 寫入檔案
output_file = exp_dir / 'perfect_niches_analysis.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("\n" + "="*100)
print(f"✓ 分析完成！結果已儲存至: {output_file}")
print("="*100)
