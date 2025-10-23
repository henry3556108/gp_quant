"""
精確分析所有 generation 中 silhouette = 1.0 的 niche
通過重新計算相似度矩陣和聚類來精確定位每個 niche 的個體
"""
import json
import dill
import numpy as np
from pathlib import Path
from deap import creator, base, gp
from gp_quant.gp.operators import pset
from gp_quant.niching.clustering import NichingClusterer
from gp_quant.similarity.similarity_matrix import SimilarityMatrix
from collections import defaultdict, Counter

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("精確分析所有 Generation 中 Silhouette = 1.0 的 Niche")
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
output_lines.append("所有 Generation 中完全相同的 Niche 分析（精確版本）")
output_lines.append("="*100)
output_lines.append("")
output_lines.append("說明：本分析通過重新計算相似度矩陣和聚類來精確定位每個 niche 的個體")
output_lines.append("")

# 只分析有完全相同 niche 的前幾個 generation（避免計算時間過長）
generations_to_analyze = [1, 2, 11, 12]  # Generation 1, 2 (1個niche), 11, 12 (2個niche)

for gen in generations_to_analyze:
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
    print(f"  族群大小: {len(population)}")
    
    # 重新計算相似度矩陣和聚類
    print(f"  計算相似度矩陣...")
    n_clusters = len(perfect_niches[0]['niche_sizes'])
    
    try:
        # 計算相似度矩陣
        sim_calculator = SimilarityMatrix(population)
        similarity_matrix = sim_calculator.compute(show_progress=True)
        
        print(f"  執行 K-means 聚類 (k={n_clusters})...")
        clusterer = NichingClusterer(n_clusters=n_clusters, algorithm='kmeans')
        clusterer.fit(similarity_matrix)
        labels = clusterer.labels_
        silhouette_avg = clusterer.silhouette_score_
        
        print(f"  聚類完成，Silhouette score: {silhouette_avg:.4f}")
        
        # 對每個完全相同的 niche 進行分析
        for i, niche_info in enumerate(perfect_niches, 1):
            niche_id = niche_info['niche_id']
            expected_size = niche_info['size']
            
            output_lines.append(f"\n{'-'*100}")
            output_lines.append(f"Niche {niche_id} (預期大小: {expected_size} 個個體)")
            output_lines.append(f"{'-'*100}")
            
            print(f"  分析 Niche {niche_id} (預期大小: {expected_size})...")
            
            # 找出屬於這個 niche 的所有個體
            niche_individuals = [ind for ind, label in zip(population, labels) if label == niche_id]
            actual_size = len(niche_individuals)
            
            output_lines.append(f"\n實際大小: {actual_size} 個個體")
            output_lines.append(f"大小匹配: {'✓' if abs(actual_size - expected_size) <= 10 else '✗'}")
            
            if actual_size == 0:
                output_lines.append("⚠️  該 niche 沒有個體")
                continue
            
            # 統計表達式
            expression_counts = Counter(str(ind) for ind in niche_individuals)
            unique_expressions = len(expression_counts)
            
            output_lines.append(f"\n不同表達式數量: {unique_expressions}")
            
            if unique_expressions == 1:
                output_lines.append("✅ 確認：該 niche 中所有個體完全相同！")
            else:
                output_lines.append(f"⚠️  該 niche 中有 {unique_expressions} 種不同的表達式")
            
            # 顯示最常見的表達式
            most_common = expression_counts.most_common(3)
            output_lines.append(f"\n最常見的表達式:")
            for rank, (expr, count) in enumerate(most_common, 1):
                output_lines.append(f"  {rank}. 出現 {count} 次 ({count/actual_size*100:.1f}%): {expr}")
            
            # 隨機抽樣 3 個個體
            sample_size = min(3, actual_size)
            indices = np.random.choice(actual_size, size=sample_size, replace=False)
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
            
    except Exception as e:
        output_lines.append(f"\n❌ 計算相似度矩陣時發生錯誤: {str(e)}")
        print(f"  ❌ 錯誤: {str(e)}")
        continue

# 寫入檔案
output_file = exp_dir / 'perfect_niches_analysis_accurate.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("\n" + "="*100)
print(f"✓ 分析完成！結果已儲存至: {output_file}")
print("="*100)
print("\n說明：由於計算相似度矩陣需要大量時間，本分析只處理了部分 generation。")
print("如需分析所有 generation，請修改 generations_to_analyze 列表。")
