"""
分析 Niching 實驗中相似度為 1 的 cluster
找出完全相同的個體並視覺化
"""
import json
import dill
import numpy as np
from pathlib import Path
from deap import creator, base, gp
from gp_quant.gp.operators import pset

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("分析 Niching 實驗中的相似度")
print("="*100)

# 讀取實驗紀錄
exp_dir = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251023_125111')
log_file = exp_dir / 'evolution_log.json'

with open(log_file, 'r') as f:
    data = json.load(f)

evolution_log = data['evolution_log']
print(f"\n總共有 {len(evolution_log)} 個 generation 的紀錄")

# 檢查每個 generation 的 niching 資訊
generations_with_high_similarity = []
all_similarities_summary = []

for gen_data in evolution_log:
    gen = gen_data['generation']
    niching_info = gen_data.get('niching_info', {})
    
    if niching_info:
        intra_similarities = niching_info.get('intra_cluster_similarities', [])
        
        if intra_similarities:
            all_similarities_summary.append({
                'generation': gen,
                'similarities': intra_similarities,
                'max_similarity': max(intra_similarities),
                'min_similarity': min(intra_similarities),
                'avg_similarity': np.mean(intra_similarities)
            })
            
            # 檢查是否有相似度 >= 0.95 的 cluster（降低閾值）
            high_sim_clusters = [(i, sim) for i, sim in enumerate(intra_similarities) if sim >= 0.95]
            
            if high_sim_clusters:
                generations_with_high_similarity.append({
                    'generation': gen,
                    'high_sim_clusters': high_sim_clusters,
                    'all_similarities': intra_similarities,
                    'silhouette_scores': niching_info.get('silhouette_scores', []),
                    'cluster_sizes': niching_info.get('cluster_sizes', [])
                })

# 顯示所有 generation 的相似度統計
print("\n所有 generation 的相似度統計:")
print("-"*100)
for info in all_similarities_summary:
    print(f"Gen {info['generation']:2d}: 最大={info['max_similarity']:.4f}, "
          f"最小={info['min_similarity']:.4f}, 平均={info['avg_similarity']:.4f}")

print(f"\n發現 {len(generations_with_high_similarity)} 個 generation 有相似度 >= 0.95 的 cluster")
print("="*100)

# 詳細顯示每個 generation
for info in generations_with_high_similarity:
    gen = info['generation']
    print(f"\n📊 Generation {gen}:")
    print(f"   高相似度的 cluster: {[f'Cluster {i} (相似度={sim:.6f})' for i, sim in info['high_sim_clusters']]}")
    print(f"   所有 cluster 的相似度: {info['all_similarities']}")
    print(f"   Silhouette scores: {info['silhouette_scores']}")
    print(f"   Cluster 大小: {info['cluster_sizes']}")

# 選擇一個 generation 來詳細分析
if generations_with_high_similarity:
    target_gen_info = generations_with_high_similarity[0]
    target_gen = target_gen_info['generation']
    
    print("\n" + "="*100)
    print(f"詳細分析 Generation {target_gen}")
    print("="*100)
    
    # 載入該 generation 的族群
    gen_dir = exp_dir / 'generations' / f'generation_{target_gen:03d}'
    pop_file = gen_dir / 'population.pkl'
    labels_file = gen_dir / 'cluster_labels.pkl'
    
    if pop_file.exists() and labels_file.exists():
        with open(pop_file, 'rb') as f:
            population = dill.load(f)
        
        with open(labels_file, 'rb') as f:
            cluster_labels = dill.load(f)
        
        print(f"\n✓ 成功載入族群 (大小: {len(population)})")
        print(f"✓ 成功載入 cluster labels (大小: {len(cluster_labels)})")
        
        # 分析每個高相似度的 cluster
        for cluster_idx, similarity in target_gen_info['high_sim_clusters']:
            print(f"\n{'='*100}")
            print(f"Cluster {cluster_idx} (相似度 = {similarity:.6f})")
            print(f"{'='*100}")
            
            # 找出屬於這個 cluster 的所有個體
            cluster_individuals = [ind for ind, label in zip(population, cluster_labels) if label == cluster_idx]
            
            print(f"\n該 cluster 有 {len(cluster_individuals)} 個個體")
            
            # 抽樣 3-5 個個體
            sample_size = min(5, len(cluster_individuals))
            sampled_individuals = np.random.choice(cluster_individuals, size=sample_size, replace=False)
            
            print(f"\n隨機抽樣 {sample_size} 個個體:")
            print("-"*100)
            
            for i, ind in enumerate(sampled_individuals, 1):
                print(f"\n個體 {i}:")
                print(f"  Fitness: {ind.fitness.values[0]:.6f}")
                print(f"  深度: {ind.height}")
                print(f"  大小: {len(ind)}")
                print(f"  表達式: {str(ind)}")
            
            # 檢查這些個體是否真的完全相同
            print(f"\n{'='*100}")
            print("驗證個體是否完全相同:")
            print("-"*100)
            
            unique_expressions = set(str(ind) for ind in cluster_individuals)
            print(f"該 cluster 中不同的表達式數量: {len(unique_expressions)}")
            
            if len(unique_expressions) == 1:
                print("✓ 確認：該 cluster 中所有個體完全相同！")
            else:
                print(f"⚠️  該 cluster 中有 {len(unique_expressions)} 種不同的表達式")
                print("\n前 5 種不同的表達式:")
                for i, expr in enumerate(list(unique_expressions)[:5], 1):
                    print(f"  {i}. {expr}")
            
            # 計算 fitness 分布
            fitnesses = [ind.fitness.values[0] for ind in cluster_individuals]
            print(f"\nFitness 統計:")
            print(f"  最小值: {min(fitnesses):.6f}")
            print(f"  最大值: {max(fitnesses):.6f}")
            print(f"  平均值: {np.mean(fitnesses):.6f}")
            print(f"  標準差: {np.std(fitnesses):.6f}")
            print(f"  不同 fitness 值數量: {len(set(fitnesses))}")
    else:
        print(f"\n❌ 找不到 Generation {target_gen} 的族群檔案")
        print(f"   期望路徑: {pop_file}")

print("\n" + "="*100)
print("分析完成")
print("="*100)
