"""
測試單個 generation 的動態 niche 選擇

快速測試腳本，只處理一個 generation
"""

import pickle
import time
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

# 設置 DEAP creator
from deap import base, creator, gp

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

from gp_quant.similarity.similarity_matrix import SimilarityMatrix
from gp_quant.niching.clustering import NichingClusterer


def main():
    # 載入第一個 generation
    gen_file = Path("portfolio_experiment_results/portfolio_exp_sharpe_20251017_122243/generations/generation_001.pkl")
    
    print("=" * 80)
    print("測試單個 Generation 的動態 Niche 選擇")
    print("=" * 80)
    print(f"檔案: {gen_file.name}\n")
    
    # 載入資料
    print("📂 載入資料...")
    with open(gen_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and 'population' in data:
        population = data['population']
        print(f"   ✓ 載入完整資料 (dict 格式)")
    else:
        population = data
        print(f"   ✓ 載入 population (list 格式)")
    
    print(f"   Population 大小: {len(population)}\n")
    
    # 計算相似度矩陣
    print("🔬 計算相似度矩陣...")
    print(f"   預計需要計算 {len(population) * (len(population) - 1) // 2} 對相似度")
    start_time = time.time()
    
    sim_matrix = SimilarityMatrix(population)
    sim_matrix.compute(show_progress=True)
    
    sim_time = time.time() - start_time
    print(f"\n   ✓ 完成！耗時: {sim_time:.2f}s ({sim_time/60:.2f} 分鐘)")
    print(f"   平均相似度: {sim_matrix.get_average_similarity():.4f}")
    print(f"   多樣性分數: {sim_matrix.get_diversity_score():.4f}\n")
    
    # 測試不同 k 值
    k_range = [2, 3, 4, 5, 6, 7, 8]
    print(f"🎯 測試 k 值範圍: {k_range}\n")
    
    results = []
    for k in k_range:
        print(f"   測試 k={k}...")
        k_start = time.time()
        
        clusterer = NichingClusterer(
            n_clusters=k,
            algorithm='kmeans',
            random_state=42
        )
        clusterer.fit(sim_matrix.matrix)
        
        k_time = time.time() - k_start
        stats = clusterer.get_statistics()
        
        results.append({
            'k': k,
            'silhouette': clusterer.silhouette_score_,
            'time': k_time,
            'niche_sizes': stats['niche_sizes']
        })
        
        print(f"      Silhouette Score: {clusterer.silhouette_score_:.4f}")
        print(f"      時間: {k_time:.3f}s")
        print(f"      Niche 大小: {list(stats['niche_sizes'].values())}\n")
    
    # 找出最佳 k
    best = max(results, key=lambda x: x['silhouette'])
    print("=" * 80)
    print(f"✨ 最佳 k 值: {best['k']}")
    print(f"   Silhouette Score: {best['silhouette']:.4f}")
    print(f"   Niche 大小分布: {list(best['niche_sizes'].values())}")
    print("=" * 80)
    
    # 總結
    total_time = sim_time + sum(r['time'] for r in results)
    print(f"\n總計算時間: {total_time:.2f}s ({total_time/60:.2f} 分鐘)")
    print(f"  - 相似度矩陣: {sim_time:.2f}s ({sim_time/total_time*100:.1f}%)")
    print(f"  - K 值測試: {sum(r['time'] for r in results):.2f}s ({sum(r['time'] for r in results)/total_time*100:.1f}%)")


if __name__ == "__main__":
    main()
