#!/usr/bin/env python3
"""
測試 TED Niche Selection 與演化引擎的整合
"""

import sys
import pickle
from pathlib import Path
from deap import creator, base, gp

sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.evolution.components.strategies import TEDNicheSelectionStrategy


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def main():
    print("="*80)
    print("🧪 測試 TED Niche Selection 整合")
    print("="*80)
    
    # 設置
    setup_deap_creator()
    
    # 載入測試族群
    records_dir = Path("/Users/hongyicheng/Downloads/gp_quant/test_evolution_11241221_records_20251125_1335")
    gen_file = records_dir / 'populations' / 'generation_000.pkl'
    
    print(f"\n📂 載入測試族群: {gen_file}")
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    print(f"   ✅ 載入 {len(population)} 個個體")
    
    # 創建策略
    print(f"\n📦 創建 TED Niche Selection Strategy")
    strategy = TEDNicheSelectionStrategy(
        max_k=5,
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        cv_criterion='min',
        n_jobs=6
    )
    
    print(f"   配置: max_k={strategy.max_k}, M={strategy.M}, cv_criterion={strategy.cv_criterion}")
    
    # 測試完整流程
    print(f"\n🚀 測試完整流程...")
    data = {'generation': 0}
    
    # 1. 選擇 Crossover pairs
    print(f"\n1️⃣  選擇 Crossover Pairs (40 對)")
    pairs = strategy.select_pairs(population, 40, data)
    print(f"   ✅ 選擇了 {len(pairs)} 對 parents")
    print(f"   ✅ 最佳 K: {strategy._optimal_k}")
    print(f"   ✅ Elite Pool 大小: {len(strategy._cached_elite_pool)}")
    
    # 2. 選擇 Mutation individuals
    print(f"\n2️⃣  選擇 Mutation Individuals (20 個)")
    individuals = strategy.select_individuals(population, 20, data)
    print(f"   ✅ 選擇了 {len(individuals)} 個 individuals")
    
    # 3. 測試快取
    print(f"\n3️⃣  測試快取機制")
    pairs_2 = strategy.select_pairs(population, 10, data)
    print(f"   ✅ 使用快取，選擇了 {len(pairs_2)} 對 parents")
    
    # 4. 測試不同世代
    print(f"\n4️⃣  測試不同世代（應重新計算）")
    data_gen1 = {'generation': 1}
    pairs_3 = strategy.select_pairs(population, 10, data_gen1)
    print(f"   ✅ 重新計算，選擇了 {len(pairs_3)} 對 parents")
    print(f"   ✅ 新的最佳 K: {strategy._optimal_k}")
    
    # 5. 顯示統計
    print(f"\n📊 策略統計")
    stats = strategy.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ 整合測試完成！")
    print("="*80)


if __name__ == "__main__":
    main()
