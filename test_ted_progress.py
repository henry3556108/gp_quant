#!/usr/bin/env python3
"""
測試 TED 計算的進度條顯示
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
    print("🧪 測試 TED 計算進度條")
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
        n_clusters=5,
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        n_jobs=6
    )
    
    # 計算 TED matrix（會顯示進度條）
    print(f"\n📊 開始計算 TED Distance Matrix...")
    print(f"   Population size: {len(population)}")
    print(f"   Total pairs: {len(population) * (len(population) - 1) // 2}")
    print(f"   Workers: {strategy.n_jobs}")
    print()
    
    ted_matrix = strategy._calculate_ted_distance_matrix(population)
    
    print(f"\n✅ 計算完成！")
    print(f"   Matrix shape: {ted_matrix.shape}")
    print(f"   Matrix 對稱: {(ted_matrix == ted_matrix.T).all()}")
    
    print("\n" + "="*80)
    print("✅ 測試完成！")
    print("="*80)


if __name__ == "__main__":
    main()
