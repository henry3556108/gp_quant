#!/usr/bin/env python3
"""
測試 Niche Selection Strategies

驗證 TEDNicheSelectionStrategy：
1. TED matrix 計算與快取
2. 階層式分群
3. Elite Pool 提取
4. Crossover pairs 選擇（同群/跨群）
5. Mutation individuals 選擇（Ranked SUS）
6. 數量整除性
"""

import sys
import pickle
from pathlib import Path
from deap import creator, base, gp

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.evolution.components.strategies import TEDNicheSelectionStrategy


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_test_population(records_dir: Path, generation: int = 0):
    """載入測試族群"""
    populations_dir = records_dir / 'populations'
    gen_file = populations_dir / f'generation_{generation:03d}.pkl'
    
    print(f"📂 載入測試族群: {gen_file}")
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    print(f"   ✅ 載入 {len(population)} 個個體")
    
    return population


def test_ted_niche_selection():
    """測試 TED Niche Selection Strategy"""
    
    print("=" * 80)
    print("🧪 測試 TED Niche Selection Strategy")
    print("=" * 80)
    
    # 1. 設置 DEAP
    setup_deap_creator()
    
    # 2. 載入測試族群
    records_dir = Path("/Users/hongyicheng/Downloads/gp_quant/test_evolution_11241221_records_20251125_1335")
    population = load_test_population(records_dir, generation=0)
    
    POP_SIZE = len(population)
    print(f"\n族群大小: {POP_SIZE}")
    
    # 3. 創建 TED Niche Selection Strategy
    print("\n" + "=" * 80)
    print("📦 創建 TED Niche Selection Strategy")
    print("=" * 80)
    
    strategy = TEDNicheSelectionStrategy(
        n_clusters=5,
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        n_jobs=6
    )
    
    print(f"策略配置: {strategy.get_stats()}")
    
    # 4. 測試三個階段的數量計算
    print("\n" + "=" * 80)
    print("🔢 測試數量計算（三個互斥階段）")
    print("=" * 80)
    
    crossover_rate = 0.75
    mutation_rate = 0.20
    reproduction_rate = 0.05
    
    # Crossover
    num_crossover_offspring = int(POP_SIZE * crossover_rate)
    if num_crossover_offspring % 2 != 0:
        print(f"⚠️  Crossover offspring 數量為奇數 ({num_crossover_offspring})，調整為偶數")
        num_crossover_offspring -= 1
    
    num_crossover_pairs = num_crossover_offspring // 2
    
    # Mutation
    num_mutation_offspring = int(POP_SIZE * mutation_rate)
    
    # Reproduction
    num_reproduction_offspring = int(POP_SIZE * reproduction_rate)
    
    # 調整以確保總和 = POP_SIZE
    total = num_crossover_offspring + num_mutation_offspring + num_reproduction_offspring
    if total != POP_SIZE:
        diff = POP_SIZE - total
        print(f"⚠️  總和 ({total}) ≠ POP_SIZE ({POP_SIZE})，差異: {diff}")
        num_mutation_offspring += diff
        print(f"   調整 mutation_offspring: {num_mutation_offspring - diff} → {num_mutation_offspring}")
    
    print(f"\n階段 1 - Crossover:")
    print(f"  目標比例: {crossover_rate * 100}%")
    print(f"  offspring 數量: {num_crossover_offspring}")
    print(f"  parent pairs 數量: {num_crossover_pairs}")
    print(f"  實際產生: {num_crossover_pairs * 2} 個 offspring")
    
    print(f"\n階段 2 - Mutation:")
    print(f"  目標比例: {mutation_rate * 100}%")
    print(f"  offspring 數量: {num_mutation_offspring}")
    
    print(f"\n階段 3 - Reproduction:")
    print(f"  目標比例: {reproduction_rate * 100}%")
    print(f"  offspring 數量: {num_reproduction_offspring}")
    
    print(f"\n總計: {num_crossover_offspring} + {num_mutation_offspring} + {num_reproduction_offspring} = {total}")
    print(f"✅ 數量檢查: {'通過' if total == POP_SIZE else '失敗'}")
    
    # 5. 測試 Crossover pairs 選擇
    print("\n" + "=" * 80)
    print("🧬 測試 Crossover Pairs 選擇")
    print("=" * 80)
    
    data = {'generation': 0}
    
    print(f"選擇 {num_crossover_pairs} 對 parents...")
    crossover_pairs = strategy.select_pairs(population, num_crossover_pairs, data)
    
    print(f"✅ 選擇了 {len(crossover_pairs)} 對 parents")
    print(f"   預期產生: {len(crossover_pairs) * 2} 個 offspring")
    
    # 檢查 pairs 的有效性
    if crossover_pairs:
        print(f"\n範例 pairs (前 3 對):")
        for i, (p1, p2) in enumerate(crossover_pairs[:3]):
            print(f"  Pair {i+1}: fitness=({p1.fitness.values[0]:.4f}, {p2.fitness.values[0]:.4f}), "
                  f"size=({len(p1)}, {len(p2)})")
    
    # 6. 測試 Mutation individuals 選擇
    print("\n" + "=" * 80)
    print("🧬 測試 Mutation Individuals 選擇")
    print("=" * 80)
    
    print(f"選擇 {num_mutation_offspring} 個 individuals...")
    mutation_individuals = strategy.select_individuals(population, num_mutation_offspring, data)
    
    print(f"✅ 選擇了 {len(mutation_individuals)} 個 individuals")
    
    if mutation_individuals:
        fitnesses = [ind.fitness.values[0] for ind in mutation_individuals]
        print(f"\nFitness 統計:")
        print(f"  平均: {sum(fitnesses) / len(fitnesses):.4f}")
        print(f"  最大: {max(fitnesses):.4f}")
        print(f"  最小: {min(fitnesses):.4f}")
    
    # 7. 測試快取機制
    print("\n" + "=" * 80)
    print("💾 測試快取機制")
    print("=" * 80)
    
    print("第二次呼叫 select_pairs (應該使用快取)...")
    crossover_pairs_2 = strategy.select_pairs(population, 10, data)
    
    print(f"✅ 選擇了 {len(crossover_pairs_2)} 對 parents (使用快取)")
    
    # 8. 測試不同世代（應該重新計算）
    print("\n測試不同世代 (應該重新計算)...")
    data_gen1 = {'generation': 1}
    crossover_pairs_3 = strategy.select_pairs(population, 10, data_gen1)
    
    print(f"✅ 選擇了 {len(crossover_pairs_3)} 對 parents (重新計算)")
    
    # 9. 顯示策略統計
    print("\n" + "=" * 80)
    print("📊 策略統計")
    print("=" * 80)
    
    stats = strategy.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("✅ 測試完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_ted_niche_selection()
