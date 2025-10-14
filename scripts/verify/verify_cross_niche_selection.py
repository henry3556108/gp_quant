"""
Cross-Niche Parent Selection 驗證腳本

演示跨群親代選擇機制的運作方式。
展示兩階段選擇過程和統計資訊。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import numpy as np
from deap import base, creator, tools, gp

from gp_quant.gp.operators import pset
from gp_quant.similarity import TreeEditDistance, SimilarityMatrix
from gp_quant.niching import NichingClusterer, CrossNicheSelector


def setup_gp():
    """設置 GP 環境"""
    # 創建 fitness 和 individual 類型
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    # 創建 toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    return toolbox


def assign_random_fitness(population, seed=42):
    """為族群分配隨機 fitness"""
    random.seed(seed)
    for ind in population:
        # 隨機 fitness，範圍 [0, 100]
        ind.fitness.values = (random.uniform(0, 100),)


def print_header(title):
    """打印標題"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_population_info(population, niche_labels, title="族群資訊"):
    """打印族群資訊"""
    print(f"\n{title}:")
    print(f"  總個體數: {len(population)}")
    print(f"  Niche 數量: {len(np.unique(niche_labels))}")
    
    # 統計每個 niche 的大小
    unique_niches, counts = np.unique(niche_labels, return_counts=True)
    print(f"\n  各 Niche 大小:")
    for niche_id, count in zip(unique_niches, counts):
        print(f"    Niche {niche_id}: {count} 個體")
    
    # 統計每個 niche 的平均 fitness
    print(f"\n  各 Niche 平均 Fitness:")
    for niche_id in unique_niches:
        niche_inds = [ind for ind, label in zip(population, niche_labels) if label == niche_id]
        avg_fitness = np.mean([ind.fitness.values[0] for ind in niche_inds])
        print(f"    Niche {niche_id}: {avg_fitness:.2f}")


def visualize_selection(population, niche_labels, selected, title="選擇結果"):
    """可視化選擇結果"""
    print(f"\n{title}:")
    print(f"  選出的個體數: {len(selected)}")
    
    # 統計選出的個體來自哪些 niches
    selected_indices = [population.index(ind) for ind in selected]
    selected_niches = [niche_labels[i] for i in selected_indices]
    
    unique_niches, counts = np.unique(selected_niches, return_counts=True)
    print(f"\n  選出個體的 Niche 分佈:")
    for niche_id, count in zip(unique_niches, counts):
        print(f"    Niche {niche_id}: {count} 個體")


def demonstrate_cross_niche_selection():
    """演示跨群選擇"""
    print_header("跨群親代選擇演示")
    
    # 1. 創建族群
    print("\n步驟 1: 創建族群")
    toolbox = setup_gp()
    population = toolbox.population(n=30)
    assign_random_fitness(population)
    print(f"  ✓ 創建了 {len(population)} 個個體")
    
    # 2. 計算相似度矩陣
    print("\n步驟 2: 計算相似度矩陣")
    sim_matrix = SimilarityMatrix(population)
    similarity_matrix = sim_matrix.compute(show_progress=False)
    print(f"  ✓ 計算完成，矩陣大小: {similarity_matrix.shape}")
    print(f"  平均相似度: {sim_matrix.get_average_similarity():.4f}")
    
    # 3. 聚類
    print("\n步驟 3: 聚類（分成 5 個 niches）")
    clusterer = NichingClusterer(n_clusters=5, algorithm='kmeans')
    niche_labels = clusterer.fit_predict(similarity_matrix)
    print(f"  ✓ 聚類完成")
    print(f"  Silhouette 分數: {clusterer.silhouette_score_:.4f}")
    
    # 打印族群資訊
    print_population_info(population, niche_labels)
    
    # 4. 跨群選擇
    print("\n步驟 4: 跨群親代選擇")
    selector = CrossNicheSelector(
        cross_niche_ratio=0.8,  # 80% 跨群配對
        tournament_size=3,
        random_state=42
    )
    
    k = 20  # 選擇 20 個個體（10 對）
    selected = selector.select(population, niche_labels, k)
    
    print(f"  ✓ 選擇完成，選出 {len(selected)} 個個體")
    
    # 5. 顯示統計資訊
    selector.print_statistics()
    
    # 6. 可視化選擇結果
    visualize_selection(population, niche_labels, selected)
    
    return selector, population, niche_labels, selected


def demonstrate_different_ratios():
    """演示不同的跨群比例"""
    print_header("不同跨群比例的比較")
    
    # 創建族群
    toolbox = setup_gp()
    population = toolbox.population(n=40)
    assign_random_fitness(population)
    
    # 計算相似度矩陣
    sim_matrix = SimilarityMatrix(population)
    similarity_matrix = sim_matrix.compute(show_progress=False)
    
    # 聚類
    clusterer = NichingClusterer(n_clusters=5, algorithm='kmeans')
    niche_labels = clusterer.fit_predict(similarity_matrix)
    
    print(f"\n族群大小: {len(population)}")
    print(f"Niche 數量: {len(np.unique(niche_labels))}")
    
    # 測試不同的跨群比例
    ratios = [0.0, 0.3, 0.5, 0.8, 1.0]
    k = 20
    
    print(f"\n選擇 {k} 個個體（{k//2} 對）:")
    print("\n" + "-" * 80)
    
    for ratio in ratios:
        selector = CrossNicheSelector(
            cross_niche_ratio=ratio,
            tournament_size=3,
            random_state=42
        )
        
        selected = selector.select(population, niche_labels, k)
        stats = selector.get_statistics()
        
        print(f"\n跨群比例設定: {ratio:.0%}")
        print(f"  實際跨群配對: {stats['cross_niche_pairs']} 對 "
              f"({stats['cross_niche_ratio_actual']:.0%})")
        print(f"  實際群內配對: {stats['within_niche_pairs']} 對 "
              f"({stats['within_niche_ratio_actual']:.0%})")
    
    print("-" * 80)


def demonstrate_pairing_details():
    """演示配對細節"""
    print_header("配對細節演示")
    
    # 創建族群
    toolbox = setup_gp()
    population = toolbox.population(n=20)
    assign_random_fitness(population)
    
    # 計算相似度矩陣
    sim_matrix = SimilarityMatrix(population)
    similarity_matrix = sim_matrix.compute(show_progress=False)
    
    # 聚類
    clusterer = NichingClusterer(n_clusters=3, algorithm='kmeans')
    niche_labels = clusterer.fit_predict(similarity_matrix)
    
    print(f"\n族群大小: {len(population)}")
    print(f"Niche 數量: 3")
    
    # 打印每個個體的資訊
    print("\n個體資訊:")
    print(f"{'ID':<5} {'Niche':<8} {'Fitness':<10}")
    print("-" * 25)
    for i, (ind, label) in enumerate(zip(population, niche_labels)):
        print(f"{i:<5} {label:<8} {ind.fitness.values[0]:<10.2f}")
    
    # 跨群選擇
    selector = CrossNicheSelector(
        cross_niche_ratio=0.8,
        tournament_size=3,
        random_state=42
    )
    
    k = 10
    selected = selector.select(population, niche_labels, k)
    
    # 顯示配對
    print(f"\n選出的配對（{k//2} 對）:")
    print(f"{'配對':<8} {'個體1':<10} {'個體2':<10} {'類型':<15}")
    print("-" * 50)
    
    for i in range(0, len(selected), 2):
        ind1 = selected[i]
        ind2 = selected[i + 1]
        
        idx1 = population.index(ind1)
        idx2 = population.index(ind2)
        
        niche1 = niche_labels[idx1]
        niche2 = niche_labels[idx2]
        
        pair_type = "跨群配對" if niche1 != niche2 else "群內配對"
        
        print(f"{i//2 + 1:<8} "
              f"#{idx1}(N{niche1})<-10 "
              f"#{idx2}(N{niche2})<-10 "
              f"{pair_type:<15}")
    
    # 統計資訊
    selector.print_statistics()


def test_edge_cases():
    """測試邊界情況"""
    print_header("邊界情況測試")
    
    toolbox = setup_gp()
    
    # 測試 1: 只有一個 niche
    print("\n測試 1: 只有一個 niche（所有個體都應該群內配對）")
    population = toolbox.population(n=10)
    assign_random_fitness(population)
    niche_labels = np.zeros(10, dtype=int)  # 所有個體都在 niche 0
    
    selector = CrossNicheSelector(cross_niche_ratio=0.8, tournament_size=3)
    selected = selector.select(population, niche_labels, k=6)
    stats = selector.get_statistics()
    
    print(f"  配對數: {stats['total_pairs']}")
    print(f"  跨群配對: {stats['cross_niche_pairs']} (應該是 0)")
    print(f"  群內配對: {stats['within_niche_pairs']} (應該是 {stats['total_pairs']})")
    assert stats['cross_niche_pairs'] == 0, "❌ 測試失敗"
    print("  ✓ 測試通過")
    
    # 測試 2: 每個 niche 只有一個個體
    print("\n測試 2: 每個 niche 只有一個個體")
    population = toolbox.population(n=5)
    assign_random_fitness(population)
    niche_labels = np.array([0, 1, 2, 3, 4])  # 每個個體一個 niche
    
    selector = CrossNicheSelector(cross_niche_ratio=1.0, tournament_size=2)
    selected = selector.select(population, niche_labels, k=4)
    stats = selector.get_statistics()
    
    print(f"  配對數: {stats['total_pairs']}")
    print(f"  跨群配對: {stats['cross_niche_pairs']}")
    print(f"  ✓ 測試通過（沒有錯誤）")
    
    # 測試 3: 跨群比例為 0（全部群內配對）
    print("\n測試 3: 跨群比例為 0（全部群內配對）")
    population = toolbox.population(n=20)
    assign_random_fitness(population)
    niche_labels = np.array([i % 3 for i in range(20)])  # 3 個 niches
    
    selector = CrossNicheSelector(cross_niche_ratio=0.0, tournament_size=3)
    selected = selector.select(population, niche_labels, k=10)
    stats = selector.get_statistics()
    
    print(f"  配對數: {stats['total_pairs']}")
    print(f"  跨群配對: {stats['cross_niche_pairs']} (應該是 0)")
    print(f"  群內配對: {stats['within_niche_pairs']} (應該是 {stats['total_pairs']})")
    assert stats['cross_niche_pairs'] == 0, "❌ 測試失敗"
    print("  ✓ 測試通過")
    
    # 測試 4: 跨群比例為 1（全部跨群配對）
    print("\n測試 4: 跨群比例為 1（全部跨群配對）")
    selector = CrossNicheSelector(cross_niche_ratio=1.0, tournament_size=3)
    selected = selector.select(population, niche_labels, k=10)
    stats = selector.get_statistics()
    
    print(f"  配對數: {stats['total_pairs']}")
    print(f"  跨群配對: {stats['cross_niche_pairs']} (應該是 {stats['total_pairs']})")
    print(f"  群內配對: {stats['within_niche_pairs']} (應該是 0)")
    assert stats['within_niche_pairs'] == 0, "❌ 測試失敗"
    print("  ✓ 測試通過")


def main():
    """主函數"""
    print("\n" + "🎯" * 40)
    print("Cross-Niche Parent Selection 驗證")
    print("🎯" * 40)
    
    # 1. 基本演示
    demonstrate_cross_niche_selection()
    
    # 2. 不同比例比較
    demonstrate_different_ratios()
    
    # 3. 配對細節
    demonstrate_pairing_details()
    
    # 4. 邊界情況測試
    test_edge_cases()
    
    # 總結
    print_header("驗證總結")
    print("\n✅ 所有測試通過！")
    print("\n主要功能:")
    print("  1. ✓ 兩階段選擇機制（Within-Niche Tournament + Cross-Niche Pairing）")
    print("  2. ✓ 可配置跨群比例（0-100%）")
    print("  3. ✓ Tournament selection 保持群內競爭")
    print("  4. ✓ 詳細的統計資訊")
    print("  5. ✓ 邊界情況處理正確")
    print("\n下一步:")
    print("  - 整合到 EvolutionEngine")
    print("  - 運行完整的 Niching 實驗")
    print("  - 分析多樣性提升效果")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
