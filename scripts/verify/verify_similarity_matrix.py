"""
Similarity Matrix 驗證腳本

展示如何使用 SimilarityMatrix 計算族群的相似度矩陣
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import gp, creator, base
from gp_quant.similarity import SimilarityMatrix, compute_similarity
import numpy as np


def setup_gp():
    """設置 GP 環境"""
    # 創建 fitness 和 individual 類型
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    # 創建 primitive set
    pset = gp.PrimitiveSet("MAIN", arity=2)
    pset.addPrimitive(lambda x, y: x + y, 2, name="add")
    pset.addPrimitive(lambda x, y: x - y, 2, name="sub")
    pset.addPrimitive(lambda x, y: x * y, 2, name="mul")
    pset.addPrimitive(lambda x, y: x / y if y != 0 else 1, 2, name="div")
    
    pset.renameArguments(ARG0='x', ARG1='y')
    
    return pset


def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("Similarity Matrix 驗證")
    print("=" * 80)
    
    # 設置 GP 環境
    pset = setup_gp()
    
    # 創建 toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", lambda: creator.Individual(toolbox.expr()))
    toolbox.register("population", lambda n: [toolbox.individual() for _ in range(n)])
    
    # ========================================================================
    # 測試 1: 小族群（10 個個體）
    # ========================================================================
    print("\n" + "=" * 80)
    print("測試 1: 小族群相似度矩陣（10 個個體）")
    print("=" * 80)
    
    population_size = 10
    population = toolbox.population(n=population_size)
    
    print(f"\n族群大小: {population_size}")
    print(f"\nIndividual 列表:")
    for i, ind in enumerate(population):
        print(f"  [{i}] {ind}")
    
    # 創建 SimilarityMatrix
    print(f"\n計算相似度矩陣...")
    sim_matrix = SimilarityMatrix(population)
    matrix = sim_matrix.compute(show_progress=True)
    
    # 打印矩陣
    print(f"\n相似度矩陣:")
    sim_matrix.print_matrix(precision=4)
    
    # 獲取統計資訊
    stats = sim_matrix.get_statistics()
    print(f"\n📊 統計資訊:")
    print(f"  族群大小: {stats['population_size']}")
    print(f"  配對總數: {stats['total_pairs']}")
    print(f"  平均相似度: {stats['mean_similarity']:.4f}")
    print(f"  標準差: {stats['std_similarity']:.4f}")
    print(f"  最小相似度: {stats['min_similarity']:.4f}")
    print(f"  最大相似度: {stats['max_similarity']:.4f}")
    print(f"  中位數: {stats['median_similarity']:.4f}")
    print(f"  多樣性分數: {stats['diversity_score']:.4f}")
    
    # 最相似的配對
    print(f"\n🔍 最相似的 3 對個體:")
    most_similar = sim_matrix.get_most_similar_pairs(n=3)
    for i, j, sim in most_similar:
        print(f"  Individual [{i}] vs [{j}]: {sim:.4f}")
        print(f"    [{i}] {population[i]}")
        print(f"    [{j}] {population[j]}")
    
    # 最不相似的配對
    print(f"\n🔍 最不相似的 3 對個體:")
    least_similar = sim_matrix.get_least_similar_pairs(n=3)
    for i, j, sim in least_similar:
        print(f"  Individual [{i}] vs [{j}]: {sim:.4f}")
        print(f"    [{i}] {population[i]}")
        print(f"    [{j}] {population[j]}")
    
    # ========================================================================
    # 測試 2: 中等族群（50 個個體）
    # ========================================================================
    print("\n" + "=" * 80)
    print("測試 2: 中等族群相似度矩陣（50 個個體）")
    print("=" * 80)
    
    population_size = 50
    population = toolbox.population(n=population_size)
    
    print(f"\n族群大小: {population_size}")
    print(f"配對總數: {population_size * (population_size - 1) // 2}")
    
    # 計算相似度矩陣
    print(f"\n計算相似度矩陣...")
    sim_matrix = SimilarityMatrix(population)
    matrix = sim_matrix.compute(show_progress=True)
    
    # 獲取統計資訊
    stats = sim_matrix.get_statistics()
    print(f"\n📊 統計資訊:")
    print(f"  族群大小: {stats['population_size']}")
    print(f"  配對總數: {stats['total_pairs']}")
    print(f"  平均相似度: {stats['mean_similarity']:.4f}")
    print(f"  標準差: {stats['std_similarity']:.4f}")
    print(f"  最小相似度: {stats['min_similarity']:.4f}")
    print(f"  最大相似度: {stats['max_similarity']:.4f}")
    print(f"  中位數: {stats['median_similarity']:.4f}")
    print(f"  多樣性分數: {stats['diversity_score']:.4f}")
    
    # 相似度分佈
    print(f"\n📈 相似度分佈:")
    similarities = []
    for i in range(population_size):
        for j in range(i + 1, population_size):
            similarities.append(matrix[i][j])
    
    similarities = np.array(similarities)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(similarities, bins=bins)
    
    for i in range(len(bins) - 1):
        percentage = (hist[i] / len(similarities)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}]: {hist[i]:>5} ({percentage:>5.1f}%) {bar}")
    
    # ========================================================================
    # 測試 3: 驗證對稱性和正確性
    # ========================================================================
    print("\n" + "=" * 80)
    print("測試 3: 驗證矩陣性質")
    print("=" * 80)
    
    # 驗證對稱性
    is_symmetric = np.allclose(matrix, matrix.T)
    print(f"\n✓ 對稱性檢查: {'通過' if is_symmetric else '失敗'}")
    
    # 驗證對角線為 1
    diagonal_ones = np.allclose(np.diag(matrix), 1.0)
    print(f"✓ 對角線為 1: {'通過' if diagonal_ones else '失敗'}")
    
    # 驗證範圍 [0, 1]
    in_range = np.all((matrix >= 0) & (matrix <= 1))
    print(f"✓ 範圍 [0, 1]: {'通過' if in_range else '失敗'}")
    
    # 隨機驗證幾個值
    print(f"\n✓ 隨機驗證:")
    for _ in range(3):
        i = np.random.randint(0, population_size)
        j = np.random.randint(0, population_size)
        if i != j:
            # 使用 compute_similarity 重新計算
            expected = compute_similarity(population[i], population[j])
            actual = matrix[i][j]
            match = np.isclose(expected, actual)
            print(f"  Individual [{i}] vs [{j}]: 矩陣={actual:.4f}, 重算={expected:.4f} {'✓' if match else '✗'}")
    
    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 總結")
    print("=" * 80)
    print("\n✨ SimilarityMatrix 功能驗證完成！")
    print("\n功能特點:")
    print("  1. ✅ 計算族群相似度矩陣")
    print("  2. ✅ 支援 DEAP Individual")
    print("  3. ✅ 對稱矩陣優化")
    print("  4. ✅ 進度條顯示")
    print("  5. ✅ 統計資訊計算")
    print("  6. ✅ 最相似/不相似配對查詢")
    print("  7. ✅ 多樣性分數計算")
    print("\n性能:")
    print(f"  - 10 個個體: 45 對比較")
    print(f"  - 50 個個體: 1,225 對比較")
    print(f"  - 100 個個體: 4,950 對比較")
    print(f"  - 500 個個體: 124,750 對比較")
    print("\n下一步:")
    print("  - 實作並行計算加速大規模族群（5000+ 個體）")
    print("  - 實作視覺化工具（熱圖、分佈圖）")
    print("  - 整合到 Niching 策略")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
