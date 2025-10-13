"""
DEAP Individual 相似度計算演示

展示如何直接使用 DEAP Individual 計算樹相似度
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import gp, creator, base
from gp_quant.similarity import (
    compute_similarity, 
    compute_ted,
    TreeEditDistance,
    deap_to_tree_node,
    tree_node_to_bracket
)


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


def create_individual(pset, expr_list):
    """從表達式列表創建 Individual"""
    ind = creator.Individual(expr_list)
    return ind


def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("DEAP Individual 相似度計算演示")
    print("=" * 80)
    
    # 設置 GP 環境
    pset = setup_gp()
    
    # 創建兩個 DEAP Individual
    print("\n📝 創建 DEAP Individual...")
    
    # 使用 DEAP 的標準方式生成樹
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", lambda: creator.Individual(toolbox.expr()))
    
    # 生成兩個隨機個體
    individual1 = toolbox.individual()
    individual2 = toolbox.individual()
    
    print(f"\nIndividual 1: {individual1}")
    print(f"Individual 2: {individual2}")
    
    # ========================================================================
    # 方式 1: 使用便捷函數（最簡單）⭐
    # ========================================================================
    print("\n" + "=" * 80)
    print("方式 1: 使用便捷函數 compute_similarity() ⭐ 推薦")
    print("=" * 80)
    
    similarity = compute_similarity(individual1, individual2)
    distance = compute_ted(individual1, individual2)
    
    print(f"\n結果:")
    print(f"  編輯距離: {distance:.2f}")
    print(f"  相似度: {similarity:.4f}")
    
    if similarity >= 0.7:
        print(f"  解釋: 🟢 非常相似")
    elif similarity >= 0.5:
        print(f"  解釋: 🟡 中等相似")
    else:
        print(f"  解釋: 🔴 不太相似")
    
    # ========================================================================
    # 方式 2: 使用 TreeEditDistance 類（更靈活）
    # ========================================================================
    print("\n" + "=" * 80)
    print("方式 2: 使用 TreeEditDistance 類（更靈活）")
    print("=" * 80)
    
    # 轉換為 TreeNode
    tree1 = deap_to_tree_node(individual1)
    tree2 = deap_to_tree_node(individual2)
    
    print(f"\n轉換後的樹結構:")
    print(f"  Tree 1: {tree_node_to_bracket(tree1)}")
    print(f"  Tree 2: {tree_node_to_bracket(tree2)}")
    
    # 創建 TED 計算器
    ted = TreeEditDistance()
    
    # 計算距離和相似度
    distance2 = ted.compute(tree1, tree2)
    similarity2 = ted.compute_similarity(tree1, tree2)
    
    print(f"\n結果:")
    print(f"  編輯距離: {distance2:.2f}")
    print(f"  相似度: {similarity2:.4f}")
    
    # ========================================================================
    # 批次計算多個 Individual
    # ========================================================================
    print("\n" + "=" * 80)
    print("批次計算：多個 Individual 之間的相似度")
    print("=" * 80)
    
    # 創建一個小族群
    population = [toolbox.individual() for _ in range(4)]
    
    print(f"\n族群大小: {len(population)}")
    print(f"Individual 列表:")
    for i, ind in enumerate(population):
        print(f"  [{i}] {ind}")
    
    # 計算相似度矩陣
    print(f"\n相似度矩陣:")
    print(f"{'':>10}", end="")
    for i in range(len(population)):
        print(f"{i:>8}", end="")
    print()
    
    for i, ind1 in enumerate(population):
        print(f"{i:>10}", end="")
        for j, ind2 in enumerate(population):
            sim = compute_similarity(ind1, ind2)
            print(f"{sim:>8.4f}", end="")
        print()
    
    # ========================================================================
    # 找出最相似和最不相似的配對
    # ========================================================================
    print("\n" + "=" * 80)
    print("分析：找出最相似和最不相似的配對")
    print("=" * 80)
    
    max_sim = 0.0
    min_sim = 1.0
    max_pair = (0, 0)
    min_pair = (0, 0)
    
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            sim = compute_similarity(population[i], population[j])
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)
            if sim < min_sim:
                min_sim = sim
                min_pair = (i, j)
    
    print(f"\n最相似的配對:")
    print(f"  Individual [{max_pair[0]}]: {population[max_pair[0]]}")
    print(f"  Individual [{max_pair[1]}]: {population[max_pair[1]]}")
    print(f"  相似度: {max_sim:.4f}")
    
    print(f"\n最不相似的配對:")
    print(f"  Individual [{min_pair[0]}]: {population[min_pair[0]]}")
    print(f"  Individual [{min_pair[1]}]: {population[min_pair[1]]}")
    print(f"  相似度: {min_sim:.4f}")
    
    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 總結")
    print("=" * 80)
    print("\n✨ DEAP Individual 可以直接使用！")
    print("\n推薦使用方式:")
    print("  1. 單次計算: compute_similarity(ind1, ind2)")
    print("  2. 批次計算: 使用 for 迴圈遍歷族群")
    print("  3. 相似度矩陣: 計算所有配對的相似度")
    print("\n下一步:")
    print("  - 實作 SimilarityMatrix 類自動化批次計算")
    print("  - 實作並行計算加速大規模族群")
    print("  - 實作視覺化工具展示相似度分佈")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
