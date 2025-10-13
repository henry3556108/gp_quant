"""
驗證 Tree Edit Distance 與相似度計算

測試兩組實驗：
1. 相似個體：兩個長得很像的 GP tree
2. 不相似個體：兩個長得不像的 GP tree
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import gp, creator, base
from gp_quant.similarity.tree_edit_distance import (
    TreeEditDistance,
    deap_to_tree_node,
    tree_node_to_bracket,
    compute_ted,
    compute_similarity
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
    pset.addPrimitive(lambda x, y: 1 if x > y else 0, 2, name="gt")
    pset.addPrimitive(lambda x, y: 1 if x < y else 0, 2, name="lt")
    
    pset.renameArguments(ARG0='x', ARG1='y')
    
    return pset


def create_tree_from_expr(pset, expr_str):
    """從表達式字符串創建 GP tree"""
    expr = gp.compile(expr_str, pset)
    tree = gp.PrimitiveTree.from_string(expr_str, pset)
    return creator.Individual(tree)


def print_tree_info(tree, label="Tree"):
    """打印樹的信息"""
    node = deap_to_tree_node(tree)
    bracket = tree_node_to_bracket(node)
    
    print(f"\n{label}:")
    print(f"  表達式: {tree}")
    print(f"  括號表示: {bracket}")
    print(f"  節點數: {len(tree)}")
    print(f"  高度: {tree.height}")


def interpret_similarity(similarity):
    """解釋相似度分數"""
    if similarity >= 0.9:
        return "非常相似 (Very Similar)"
    elif similarity >= 0.7:
        return "相似 (Similar)"
    elif similarity >= 0.5:
        return "中等相似 (Moderately Similar)"
    elif similarity >= 0.3:
        return "不太相似 (Somewhat Different)"
    else:
        return "非常不同 (Very Different)"


def experiment_1_similar_trees(pset):
    """實驗 1: 相似的兩棵樹"""
    print("=" * 80)
    print("實驗 1: 相似個體測試")
    print("=" * 80)
    
    # 創建兩棵非常相似的樹
    # Tree A: add(x, y)
    tree_a = creator.Individual([
        pset.primitiveMap['add'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    # Tree B: add(x, y) - 完全相同
    tree_b = creator.Individual([
        pset.primitiveMap['add'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    print_tree_info(tree_a, "Tree A")
    print_tree_info(tree_b, "Tree B")
    
    # 計算距離和相似度
    distance = compute_ted(tree_a, tree_b)
    similarity = compute_similarity(tree_a, tree_b)
    
    print(f"\n📊 結果:")
    print(f"  編輯距離 (TED): {distance:.2f}")
    print(f"  相似度分數: {similarity:.4f}")
    print(f"  相似程度: {interpret_similarity(similarity)}")
    
    # 驗證
    assert distance == 0.0, "相同的樹距離應該為 0"
    assert similarity == 1.0, "相同的樹相似度應該為 1.0"
    print(f"\n✅ 驗證通過：相同的樹具有最高相似度")
    
    return distance, similarity


def experiment_2_slightly_different_trees(pset):
    """實驗 1.5: 稍微不同的兩棵樹"""
    print("\n" + "=" * 80)
    print("實驗 1.5: 稍微不同的個體測試")
    print("=" * 80)
    
    # Tree A: add(x, y)
    tree_a = creator.Individual([
        pset.primitiveMap['add'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    # Tree B: sub(x, y) - 只有根節點不同
    tree_b = creator.Individual([
        pset.primitiveMap['sub'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    print_tree_info(tree_a, "Tree A")
    print_tree_info(tree_b, "Tree B")
    
    # 計算距離和相似度
    distance = compute_ted(tree_a, tree_b)
    similarity = compute_similarity(tree_a, tree_b)
    
    print(f"\n📊 結果:")
    print(f"  編輯距離 (TED): {distance:.2f}")
    print(f"  相似度分數: {similarity:.4f}")
    print(f"  相似程度: {interpret_similarity(similarity)}")
    
    # 驗證
    assert distance == 1.0, "只有根節點不同，距離應該為 1"
    assert similarity == 0.5, "只有根節點不同，相似度應該為 0.5"
    print(f"\n✅ 驗證通過：結構相同但根節點不同的樹具有中等相似度")
    
    return distance, similarity


def experiment_3_dissimilar_trees(pset):
    """實驗 2: 不相似的兩棵樹"""
    print("\n" + "=" * 80)
    print("實驗 2: 不相似個體測試")
    print("=" * 80)
    
    # Tree A: add(x, y) - 簡單的樹
    tree_a = creator.Individual([
        pset.primitiveMap['add'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    # Tree B: mul(div(x, sub(y, x)), gt(x, y)) - 複雜的樹
    tree_b = creator.Individual([
        pset.primitiveMap['mul'],
        pset.primitiveMap['div'],
        pset.arguments[0],  # ARG0 (x)
        pset.primitiveMap['sub'],
        pset.arguments[1],  # ARG1 (y)
        pset.arguments[0],  # ARG0 (x)
        pset.primitiveMap['gt'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1]   # ARG1 (y)
    ])
    
    print_tree_info(tree_a, "Tree A (簡單)")
    print_tree_info(tree_b, "Tree B (複雜)")
    
    # 計算距離和相似度
    distance = compute_ted(tree_a, tree_b)
    similarity = compute_similarity(tree_a, tree_b)
    
    print(f"\n📊 結果:")
    print(f"  編輯距離 (TED): {distance:.2f}")
    print(f"  相似度分數: {similarity:.4f}")
    print(f"  相似程度: {interpret_similarity(similarity)}")
    
    # 驗證
    assert distance > 5.0, "非常不同的樹距離應該較大"
    assert similarity < 0.2, "非常不同的樹相似度應該較低"
    print(f"\n✅ 驗證通過：結構完全不同的樹具有低相似度")
    
    return distance, similarity


def experiment_4_medium_similarity(pset):
    """實驗 3: 中等相似度的兩棵樹"""
    print("\n" + "=" * 80)
    print("實驗 3: 中等相似度個體測試")
    print("=" * 80)
    
    # Tree A: add(mul(x, y), x)
    tree_a = creator.Individual([
        pset.primitiveMap['add'],
        pset.primitiveMap['mul'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1],  # ARG1 (y)
        pset.arguments[0]   # ARG0 (x)
    ])
    
    # Tree B: add(mul(x, y), y) - 只有最後一個 terminal 不同
    tree_b = creator.Individual([
        pset.primitiveMap['add'],
        pset.primitiveMap['mul'],
        pset.arguments[0],  # ARG0 (x)
        pset.arguments[1],  # ARG1 (y)
        pset.arguments[1]   # ARG1 (y) - 這裡不同
    ])
    
    print_tree_info(tree_a, "Tree A")
    print_tree_info(tree_b, "Tree B")
    
    # 計算距離和相似度
    distance = compute_ted(tree_a, tree_b)
    similarity = compute_similarity(tree_a, tree_b)
    
    print(f"\n📊 結果:")
    print(f"  編輯距離 (TED): {distance:.2f}")
    print(f"  相似度分數: {similarity:.4f}")
    print(f"  相似程度: {interpret_similarity(similarity)}")
    
    # 驗證
    assert 0.3 < similarity < 0.7, "中等相似的樹相似度應該在中間範圍"
    print(f"\n✅ 驗證通過：結構相似但部分不同的樹具有中等相似度")
    
    return distance, similarity


def main():
    """主函數"""
    print("\n" + "🌳" * 40)
    print("Tree Edit Distance (TED) 與相似度計算驗證")
    print("🌳" * 40)
    
    # 設置 GP 環境
    pset = setup_gp()
    
    # 運行實驗
    results = []
    
    # 實驗 1: 完全相同的樹
    dist1, sim1 = experiment_1_similar_trees(pset)
    results.append(("完全相同", dist1, sim1))
    
    # 實驗 1.5: 稍微不同的樹
    dist2, sim2 = experiment_2_slightly_different_trees(pset)
    results.append(("稍微不同", dist2, sim2))
    
    # 實驗 2: 非常不同的樹
    dist3, sim3 = experiment_3_dissimilar_trees(pset)
    results.append(("非常不同", dist3, sim3))
    
    # 實驗 3: 中等相似度
    dist4, sim4 = experiment_4_medium_similarity(pset)
    results.append(("中等相似", dist4, sim4))
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 實驗總結")
    print("=" * 80)
    print(f"\n{'實驗類型':<15} {'編輯距離':>12} {'相似度':>12} {'相似程度':<25}")
    print("-" * 80)
    
    for exp_type, dist, sim in results:
        interpretation = interpret_similarity(sim)
        print(f"{exp_type:<15} {dist:>12.2f} {sim:>12.4f} {interpretation:<25}")
    
    print("\n" + "=" * 80)
    print("✅ 所有實驗驗證通過！")
    print("=" * 80)
    
    print("\n📝 結論:")
    print("  1. TED 演算法正確實作，能夠計算樹之間的編輯距離")
    print("  2. 相似度轉換公式有效，能夠將距離轉換為 [0, 1] 範圍的相似度")
    print("  3. 相似度分數符合直覺：")
    print("     - 完全相同的樹：相似度 = 1.0")
    print("     - 結構相似的樹：相似度 > 0.5")
    print("     - 完全不同的樹：相似度 < 0.2")
    print("  4. 可以清楚地區分不同程度的相似性")


if __name__ == "__main__":
    main()
