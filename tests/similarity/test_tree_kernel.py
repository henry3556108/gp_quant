"""
Tree Kernel 單元測試

Tests for SubtreeKernel and related functions.
"""

import sys
import time
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gp_quant.similarity.tree_edit_distance import TreeNode, TreeEditDistance
from gp_quant.similarity.tree_kernel import (
    SubtreeKernel,
    compute_subtree_kernel,
    compute_kernel_similarity,
    compute_kernel_distance
)


def test_identical_trees_kernel():
    """測試相同樹的 kernel 值"""
    tree1 = TreeNode("a", [TreeNode("b"), TreeNode("c")])
    tree2 = TreeNode("a", [TreeNode("b"), TreeNode("c")])
    
    kernel = SubtreeKernel(lambda_decay=0.5)
    k_value = kernel.compute(tree1, tree2)
    k_self = kernel.compute(tree1, tree1)
    similarity = kernel.compute_normalized(tree1, tree2)
    distance = kernel.compute_distance(tree1, tree2)
    
    print(f"測試 1: 相同的樹")
    print(f"  Tree 1: {tree1}")
    print(f"  Tree 2: {tree2}")
    print(f"  Kernel K(T1,T2): {k_value:.4f}")
    print(f"  Kernel K(T1,T1): {k_self:.4f}")
    print(f"  Normalized Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    assert similarity == 1.0, f"相同的樹 normalized similarity 應該為 1.0，但得到 {similarity}"
    assert distance == 0.0, f"相同的樹距離應該為 0，但得到 {distance}"
    print("  ✅ 通過\n")


def test_single_node_difference():
    """測試只有一個節點不同"""
    tree1 = TreeNode("a", [TreeNode("b"), TreeNode("c")])
    tree2 = TreeNode("a", [TreeNode("b"), TreeNode("d")])  # c -> d
    
    kernel = SubtreeKernel(lambda_decay=0.5)
    similarity = kernel.compute_normalized(tree1, tree2)
    distance = kernel.compute_distance(tree1, tree2)
    
    print(f"測試 2: 只有一個節點不同")
    print(f"  Tree 1: {tree1}")
    print(f"  Tree 2: {tree2}")
    print(f"  Normalized Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    assert 0 < similarity < 1, f"部分相同的樹相似度應該在 0-1 之間，但得到 {similarity}"
    assert distance > 0, f"不同的樹距離應該 > 0，但得到 {distance}"
    print("  ✅ 通過\n")


def test_completely_different_trees():
    """測試完全不同的樹"""
    tree1 = TreeNode("a", [TreeNode("b")])
    tree2 = TreeNode("x", [TreeNode("y", [TreeNode("z")])])
    
    kernel = SubtreeKernel(lambda_decay=0.5)
    k_value = kernel.compute(tree1, tree2)
    similarity = kernel.compute_normalized(tree1, tree2)
    distance = kernel.compute_distance(tree1, tree2)
    
    print(f"測試 3: 完全不同的樹")
    print(f"  Tree 1: {tree1}")
    print(f"  Tree 2: {tree2}")
    print(f"  Kernel K(T1,T2): {k_value:.4f}")
    print(f"  Normalized Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    assert k_value == 0.0, f"完全不同的樹 kernel 應該為 0，但得到 {k_value}"
    assert similarity == 0.0, f"完全不同的樹相似度應該為 0，但得到 {similarity}"
    assert distance > 0, f"完全不同的樹距離應該 > 0，但得到 {distance}"
    print("  ✅ 通過\n")


def test_kernel_to_distance_conversion():
    """測試 kernel 到距離的轉換正確性"""
    tree1 = TreeNode("add", [TreeNode("x"), TreeNode("y")])
    tree2 = TreeNode("add", [TreeNode("x"), TreeNode("z")])
    
    kernel = SubtreeKernel(lambda_decay=0.5)
    
    k11 = kernel.compute(tree1, tree1)
    k22 = kernel.compute(tree2, tree2)
    k12 = kernel.compute(tree1, tree2)
    
    # 手動計算距離
    expected_distance = (k11 + k22 - 2 * k12) ** 0.5
    actual_distance = kernel.compute_distance(tree1, tree2)
    
    print(f"測試 4: Kernel 到距離轉換")
    print(f"  K(T1,T1): {k11:.4f}")
    print(f"  K(T2,T2): {k22:.4f}")
    print(f"  K(T1,T2): {k12:.4f}")
    print(f"  Expected distance: {expected_distance:.4f}")
    print(f"  Actual distance: {actual_distance:.4f}")
    
    assert abs(expected_distance - actual_distance) < 1e-6, \
        f"距離計算不正確: expected {expected_distance}, got {actual_distance}"
    print("  ✅ 通過\n")


def test_lambda_decay_effect():
    """測試 λ 衰減因子的效果"""
    tree1 = TreeNode("a", [TreeNode("b", [TreeNode("c")])])
    tree2 = TreeNode("a", [TreeNode("b", [TreeNode("c")])])
    
    kernel_high_lambda = SubtreeKernel(lambda_decay=0.9)
    kernel_low_lambda = SubtreeKernel(lambda_decay=0.1)
    
    k_high = kernel_high_lambda.compute(tree1, tree2)
    k_low = kernel_low_lambda.compute(tree1, tree2)
    
    print(f"測試 5: Lambda 衰減因子效果")
    print(f"  Tree: {tree1}")
    print(f"  K(λ=0.9): {k_high:.4f}")
    print(f"  K(λ=0.1): {k_low:.4f}")
    
    assert k_high > k_low, f"較高的 λ 應該產生較大的 kernel 值，但 {k_high} <= {k_low}"
    print("  ✅ 通過\n")


def test_complex_tree():
    """測試複雜樹結構"""
    # GP-like tree: add(mul(x, y), x)
    tree1 = TreeNode("add", [
        TreeNode("mul", [TreeNode("x"), TreeNode("y")]),
        TreeNode("x")
    ])
    
    # Similar tree: add(mul(x, y), y)
    tree2 = TreeNode("add", [
        TreeNode("mul", [TreeNode("x"), TreeNode("y")]),
        TreeNode("y")
    ])
    
    kernel = SubtreeKernel(lambda_decay=0.5)
    similarity = kernel.compute_normalized(tree1, tree2)
    distance = kernel.compute_distance(tree1, tree2)
    
    print(f"測試 6: 複雜 GP 樹結構")
    print(f"  Tree 1: {tree1}")
    print(f"  Tree 2: {tree2}")
    print(f"  Normalized Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    assert 0.3 < similarity < 0.9, f"結構相似但部分不同的樹相似度應該中等，但得到 {similarity}"
    print("  ✅ 通過\n")


def test_performance_vs_ted():
    """效能比較：Tree Kernel vs TED"""
    # 建立較大的測試樹
    def build_deep_tree(depth: int, label_prefix: str = "n") -> TreeNode:
        if depth == 0:
            return TreeNode(f"{label_prefix}_leaf")
        return TreeNode(f"{label_prefix}_{depth}", [
            build_deep_tree(depth - 1, f"{label_prefix}L"),
            build_deep_tree(depth - 1, f"{label_prefix}R")
        ])
    
    tree1 = build_deep_tree(4, "A")
    tree2 = build_deep_tree(4, "B")
    
    # 計算節點數
    def count_nodes(node: TreeNode) -> int:
        return 1 + sum(count_nodes(c) for c in node.children)
    
    n1, n2 = count_nodes(tree1), count_nodes(tree2)
    
    print(f"測試 7: 效能比較 (Tree Kernel vs TED)")
    print(f"  Tree 1 nodes: {n1}")
    print(f"  Tree 2 nodes: {n2}")
    
    # Benchmark Tree Kernel
    kernel = SubtreeKernel(lambda_decay=0.5)
    start = time.perf_counter()
    for _ in range(100):
        kernel.compute_distance(tree1, tree2)
    kernel_time = (time.perf_counter() - start) / 100
    
    # Benchmark TED
    ted = TreeEditDistance()
    start = time.perf_counter()
    for _ in range(100):
        ted.compute(tree1, tree2)
    ted_time = (time.perf_counter() - start) / 100
    
    speedup = ted_time / kernel_time if kernel_time > 0 else float('inf')
    
    print(f"  Tree Kernel avg time: {kernel_time*1000:.4f} ms")
    print(f"  TED avg time: {ted_time*1000:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    
    assert kernel_time < ted_time, f"Tree Kernel 應該比 TED 快，但 {kernel_time} >= {ted_time}"
    print("  ✅ 通過\n")


def test_convenience_functions():
    """測試便捷函數"""
    tree1 = TreeNode("a", [TreeNode("b")])
    tree2 = TreeNode("a", [TreeNode("c")])
    
    k = compute_subtree_kernel(tree1, tree2)
    sim = compute_kernel_similarity(tree1, tree2)
    dist = compute_kernel_distance(tree1, tree2)
    
    print(f"測試 8: 便捷函數")
    print(f"  compute_subtree_kernel: {k:.4f}")
    print(f"  compute_kernel_similarity: {sim:.4f}")
    print(f"  compute_kernel_distance: {dist:.4f}")
    
    assert isinstance(k, float), "compute_subtree_kernel 應返回 float"
    assert isinstance(sim, float), "compute_kernel_similarity 應返回 float"
    assert isinstance(dist, float), "compute_kernel_distance 應返回 float"
    assert 0 <= sim <= 1, f"Similarity 應在 [0,1]，但得到 {sim}"
    print("  ✅ 通過\n")


def test_large_scale_matrix_benchmark():
    """
    大規模 Distance Matrix 效能測試
    
    模擬 niching 場景：計算 population 的 pairwise distance matrix
    測試規模：1000 x 1000 (499,500 pairs)
    """
    import random
    
    def build_random_gp_tree(max_depth: int = 4) -> TreeNode:
        """Build random GP-like tree."""
        ops = ['add', 'sub', 'mul', 'div', 'gt', 'lt', 'and_', 'or_']
        terminals = ['open', 'high', 'low', 'close', 'volume', 'x', 'y', 'c1', 'c2']
        
        if max_depth == 0 or (max_depth < 3 and random.random() < 0.3):
            return TreeNode(random.choice(terminals))
        
        return TreeNode(random.choice(ops), [
            build_random_gp_tree(max_depth - 1),
            build_random_gp_tree(max_depth - 1)
        ])
    
    def count_nodes(node: TreeNode) -> int:
        return 1 + sum(count_nodes(c) for c in node.children)
    
    print(f"測試 9: 大規模 Distance Matrix 效能測試")
    print(f"=" * 60)
    
    # Generate population
    pop_size = 100  # Use smaller sample for actual calculation
    target_pop = 1000  # Target population for estimation
    
    random.seed(42)
    print(f"  生成 {pop_size} 棵隨機 GP 樹...")
    population = [build_random_gp_tree(max_depth=5) for _ in range(pop_size)]
    
    avg_nodes = sum(count_nodes(t) for t in population) / len(population)
    print(f"  平均樹大小: {avg_nodes:.1f} nodes")
    
    # Sample pairs for benchmark
    num_sample_pairs = 500
    pairs = [(random.randint(0, pop_size-1), random.randint(0, pop_size-1)) 
             for _ in range(num_sample_pairs)]
    
    print(f"\n  Benchmarking with {num_sample_pairs} sample pairs...")
    
    # Benchmark Tree Kernel
    kernel = SubtreeKernel(lambda_decay=0.5)
    start = time.perf_counter()
    for i, j in pairs:
        kernel.compute_distance(population[i], population[j])
    kernel_time = time.perf_counter() - start
    kernel_per_pair = kernel_time / num_sample_pairs
    
    # Benchmark TED
    ted = TreeEditDistance()
    start = time.perf_counter()
    for i, j in pairs:
        ted.compute(population[i], population[j])
    ted_time = time.perf_counter() - start
    ted_per_pair = ted_time / num_sample_pairs
    
    speedup = ted_per_pair / kernel_per_pair if kernel_per_pair > 0 else float('inf')
    
    print(f"\n  --- Sample Results ({num_sample_pairs} pairs) ---")
    print(f"  Tree Kernel: {kernel_time:.3f}s ({kernel_per_pair*1000:.3f} ms/pair)")
    print(f"  TED:         {ted_time:.3f}s ({ted_per_pair*1000:.3f} ms/pair)")
    print(f"  Speedup:     {speedup:.2f}x")
    
    # Estimate full matrix time for different population sizes
    print(f"\n  --- 預估完整 Distance Matrix 時間 ---")
    print(f"  {'Population':<12} {'Pairs':>12} {'Tree Kernel':>15} {'TED':>15} {'Time Saved':>15}")
    print(f"  {'-'*12} {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    
    for pop in [200, 500, 1000]:
        num_pairs = (pop * (pop - 1)) // 2
        kernel_est = kernel_per_pair * num_pairs
        ted_est = ted_per_pair * num_pairs
        saved = ted_est - kernel_est
        
        kernel_str = f"{kernel_est:.1f}s" if kernel_est < 60 else f"{kernel_est/60:.1f}min"
        ted_str = f"{ted_est:.1f}s" if ted_est < 60 else f"{ted_est/60:.1f}min"
        saved_str = f"{saved:.1f}s" if saved < 60 else f"{saved/60:.1f}min"
        
        print(f"  {pop:<12} {num_pairs:>12,} {kernel_str:>15} {ted_str:>15} {saved_str:>15}")
    
    print()
    assert kernel_per_pair < ted_per_pair, "Tree Kernel should be faster than TED"
    print("  ✅ 通過\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Tree Kernel 單元測試")
    print("=" * 80)
    print()
    
    test_identical_trees_kernel()
    test_single_node_difference()
    test_completely_different_trees()
    test_kernel_to_distance_conversion()
    test_lambda_decay_effect()
    test_complex_tree()
    test_performance_vs_ted()
    test_convenience_functions()
    test_large_scale_matrix_benchmark()
    
    print("=" * 80)
    print("✅ 所有測試通過！")
    print("=" * 80)
    print()
    print("總結:")
    print("  1. SubtreeKernel 正確實作")
    print("  2. 距離轉換公式正確")
    print("  3. Tree Kernel 比 TED 顯著更快")
    print("  4. 大規模 Matrix 計算顯著節省時間")


if __name__ == "__main__":
    main()

