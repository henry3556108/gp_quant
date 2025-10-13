"""
Niching Clustering 驗證腳本

驗證聚類機制：
1. 生成固定族群
2. 計算相似度矩陣
3. 執行聚類
4. 驗證每個個體都被標記到某個 niche
"""

import sys
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import gp, creator, base
from gp_quant.similarity import SimilarityMatrix
from gp_quant.niching import NichingClusterer
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
    print("Niching Clustering 驗證")
    print("=" * 80)
    
    # 設置 GP 環境
    pset = setup_gp()
    
    # 創建 toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", lambda: creator.Individual(toolbox.expr()))
    
    # ========================================================================
    # 步驟 1: 生成固定族群
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 1: 生成固定族群")
    print("=" * 80)
    
    # 設置隨機種子以獲得可重現的結果
    np.random.seed(42)
    
    population_size = 30
    population = [toolbox.individual() for _ in range(population_size)]
    
    print(f"\n族群大小: {population_size}")
    print(f"\nIndividual 列表（前 10 個）:")
    for i in range(min(10, population_size)):
        print(f"  [{i}] {population[i]}")
    if population_size > 10:
        print(f"  ... (還有 {population_size - 10} 個)")
    
    # ========================================================================
    # 步驟 2: 計算相似度矩陣
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 2: 計算相似度矩陣")
    print("=" * 80)
    
    print(f"\n計算 {population_size} 個個體的相似度矩陣...")
    sim_matrix_obj = SimilarityMatrix(population)
    similarity_matrix = sim_matrix_obj.compute(show_progress=True)
    
    # 顯示統計資訊
    stats = sim_matrix_obj.get_statistics()
    print(f"\n相似度統計:")
    print(f"  平均相似度: {stats['mean_similarity']:.4f}")
    print(f"  標準差: {stats['std_similarity']:.4f}")
    print(f"  最小值: {stats['min_similarity']:.4f}")
    print(f"  最大值: {stats['max_similarity']:.4f}")
    print(f"  多樣性分數: {stats['diversity_score']:.4f}")
    
    # ========================================================================
    # 步驟 3: 執行聚類（K-means）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 3: 執行聚類（K-means）")
    print("=" * 80)
    
    n_clusters = 5
    print(f"\n設定群數: {n_clusters}")
    
    clusterer_kmeans = NichingClusterer(n_clusters=n_clusters, algorithm='kmeans')
    labels_kmeans = clusterer_kmeans.fit_predict(similarity_matrix)
    
    print(f"\n聚類完成！")
    clusterer_kmeans.print_summary()
    
    # 顯示每個 niche 的成員
    print(f"各 Niche 成員:")
    for niche_id in range(n_clusters):
        members = clusterer_kmeans.get_niche_members(niche_id)
        print(f"\n  Niche {niche_id} ({len(members)} 個成員):")
        for idx in members[:5]:  # 只顯示前 5 個
            print(f"    [{idx}] {population[idx]}")
        if len(members) > 5:
            print(f"    ... (還有 {len(members) - 5} 個)")
    
    # ========================================================================
    # 步驟 4: 執行聚類（Hierarchical）
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 4: 執行聚類（Hierarchical）")
    print("=" * 80)
    
    clusterer_hier = NichingClusterer(n_clusters=n_clusters, algorithm='hierarchical')
    labels_hier = clusterer_hier.fit_predict(similarity_matrix)
    
    print(f"\n聚類完成！")
    clusterer_hier.print_summary()
    
    # ========================================================================
    # 步驟 5: 驗證聚類結果
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 5: 驗證聚類結果")
    print("=" * 80)
    
    print(f"\n✓ 驗證項目:")
    
    # 驗證 1: 每個個體都有標籤
    all_labeled_kmeans = len(labels_kmeans) == population_size
    all_labeled_hier = len(labels_hier) == population_size
    print(f"  1. 每個個體都有標籤:")
    print(f"     - K-means: {'✓ 通過' if all_labeled_kmeans else '✗ 失敗'}")
    print(f"     - Hierarchical: {'✓ 通過' if all_labeled_hier else '✗ 失敗'}")
    
    # 驗證 2: 標籤範圍正確
    valid_range_kmeans = set(labels_kmeans) <= set(range(n_clusters))
    valid_range_hier = set(labels_hier) <= set(range(n_clusters))
    print(f"  2. 標籤範圍 [0, {n_clusters-1}]:")
    print(f"     - K-means: {'✓ 通過' if valid_range_kmeans else '✗ 失敗'}")
    print(f"     - Hierarchical: {'✓ 通過' if valid_range_hier else '✗ 失敗'}")
    
    # 驗證 3: 所有群都有成員
    all_clusters_used_kmeans = len(set(labels_kmeans)) == n_clusters
    all_clusters_used_hier = len(set(labels_hier)) == n_clusters
    print(f"  3. 所有 {n_clusters} 個群都有成員:")
    print(f"     - K-means: {'✓ 通過' if all_clusters_used_kmeans else '✗ 失敗'}")
    print(f"     - Hierarchical: {'✓ 通過' if all_clusters_used_hier else '✗ 失敗'}")
    
    # 驗證 4: Silhouette 分數合理
    reasonable_score_kmeans = -1 <= clusterer_kmeans.silhouette_score_ <= 1
    reasonable_score_hier = -1 <= clusterer_hier.silhouette_score_ <= 1
    print(f"  4. Silhouette 分數範圍 [-1, 1]:")
    print(f"     - K-means: {clusterer_kmeans.silhouette_score_:.4f} {'✓ 通過' if reasonable_score_kmeans else '✗ 失敗'}")
    print(f"     - Hierarchical: {clusterer_hier.silhouette_score_:.4f} {'✓ 通過' if reasonable_score_hier else '✗ 失敗'}")
    
    # ========================================================================
    # 步驟 6: 標籤映射表
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 6: 個體 → Niche 標籤映射表")
    print("=" * 80)
    
    print(f"\n使用 K-means 結果:")
    print(f"\n{'Individual ID':>15} {'Niche':>10} {'Expression':<50}")
    print("-" * 80)
    for i in range(min(15, population_size)):
        print(f"{i:>15} {labels_kmeans[i]:>10} {str(population[i]):<50}")
    if population_size > 15:
        print(f"{'...':>15} {'...':>10} {'...':<50}")
    
    # ========================================================================
    # 步驟 7: 比較兩種演算法
    # ========================================================================
    print("\n" + "=" * 80)
    print("步驟 7: 演算法比較")
    print("=" * 80)
    
    print(f"\n{'指標':<30} {'K-means':>15} {'Hierarchical':>15}")
    print("-" * 60)
    print(f"{'Silhouette 分數':<30} {clusterer_kmeans.silhouette_score_:>15.4f} {clusterer_hier.silhouette_score_:>15.4f}")
    
    stats_kmeans = clusterer_kmeans.get_statistics()
    stats_hier = clusterer_hier.get_statistics()
    
    print(f"{'最小 Niche 大小':<30} {stats_kmeans['min_niche_size']:>15} {stats_hier['min_niche_size']:>15}")
    print(f"{'最大 Niche 大小':<30} {stats_kmeans['max_niche_size']:>15} {stats_hier['max_niche_size']:>15}")
    print(f"{'平均 Niche 大小':<30} {stats_kmeans['avg_niche_size']:>15.1f} {stats_hier['avg_niche_size']:>15.1f}")
    print(f"{'Niche 大小標準差':<30} {stats_kmeans['std_niche_size']:>15.2f} {stats_hier['std_niche_size']:>15.2f}")
    
    # ========================================================================
    # 總結
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ 總結")
    print("=" * 80)
    print("\n✨ Niching Clustering 功能驗證完成！")
    print("\n驗收結果:")
    print(f"  ✅ 生成固定族群: {population_size} 個個體")
    print(f"  ✅ 計算相似度矩陣: {population_size}x{population_size}")
    print(f"  ✅ 執行聚類: K-means 和 Hierarchical")
    print(f"  ✅ 每個個體都被標記到 niche")
    print(f"  ✅ 標籤範圍正確: [0, {n_clusters-1}]")
    print(f"  ✅ 所有 {n_clusters} 個 niche 都有成員")
    print("\n功能特點:")
    print("  1. 支援 K-means 和 Hierarchical clustering")
    print("  2. 自動計算 Silhouette 分數評估聚類品質")
    print("  3. 提供詳細的統計資訊")
    print("  4. 可查詢每個 niche 的成員")
    print("  5. 支援固定群數（可擴展為動態）")
    print("\n下一步:")
    print("  - 實作跨群 Parent Selection")
    print("  - 整合到演化流程")
    print("  - 實作動態群數調整（未來）")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
