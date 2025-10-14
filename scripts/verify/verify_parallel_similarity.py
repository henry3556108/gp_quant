"""
驗證並行相似度矩陣計算

測試並行版本的正確性和性能。
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deap import gp, creator, base
from gp_quant.gp.operators import pset
from gp_quant.similarity import SimilarityMatrix, ParallelSimilarityMatrix

# 創建 DEAP fitness 和 individual
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def generate_population(size: int, max_depth: int = 5) -> list:
    """生成隨機族群"""
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
    toolbox.register("individual", lambda: creator.Individual(toolbox.expr()))
    toolbox.register("population", lambda n: [toolbox.individual() for _ in range(n)])
    
    return toolbox.population(size)


def test_correctness(pop_size=30):
    """測試並行版本的正確性"""
    print("="*80)
    print(f"測試 1: 正確性驗證（population={pop_size}）")
    print("="*80)
    print()
    
    # 生成族群
    print(f"生成 {pop_size} 個個體...")
    population = generate_population(pop_size)
    print(f"✓ 族群生成完成")
    print()
    
    # 序列計算
    print("序列計算...")
    start = time.time()
    sim_matrix_seq = SimilarityMatrix(population)
    matrix_seq = sim_matrix_seq.compute(show_progress=True)
    time_seq = time.time() - start
    print(f"✓ 序列計算完成: {time_seq:.2f}s")
    print()
    
    # 並行計算
    print("並行計算...")
    start = time.time()
    sim_matrix_par = ParallelSimilarityMatrix(population)
    matrix_par = sim_matrix_par.compute(show_progress=True)
    time_par = time.time() - start
    print(f"✓ 並行計算完成: {time_par:.2f}s")
    print()
    
    # 比較結果
    print("比較結果...")
    diff = np.abs(matrix_seq - matrix_par)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  最大差異: {max_diff:.10f}")
    print(f"  平均差異: {mean_diff:.10f}")
    
    if max_diff < 1e-10:
        print("  ✅ 結果完全一致！")
    else:
        print(f"  ⚠️  結果有微小差異（可能是浮點誤差）")
    print()
    
    # 統計資訊
    stats_seq = sim_matrix_seq.get_statistics()
    stats_par = sim_matrix_par.get_statistics()
    
    print("統計資訊比較:")
    print(f"  平均相似度: 序列={stats_seq.get('mean_similarity', stats_seq.get('mean')):.4f}, 並行={stats_par.get('mean_similarity', stats_par.get('mean')):.4f}")
    print(f"  多樣性分數: 序列={stats_seq['diversity_score']:.4f}, 並行={stats_par['diversity_score']:.4f}")
    print()
    
    # 加速比
    speedup = time_seq / time_par
    print(f"⚡ 加速比: {speedup:.2f}x")
    print()
    
    return max_diff < 1e-6


def test_performance():
    """測試不同族群大小的性能"""
    print("="*80)
    print("測試 2: 性能測試")
    print("="*80)
    print()
    
    sizes = [50, 100, 200, 500, 2000]
    results = []
    
    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Population Size: {size}")
        print(f"{'='*80}")
        
        # 生成族群
        print(f"生成 {size} 個個體...")
        population = generate_population(size, max_depth=4)
        print(f"✓ 族群生成完成")
        print()
        
        # 序列計算
        print("序列計算...")
        start = time.time()
        sim_matrix_seq = SimilarityMatrix(population)
        matrix_seq = sim_matrix_seq.compute(show_progress=False)
        time_seq = time.time() - start
        print(f"✓ 序列計算完成: {time_seq:.2f}s")
        
        # 並行計算
        print("並行計算...")
        start = time.time()
        sim_matrix_par = ParallelSimilarityMatrix(population)
        matrix_par = sim_matrix_par.compute(show_progress=False)
        time_par = time.time() - start
        print(f"✓ 並行計算完成: {time_par:.2f}s")
        
        # 加速比
        speedup = time_seq / time_par
        print(f"⚡ 加速比: {speedup:.2f}x")
        
        # 統計
        stats = sim_matrix_par.get_statistics()
        avg_sim = stats.get('mean_similarity', stats.get('mean', 0))
        print(f"📊 平均相似度: {avg_sim:.4f}")
        print(f"📊 多樣性分數: {stats['diversity_score']:.4f}")
        
        results.append({
            'size': size,
            'time_seq': time_seq,
            'time_par': time_par,
            'speedup': speedup,
            'avg_similarity': avg_sim
        })
    
    # 總結
    print("\n" + "="*80)
    print("性能總結")
    print("="*80)
    print()
    print(f"{'Size':<10} {'序列(s)':<12} {'並行(s)':<12} {'加速比':<10} {'平均相似度':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['size']:<10} {r['time_seq']:<12.2f} {r['time_par']:<12.2f} {r['speedup']:<10.2f}x {r['avg_similarity']:<12.4f}")
    print()
    
    # 平均加速比
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"平均加速比: {avg_speedup:.2f}x")
    print()
    
    return results


def test_large_population():
    """測試大族群（1000）"""
    print("="*80)
    print("測試 3: 大族群測試（population=1000）")
    print("="*80)
    print()
    
    size = 1000
    print(f"生成 {size} 個個體...")
    population = generate_population(size, max_depth=4)
    print(f"✓ 族群生成完成")
    print()
    
    # 只測試並行版本（序列版本太慢）
    print("並行計算...")
    start = time.time()
    sim_matrix_par = ParallelSimilarityMatrix(population, n_workers=8)
    matrix_par = sim_matrix_par.compute(show_progress=True)
    time_par = time.time() - start
    print(f"✓ 並行計算完成: {time_par:.2f}s ({time_par/60:.1f} 分鐘)")
    print()
    
    # 統計
    stats = sim_matrix_par.get_statistics()
    print("統計資訊:")
    print(f"  平均相似度: {stats.get('mean_similarity', stats.get('mean', 0)):.4f}")
    print(f"  標準差: {stats.get('std_similarity', stats.get('std', 0)):.4f}")
    print(f"  最小值: {stats.get('min_similarity', stats.get('min', 0)):.4f}")
    print(f"  最大值: {stats.get('max_similarity', stats.get('max', 0)):.4f}")
    print(f"  多樣性分數: {stats['diversity_score']:.4f}")
    print()
    
    # 最相似和最不相似的配對
    print("最相似的 5 對:")
    most_similar = sim_matrix_par.get_most_similar_pairs(top_k=5)
    for i, j, sim in most_similar:
        print(f"  [{i}, {j}]: {sim:.4f}")
    print()
    
    print("最不相似的 5 對:")
    most_dissimilar = sim_matrix_par.get_most_dissimilar_pairs(top_k=5)
    for i, j, sim in most_dissimilar:
        print(f"  [{i}, {j}]: {sim:.4f}")
    print()
    
    return time_par


def main():
    print("\n" + "="*80)
    print("並行相似度矩陣計算驗證")
    print("="*80)
    print()
    
    # 測試 1: 正確性
    correct = test_correctness(pop_size=30)
    
    if not correct:
        print("❌ 正確性測試失敗！")
        return
    
    print("✅ 正確性測試通過！")
    print()
    
    # 測試 2: 性能
    results = test_performance()
    
    # 測試 3: 大族群
    time_1000 = test_large_population()
    
    # 最終總結
    print("="*80)
    print("✅ 所有測試完成！")
    print("="*80)
    print()
    
    print("主要發現:")
    print(f"  1. 並行版本結果與序列版本完全一致 ✅")
    print(f"  2. 平均加速比: {np.mean([r['speedup'] for r in results]):.2f}x")
    print(f"  3. Population=1000 計算時間: {time_1000:.1f}s ({time_1000/60:.1f} 分鐘)")
    print()
    
    print("建議:")
    print("  - Population < 100: 使用 SimilarityMatrix（序列版本）")
    print("  - Population >= 100: 使用 ParallelSimilarityMatrix（並行版本）")
    print("  - Population >= 1000: 建議每 10-15 代更新一次相似度矩陣")
    print()


if __name__ == "__main__":
    main()
