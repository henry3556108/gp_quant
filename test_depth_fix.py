"""
測試深度限制修復是否有效
運行一個小規模實驗並檢查所有 generation 的深度
"""
import subprocess
import os
import dill
from deap import creator, base, gp
from gp_quant.gp.operators import pset

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("測試深度限制修復")
print("="*100)

# 運行一個小規模實驗
print("\n步驟 1: 運行小規模實驗（1 ticker, 1 run, 50 generations）")
print("-"*100)

ticker = "ABX.TO"
test_dir = "test_depth_fix_results"
individual_records_dir = f"{test_dir}/individual_records_test"

# 清理舊的測試結果
if os.path.exists(test_dir):
    import shutil
    shutil.rmtree(test_dir)

os.makedirs(test_dir, exist_ok=True)

# 運行實驗
result = subprocess.run([
    'python', 'main.py',
    '--tickers', ticker,
    '--mode', 'portfolio',
    '--generations', '50',
    '--population', '100',  # 較小的族群以加快測試
    '--train_data_start', '1997-06-25',
    '--train_backtest_start', '1998-06-22',
    '--train_backtest_end', '1999-06-25',
    '--test_data_start', '1998-07-07',
    '--test_backtest_start', '1999-06-28',
    '--test_backtest_end', '2000-06-30',
    '--individual_records_dir', individual_records_dir
], capture_output=True, text=True)

if result.returncode != 0:
    print("❌ 實驗運行失敗！")
    print(result.stderr)
    exit(1)

print("✅ 實驗運行完成")

# 檢查深度
print("\n步驟 2: 檢查所有 generation 的深度")
print("-"*100)

violations = []
gen_stats = []

for gen in range(51):
    gen_dir = os.path.join(individual_records_dir, f"generation_{gen:03d}")
    population_file = os.path.join(gen_dir, "population.pkl")
    
    if not os.path.exists(population_file):
        print(f"⚠️  Generation {gen}: 檔案不存在")
        continue
    
    try:
        with open(population_file, 'rb') as f:
            population = dill.load(f)
        
        depths = [ind.height for ind in population]
        min_depth = min(depths)
        max_depth = max(depths)
        avg_depth = sum(depths) / len(depths)
        
        # 檢查是否符合限制
        if gen == 0:
            expected_max = 6
            compliant = max_depth <= expected_max
        else:
            expected_max = 17
            compliant = max_depth <= expected_max
        
        status = "✅" if compliant else "❌"
        
        gen_stats.append({
            'gen': gen,
            'min': min_depth,
            'max': max_depth,
            'avg': avg_depth,
            'expected_max': expected_max,
            'compliant': compliant
        })
        
        print(f"Gen {gen:2d}: min={min_depth:2d}, max={max_depth:3d}, avg={avg_depth:5.2f}, "
              f"expected_max={expected_max:2d} {status}")
        
        if not compliant:
            violations.append({
                'generation': gen,
                'max_depth': max_depth,
                'expected_max': expected_max
            })
    
    except Exception as e:
        print(f"❌ Generation {gen}: 載入失敗 - {e}")

# 總結
print("\n" + "="*100)
print("測試結果總結")
print("="*100)

total_gens = len(gen_stats)
compliant_gens = sum(1 for g in gen_stats if g['compliant'])
violation_count = len(violations)

print(f"\n總 generation 數: {total_gens}")
print(f"符合限制: {compliant_gens} ({compliant_gens/total_gens*100:.2f}%)")
print(f"違反限制: {violation_count} ({violation_count/total_gens*100:.2f}%)")

if violation_count == 0:
    print("\n🎉 ✅ 修復成功！所有 generation 都符合深度限制！")
    print("\n建議:")
    print("  1. 可以重新運行完整實驗")
    print("  2. 重新運行後，結果應該符合論文要求")
else:
    print(f"\n❌ 仍有 {violation_count} 個 generation 違反深度限制")
    print("\n違規詳情:")
    for v in violations:
        print(f"  Generation {v['generation']}: max_depth={v['max_depth']} > {v['expected_max']}")
    print("\n需要進一步調查問題")

# 顯示深度演化趨勢
print("\n" + "="*100)
print("深度演化趨勢")
print("="*100)

print("\nGeneration 0-10:")
for g in gen_stats[:11]:
    print(f"  Gen {g['gen']:2d}: max={g['max']:3d}, avg={g['avg']:5.2f}")

print("\nGeneration 40-50:")
for g in gen_stats[40:]:
    print(f"  Gen {g['gen']:2d}: max={g['max']:3d}, avg={g['avg']:5.2f}")

# 清理測試文件（可選）
print("\n" + "="*100)
cleanup = input("是否刪除測試文件？(y/n): ")
if cleanup.lower() == 'y':
    import shutil
    shutil.rmtree(test_dir)
    print("✅ 測試文件已刪除")
else:
    print(f"📁 測試文件保留在: {test_dir}")

print("\n" + "="*100)
print("測試完成")
print("="*100)
