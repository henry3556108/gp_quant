"""
測試 cluster_labels 和 niching_info 的儲存與載入

這個腳本測試：
1. 向後相容性：能否正確載入舊格式的 pkl 檔案
2. 新格式儲存：能否正確儲存和載入包含 cluster_labels 的新格式
3. 工具函數：測試 generation_loader 中的各種工具函數
"""
import sys
from pathlib import Path
from deap import creator, base, gp
from gp_quant.gp.operators import pset
from gp_quant.utils import (
    load_generation,
    has_niching_info,
    get_niche_individuals,
    get_niche_statistics
)

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

print("="*100)
print("測試 cluster_labels 和 niching_info 的儲存與載入")
print("="*100)

# ============================================================================
# 測試 1：載入舊格式（沒有 cluster_labels）
# ============================================================================
print("\n" + "="*100)
print("測試 1：載入舊格式的 generation pkl")
print("="*100)

old_format_file = Path('portfolio_experiment_results/portfolio_exp_sharpe_20251023_125111/generations/generation_001.pkl')

if old_format_file.exists():
    print(f"\n載入檔案: {old_format_file}")
    
    try:
        gen_data = load_generation(old_format_file)
        
        print(f"✓ 成功載入")
        print(f"  Generation: {gen_data['generation']}")
        print(f"  Population size: {len(gen_data['population'])}")
        print(f"  Has niching info: {has_niching_info(gen_data)}")
        print(f"  cluster_labels: {gen_data['cluster_labels']}")
        print(f"  niching_info: {gen_data['niching_info']}")
        
        if not has_niching_info(gen_data):
            print("\n✅ 向後相容測試通過：舊格式正確載入，cluster_labels 為 None")
        else:
            print("\n⚠️  警告：舊格式檔案不應該包含 niching 資訊")
            
    except Exception as e:
        print(f"✗ 載入失敗: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"⚠️  測試檔案不存在: {old_format_file}")

# ============================================================================
# 測試 2：測試工具函數
# ============================================================================
print("\n" + "="*100)
print("測試 2：測試工具函數")
print("="*100)

# 尋找有 niching 資訊的檔案（如果有的話）
test_dirs = [
    'k_comparison_experiments/exp_1_fixed_k3/generations',
    'k_comparison_experiments/exp_2_fixed_k8/generations',
    'k_comparison_experiments/exp_3_dynamic_calibration/generations'
]

niching_file = None
for test_dir in test_dirs:
    test_path = Path(test_dir)
    if test_path.exists():
        gen_files = list(test_path.glob('generation_*.pkl'))
        if gen_files:
            # 嘗試載入第一個檔案
            try:
                test_data = load_generation(gen_files[0])
                if has_niching_info(test_data):
                    niching_file = gen_files[0]
                    break
            except:
                continue

if niching_file:
    print(f"\n找到包含 niching 資訊的檔案: {niching_file}")
    
    try:
        gen_data = load_generation(niching_file)
        
        print(f"\n基本資訊:")
        print(f"  Generation: {gen_data['generation']}")
        print(f"  Population size: {len(gen_data['population'])}")
        print(f"  Has niching info: {has_niching_info(gen_data)}")
        
        if has_niching_info(gen_data):
            niching_info = gen_data['niching_info']
            print(f"\nNiching 資訊:")
            print(f"  n_clusters: {niching_info['n_clusters']}")
            print(f"  algorithm: {niching_info['algorithm']}")
            print(f"  silhouette_score: {niching_info['silhouette_score']}")
            
            # 測試 get_niche_statistics
            print(f"\n測試 get_niche_statistics():")
            stats = get_niche_statistics(gen_data)
            for niche_id, niche_stats in stats.items():
                print(f"  Niche {niche_id}:")
                print(f"    Size: {niche_stats['size']}")
                print(f"    Fitness mean: {niche_stats['fitness_mean']:.4f}")
                print(f"    Fitness std: {niche_stats['fitness_std']:.4f}")
                print(f"    Fitness range: [{niche_stats['fitness_min']:.4f}, {niche_stats['fitness_max']:.4f}]")
            
            # 測試 get_niche_individuals
            print(f"\n測試 get_niche_individuals():")
            niche_0_inds = get_niche_individuals(gen_data, 0)
            print(f"  Niche 0 有 {len(niche_0_inds)} 個個體")
            if niche_0_inds:
                print(f"  第一個個體:")
                print(f"    Fitness: {niche_0_inds[0].fitness.values[0]:.4f}")
                print(f"    Expression: {str(niche_0_inds[0])[:100]}...")
            
            print("\n✅ 工具函數測試通過")
        
    except Exception as e:
        print(f"✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n⚠️  沒有找到包含 niching 資訊的檔案")
    print("   這是正常的，因為舊的實驗沒有儲存 cluster_labels")
    print("   請執行新的實驗來測試新格式")

# ============================================================================
# 測試 3：說明如何在新實驗中使用
# ============================================================================
print("\n" + "="*100)
print("測試 3：新實驗使用說明")
print("="*100)

print("""
要測試新格式的儲存，請執行一個啟用 niching 的實驗：

1. 在 run_portfolio_experiment.py 中設定：
   CONFIG = {
       ...
       'niching_enabled': True,
       'niching_n_clusters': 3,
       ...
   }

2. 執行實驗後，使用以下程式碼載入：

   from gp_quant.utils import load_generation, has_niching_info, get_niche_individuals
   
   # 載入 generation
   gen_data = load_generation('path/to/generation_001.pkl')
   
   # 檢查是否有 niching 資訊
   if has_niching_info(gen_data):
       print(f"Niching info: {gen_data['niching_info']}")
       print(f"Cluster labels: {gen_data['cluster_labels']}")
       
       # 獲取特定 niche 的個體
       niche_0_individuals = get_niche_individuals(gen_data, 0)
       print(f"Niche 0 has {len(niche_0_individuals)} individuals")
""")

print("\n" + "="*100)
print("測試完成")
print("="*100)
