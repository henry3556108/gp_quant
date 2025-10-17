"""
動態 K 值選擇示例

展示如何使用 DynamicKSelector 進行動態 niche 數量選擇
"""

import sys
from pathlib import Path
import numpy as np

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.niching import DynamicKSelector, create_k_selector


def example_1_fixed_k():
    """示例 1：固定 k 值（向下兼容）"""
    print("=" * 80)
    print("示例 1：固定 k 值（向下兼容）")
    print("=" * 80)
    
    # 方式 1：直接創建
    selector = DynamicKSelector(mode='fixed', fixed_k=3)
    
    # 方式 2：從配置創建（向下兼容舊代碼）
    config = {'niching_n_clusters': 3}
    selector = create_k_selector(config)
    
    # 模擬相似度矩陣
    similarity_matrix = np.random.rand(100, 100)
    
    result = selector.select_k(similarity_matrix, population_size=100)
    
    print(f"選擇的 k 值: {result['k']}")
    print(f"模式: {result['mode']}")
    print()


def example_2_dynamic_k():
    """示例 2：動態選擇 k 值"""
    print("=" * 80)
    print("示例 2：動態選擇 k 值")
    print("=" * 80)
    
    # 創建動態選擇器
    selector = DynamicKSelector(
        mode='dynamic',
        k_min=2,
        k_max=8,
        verbose=True
    )
    
    # 模擬相似度矩陣
    similarity_matrix = np.random.rand(100, 100)
    
    result = selector.select_k(similarity_matrix, population_size=100)
    
    print(f"\n選擇的 k 值: {result['k']}")
    print(f"測試範圍: {result['k_range']}")
    print(f"各 k 值分數:")
    for k, score in result['scores'].items():
        print(f"  k={k}: {score:.4f}")
    print()


def example_3_auto_k_max():
    """示例 3：自動 k 上限（ln(n)）"""
    print("=" * 80)
    print("示例 3：自動 k 上限（ln(n)）")
    print("=" * 80)
    
    # 創建自動上限選擇器
    config = {
        'niching_k_selection': 'auto',
        'niching_k_min': 2,
        'niching_k_max': 'auto',  # 使用 ln(n)
        'niching_algorithm': 'kmeans'
    }
    
    selector = create_k_selector(config)
    
    # 測試不同 population sizes
    for pop_size in [500, 1000, 2000, 5000]:
        print(f"\nPopulation size: {pop_size}")
        print(f"  ln({pop_size}) = {np.log(pop_size):.2f}")
        
        similarity_matrix = np.random.rand(pop_size, pop_size)
        result = selector.select_k(similarity_matrix, population_size=pop_size)
        
        print(f"  k_max: {max(result['k_range'])}")
        print(f"  選擇的 k: {result['k']}")
    
    print()


def example_4_calibration():
    """示例 4：階段性校準"""
    print("=" * 80)
    print("示例 4：階段性校準")
    print("=" * 80)
    
    # 創建校準模式選擇器
    config = {
        'niching_k_selection': 'calibration',
        'niching_k_min': 2,
        'niching_k_max': 'auto',
        'niching_k_calibration_gens': 3,  # 前 3 代校準
        'niching_algorithm': 'kmeans'
    }
    
    selector = create_k_selector(config)
    
    # 模擬 10 代演化
    for gen in range(1, 11):
        print(f"\n{'='*60}")
        print(f"Generation {gen}")
        print(f"{'='*60}")
        
        # 模擬相似度矩陣
        similarity_matrix = np.random.rand(100, 100)
        
        result = selector.select_k(
            similarity_matrix, 
            population_size=100,
            generation=gen
        )
        
        print(f"選擇的 k: {result['k']}")
        print(f"模式: {result['mode']}")
        
        if 'calibration_progress' in result:
            print(f"校準進度: {result['calibration_progress']}")
    
    # 顯示校準歷史
    stats = selector.get_statistics()
    print(f"\n{'='*60}")
    print("校準完成！")
    print(f"{'='*60}")
    print(f"校準後固定 k: {stats['calibrated_k']}")
    print(f"\n校準歷史:")
    for h in stats['calibration_history']:
        print(f"  Generation {h['generation']}: k={h['best_k']}, score={h['score']:.4f}")
    print()


def example_5_integration_with_experiment():
    """示例 5：與實驗腳本整合"""
    print("=" * 80)
    print("示例 5：與實驗腳本整合")
    print("=" * 80)
    
    # 舊的配置（向下兼容）
    print("\n舊配置（向下兼容）:")
    old_config = {
        'niching_enabled': True,
        'niching_n_clusters': 3,
        'niching_algorithm': 'kmeans'
    }
    print(f"  {old_config}")
    
    selector = create_k_selector(old_config)
    print(f"  → 創建固定 k={old_config['niching_n_clusters']} 的選擇器")
    
    # 新的配置（動態選擇）
    print("\n新配置（動態選擇）:")
    new_config = {
        'niching_enabled': True,
        'niching_k_selection': 'auto',
        'niching_k_min': 2,
        'niching_k_max': 'auto',
        'niching_algorithm': 'kmeans'
    }
    print(f"  {new_config}")
    
    selector = create_k_selector(new_config)
    print(f"  → 創建動態選擇器（k_max = ln(n)）")
    
    # 新的配置（階段性校準）
    print("\n新配置（階段性校準）:")
    calibration_config = {
        'niching_enabled': True,
        'niching_k_selection': 'calibration',
        'niching_k_min': 2,
        'niching_k_max': 'auto',
        'niching_k_calibration_gens': 3,
        'niching_algorithm': 'kmeans'
    }
    print(f"  {calibration_config}")
    
    selector = create_k_selector(calibration_config)
    print(f"  → 創建校準模式選擇器（前 3 代動態，之後固定）")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("動態 K 值選擇示例")
    print("=" * 80 + "\n")
    
    # 運行所有示例
    example_1_fixed_k()
    example_2_dynamic_k()
    example_3_auto_k_max()
    example_4_calibration()
    example_5_integration_with_experiment()
    
    print("=" * 80)
    print("所有示例完成！")
    print("=" * 80)
