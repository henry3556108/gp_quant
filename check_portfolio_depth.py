"""
檢查 portfolio_experiment_results 中所有實驗的 GP 樹深度
論文要求：
- Generation 0（初始族群）：最大深度 6
- Generation 1-50（後續世代）：最大深度 17
"""
import os
import sys
import dill
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 設置 DEAP 環境（載入 pickle 需要）
from deap import creator, base, gp
from gp_quant.gp.operators import pset

# 初始化 DEAP creator（如果還沒有的話）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def check_generation_depth(generation_file):
    """
    檢查一個 generation pkl 檔案中所有個體的深度
    支援新舊兩種格式
    
    Returns:
        dict: {
            'min_depth': 最小深度,
            'max_depth': 最大深度,
            'avg_depth': 平均深度,
            'population_size': 族群大小,
            'depths': 所有深度的列表,
            'has_niching': 是否包含 niching 資訊
        }
    """
    try:
        with open(generation_file, 'rb') as f:
            data = dill.load(f)
        
        # 支援新舊兩種格式
        if isinstance(data, dict):
            population = data.get('population', [])
            has_niching = 'cluster_labels' in data and data['cluster_labels'] is not None
        else:
            # 舊格式：直接是 population list
            population = data
            has_niching = False
        
        if not population:
            return None
        
        depths = [ind.height for ind in population]
        
        return {
            'min_depth': min(depths),
            'max_depth': max(depths),
            'avg_depth': sum(depths) / len(depths),
            'population_size': len(population),
            'depths': depths,
            'has_niching': has_niching
        }
    except Exception as e:
        print(f"Error loading {generation_file}: {e}")
        return None

def find_all_experiments(base_dir='portfolio_experiment_results'):
    """
    找出所有實驗目錄
    
    Returns:
        list: 實驗目錄列表
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 目錄不存在: {base_dir}")
        return []
    
    # 找出所有以 portfolio_exp 開頭的目錄
    exp_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith('portfolio_exp'):
            generations_dir = item / 'generations'
            if generations_dir.exists():
                exp_dirs.append(item)
    
    return sorted(exp_dirs)

def check_all_portfolio_experiments(base_dir='portfolio_experiment_results'):
    """檢查所有 portfolio 實驗的深度限制"""
    
    results = []
    
    # 找出所有實驗
    exp_dirs = find_all_experiments(base_dir)
    
    if not exp_dirs:
        print(f"❌ 在 {base_dir} 中沒有找到任何實驗目錄")
        return pd.DataFrame()
    
    print(f"找到 {len(exp_dirs)} 個實驗目錄")
    print()
    
    # 計算總檢查數
    total_checks = 0
    for exp_dir in exp_dirs:
        generations_dir = exp_dir / 'generations'
        gen_files = list(generations_dir.glob('generation_*.pkl'))
        total_checks += len(gen_files)
    
    print(f"總共需要檢查 {total_checks} 個 generation 檔案")
    print("開始深度檢查...\n")
    
    # 使用進度條
    with tqdm(total=total_checks, desc="Checking depths") as pbar:
        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            generations_dir = exp_dir / 'generations'
            
            # 找出所有 generation 檔案
            gen_files = sorted(generations_dir.glob('generation_*.pkl'))
            
            for gen_file in gen_files:
                # 解析 generation 編號
                try:
                    # generation_001.pkl 或 generation_024_final.pkl
                    gen_num_str = gen_file.stem.split('_')[1]
                    gen = int(gen_num_str)
                    is_final = 'final' in gen_file.stem
                except (IndexError, ValueError):
                    pbar.update(1)
                    continue
                
                # 檢查深度
                depth_info = check_generation_depth(gen_file)
                
                if depth_info is None:
                    pbar.update(1)
                    continue
                
                # 判斷是否符合限制
                if gen == 0 or gen == 1:
                    # 初始族群：最大深度應該 <= 6
                    expected_max = 6
                    compliant = depth_info['max_depth'] <= expected_max
                else:
                    # 後續世代：最大深度應該 <= 17
                    expected_max = 17
                    compliant = depth_info['max_depth'] <= expected_max
                
                # 記錄結果
                results.append({
                    'experiment': exp_name,
                    'generation': gen,
                    'is_final': is_final,
                    'min_depth': depth_info['min_depth'],
                    'max_depth': depth_info['max_depth'],
                    'avg_depth': round(depth_info['avg_depth'], 2),
                    'population_size': depth_info['population_size'],
                    'has_niching': depth_info['has_niching'],
                    'expected_max_depth': expected_max,
                    'compliant': compliant,
                    'violation': 'No' if compliant else f'Yes (max={depth_info["max_depth"]} > {expected_max})'
                })
                
                pbar.update(1)
    
    return pd.DataFrame(results)

def generate_summary(df):
    """生成檢查摘要"""
    print("\n" + "="*100)
    print("深度限制檢查摘要")
    print("="*100)
    
    total_checks = len(df)
    violations = df[~df['compliant']]
    num_violations = len(violations)
    
    print(f"\n總檢查數: {total_checks}")
    print(f"符合限制: {total_checks - num_violations} ({(total_checks - num_violations)/total_checks*100:.2f}%)")
    print(f"違反限制: {num_violations} ({num_violations/total_checks*100:.2f}%)")
    
    if num_violations > 0:
        print("\n❌ 發現違規情況:")
        print(violations[['experiment', 'generation', 'max_depth', 'expected_max_depth', 'violation']].to_string())
    else:
        print("\n✅ 所有實驗都符合深度限制！")
    
    # 按實驗統計
    print("\n" + "="*100)
    print("按實驗統計")
    print("="*100)
    
    exp_stats = df.groupby('experiment').agg({
        'generation': 'count',
        'max_depth': ['min', 'max', 'mean'],
        'avg_depth': 'mean',
        'compliant': lambda x: (x.sum() / len(x) * 100),
        'has_niching': 'any'
    }).round(2)
    
    exp_stats.columns = ['Gen Count', 'Min Max', 'Max Max', 'Avg Max', 'Avg Depth', 'Compliance %', 'Has Niching']
    print(exp_stats.to_string())
    
    # 按 generation 統計
    print("\n" + "="*100)
    print("按 Generation 統計（所有實驗）")
    print("="*100)
    
    gen_stats = df.groupby('generation').agg({
        'max_depth': ['min', 'max', 'mean'],
        'avg_depth': 'mean',
        'compliant': lambda x: (x.sum() / len(x) * 100)
    }).round(2)
    
    gen_stats.columns = ['Min of Max', 'Max of Max', 'Avg of Max', 'Avg Depth', 'Compliance %']
    print(gen_stats.head(20).to_string())  # 只顯示前 20 個 generation
    
    # Generation 1 特別檢查（初始族群）
    print("\n" + "="*100)
    print("Generation 1（初始族群）深度檢查（應 <= 6）")
    print("="*100)
    
    gen1 = df[df['generation'] == 1]
    if not gen1.empty:
        gen1_summary = gen1.groupby('experiment').agg({
            'max_depth': ['min', 'max', 'mean'],
            'compliant': 'all'
        }).round(2)
        
        gen1_summary.columns = ['Min Max', 'Max Max', 'Avg Max', 'All Compliant']
        print(gen1_summary.to_string())
    else:
        print("沒有 Generation 1 的資料")
    
    # Generation 2+ 檢查
    print("\n" + "="*100)
    print("Generation 2+（後續世代）深度檢查（應 <= 17）")
    print("="*100)
    
    gen_later = df[df['generation'] > 1]
    if not gen_later.empty:
        gen_later_summary = gen_later.groupby('experiment').agg({
            'max_depth': ['min', 'max', 'mean'],
            'compliant': 'all'
        }).round(2)
        
        gen_later_summary.columns = ['Min Max', 'Max Max', 'Avg Max', 'All Compliant']
        print(gen_later_summary.to_string())
    else:
        print("沒有 Generation 2+ 的資料")
    
    # Niching 資訊統計
    print("\n" + "="*100)
    print("Niching 資訊統計")
    print("="*100)
    
    niching_stats = df.groupby('experiment')['has_niching'].any()
    print(f"\n包含 Niching 資訊的實驗: {niching_stats.sum()} / {len(niching_stats)}")
    if niching_stats.any():
        print("\n有 Niching 資訊的實驗:")
        for exp_name in niching_stats[niching_stats].index:
            print(f"  - {exp_name}")

def main():
    """主函數"""
    print("="*100)
    print("開始檢查 Portfolio 實驗的 GP 深度限制")
    print("="*100)
    print("\n論文要求:")
    print("  - Generation 0-1（初始族群）: 最大深度 <= 6")
    print("  - Generation 2+（後續世代）: 最大深度 <= 17")
    print()
    
    # 檢查所有實驗
    df = check_all_portfolio_experiments()
    
    if df.empty:
        print("❌ 沒有找到任何實驗數據！")
        return
    
    # 儲存詳細結果
    output_file = 'portfolio_depth_check_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ 詳細結果已儲存至: {output_file}")
    
    # 生成摘要
    generate_summary(df)
    
    # 儲存違規記錄（如果有）
    violations = df[~df['compliant']]
    if not violations.empty:
        violations_file = 'portfolio_depth_violations.csv'
        violations.to_csv(violations_file, index=False)
        print(f"\n⚠️  違規記錄已儲存至: {violations_file}")
    
    print("\n" + "="*100)
    print("檢查完成！")
    print("="*100)

if __name__ == "__main__":
    main()
