"""
檢查所有實驗中每個 generation 的 GP 樹深度是否符合論文限制
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

# 設置 DEAP 環境（載入 pickle 需要）
from deap import creator, base, gp
from gp_quant.gp.operators import pset

# 初始化 DEAP creator（如果還沒有的話）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def check_population_depth(population_file):
    """
    檢查一個族群檔案中所有個體的深度
    
    Returns:
        dict: {
            'min_depth': 最小深度,
            'max_depth': 最大深度,
            'avg_depth': 平均深度,
            'population_size': 族群大小,
            'depths': 所有深度的列表
        }
    """
    try:
        with open(population_file, 'rb') as f:
            population = dill.load(f)
        
        if not population:
            return None
        
        depths = [ind.height for ind in population]
        
        return {
            'min_depth': min(depths),
            'max_depth': max(depths),
            'avg_depth': sum(depths) / len(depths),
            'population_size': len(population),
            'depths': depths
        }
    except Exception as e:
        print(f"Error loading {population_file}: {e}")
        return None

def check_all_experiments(base_dir='experiments_results'):
    """檢查所有實驗的深度限制"""
    
    results = []
    
    # 遍歷所有 ticker
    tickers = ['ABX_TO', 'BBD-B_TO', 'RY_TO', 'TRP_TO']
    
    total_checks = 0
    for ticker in tickers:
        ticker_dir = os.path.join(base_dir, ticker)
        if not os.path.exists(ticker_dir):
            print(f"Warning: {ticker_dir} not found")
            continue
        
        # 遍歷所有實驗（short 和 long，run 1-10）
        for period in ['short', 'long']:
            for run in range(1, 11):
                run_name = f"{period}_run{run:02d}"
                individual_records_dir = os.path.join(ticker_dir, f"individual_records_{run_name}")
                
                if not os.path.exists(individual_records_dir):
                    print(f"Warning: {individual_records_dir} not found")
                    continue
                
                # 遍歷所有 generation (0-50)
                for gen in range(51):
                    gen_dir = os.path.join(individual_records_dir, f"generation_{gen:03d}")
                    population_file = os.path.join(gen_dir, "population.pkl")
                    
                    if not os.path.exists(population_file):
                        print(f"Warning: {population_file} not found")
                        continue
                    
                    total_checks += 1
    
    print(f"Total files to check: {total_checks}")
    print("Starting depth check...\n")
    
    # 使用進度條
    with tqdm(total=total_checks, desc="Checking depths") as pbar:
        for ticker in tickers:
            ticker_dir = os.path.join(base_dir, ticker)
            if not os.path.exists(ticker_dir):
                continue
            
            for period in ['short', 'long']:
                period_name = '短訓練期' if period == 'short' else '長訓練期'
                
                for run in range(1, 11):
                    run_name = f"{period}_run{run:02d}"
                    individual_records_dir = os.path.join(ticker_dir, f"individual_records_{run_name}")
                    
                    if not os.path.exists(individual_records_dir):
                        continue
                    
                    for gen in range(51):
                        gen_dir = os.path.join(individual_records_dir, f"generation_{gen:03d}")
                        population_file = os.path.join(gen_dir, "population.pkl")
                        
                        if not os.path.exists(population_file):
                            continue
                        
                        # 檢查深度
                        depth_info = check_population_depth(population_file)
                        
                        if depth_info is None:
                            pbar.update(1)
                            continue
                        
                        # 判斷是否符合限制
                        if gen == 0:
                            # 初始族群：最大深度應該 <= 6
                            expected_max = 6
                            compliant = depth_info['max_depth'] <= expected_max
                        else:
                            # 後續世代：最大深度應該 <= 17
                            expected_max = 17
                            compliant = depth_info['max_depth'] <= expected_max
                        
                        # 記錄結果
                        results.append({
                            'ticker': ticker,
                            'period': period_name,
                            'run': run,
                            'generation': gen,
                            'min_depth': depth_info['min_depth'],
                            'max_depth': depth_info['max_depth'],
                            'avg_depth': round(depth_info['avg_depth'], 2),
                            'population_size': depth_info['population_size'],
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
        print(violations[['ticker', 'period', 'run', 'generation', 'max_depth', 'expected_max_depth', 'violation']])
    else:
        print("\n✅ 所有實驗都符合深度限制！")
    
    # 按 generation 統計
    print("\n" + "="*100)
    print("按 Generation 統計")
    print("="*100)
    
    gen_stats = df.groupby('generation').agg({
        'max_depth': ['min', 'max', 'mean'],
        'avg_depth': 'mean',
        'compliant': lambda x: (x.sum() / len(x) * 100)
    }).round(2)
    
    gen_stats.columns = ['Min of Max', 'Max of Max', 'Avg of Max', 'Avg Depth', 'Compliance %']
    print(gen_stats)
    
    # 按 ticker 和 period 統計
    print("\n" + "="*100)
    print("按 Ticker 和 Period 統計")
    print("="*100)
    
    ticker_stats = df.groupby(['ticker', 'period']).agg({
        'max_depth': 'max',
        'avg_depth': 'mean',
        'compliant': lambda x: (x.sum() / len(x) * 100)
    }).round(2)
    
    ticker_stats.columns = ['Max Depth', 'Avg Depth', 'Compliance %']
    print(ticker_stats)
    
    # Generation 0 特別檢查
    print("\n" + "="*100)
    print("Generation 0（初始族群）深度檢查（應 <= 6）")
    print("="*100)
    
    gen0 = df[df['generation'] == 0]
    gen0_summary = gen0.groupby(['ticker', 'period']).agg({
        'max_depth': ['min', 'max', 'mean'],
        'compliant': 'all'
    }).round(2)
    
    gen0_summary.columns = ['Min Max', 'Max Max', 'Avg Max', 'All Compliant']
    print(gen0_summary)
    
    # Generation 1-50 檢查
    print("\n" + "="*100)
    print("Generation 1-50（後續世代）深度檢查（應 <= 17）")
    print("="*100)
    
    gen_later = df[df['generation'] > 0]
    gen_later_summary = gen_later.groupby(['ticker', 'period']).agg({
        'max_depth': ['min', 'max', 'mean'],
        'compliant': 'all'
    }).round(2)
    
    gen_later_summary.columns = ['Min Max', 'Max Max', 'Avg Max', 'All Compliant']
    print(gen_later_summary)

def main():
    """主函數"""
    print("="*100)
    print("開始檢查 GP 深度限制")
    print("="*100)
    print("\n論文要求:")
    print("  - Generation 0（初始族群）: 最大深度 <= 6")
    print("  - Generation 1-50（後續世代）: 最大深度 <= 17")
    print()
    
    # 檢查所有實驗
    df = check_all_experiments()
    
    if df.empty:
        print("❌ 沒有找到任何實驗數據！")
        return
    
    # 儲存詳細結果
    output_file = 'depth_limit_check_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ 詳細結果已儲存至: {output_file}")
    
    # 生成摘要
    generate_summary(df)
    
    # 儲存違規記錄（如果有）
    violations = df[~df['compliant']]
    if not violations.empty:
        violations_file = 'depth_limit_violations.csv'
        violations.to_csv(violations_file, index=False)
        print(f"\n⚠️  違規記錄已儲存至: {violations_file}")
    
    print("\n" + "="*100)
    print("檢查完成！")
    print("="*100)

if __name__ == "__main__":
    main()
