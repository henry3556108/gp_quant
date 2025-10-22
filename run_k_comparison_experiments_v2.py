"""
K 值對比實驗批次運行腳本

運行 3 個實驗：
1. 固定 k=3 (baseline)
2. 固定 k=8
3. 動態選擇 (calibration)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from deap import creator, base, gp, tools
import random
import json
import os
import dill

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.gp.operators import pset
from gp_quant.evolution.early_stopping import EarlyStopping
from gp_quant.similarity import SimilarityMatrix, ParallelSimilarityMatrix
from gp_quant.niching import NichingClusterer, CrossNicheSelector, create_k_selector


def run_experiment(exp_config, exp_name):
    """運行單個實驗（複製自 run_portfolio_experiment.py 的 main 函數）"""
    
    print("\n" + "="*100)
    print(f"🚀 實驗: {exp_name}")
    print("="*100)
    print()
    
    CONFIG = exp_config
    
    print("📋 實驗配置:")
    print(f"  股票組合: {', '.join(CONFIG['tickers'])}")
    print(f"  族群大小: {CONFIG['population_size']}")
    print(f"  演化世代: {CONFIG['generations']}")
    print(f"  Fitness 指標: {CONFIG['fitness_metric']}")
    
    if CONFIG['niching_enabled']:
        print(f"  Niching 策略: 啟用")
        if 'niching_k_selection' in CONFIG:
            print(f"    - K 值選擇: {CONFIG['niching_k_selection']} 模式")
            if CONFIG['niching_k_selection'] == 'calibration':
                print(f"    - 校準期: 前 {CONFIG.get('niching_k_calibration_gens', 3)} 代")
            print(f"    - K 範圍: [{CONFIG.get('niching_k_min', 2)}, {CONFIG.get('niching_k_max', 'auto')}]")
        else:
            print(f"    - Niche 數量: {CONFIG['niching_n_clusters']} (固定)")
        print(f"    - 跨群比例: {CONFIG['niching_cross_ratio']:.0%}")
        print(f"    - 更新頻率: 每 {CONFIG['niching_update_frequency']} 代")
    print()
    
    # 創建輸出目錄
    exp_dir = Path(CONFIG['output_dir']) / CONFIG['experiment_name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    generations_dir = exp_dir / "generations"
    generations_dir.mkdir(exist_ok=True)
    
    print(f"📁 輸出目錄: {exp_dir}")
    print()
    
    # 儲存配置
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # 載入數據
    print("1️⃣  載入市場數據...")
    data = {}
    
    for ticker in CONFIG['tickers']:
        file_path = project_root / f"TSE300_selected/{ticker}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            data[ticker] = df
            print(f"   ✓ {ticker}: {len(df)} 天")
        else:
            print(f"   ✗ {ticker}: 文件不存在")
            return None
    
    print()
    
    # 初始化 Engine
    print("2️⃣  初始化 Backtesting Engine...")
    
    try:
        train_engine = PortfolioBacktestingEngine(
            data=data,
            backtest_start=CONFIG['train_backtest_start'],
            backtest_end=CONFIG['train_backtest_end'],
            initial_capital=CONFIG['initial_capital'],
            pset=pset
        )
        print(f"   ✓ 交易日數: {len(train_engine.common_dates)}")
    except Exception as e:
        print(f"   ✗ 初始化失敗: {e}")
        return None
    
    print()
    
    # 設置 DEAP
    print("3️⃣  設置 DEAP...")
    
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=CONFIG['tournament_size'])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    def evaluate_individual(individual):
        try:
            fitness = train_engine.get_fitness(individual, fitness_metric=CONFIG['fitness_metric'])
            return (fitness,)
        except:
            return (-1000000.0,)
    
    toolbox.register("evaluate", evaluate_individual)
    
    print("   ✓ DEAP 設置完成")
    print()
    
    # 創建統計和 Hall of Fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(10)
    
    # 創建初始族群
    print("4️⃣  創建初始族群...")
    population = toolbox.population(n=CONFIG['population_size'])
    print(f"   ✓ 創建 {len(population)} 個個體")
    print()
    
    # 開始演化
    print("5️⃣  開始 GP 演化...")
    print("="*100)
    
    evolution_log = []
    niching_log = []
    start_time = datetime.now()
    
    # 初始化 Niching 機制
    niching_selector = None
    k_selector = None
    niche_labels = None
    
    if CONFIG['niching_enabled']:
        niching_selector = CrossNicheSelector(
            cross_niche_ratio=CONFIG['niching_cross_ratio'],
            tournament_size=CONFIG['tournament_size'],
            random_state=42
        )
        
        # 創建 k 值選擇器
        k_selector = create_k_selector(CONFIG)
        
        print(f"✓ Niching 策略已啟用")
        if 'niching_k_selection' in CONFIG:
            print(f"  - K 值選擇: {CONFIG['niching_k_selection']} 模式")
        else:
            print(f"  - Niche 數量: {CONFIG['niching_n_clusters']} (固定)")
        print()
    
    for gen in range(CONFIG['generations']):
        gen_start_time = datetime.now()
        
        print(f"\n{'='*100}")
        print(f"📊 Generation {gen + 1}/{CONFIG['generations']}")
        print(f"{'='*100}")
        
        # 評估族群
        print(f"⏳ 評估 {len(population)} 個個體...")
        eval_start = datetime.now()
        
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        eval_time = (datetime.now() - eval_start).total_seconds()
        print(f"✓ 評估完成 ({eval_time:.1f}s)")
        
        # 更新統計
        hof.update(population)
        record = stats.compile(population)
        
        # 顯示統計
        print(f"\n📈 Fitness 統計:")
        print(f"   Avg: {record['avg']:.4f} | Max: {record['max']:.4f} | Std: {record['std']:.4f}")
        
        # 記錄到日誌
        gen_log = {
            'generation': gen + 1,
            'min_fitness': float(record['min']),
            'avg_fitness': float(record['avg']),
            'max_fitness': float(record['max']),
            'std_fitness': float(record['std']),
            'eval_time': eval_time,
        }
        evolution_log.append(gen_log)
        
        # 儲存當前世代
        print(f"\n💾 儲存 Generation {gen + 1} 族群...")
        gen_file = generations_dir / f"generation_{gen+1:03d}.pkl"
        
        try:
            with open(gen_file, 'wb') as f:
                dill.dump({
                    'generation': gen + 1,
                    'population': population,
                    'hall_of_fame': list(hof),
                    'statistics': record,
                }, f)
            
            file_size = gen_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ 已儲存: {gen_file.name} ({file_size:.2f} MB)")
        except Exception as e:
            print(f"   ✗ 儲存失敗: {e}")
        
        # 選擇和繁殖
        if gen < CONFIG['generations'] - 1:
            print(f"\n🔄 選擇和繁殖...")
            
            # Niching: 計算相似度矩陣並聚類
            if CONFIG['niching_enabled'] and gen % CONFIG['niching_update_frequency'] == 0:
                print(f"\n🔬 Niching: 計算相似度矩陣...")
                sim_start = datetime.now()
                
                try:
                    if len(population) >= 200:
                        sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
                        similarity_matrix = sim_matrix.compute(show_progress=False)
                    else:
                        sim_matrix = SimilarityMatrix(population)
                        similarity_matrix = sim_matrix.compute(show_progress=False)
                    
                    sim_time = (datetime.now() - sim_start).total_seconds()
                    
                    print(f"   ✓ 相似度矩陣計算完成 ({sim_time:.1f}s)")
                    print(f"   平均相似度: {sim_matrix.get_average_similarity():.4f}")
                    
                    # 動態選擇 k 值
                    if k_selector is not None:
                        print(f"\n🎯 選擇 K 值...")
                        k_result = k_selector.select_k(
                            similarity_matrix,
                            population_size=len(population),
                            generation=gen + 1
                        )
                        selected_k = k_result['k']
                        print(f"   ✓ 選擇的 K: {selected_k}")
                        if k_result.get('scores'):
                            best_score = k_result['scores'][selected_k]
                            print(f"   Silhouette Score: {best_score:.4f}")
                    else:
                        selected_k = CONFIG['niching_n_clusters']
                    
                    # 聚類
                    print(f"\n🔬 Niching: 聚類（k={selected_k}）...")
                    clusterer = NichingClusterer(
                        n_clusters=selected_k,
                        algorithm=CONFIG['niching_algorithm']
                    )
                    niche_labels = clusterer.fit_predict(similarity_matrix)
                    
                    print(f"   ✓ 聚類完成")
                    print(f"   Silhouette 分數: {clusterer.silhouette_score_:.4f}")
                    
                    # 統計各 niche 大小
                    unique_niches, counts = np.unique(niche_labels, return_counts=True)
                    print(f"   各 Niche 大小: {dict(zip(unique_niches, counts))}")
                    
                    # 顯示每個 niche 的 silhouette score
                    if clusterer.per_cluster_silhouette_:
                        print(f"\n   各 Niche Silhouette Score:")
                        for niche_id, niche_stats in clusterer.per_cluster_silhouette_.items():
                            print(f"     Niche {niche_id}: {niche_stats['mean']:.4f} (size={niche_stats['size']}, std={niche_stats['std']:.4f})")
                    
                    # 記錄 niching 統計
                    niching_stats = {
                        'generation': gen + 1,
                        'selected_k': int(selected_k),
                        'avg_similarity': float(sim_matrix.get_average_similarity()),
                        'diversity_score': float(sim_matrix.get_diversity_score()),
                        'silhouette_score': float(clusterer.silhouette_score_),
                        'niche_sizes': {int(k): int(v) for k, v in zip(unique_niches, counts)},
                        'per_niche_silhouette': clusterer.per_cluster_silhouette_,  # 新增：每個 niche 的詳細信息
                        'computation_time': sim_time
                    }
                    if k_selector is not None and k_result.get('mode'):
                        niching_stats['k_selection_mode'] = k_result['mode']
                    niching_log.append(niching_stats)
                    
                except Exception as e:
                    print(f"   ✗ Niching 計算失敗: {e}")
                    niche_labels = None
            
            # Selection
            if CONFIG['niching_enabled'] and niche_labels is not None:
                offspring = niching_selector.select(population, niche_labels, len(population))
                offspring = list(map(toolbox.clone, offspring))
            else:
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CONFIG['crossover_prob']:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < CONFIG['mutation_prob']:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            population[:] = offspring
        
        gen_time = (datetime.now() - gen_start_time).total_seconds()
        print(f"\n⏱️  Generation {gen + 1} 耗時: {gen_time:.1f}s")
    
    # 演化完成
    total_time = (datetime.now() - start_time).total_seconds()
    
    print()
    print("="*100)
    print("✅ 演化完成！")
    print("="*100)
    print(f"⏱️  總耗時: {total_time/60:.2f} 分鐘")
    print(f"🏆 最佳 Fitness: {hof[0].fitness.values[0]:.4f}")
    print()
    
    # 儲存結果
    result = {
        'experiment_name': exp_name,
        'config': CONFIG,
        'evolution_log': evolution_log,
        'niching_log': niching_log,
        'total_time': total_time,
        'best_fitness': float(hof[0].fitness.values[0]),
    }
    
    # 儲存 JSON
    with open(exp_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    # 儲存 CSV
    log_df = pd.DataFrame(evolution_log)
    log_df.to_csv(exp_dir / "evolution_log.csv", index=False)
    
    if niching_log:
        niching_df = pd.DataFrame(niching_log)
        niching_df.to_csv(exp_dir / "niching_log.csv", index=False)
    
    print(f"💾 結果已儲存至: {exp_dir}")
    
    return result


def main():
    print("\n" + "="*100)
    print("🔬 K 值對比實驗批次運行")
    print("="*100)
    print()
    
    # 基礎配置
    BASE_CONFIG = {
        'tickers': ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO'],
        'train_data_start': '1995-01-03',
        'train_backtest_start': '1997-06-25',
        'train_backtest_end': '1999-06-25',
        'initial_capital': 100000.0,
        'population_size': 1000,
        'generations': 20,
        'crossover_prob': 0.8,
        'mutation_prob': 0.2,
        'tournament_size': 3,
        'fitness_metric': 'sharpe_ratio',
        'niching_enabled': True,
        'niching_cross_ratio': 0.8,
        'niching_update_frequency': 1,  # 每代都更新
        'niching_algorithm': 'kmeans',
        'output_dir': 'k_comparison_experiments',
    }
    
    # 實驗配置
    experiments = [
        # 實驗 1: 固定 k=3 (baseline)
        {
            **BASE_CONFIG,
            'experiment_name': 'exp_1_fixed_k3',
            'niching_n_clusters': 3,
        },
        
        # 實驗 2: 固定 k=8
        {
            **BASE_CONFIG,
            'experiment_name': 'exp_2_fixed_k8',
            'niching_n_clusters': 8,
        },
        
        # 實驗 3: 動態選擇 (calibration)
        {
            **BASE_CONFIG,
            'experiment_name': 'exp_3_dynamic_calibration',
            'niching_k_selection': 'calibration',
            'niching_k_min': 2,
            'niching_k_max': 'auto',
            'niching_k_calibration_gens': 3,
        },
    ]
    
    # 運行所有實驗
    results = []
    
    for i, exp_config in enumerate(experiments, 1):
        exp_name = exp_config['experiment_name']
        print(f"\n{'='*100}")
        print(f"🚀 開始實驗 {i}/3: {exp_name}")
        print(f"{'='*100}")
        
        result = run_experiment(exp_config, exp_name)
        if result:
            results.append(result)
    
    # 比較結果
    print("\n" + "="*100)
    print("📊 實驗結果比較")
    print("="*100)
    print()
    
    comparison_data = []
    for result in results:
        comparison_data.append({
            '實驗': result['experiment_name'],
            '最佳 Fitness': result['best_fitness'],
            '總時間 (分鐘)': result['total_time'] / 60,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # 儲存比較結果
    comparison_df.to_csv('k_comparison_experiments/comparison_summary.csv', index=False)
    
    print("="*100)
    print("✅ 所有實驗完成！")
    print(f"📁 結果保存在: k_comparison_experiments/")
    print("="*100)


if __name__ == "__main__":
    main()
