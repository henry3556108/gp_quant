"""
Phase 1: Portfolio-Based GP Evolution Experiment (with Norm Operator)

多股票組合的 GP 演化實驗
- 使用 PortfolioBacktestingEngine 同時評估多個股票
- 包含新實作的 Norm operator
- 儲存每個 generation 的族群快照
- 5000 個體，50 代演化（大規模實驗）
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
from gp_quant.similarity import SimilarityMatrix
from gp_quant.niching import NichingClusterer, CrossNicheSelector

def main():
    print("="*100)
    print("🚀 Phase 1: Portfolio-Based GP Evolution Experiment")
    print("="*100)
    print()
    
    # ============================================================================
    # 實驗配置
    # ============================================================================
    
    CONFIG = {
        # 股票組合
        'tickers': ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO'],
        
        # 訓練期（樣本內）
        'train_data_start': '1995-01-03',
        'train_backtest_start': '1997-06-25',
        'train_backtest_end': '1999-06-25',
        
        # 測試期（樣本外）
        'test_data_start': '1997-06-25',
        'test_backtest_start': '1999-06-26',
        'test_backtest_end': '2001-06-26',
        
        'initial_capital': 100000.0,
        
        # GP 參數
        'population_size': 100,
        'generations': 50,
        
        # 演化參數
        'crossover_prob': 0.8,
        'mutation_prob': 0.2,
        'tournament_size': 3,
        
        # Fitness 計算方式
        'fitness_metric': 'sharpe_ratio',  # 'excess_return', 'sharpe_ratio', 'avg_sharpe'
        'risk_free_rate': 0.0,  # 年化無風險利率
        
        # 早停配置
        'early_stopping_enabled': True,      # 是否啟用早停
        'early_stopping_patience': 5,       # 連續無進步的代數
        'early_stopping_min_delta': 0.001,   # 最小改進閾值（根據 fitness_metric 調整）
        
        # Niching 配置
        'niching_enabled': True,            # 是否啟用 Niching 策略
        'niching_n_clusters': 5,            # Niche 數量
        'niching_cross_ratio': 0.8,         # 跨群交配比例 (0.8 = 80%)
        'niching_update_frequency': 5,      # 每 N 代重新計算相似度矩陣
        'niching_algorithm': 'kmeans',      # 聚類演算法 ('kmeans' 或 'hierarchical')
        
        # 輸出目錄
        'output_dir': 'portfolio_experiment_results',
        'experiment_name': f'portfolio_exp_sharpe_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    print("📋 實驗配置:")
    print(f"  股票組合: {', '.join(CONFIG['tickers'])}")
    print(f"\n  訓練期（樣本內）:")
    print(f"    數據期間: {CONFIG['train_data_start']} 到 {CONFIG['train_backtest_start']}")
    print(f"    回測期間: {CONFIG['train_backtest_start']} 到 {CONFIG['train_backtest_end']}")
    print(f"\n  測試期（樣本外）:")
    print(f"    數據期間: {CONFIG['test_data_start']} 到 {CONFIG['test_backtest_start']}")
    print(f"    回測期間: {CONFIG['test_backtest_start']} 到 {CONFIG['test_backtest_end']}")
    print(f"\n  初始資金: ${CONFIG['initial_capital']:,.0f}")
    print(f"  族群大小: {CONFIG['population_size']}")
    print(f"  演化世代: {CONFIG['generations']}")
    print(f"  Fitness 指標: {CONFIG['fitness_metric']}")
    if CONFIG['early_stopping_enabled']:
        print(f"  早停機制: 啟用（patience={CONFIG['early_stopping_patience']}, min_delta={CONFIG['early_stopping_min_delta']}）")
    else:
        print(f"  早停機制: 停用")
    if CONFIG['niching_enabled']:
        print(f"  Niching 策略: 啟用")
        print(f"    - Niche 數量: {CONFIG['niching_n_clusters']}")
        print(f"    - 跨群比例: {CONFIG['niching_cross_ratio']:.0%}")
        print(f"    - 更新頻率: 每 {CONFIG['niching_update_frequency']} 代")
        print(f"    - 聚類演算法: {CONFIG['niching_algorithm']}")
    else:
        print(f"  Niching 策略: 停用")
    print()
    
    # ============================================================================
    # 創建輸出目錄
    # ============================================================================
    
    exp_dir = Path(CONFIG['output_dir']) / CONFIG['experiment_name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建子目錄
    generations_dir = exp_dir / "generations"
    generations_dir.mkdir(exist_ok=True)
    
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    print(f"📁 輸出目錄: {exp_dir}")
    print()
    
    # 儲存配置
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    # ============================================================================
    # 載入數據
    # ============================================================================
    
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
            sys.exit(1)
    
    print()
    
    # ============================================================================
    # 創建訓練和測試 Engine
    # ============================================================================
    
    print("2️⃣  初始化 Portfolio Backtesting Engines...")
    
    # 訓練 Engine（樣本內）
    print("\n   訓練期 Engine:")
    try:
        train_engine = PortfolioBacktestingEngine(
            data=data,
            backtest_start=CONFIG['train_backtest_start'],
            backtest_end=CONFIG['train_backtest_end'],
            initial_capital=CONFIG['initial_capital'],
            pset=pset
        )
        print(f"     ✓ 初始化成功")
        print(f"     ✓ 交易日數: {len(train_engine.common_dates)}")
        print(f"     ✓ 日期範圍: {train_engine.common_dates[0].date()} 到 {train_engine.common_dates[-1].date()}")
    except Exception as e:
        print(f"     ✗ 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 測試 Engine（樣本外）
    print("\n   測試期 Engine:")
    try:
        test_engine = PortfolioBacktestingEngine(
            data=data,
            backtest_start=CONFIG['test_backtest_start'],
            backtest_end=CONFIG['test_backtest_end'],
            initial_capital=CONFIG['initial_capital'],
            pset=pset
        )
        print(f"     ✓ 初始化成功")
        print(f"     ✓ 交易日數: {len(test_engine.common_dates)}")
        print(f"     ✓ 日期範圍: {test_engine.common_dates[0].date()} 到 {test_engine.common_dates[-1].date()}")
    except Exception as e:
        print(f"     ✗ 初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    
    # ============================================================================
    # 設置 DEAP
    # ============================================================================
    
    print("3️⃣  設置 DEAP...")
    
    # 創建 Fitness 和 Individual 類型
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    # 創建 toolbox
    toolbox = base.Toolbox()
    
    # 註冊 GP 操作
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # 註冊演化操作
    toolbox.register("select", tools.selTournament, tournsize=CONFIG['tournament_size'])
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
    
    # 定義 fitness 評估函數（使用訓練期數據）
    def evaluate_individual(individual):
        """評估單個個體的 fitness（訓練期）"""
        try:
            fitness = train_engine.get_fitness(individual, fitness_metric=CONFIG['fitness_metric'])
            return (fitness,)
        except Exception as e:
            return (-1000000.0,)
    
    toolbox.register("evaluate", evaluate_individual)
    
    print("   ✓ DEAP 設置完成")
    print()
    
    # ============================================================================
    # 創建統計和 Hall of Fame
    # ============================================================================
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    hof = tools.HallOfFame(10)  # 保存前 10 個最佳個體
    
    # ============================================================================
    # 創建初始族群
    # ============================================================================
    
    print("4️⃣  創建初始族群...")
    population = toolbox.population(n=CONFIG['population_size'])
    print(f"   ✓ 創建 {len(population)} 個個體")
    print()
    
    # ============================================================================
    # 開始演化
    # ============================================================================
    
    print("5️⃣  開始 GP 演化...")
    print(f"   族群大小: {CONFIG['population_size']}")
    print(f"   演化世代: {CONFIG['generations']}")
    print()
    print("="*100)
    
    # 記錄演化歷史
    evolution_log = []
    start_time = datetime.now()
    
    # 初始化早停機制
    early_stopping = None
    if CONFIG['early_stopping_enabled']:
        early_stopping = EarlyStopping(
            patience=CONFIG['early_stopping_patience'],
            min_delta=CONFIG['early_stopping_min_delta'],
            mode='max'  # Fitness 越大越好
        )
        print(f"✓ 早停機制已啟用（patience={CONFIG['early_stopping_patience']}, min_delta={CONFIG['early_stopping_min_delta']}）")
        print()
    
    # 初始化 Niching 機制
    niching_selector = None
    niche_labels = None
    niching_log = []
    
    if CONFIG['niching_enabled']:
        niching_selector = CrossNicheSelector(
            cross_niche_ratio=CONFIG['niching_cross_ratio'],
            tournament_size=CONFIG['tournament_size'],
            random_state=42
        )
        print(f"✓ Niching 策略已啟用")
        print(f"  - Niche 數量: {CONFIG['niching_n_clusters']}")
        print(f"  - 跨群比例: {CONFIG['niching_cross_ratio']:.0%}")
        print(f"  - 更新頻率: 每 {CONFIG['niching_update_frequency']} 代")
        print()
    
    for gen in range(CONFIG['generations']):
        gen_start_time = datetime.now()
        
        print(f"\n{'='*100}")
        print(f"📊 Generation {gen + 1}/{CONFIG['generations']}")
        print(f"{'='*100}")
        
        # 評估族群
        print(f"⏳ 評估 {len(population)} 個個體...")
        eval_start = datetime.now()
        
        # 評估所有個體
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
        min_fit = record['min']
        avg_fit = record['avg']
        max_fit = record['max']
        std_fit = record['std']
        
        print(f"\n📈 Fitness 統計:")
        print(f"   Min: {min_fit:.4f} ({min_fit*100:+.2f}%) | PnL: ${min_fit*CONFIG['initial_capital']:+,.0f}")
        print(f"   Avg: {avg_fit:.4f} ({avg_fit*100:+.2f}%) | PnL: ${avg_fit*CONFIG['initial_capital']:+,.0f}")
        print(f"   Max: {max_fit:.4f} ({max_fit*100:+.2f}%) | PnL: ${max_fit*CONFIG['initial_capital']:+,.0f}")
        print(f"   Std: {std_fit:.4f}")
        
        # 記錄到日誌
        gen_log = {
            'generation': gen + 1,
            'min_fitness': float(min_fit),
            'avg_fitness': float(avg_fit),
            'max_fitness': float(max_fit),
            'std_fitness': float(std_fit),
            'eval_time': eval_time,
            'timestamp': datetime.now().isoformat()
        }
        evolution_log.append(gen_log)
        
        # ========================================================================
        # 早停檢查
        # ========================================================================
        
        if early_stopping is not None:
            current_best = hof[0].fitness.values[0]
            
            if early_stopping.step(current_best):
                print(f"\n⏹️  早停觸發！")
                print(f"   連續 {early_stopping.counter} 代無顯著進步")
                print(f"   最佳 fitness: {early_stopping.best_fitness:.4f}")
                print(f"   最終 generation: {gen + 1}/{CONFIG['generations']}")
                print(f"   早停狀態: {early_stopping.get_status()}")
                
                # 記錄早停資訊
                gen_log['early_stopped'] = True
                gen_log['early_stop_reason'] = f'No improvement for {early_stopping.counter} generations'
                
                # 儲存最後一代後跳出循環
                print(f"\n💾 儲存最終 Generation {gen + 1} 族群...")
                gen_file = generations_dir / f"generation_{gen+1:03d}_final.pkl"
                
                try:
                    with open(gen_file, 'wb') as f:
                        dill.dump({
                            'generation': gen + 1,
                            'population': population,
                            'hall_of_fame': list(hof),
                            'statistics': record,
                            'early_stopped': True,
                            'early_stopping_status': early_stopping.get_status(),
                            'timestamp': datetime.now().isoformat()
                        }, f)
                    
                    file_size = gen_file.stat().st_size / (1024 * 1024)
                    print(f"   ✓ 已儲存: {gen_file.name} ({file_size:.2f} MB)")
                except Exception as e:
                    print(f"   ✗ 儲存失敗: {e}")
                
                break  # 跳出演化循環
            else:
                # 顯示早停狀態
                if gen > 0:  # 第一代不顯示
                    print(f"\n⏸️  早停狀態: {early_stopping.counter}/{early_stopping.patience} 代無進步")
        
        # ========================================================================
        # 儲存當前世代的族群
        # ========================================================================
        
        print(f"\n💾 儲存 Generation {gen + 1} 族群...")
        gen_file = generations_dir / f"generation_{gen+1:03d}.pkl"
        
        try:
            # 儲存整個族群
            with open(gen_file, 'wb') as f:
                dill.dump({
                    'generation': gen + 1,
                    'population': population,
                    'hall_of_fame': list(hof),
                    'statistics': record,
                    'timestamp': datetime.now().isoformat()
                }, f)
            
            file_size = gen_file.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✓ 已儲存: {gen_file.name} ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"   ✗ 儲存失敗: {e}")
        
        # 顯示最佳個體
        best_ind = hof[0]
        print(f"\n🏆 當前最佳個體:")
        print(f"   Fitness: {best_ind.fitness.values[0]:.4f} ({best_ind.fitness.values[0]*100:+.2f}%)")
        print(f"   PnL: ${best_ind.fitness.values[0]*CONFIG['initial_capital']:+,.0f}")
        print(f"   深度: {best_ind.height}, 節點數: {len(best_ind)}")
        print(f"   規則: {str(best_ind)[:100]}{'...' if len(str(best_ind)) > 100 else ''}")
        
        # ========================================================================
        # 選擇和繁殖（如果不是最後一代）
        # ========================================================================
        
        if gen < CONFIG['generations'] - 1:
            print(f"\n🔄 選擇和繁殖...")
            
            # ====================================================================
            # Niching: 計算相似度矩陣並聚類（每 N 代更新一次）
            # ====================================================================
            if CONFIG['niching_enabled'] and gen % CONFIG['niching_update_frequency'] == 0:
                print(f"\n🔬 Niching: 計算相似度矩陣...")
                sim_start = datetime.now()
                
                try:
                    sim_matrix = SimilarityMatrix(population)
                    similarity_matrix = sim_matrix.compute(show_progress=False)
                    sim_time = (datetime.now() - sim_start).total_seconds()
                    
                    print(f"   ✓ 相似度矩陣計算完成 ({sim_time:.1f}s)")
                    print(f"   平均相似度: {sim_matrix.get_average_similarity():.4f}")
                    print(f"   多樣性分數: {sim_matrix.get_diversity_score():.4f}")
                    
                    # 聚類
                    print(f"\n🔬 Niching: 聚類（k={CONFIG['niching_n_clusters']}）...")
                    clusterer = NichingClusterer(
                        n_clusters=CONFIG['niching_n_clusters'],
                        algorithm=CONFIG['niching_algorithm']
                    )
                    niche_labels = clusterer.fit_predict(similarity_matrix)
                    
                    print(f"   ✓ 聚類完成")
                    print(f"   Silhouette 分數: {clusterer.silhouette_score_:.4f}")
                    
                    # 統計各 niche 大小
                    unique_niches, counts = np.unique(niche_labels, return_counts=True)
                    print(f"   各 Niche 大小: {dict(zip(unique_niches, counts))}")
                    
                    # 記錄 niching 統計
                    niching_log.append({
                        'generation': gen + 1,
                        'avg_similarity': float(sim_matrix.get_average_similarity()),
                        'diversity_score': float(sim_matrix.get_diversity_score()),
                        'silhouette_score': float(clusterer.silhouette_score_),
                        'niche_sizes': {int(k): int(v) for k, v in zip(unique_niches, counts)},
                        'computation_time': sim_time
                    })
                    
                except Exception as e:
                    print(f"   ✗ Niching 計算失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    # 失敗時使用傳統選擇
                    niche_labels = None
            
            # ====================================================================
            # Selection: 使用 Niching 或傳統選擇
            # ====================================================================
            if CONFIG['niching_enabled'] and niche_labels is not None:
                # 使用跨群選擇
                print(f"\n🎯 使用跨群選擇...")
                try:
                    offspring = niching_selector.select(population, niche_labels, len(population))
                    offspring = list(map(toolbox.clone, offspring))
                    
                    # 顯示選擇統計
                    selection_stats = niching_selector.get_statistics()
                    print(f"   ✓ 選擇完成")
                    print(f"   跨群配對: {selection_stats['cross_niche_pairs']} ({selection_stats['cross_niche_ratio_actual']:.0%})")
                    print(f"   群內配對: {selection_stats['within_niche_pairs']} ({selection_stats['within_niche_ratio_actual']:.0%})")
                    
                    # 記錄選擇統計
                    gen_log['niching_selection'] = {
                        'cross_niche_pairs': selection_stats['cross_niche_pairs'],
                        'within_niche_pairs': selection_stats['within_niche_pairs'],
                        'cross_niche_ratio': selection_stats['cross_niche_ratio_actual']
                    }
                    
                except Exception as e:
                    print(f"   ✗ 跨群選擇失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    # 失敗時使用傳統選擇
                    offspring = toolbox.select(population, len(population))
                    offspring = list(map(toolbox.clone, offspring))
            else:
                # 使用傳統 tournament selection
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
            print(f"   ✓ 新一代族群已準備")
        
        # 顯示世代耗時
        gen_time = (datetime.now() - gen_start_time).total_seconds()
        print(f"\n⏱️  Generation {gen + 1} 耗時: {gen_time:.1f}s")
    
    # ============================================================================
    # 演化完成
    # ============================================================================
    
    total_time = (datetime.now() - start_time).total_seconds()
    actual_generations = gen + 1  # 實際運行的代數
    
    print()
    print("="*100)
    print("✅ 演化完成！")
    print("="*100)
    print()
    
    print(f"⏱️  總耗時: {total_time/60:.2f} 分鐘 ({total_time:.1f} 秒)")
    print(f"📊 總世代數: {actual_generations}/{CONFIG['generations']}")
    
    # 顯示早停資訊
    if early_stopping is not None and early_stopping.should_stop:
        print(f"⏹️  早停: 是（第 {actual_generations} 代觸發）")
        print(f"   原因: 連續 {early_stopping.patience} 代無顯著進步（min_delta={early_stopping.min_delta}）")
        print(f"   最佳 fitness: {early_stopping.best_fitness:.4f}")
    else:
        print(f"⏹️  早停: 否（完整運行）")
    
    print(f"⚡ 平均每代: {total_time/actual_generations:.1f} 秒")
    print()
    
    # ============================================================================
    # 儲存演化日誌
    # ============================================================================
    
    print("💾 儲存演化日誌...")
    
    # 儲存 JSON 日誌
    log_file = exp_dir / "evolution_log.json"
    log_data = {
        'config': CONFIG,
        'evolution_log': evolution_log,
        'total_time': total_time,
        'actual_generations': actual_generations,
        'final_statistics': {
            'best_fitness': float(hof[0].fitness.values[0]),
            'best_pnl': float(hof[0].fitness.values[0] * CONFIG['initial_capital'])
        }
    }
    
    # 添加早停資訊
    if early_stopping is not None:
        log_data['early_stopping'] = {
            'enabled': True,
            'triggered': early_stopping.should_stop,
            'status': early_stopping.get_status()
        }
    else:
        log_data['early_stopping'] = {
            'enabled': False,
            'triggered': False
        }
    
    # 添加 Niching 資訊
    if CONFIG['niching_enabled']:
        log_data['niching'] = {
            'enabled': True,
            'n_clusters': CONFIG['niching_n_clusters'],
            'cross_ratio': CONFIG['niching_cross_ratio'],
            'update_frequency': CONFIG['niching_update_frequency'],
            'algorithm': CONFIG['niching_algorithm'],
            'log': niching_log
        }
    else:
        log_data['niching'] = {
            'enabled': False
        }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"   ✓ {log_file}")
    
    # 儲存 CSV 日誌
    log_df = pd.DataFrame(evolution_log)
    csv_file = exp_dir / "evolution_log.csv"
    log_df.to_csv(csv_file, index=False)
    print(f"   ✓ {csv_file}")
    
    # ============================================================================
    # 最終分析
    # ============================================================================
    
    print()
    print("="*100)
    print("📊 最終分析")
    print("="*100)
    print()
    
    print("🏆 Top 10 最佳個體:")
    for i, ind in enumerate(hof, 1):
        fitness = ind.fitness.values[0]
        pnl = fitness * CONFIG['initial_capital']
        print(f"   {i:2d}. Fitness: {fitness:+.4f} ({fitness*100:+.2f}%) | "
              f"PnL: ${pnl:+,.0f} | "
              f"深度: {ind.height} | 節點: {len(ind)}")
    
    print()
    
    # 詳細回測最佳個體
    print("🔍 詳細回測最佳個體...")
    best_individual = hof[0]
    
    # ========================================================================
    # 訓練期（樣本內）回測
    # ========================================================================
    print("\n📊 訓練期（樣本內）績效:")
    print("="*80)
    
    try:
        train_result = train_engine.backtest(best_individual)
        train_metrics = train_result['metrics']
        train_per_stock_pnl = train_result['per_stock_pnl']
        
        print(f"\n組合績效:")
        print(f"  Total Return: {train_metrics['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {train_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {train_metrics['max_drawdown']*100:.2f}%")
        print(f"  Volatility: {train_metrics['volatility']*100:.2f}%")
        print(f"  Win Rate: {train_metrics['win_rate']*100:.2f}%")
        
        print(f"\n各股票 PnL 貢獻:")
        for ticker, pnl in train_per_stock_pnl.items():
            status = "✅" if pnl > 0 else "❌"
            print(f"  {ticker}: ${pnl:+,.2f} {status}")
        
        train_transactions = train_result['transactions']
        if len(train_transactions) > 0:
            buy_trades = len(train_transactions[train_transactions['action'] == 'BUY'])
            sell_trades = len(train_transactions[train_transactions['action'] == 'SELL'])
            print(f"\n交易統計: 總數 {len(train_transactions)} (買入: {buy_trades}, 賣出: {sell_trades})")
            
            # 儲存訓練期交易記錄
            train_trades_file = exp_dir / "best_individual_train_trades.csv"
            train_transactions.to_csv(train_trades_file, index=False)
        
    except Exception as e:
        print(f"   ✗ 訓練期回測失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # 測試期（樣本外）回測
    # ========================================================================
    print("\n📊 測試期（樣本外）績效:")
    print("="*80)
    
    try:
        test_result = test_engine.backtest(best_individual)
        test_metrics = test_result['metrics']
        test_per_stock_pnl = test_result['per_stock_pnl']
        
        print(f"\n組合績效:")
        print(f"  Total Return: {test_metrics['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {test_metrics['max_drawdown']*100:.2f}%")
        print(f"  Volatility: {test_metrics['volatility']*100:.2f}%")
        print(f"  Win Rate: {test_metrics['win_rate']*100:.2f}%")
        
        print(f"\n各股票 PnL 貢獻:")
        for ticker, pnl in test_per_stock_pnl.items():
            status = "✅" if pnl > 0 else "❌"
            print(f"  {ticker}: ${pnl:+,.2f} {status}")
        
        test_transactions = test_result['transactions']
        if len(test_transactions) > 0:
            buy_trades = len(test_transactions[test_transactions['action'] == 'BUY'])
            sell_trades = len(test_transactions[test_transactions['action'] == 'SELL'])
            print(f"\n交易統計: 總數 {len(test_transactions)} (買入: {buy_trades}, 賣出: {sell_trades})")
            
            # 儲存測試期交易記錄
            test_trades_file = exp_dir / "best_individual_test_trades.csv"
            test_transactions.to_csv(test_trades_file, index=False)
        
        # ====================================================================
        # 比較訓練期 vs 測試期
        # ====================================================================
        print("\n📈 訓練期 vs 測試期比較:")
        print("="*80)
        print(f"  {'指標':<20} {'訓練期':>15} {'測試期':>15} {'差異':>15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        print(f"  {'Total Return':<20} {train_metrics['total_return']*100:>14.2f}% {test_metrics['total_return']*100:>14.2f}% {(test_metrics['total_return']-train_metrics['total_return'])*100:>+14.2f}%")
        print(f"  {'Sharpe Ratio':<20} {train_metrics['sharpe_ratio']:>15.3f} {test_metrics['sharpe_ratio']:>15.3f} {test_metrics['sharpe_ratio']-train_metrics['sharpe_ratio']:>+15.3f}")
        print(f"  {'Max Drawdown':<20} {train_metrics['max_drawdown']*100:>14.2f}% {test_metrics['max_drawdown']*100:>14.2f}% {(test_metrics['max_drawdown']-train_metrics['max_drawdown'])*100:>+14.2f}%")
        print(f"  {'Volatility':<20} {train_metrics['volatility']*100:>14.2f}% {test_metrics['volatility']*100:>14.2f}% {(test_metrics['volatility']-train_metrics['volatility'])*100:>+14.2f}%")
        
        # 儲存完整結果
        best_result_file = exp_dir / "best_individual_result.json"
        with open(best_result_file, 'w') as f:
            json.dump({
                'individual': str(best_individual),
                'train_fitness': float(best_individual.fitness.values[0]),
                'train_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in train_metrics.items()},
                'train_per_stock_pnl': {k: float(v) for k, v in train_per_stock_pnl.items()},
                'train_total_trades': len(train_transactions),
                'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                              for k, v in test_metrics.items()},
                'test_per_stock_pnl': {k: float(v) for k, v in test_per_stock_pnl.items()},
                'test_total_trades': len(test_transactions)
            }, f, indent=2)
        
        print(f"\n💾 結果已儲存:")
        print(f"   ✓ {best_result_file}")
        if len(train_transactions) > 0:
            print(f"   ✓ {train_trades_file}")
        if len(test_transactions) > 0:
            print(f"   ✓ {test_trades_file}")
        
    except Exception as e:
        print(f"   ✗ 回測失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("="*100)
    print("🎉 實驗完成！")
    print("="*100)
    print()
    
    print(f"📁 所有結果已儲存至: {exp_dir}")
    print(f"\n目錄結構:")
    print(f"  {exp_dir}/")
    print(f"  ├── config.json                         (實驗配置)")
    print(f"  ├── evolution_log.json                  (演化日誌)")
    print(f"  ├── evolution_log.csv                   (演化日誌 CSV)")
    print(f"  ├── best_individual_result.json         (最佳個體完整結果)")
    print(f"  ├── best_individual_train_trades.csv    (訓練期交易記錄)")
    print(f"  ├── best_individual_test_trades.csv     (測試期交易記錄)")
    print(f"  └── generations/                        (族群快照)")
    print(f"      ├── generation_001.pkl")
    print(f"      ├── generation_002.pkl")
    print(f"      ├── ...")
    print(f"      └── generation_{CONFIG['generations']:03d}.pkl")
    print()

if __name__ == '__main__':
    main()
