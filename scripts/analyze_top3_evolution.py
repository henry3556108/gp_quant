#!/usr/bin/env python3
"""
Top3 個體演化分析腳本 - 分析每世代前三名個體的回測績效
"""
import argparse
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gp_quant.data.loader import load_and_process_data
from gp_quant.evolution.components.loader import EvolutionLoader
from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine


def setup_deap_creator():
    """初始化 DEAP creator"""
    from deap import creator, base
    
    # 檢查是否已經創建
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    if not hasattr(creator, 'Individual'):
        from deap import gp
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def run_backtest_for_individual(individual, train_data, test_data, config):
    """
    為單個個體運行完整回測
    
    Args:
        individual: 要回測的個體
        train_data: 訓練數據
        test_data: 測試數據
        config: 配置
        
    Returns:
        包含訓練和測試結果的字典
    """
    from gp_quant.evolution.components.gp import pset
    
    results = {
        'individual_id': individual.id,
        'fitness': individual.fitness.values[0] if hasattr(individual.fitness, 'values') and individual.fitness.values else None,
        'tree_size': len(individual),
        'tree_depth': individual.height,
        'tree_expression': str(individual),
        'train_backtest': None,
        'test_backtest': None
    }
    
    # 訓練數據回測
    try:
        # 處理數據格式 - 提取 DataFrame
        processed_train_data = {}
        for ticker, ticker_data in train_data.items():
            if isinstance(ticker_data, dict) and 'data' in ticker_data:
                processed_train_data[ticker] = ticker_data['data']
            else:
                processed_train_data[ticker] = ticker_data
        
        train_engine = PortfolioBacktestingEngine(
            data=processed_train_data,
            backtest_start=config['data']['train_backtest_start'],
            backtest_end=config['data']['train_backtest_end'],
            initial_capital=100000.0,
            pset=pset
        )
        
        train_results = train_engine.backtest(individual)
        
        # 從 equity_curve 提取數據
        equity_curve = train_results.get('equity_curve')
        if equity_curve is not None and len(equity_curve) > 0:
            portfolio_values = equity_curve.values.tolist()
            dates = equity_curve.index.tolist()
        else:
            portfolio_values = []
            dates = []
        
        results['train_backtest'] = {
            'fitness': train_results['metrics']['excess_return'],
            'metrics': train_results['metrics'],
            'portfolio_values': portfolio_values,
            'dates': dates,
            'equity_curve': equity_curve
        }
        
    except Exception as e:
        print(f"   ⚠️ 訓練回測失敗: {e}")
        results['train_backtest'] = {'error': str(e)}
    
    # 測試數據回測
    try:
        # 處理數據格式 - 提取 DataFrame
        processed_test_data = {}
        for ticker, ticker_data in test_data.items():
            if isinstance(ticker_data, dict) and 'data' in ticker_data:
                processed_test_data[ticker] = ticker_data['data']
            else:
                processed_test_data[ticker] = ticker_data
        
        test_engine = PortfolioBacktestingEngine(
            data=processed_test_data,
            backtest_start=config['data']['test_backtest_start'],
            backtest_end=config['data']['test_backtest_end'],
            initial_capital=100000.0,
            pset=pset
        )
        
        test_results = test_engine.backtest(individual)
        
        # 從 equity_curve 提取數據
        equity_curve = test_results.get('equity_curve')
        if equity_curve is not None and len(equity_curve) > 0:
            portfolio_values = equity_curve.values.tolist()
            dates = equity_curve.index.tolist()
        else:
            portfolio_values = []
            dates = []
        
        results['test_backtest'] = {
            'fitness': test_results['metrics']['excess_return'],
            'metrics': test_results['metrics'],
            'portfolio_values': portfolio_values,
            'dates': dates,
            'equity_curve': equity_curve
        }
        
    except Exception as e:
        print(f"   ⚠️ 測試回測失敗: {e}")
        results['test_backtest'] = {'error': str(e)}
    
    return results


def plot_individual_performance(individual_results, output_dir, generation, rank):
    """
    繪製單個個體的績效圖表
    
    Args:
        individual_results: 個體回測結果
        output_dir: 輸出目錄
        generation: 世代
        rank: 排名
    """
    individual_id = individual_results['individual_id']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Gen {generation} Rank {rank} - Individual {individual_id[:8]}...\n'
                f'Fitness: {individual_results["fitness"]:.6f} | '
                f'Tree: {individual_results["tree_size"]} nodes, depth {individual_results["tree_depth"]}', 
                fontsize=14, fontweight='bold')
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 訓練數據績效
    train_data = individual_results.get('train_backtest')
    if train_data and 'portfolio_values' in train_data and train_data['portfolio_values']:
        portfolio_values = train_data['portfolio_values']
        dates = train_data['dates']
        
        # 投資組合價值
        axes[0, 0].plot(dates, portfolio_values, 'b-', linewidth=2)
        axes[0, 0].set_title(f'Training Portfolio Value\nFitness: {train_data["fitness"]:.6f}')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 累積報酬
        cumulative_returns = (np.array(portfolio_values) / portfolio_values[0] - 1) * 100
        axes[0, 1].plot(dates, cumulative_returns, 'g-', linewidth=2)
        axes[0, 1].set_title('Training Cumulative Returns')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 顯示指標
        metrics = train_data.get('metrics', {})
        axes[0, 0].text(0.02, 0.98, 
                       f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
                       f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}\n"
                       f"Max DD: {metrics.get('max_drawdown', 0):.2f}%",
                       transform=axes[0, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[0, 0].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 1].text(0.5, 0.5, 'No Training Data', ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # 測試數據績效
    test_data = individual_results.get('test_backtest')
    if test_data and 'portfolio_values' in test_data and test_data['portfolio_values']:
        portfolio_values = test_data['portfolio_values']
        # 提取績效數據
        performance_data = None
        try:
            # 從回測結果中提取績效數據 - 使用 equity_curve
            equity_curve = test_data.get('equity_curve')
            
            if equity_curve is not None and len(equity_curve) > 0:
                # equity_curve 是一個 pandas Series，索引是日期
                dates = equity_curve.index.tolist()
                portfolio_values = equity_curve.values.tolist()
                
                performance_data = {
                    'dates': dates,
                    'portfolio_values': portfolio_values,
                    'returns': np.diff(portfolio_values) / portfolio_values[:-1],
                    'cumulative_returns': (np.array(portfolio_values) / portfolio_values[0] - 1) * 100
                }
                
                print(f"   📊 績效數據: {len(dates)} 個交易日")
                print(f"   💰 最終價值: ${portfolio_values[-1]:,.2f}")
                print(f"   📈 總報酬: {performance_data['cumulative_returns'][-1]:.2f}%")
                
                # 顯示其他指標
                metrics = test_data.get('metrics', {})
                print(f"   📈 總收益率: {metrics.get('total_return', 0):.2f}%")
                print(f"   📊 夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
                print(f"   📉 最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
                
                # 檢查是否有交易
                transactions = test_data.get('transactions')
                if transactions is not None and len(transactions) > 0:
                    print(f"   💼 交易次數: {len(transactions)}")
                else:
                    print(f"   ⚠️ 警告: 沒有產生任何交易信號")
            else:
                print(f"   ⚠️ 無法提取 equity_curve")
            
        except Exception as e:
            print(f"   ⚠️ 無法提取詳細績效數據: {e}")
            import traceback
            traceback.print_exc()
        
        # 投資組合價值
        axes[1, 0].plot(performance_data['dates'], performance_data['portfolio_values'], 'r-', linewidth=2)
        axes[1, 0].set_title(f'Test Portfolio Value\nFitness: {test_data["fitness"]:.6f}')
        axes[1, 0].set_ylabel('Portfolio Value ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累積報酬
        axes[1, 1].plot(performance_data['dates'], performance_data['cumulative_returns'], 'orange', linewidth=2)
        axes[1, 1].set_title('Test Cumulative Returns')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 顯示指標
        metrics = test_data.get('metrics', {})
        axes[1, 0].text(0.02, 0.98, 
                       f"Total Return: {metrics.get('total_return', 0):.2f}%\n"
                       f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}\n"
                       f"Max DD: {metrics.get('max_drawdown', 0):.2f}%",
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        axes[1, 0].text(0.5, 0.5, 'No Test Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 1].text(0.5, 0.5, 'No Test Data', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # 保存圖表
    plot_file = output_dir / f"gen_{generation:02d}_rank_{rank}_individual_{individual_id[:8]}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()  # 關閉圖表以節省記憶體
    
    return plot_file


def analyze_top3_evolution(records_dir: str, config: dict, train_data: dict, test_data: dict, 
                          generate_plots: bool = True, top_n: int = 3):
    """
    分析每世代 Top N 個體的演化過程
    
    Args:
        records_dir: 記錄目錄
        config: 配置
        train_data: 訓練數據
        test_data: 測試數據
        generate_plots: 是否生成績效圖表
        top_n: 分析前N名個體
    """
    print(f"📊 開始分析每世代 Top{top_n} 個體演化過程...")
    
    loader = EvolutionLoader(records_dir)
    available_generations = loader.get_available_generations()
    
    if not available_generations:
        print("❌ 沒有可用的世代數據")
        return
    
    # 創建輸出目錄
    output_dir = Path(records_dir) / "top3_analysis"
    output_dir.mkdir(exist_ok=True)
    
    if generate_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for generation in available_generations:
        print(f"\n🔄 分析世代 {generation}...")
        
        try:
            # 載入族群
            population = loader.load_population(generation)
            
            # 找到有效個體並排序
            valid_individuals = [ind for ind in population 
                               if hasattr(ind.fitness, 'values') and ind.fitness.values]
            
            if len(valid_individuals) < top_n:
                print(f"   ⚠️ 世代 {generation} 只有 {len(valid_individuals)} 個有效個體")
                continue
            
            # 按適應度排序，取前N名
            top_individuals = sorted(valid_individuals, 
                                   key=lambda ind: ind.fitness.values[0], 
                                   reverse=True)[:top_n]
            
            generation_results = []
            
            for rank, individual in enumerate(top_individuals, 1):
                print(f"   📈 分析第 {rank} 名個體 (ID: {individual.id[:8]}..., "
                      f"Fitness: {individual.fitness.values[0]:.6f})")
                
                # 運行回測
                individual_results = run_backtest_for_individual(
                    individual, train_data, test_data, config)
                individual_results['generation'] = generation
                individual_results['rank'] = rank
                
                # 生成績效圖表
                if generate_plots:
                    plot_file = plot_individual_performance(
                        individual_results, plots_dir, generation, rank)
                    individual_results['plot_file'] = str(plot_file)
                    print(f"      📊 績效圖表已保存: {plot_file.name}")
                
                generation_results.append(individual_results)
            
            all_results.extend(generation_results)
            
        except Exception as e:
            print(f"   ❌ 世代 {generation} 分析失敗: {e}")
            continue
    
    # 保存完整結果
    results_file = output_dir / "top3_evolution_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    # 生成摘要報告
    summary = generate_summary_report(all_results, output_dir)
    
    print(f"\n✅ Top{top_n} 演化分析完成!")
    print(f"   📊 分析了 {len(available_generations)} 個世代")
    print(f"   📈 總共分析了 {len(all_results)} 個個體")
    print(f"   📄 詳細結果: {results_file}")
    print(f"   📋 摘要報告: {output_dir / 'evolution_summary.json'}")
    if generate_plots:
        print(f"   🎨 績效圖表: {plots_dir}")
    
    return all_results


def generate_summary_report(all_results, output_dir):
    """生成摘要報告"""
    summary = {
        'analysis_time': datetime.now().isoformat(),
        'total_individuals': len(all_results),
        'generations_analyzed': len(set(r['generation'] for r in all_results)),
        'fitness_evolution': {},
        'best_performers': {}
    }
    
    # 按世代分組分析適應度演化
    by_generation = {}
    for result in all_results:
        gen = result['generation']
        if gen not in by_generation:
            by_generation[gen] = []
        by_generation[gen].append(result)
    
    # 適應度演化趨勢
    for gen, results in by_generation.items():
        fitness_values = [r['fitness'] for r in results if r['fitness'] is not None]
        if fitness_values:
            summary['fitness_evolution'][f'generation_{gen}'] = {
                'best_fitness': max(fitness_values),
                'avg_fitness': sum(fitness_values) / len(fitness_values),
                'worst_fitness': min(fitness_values)
            }
    
    # 找出最佳表現者
    train_performers = [r for r in all_results 
                       if r.get('train_backtest') and 'fitness' in r['train_backtest']]
    test_performers = [r for r in all_results 
                      if r.get('test_backtest') and 'fitness' in r['test_backtest']]
    
    if train_performers:
        best_train = max(train_performers, 
                        key=lambda x: x['train_backtest']['fitness'])
        summary['best_performers']['training'] = {
            'individual_id': best_train['individual_id'],
            'generation': best_train['generation'],
            'rank': best_train['rank'],
            'fitness': best_train['train_backtest']['fitness']
        }
    
    if test_performers:
        best_test = max(test_performers, 
                       key=lambda x: x['test_backtest']['fitness'])
        summary['best_performers']['testing'] = {
            'individual_id': best_test['individual_id'],
            'generation': best_test['generation'],
            'rank': best_test['rank'],
            'fitness': best_test['test_backtest']['fitness']
        }
    
    # 保存摘要
    summary_file = output_dir / "evolution_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='分析每世代 Top3 個體的演化過程和回測績效',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python analyze_top3_evolution.py --records small_test_500x10_records --plots
  python analyze_top3_evolution.py --records small_test_500x10_records --no-plots --top-n 5
        """
    )
    
    parser.add_argument('--records', required=True,
                       help='演化記錄目錄路徑')
    parser.add_argument('--config',
                       help='配置文件路徑')
    parser.add_argument('--plots', action='store_true', default=True,
                       help='生成績效圖表 (默認開啟)')
    parser.add_argument('--no-plots', action='store_true',
                       help='不生成績效圖表')
    parser.add_argument('--top-n', type=int, default=3,
                       help='分析前N名個體 (默認: 3)')
    
    args = parser.parse_args()
    
    # 設置 DEAP creator
    setup_deap_creator()
    
    print("📊" * 60)
    print("📊 Top3 個體演化分析")
    print("📊" * 60)
    
    try:
        # 載入配置
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = Path(args.records) / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📄 載入配置: {config_path}")
        
        # 載入數據
        print("📊 載入數據...")
        from gp_quant.data.loader import split_train_test_data
        
        tickers_dir = Path(config['data']['tickers_dir'])
        if not tickers_dir.is_absolute():
            tickers_dir = Path.cwd() / tickers_dir
        
        tickers = [f.stem for f in tickers_dir.glob("*.csv")]
        raw_data = load_and_process_data(str(tickers_dir), tickers)
        
        train_data, test_data = split_train_test_data(
            raw_data,
            train_data_start=config['data']['train_data_start'],
            train_backtest_start=config['data']['train_backtest_start'],
            train_backtest_end=config['data']['train_backtest_end'],
            test_data_start=config['data']['test_data_start'],
            test_backtest_start=config['data']['test_backtest_start'],
            test_backtest_end=config['data']['test_backtest_end']
        )
        
        print(f"✅ 數據載入完成: 訓練({len(train_data)}), 測試({len(test_data)})")
        
        # 決定是否生成圖表
        generate_plots = args.plots and not args.no_plots
        
        # 執行分析
        results = analyze_top3_evolution(
            args.records, config, train_data, test_data, 
            generate_plots=generate_plots, top_n=args.top_n)
        
        if results:
            print(f"\n🎉 分析成功完成!")
            return 0
        else:
            print(f"\n❌ 分析失敗!")
            return 1
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
