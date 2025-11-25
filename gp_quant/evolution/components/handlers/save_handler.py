"""
保存處理器 - 負責保存演化過程中的數據
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .base import EventHandler
from ..individual import EvolutionIndividual


class SaveHandler(EventHandler):
    """保存處理器 - 保存演化數據到文件"""
    
    def __init__(self, records_dir: str = "evolution_records", save_populations: bool = True, 
                 save_genealogy: bool = True, save_format: str = "json", **kwargs):
        """
        初始化保存處理器
        
        Args:
            records_dir: 記錄保存目錄
            save_populations: 是否保存每世代族群
            save_genealogy: 是否保存譜系信息
            save_format: 保存格式 ("json" 或 "pickle")
        """
        super().__init__()
        self.records_dir = Path(records_dir)
        self.save_populations = save_populations
        self.save_genealogy = save_genealogy
        self.save_format = save_format
        
        # 創建保存目錄
        self.records_dir.mkdir(exist_ok=True)
        if self.save_populations:
            (self.records_dir / "populations").mkdir(exist_ok=True)
        if self.save_genealogy:
            (self.records_dir / "genealogy").mkdir(exist_ok=True)
            
        # 保存統計數據
        self.generation_stats = []
        # 追蹤 global best 個體 ID，用於判斷何時更新全局最佳訊號
        self.global_best_id: str | None = None
        
    def handle_event(self, event_name: str, **kwargs):
        """處理事件的通用方法"""
        if event_name == 'evolution_start':
            self.on_evolution_start(**kwargs)
        elif event_name == 'generation_complete':
            self.on_generation_complete(**kwargs)
        elif event_name == 'evolution_complete':
            self.on_evolution_complete(**kwargs)
        
    def on_evolution_start(self, engine, **kwargs):
        """演化開始時的處理"""
        print(f"💾 保存處理器啟動")
        print(f"   📁 記錄目錄: {self.records_dir}")
        print(f"   👥 保存族群: {'✅' if self.save_populations else '❌'}")
        print(f"   🧬 保存譜系: {'✅' if self.save_genealogy else '❌'}")
        print(f"   📄 保存格式: {self.save_format}")
        
        # 保存初始配置
        config_file = self.records_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(engine.config, f, indent=2, ensure_ascii=False)
            
    def on_generation_complete(self, generation: int, population: List[EvolutionIndividual], 
                             best_individual: EvolutionIndividual, engine=None, **kwargs):
        """每世代完成時的處理"""
        print(f"💾 保存第 {generation} 世代數據...")
        
        # 保存族群數據
        if self.save_populations:
            self._save_population(generation, population)
            
        # 保存統計數據
        self._save_generation_stats(generation, population, best_individual, engine)
        
        # 保存譜系數據
        if self.save_genealogy:
            self._save_genealogy(generation, population)
        
        # 1) 保存當代 generation best 的交易訊號
        if engine is not None:
            # 基於當代族群 fitness 計算當代最佳個體
            valid_inds = [ind for ind in population
                          if hasattr(ind, 'fitness') and getattr(ind.fitness, 'values', None)]
            gen_best = None
            if valid_inds:
                gen_best = max(valid_inds, key=lambda ind: ind.fitness.values[0])

            if gen_best is not None:
                # 每一代都輸出對應的 generation_XYZ 目錄
                self._save_best_individual_signals(
                    generation,
                    gen_best,
                    engine,
                    subdir_name=f"generation_{generation:03d}"
                )

        # 2) 保存 / 更新 global best 的交易訊號
        # best_individual 由引擎提供，預期為 global best so far
        if best_individual is not None and engine is not None:
            current_id = getattr(best_individual, 'id', None)
            if current_id is not None and current_id != self.global_best_id:
                # global best 發生更新，重新輸出 global 目錄
                self.global_best_id = current_id
                self._save_best_individual_signals(
                    generation,
                    best_individual,
                    engine,
                    subdir_name="global"
                )
            
    def on_evolution_complete(self, engine, result, **kwargs):
        """演化完成時的處理"""
        print(f"💾 保存最終結果...")
        
        # 保存完整統計數據
        stats_file = self.records_dir / "generation_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.generation_stats, f, indent=2, ensure_ascii=False)
            
        # 保存最終結果
        result_file = self.records_dir / "final_result.json"
        result_data = {
            'experiment_name': engine.config.get('experiment', {}).get('name', 'unknown'),
            'final_generation': result.final_generation,
            'best_fitness': result.best_fitness,
            'total_evaluations': result.total_evaluations,
            'execution_time': result.execution_time,
            'convergence_generation': result.convergence_generation,
            'improvement_rate': result.improvement_rate,
            'fitness_statistics': result.get_fitness_statistics()
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
            
        # 保存完整的演化引擎狀態 (可重載)
        self._save_engine_state(engine, result)
            
        print(f"✅ 數據保存完成!")
        print(f"   📊 統計數據: {stats_file}")
        print(f"   🏆 最終結果: {result_file}")
        print(f"   🔄 演化狀態: {self.records_dir / 'engine_state.pkl'}")
        
    def _save_engine_state(self, engine, result):
        """保存完整的演化引擎狀態"""
        engine_state = {
            'engine': engine,
            'result': result,
            'current_generation': engine.current_generation,
            'population': engine.population,
            'best_individual': engine.best_individual,
            'fitness_history': engine.fitness_history,
            'config': engine.config,
            'generation_stats': self.generation_stats
        }
        
        state_file = self.records_dir / "engine_state.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(engine_state, f)
        
    def _save_population(self, generation: int, population: List[EvolutionIndividual]):
        """保存族群數據"""
        # 1. 保存完整的個體對象 (Pickle格式 - 可重載)
        pickle_file = self.records_dir / "populations" / f"generation_{generation:03d}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(population, f)
            
        # 2. 保存可讀的 JSON 統計數據
        pop_data = []
        for i, individual in enumerate(population):
            ind_data = {
                'index': i,
                'id': individual.id,
                'generation': individual.generation,
                'fitness': individual.fitness.values[0] if hasattr(individual.fitness, 'values') and individual.fitness.values else None,
                'operation': individual.operation,
                'parents': individual.parents,
                'tree_size': len(individual),
                'tree_depth': individual.height,
                'tree_str': str(individual),
                'evaluation_count': individual.evaluation_count
            }
            pop_data.append(ind_data)
            
        pop_json_file = self.records_dir / "populations" / f"generation_{generation:03d}_stats.json"
        with open(pop_json_file, 'w', encoding='utf-8') as f:
            json.dump(pop_data, f, indent=2, ensure_ascii=False)
                
    def _save_generation_stats(self, generation: int, population: List[EvolutionIndividual], 
                             best_individual: EvolutionIndividual, engine=None):
        """保存世代統計數據"""
        # 計算統計數據
        valid_individuals = [ind for ind in population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        
        if valid_individuals:
            fitness_values = [ind.fitness.values[0] for ind in valid_individuals]
            stats = {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'population_size': len(population),
                'valid_individuals': len(valid_individuals),
                'best_fitness': max(fitness_values),
                'worst_fitness': min(fitness_values),
                'avg_fitness': sum(fitness_values) / len(fitness_values),
                'fitness_std': self._calculate_std(fitness_values),
                'avg_tree_size': sum(len(ind) for ind in valid_individuals) / len(valid_individuals),
                'avg_tree_depth': sum(ind.height for ind in valid_individuals) / len(valid_individuals),
                'best_individual_id': best_individual.id if best_individual else None,
                'best_tree_size': len(best_individual) if best_individual else None,
                'best_tree_depth': best_individual.height if best_individual else None
            }
        else:
            stats = {
                'generation': generation,
                'timestamp': datetime.now().isoformat(),
                'population_size': len(population),
                'valid_individuals': 0,
                'best_fitness': None,
                'worst_fitness': None,
                'avg_fitness': None,
                'fitness_std': None,
                'avg_tree_size': None,
                'avg_tree_depth': None,
                'best_individual_id': None,
                'best_tree_size': None,
                'best_tree_depth': None
            }
        
        # 添加選擇策略的統計信息（如果可用）
        if engine and hasattr(engine, 'strategies'):
            selection_strategy = engine.strategies.get('selection')
            if selection_strategy and hasattr(selection_strategy, 'get_stats'):
                strategy_stats = selection_strategy.get_stats()
                stats['selection_strategy'] = strategy_stats
            
        self.generation_stats.append(stats)
        
    def _save_genealogy(self, generation: int, population: List[EvolutionIndividual]):
        """保存譜系數據"""
        genealogy_data = []
        for individual in population:
            genealogy_entry = {
                'id': individual.id,
                'generation': individual.generation,
                'operation': individual.operation,
                'parents': individual.parents,
                'fitness': individual.fitness.values[0] if hasattr(individual.fitness, 'values') and individual.fitness.values else None,
                'created_at': datetime.now().isoformat()
            }
            genealogy_data.append(genealogy_entry)
            
        genealogy_file = self.records_dir / "genealogy" / f"generation_{generation:03d}.json"
        with open(genealogy_file, 'w', encoding='utf-8') as f:
            json.dump(genealogy_data, f, indent=2, ensure_ascii=False)
            
    def _calculate_std(self, values: List[float]) -> float:
        """計算標準差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _save_best_individual_signals(self, generation: int, best_individual: EvolutionIndividual, engine, subdir_name: str | None = None):
        """
        保存最佳個體的交易訊號和回測結果
        
        Args:
            generation: 當前世代
            best_individual: 最佳個體
            engine: 演化引擎（包含評估器）
        """
        try:
            import pandas as pd
            import numpy as np
            
            # 創建訊號保存目錄
            signals_dir = self.records_dir / "best_signals"
            signals_dir.mkdir(exist_ok=True)

            # 允許指定子目錄名稱：
            # - generation_XXX：當代最佳
            # - global：全局最佳
            target_subdir = subdir_name if subdir_name is not None else f"generation_{generation:03d}"
            gen_dir = signals_dir / target_subdir
            gen_dir.mkdir(exist_ok=True)
            
            # 獲取評估器
            evaluator = engine.evaluator
            if not hasattr(evaluator, 'backtest_engine') or evaluator.backtest_engine is None:
                return
            
            backtest_engine = evaluator.backtest_engine
            
            # 執行回測
            result = backtest_engine.backtest(best_individual)
            
            # 1. 保存交易記錄 (entry/exit points)
            transactions = result.get('transactions', [])
            if len(transactions) > 0:
                tx_df = pd.DataFrame(transactions)
                tx_csv_path = gen_dir / 'entry_exit_points.csv'
                tx_df.to_csv(tx_csv_path, index=False)
            
            # 2. 保存每個股票的每日訊號
            all_signals = backtest_engine._generate_signals_for_all_stocks(best_individual)
            
            for ticker, ticker_signals_dict in all_signals.items():
                # 獲取該股票的數據
                ticker_df = backtest_engine.backtest_data[ticker]
                
                # 創建 DataFrame
                backtest_dates = ticker_df.index
                backtest_prices = ticker_df['Close'].values
                
                # 轉換訊號字典為數組
                signals_array = np.array([ticker_signals_dict.get(date, 0) for date in backtest_dates])
                
                signal_df = pd.DataFrame({
                    'Date': backtest_dates,
                    'Close': backtest_prices,
                    'Signal': signals_array
                })
                
                # 保存到 CSV
                signal_csv_path = gen_dir / f'signals_{ticker}.csv'
                signal_df.to_csv(signal_csv_path, index=False)
            
            # 3. 保存回測摘要
            summary = {
                'generation': generation,
                'individual_id': best_individual.id,
                'fitness': best_individual.fitness.values[0] if hasattr(best_individual.fitness, 'values') and best_individual.fitness.values else None,
                'rule': str(best_individual),
                'metrics': result.get('metrics', {}),
                'total_transactions': len(transactions),
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = gen_dir / 'backtest_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"   📊 最佳個體訊號已保存: {gen_dir.name}")
            
        except Exception as e:
            print(f"   ⚠️ 保存訊號時出錯: {e}")
            import traceback
            traceback.print_exc()
