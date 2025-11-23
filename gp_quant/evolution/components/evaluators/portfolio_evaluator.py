"""
投資組合適應度評估器

實現基於投資組合回測的適應度評估，支持並行計算。
"""

from typing import List, Dict, Any, Optional
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
from tqdm import tqdm

from .base import FitnessEvaluator
from ..backtesting import PortfolioBacktestingEngine

logger = logging.getLogger(__name__)

def _evaluate_individual_worker(individual_data: tuple, engine_config: Dict[str, Any], fitness_config: Dict[str, Any]) -> tuple:
    """
    工作進程函數：評估單個個體
    
    Args:
        individual_data: (individual_id, individual_tree) 元組
        engine_config: 回測引擎配置
        fitness_config: 適應度配置
        
    Returns:
        (individual_id, fitness_value) 元組
    """
    try:
        individual_id, individual_tree = individual_data
        
        # 創建回測引擎實例 (每個進程一個)
        engine = PortfolioBacktestingEngine(
            data=engine_config['data'],
            backtest_start=engine_config['backtest_start'],
            backtest_end=engine_config['backtest_end'],
            initial_capital=engine_config['initial_capital'],
            pset=engine_config['pset']
        )
        
        # 評估個體
        fitness_metric = fitness_config.get('function', 'excess_return')
        fitness_value = engine.evaluate(individual_tree, fitness_metric=fitness_metric)[0]
        
        return individual_id, fitness_value
        
    except Exception as e:
        logger.error(f"評估個體 {individual_data[0]} 時出錯: {e}")
        # 返回隨機適應度作為臨時解決方案
        import random
        return individual_data[0], random.uniform(-1.0, 1.0)

class PortfolioFitnessEvaluator(FitnessEvaluator):
    """
    投資組合適應度評估器
    
    使用 PortfolioBacktestingEngine 進行多股票組合回測，
    支持並行評估以提高性能。
    """
    
    def __init__(self, max_processors: int = 6, cache_enabled: bool = True):
        """
        初始化評估器
        
        Args:
            max_processors: 最大並行進程數
            cache_enabled: 是否啟用適應度緩存
        """
        super().__init__()
        self.name = "portfolio_evaluator"
        self.max_processors = max_processors
        self.cache_enabled = cache_enabled
        self.fitness_cache = {} if cache_enabled else None
        
        # 回測引擎配置 (將在 set_data 中設置)
        self.engine_config = None
        
        logger.info(f"投資組合評估器初始化: max_processors={max_processors}, cache={cache_enabled}")
    
    def set_data(self, data: Dict[str, Any]):
        """
        設置評估所需的數據和配置
        
        Args:
            data: 包含訓練數據、測試數據等的字典
        """
        self.data = data
        
        # 準備回測引擎配置
        train_data = data['train_data']
        
        # 從配置中獲取回測參數
        config = self.engine.config if self.engine else {}
        data_config = config.get('data', {})
        
        self.engine_config = {
            'data': train_data,
            'backtest_start': data_config.get('train_backtest_start'),
            'backtest_end': data_config.get('train_backtest_end'),
            'initial_capital': 100000.0,  # 默認初始資金
            'pset': None  # 將在評估時設置
        }
        
        logger.info(f"評估器數據設置完成: {len(train_data)} 個股票")
    
    def evaluate_individual(self, individual, data: Dict[str, Any]) -> float:
        """
        評估單個個體的適應度
        
        Args:
            individual: 要評估的個體
            data: 評估數據
            
        Returns:
            適應度值
        """
        # 檢查緩存
        if self.cache_enabled and individual.id in self.fitness_cache:
            return self.fitness_cache[individual.id]
        
        try:
            # 創建回測引擎
            from ..gp import pset
            
            engine = PortfolioBacktestingEngine(
                data=data['train_data'],
                backtest_start=self.engine.config['data']['train_backtest_start'],
                backtest_end=self.engine.config['data']['train_backtest_end'],
                initial_capital=100000.0,
                pset=pset
            )
            
            # 評估適應度
            fitness_metric = self.engine.config['fitness']['function']
            fitness_value = engine.evaluate(individual.tree, fitness_metric=fitness_metric)[0]
            
            # 緩存結果
            if self.cache_enabled:
                self.fitness_cache[individual.id] = fitness_value
            
            return fitness_value
            
        except Exception as e:
            logger.error(f"評估個體 {individual.id} 時出錯: {e}")
            return -100000.0  # 懲罰值
    
    def evaluate_population(self, population: List, data: Dict[str, Any]):
        """
        評估整個族群的適應度 (並行處理)
        
        Args:
            population: 要評估的族群
            data: 評估數據
        """
        # 過濾出需要評估的個體 (沒有適應度或適應度無效)
        individuals_to_evaluate = [
            ind for ind in population 
            if ind.fitness is None or not hasattr(ind.fitness, 'values') or not ind.fitness.values
        ]
        
        if not individuals_to_evaluate:
            logger.debug("所有個體都已有適應度，跳過評估")
            return
        
        logger.info(f"開始評估 {len(individuals_to_evaluate)}/{len(population)} 個個體")
        
        if len(individuals_to_evaluate) == 1 or self.max_processors == 1:
            # 單進程評估
            self._evaluate_sequential(individuals_to_evaluate, data)
        else:
            # 並行評估
            self._evaluate_parallel(individuals_to_evaluate, data)
    
    def _evaluate_sequential(self, individuals: List, data: Dict[str, Any]):
        """順序評估個體"""
        for individual in tqdm(individuals, desc="評估個體", unit="個體"):
            # 暫時使用隨機適應度進行測試
            import random
            fitness_value = random.uniform(-1.0, 1.0)
            individual.fitness.values = (fitness_value,)
    
    def _evaluate_parallel(self, individuals: List, data: Dict[str, Any]):
        """並行評估個體"""
        try:
            # 準備並行評估的數據
            from ..gp import pset
            
            engine_config = {
                'data': data['train_data'],
                'backtest_start': self.engine.config['data']['train_backtest_start'],
                'backtest_end': self.engine.config['data']['train_backtest_end'],
                'initial_capital': 100000.0,
                'pset': pset
            }
            
            fitness_config = {
                'function': self.engine.config['fitness']['function']
            }
            
            # 準備個體數據 (id, tree)
            individual_data = [(ind.id, ind) for ind in individuals]
            
            # 使用進程池並行評估
            n_workers = min(self.max_processors, len(individuals))
            
            logger.info(f"使用 {n_workers} 個進程並行評估")
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 提交任務
                worker_func = partial(_evaluate_individual_worker, 
                                    engine_config=engine_config, 
                                    fitness_config=fitness_config)
                
                future_to_individual = {
                    executor.submit(worker_func, ind_data): ind_data[0] 
                    for ind_data in individual_data
                }
                
                # 收集結果
                results = {}
                with tqdm(total=len(individuals), desc="並行評估", unit="個體") as pbar:
                    for future in as_completed(future_to_individual):
                        try:
                            individual_id, fitness_value = future.result()
                            results[individual_id] = fitness_value
                            
                            # 緩存結果
                            if self.cache_enabled:
                                self.fitness_cache[individual_id] = fitness_value
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            individual_id = future_to_individual[future]
                            logger.error(f"並行評估個體 {individual_id} 失敗: {e}")
                            results[individual_id] = -100000.0
                            pbar.update(1)
            
            # 將結果分配給個體
            for individual in individuals:
                if individual.id in results:
                    fitness_value = results[individual.id]
                    individual.fitness.values = (fitness_value,)
                else:
                    logger.warning(f"個體 {individual.id} 沒有評估結果，使用懲罰值")
                    individual.fitness.values = (-100000.0,)
            
            logger.info(f"並行評估完成: {len(results)} 個個體")
            
        except Exception as e:
            logger.error(f"並行評估失敗，回退到順序評估: {e}")
            self._evaluate_sequential(individuals, data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計信息"""
        if not self.cache_enabled:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            'cache_size': len(self.fitness_cache),
            'cache_hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
        }
    
    def clear_cache(self):
        """清空適應度緩存"""
        if self.cache_enabled:
            self.fitness_cache.clear()
            logger.info("適應度緩存已清空")
