"""
交配策略模組

實現基於遺傳程式設計的交配操作，包括重試邏輯和深度限制。
"""

from typing import List, Dict, Any, Tuple
import logging
import random
from deap import gp, tools

from .base import EvolutionStrategy
from ..gp import pset

logger = logging.getLogger(__name__)

class CrossoverStrategy(EvolutionStrategy):
    """
    交配策略基類
    """
    
    def __init__(self, terminal_prob: float = 0.1, max_retries: int = 10, max_depth: int = 17):
        """
        初始化交配策略
        
        Args:
            terminal_prob: 終端節點交配機率 (葉偏向交配)
            max_retries: 最大重試次數
            max_depth: 最大樹深度限制
        """
        super().__init__()
        self.name = "crossover_strategy"
        self.terminal_prob = terminal_prob
        self.max_retries = max_retries
        self.max_depth = max_depth
        self.retry_stats = {
            'total_crossovers': 0,
            'successful_crossovers': 0,
            'failed_crossovers': 0
        }
    
    def crossover(self, parent_pairs: List[Tuple], data: Dict[str, Any]) -> List:
        """
        執行交配操作
        
        Args:
            parent_pairs: 父母對列表 [(parent1, parent2), ...]
            data: 演化數據
            
        Returns:
            子代列表
        """
        offspring = []
        
        for parent1, parent2 in parent_pairs:
            self.retry_stats['total_crossovers'] += 1
            
            # 嘗試交配，帶重試邏輯
            success = False
            for attempt in range(self.max_retries):
                # 複製父母
                child1 = self._clone_individual(parent1)
                child2 = self._clone_individual(parent2)
                
                # 執行交配操作
                self._perform_crossover(child1, child2)
                
                # 檢查深度限制
                if self._check_depth_constraint(child1) and self._check_depth_constraint(child2):
                    # 清除適應度 (需要重新評估)
                    self._invalidate_fitness(child1)
                    self._invalidate_fitness(child2)
                    
                    # 記錄父母資訊 (譜系追蹤)
                    self._record_parents(child1, [parent1, parent2], operation='crossover')
                    self._record_parents(child2, [parent1, parent2], operation='crossover')
                    
                    offspring.extend([child1, child2])
                    self.retry_stats['successful_crossovers'] += 1
                    success = True
                    break
            
            if not success:
                # 重試失敗，生成新的隨機個體
                logger.warning(f"交配重試 {self.max_retries} 次失敗，生成隨機個體")
                child1 = self._generate_random_individual()
                child2 = self._generate_random_individual()
                offspring.extend([child1, child2])
                self.retry_stats['failed_crossovers'] += 1
        
        return offspring
    
    def _perform_crossover(self, individual1, individual2):
        """執行具體的交配操作"""
        # 使用 DEAP 的葉偏向單點交配
        gp.cxOnePointLeafBiased(individual1, individual2, termpb=self.terminal_prob)
    
    def _clone_individual(self, individual):
        """複製個體"""
        import copy
        return copy.deepcopy(individual)
    
    def _check_depth_constraint(self, individual) -> bool:
        """檢查深度約束"""
        return individual.height <= self.max_depth
    
    def _invalidate_fitness(self, individual):
        """清除個體的適應度"""
        if hasattr(individual, 'fitness') and hasattr(individual.fitness, 'values'):
            del individual.fitness.values
    
    def _record_parents(self, child, parents: List, operation: str):
        """
        記錄父母資訊到子代
        
        Args:
            child: 子代個體
            parents: 父母列表
            operation: 操作類型 ('crossover', 'mutation', 'reproduction')
        """
        # 如果個體有 ID 屬性，記錄譜系資訊
        if hasattr(child, 'parents'):
            child.parents = []
            for parent in parents:
                if hasattr(parent, 'id'):
                    child.parents.append(parent.id)
                else:
                    child.parents.append(id(parent))  # 使用內存地址作為備用 ID
        
        # 記錄操作類型
        if hasattr(child, 'operation'):
            child.operation = operation
        
        # 記錄世代信息
        if hasattr(child, 'generation') and self.engine:
            child.generation = getattr(self.engine, 'current_generation', 0) + 1
    
    def _generate_random_individual(self):
        """生成隨機個體"""
        # 使用引擎的初始化策略生成新個體
        if self.engine and 'initialization' in self.engine.strategies:
            return self.engine.strategies['initialization'].create_individual()
        else:
            # 回退方案：使用 DEAP 的標準生成
            from ..individual import EvolutionIndividual
            expr = gp.genHalfAndHalf(pset, min_=2, max_=6)
            individual = EvolutionIndividual(expr)
            # 為隨機個體記錄資訊
            self._record_parents(individual, [], operation='random_generation')
            return individual
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取交配統計信息"""
        total = self.retry_stats['total_crossovers']
        if total > 0:
            success_rate = self.retry_stats['successful_crossovers'] / total
        else:
            success_rate = 0.0
        
        return {
            'total_crossovers': total,
            'successful_crossovers': self.retry_stats['successful_crossovers'],
            'failed_crossovers': self.retry_stats['failed_crossovers'],
            'success_rate': success_rate
        }

class OnePointCrossoverStrategy(CrossoverStrategy):
    """
    單點交配策略
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "one_point_crossover"
    
    def _perform_crossover(self, individual1, individual2):
        """執行單點交配"""
        gp.cxOnePoint(individual1, individual2)

class UniformCrossoverStrategy(CrossoverStrategy):
    """
    均勻交配策略
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "uniform_crossover"
    
    def _perform_crossover(self, individual1, individual2):
        """執行均勻交配"""
        gp.cxOnePointLeafBiased(individual1, individual2, termpb=0.5)
