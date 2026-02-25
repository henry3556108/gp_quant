"""
變異策略模組

實現基於遺傳程式設計的變異操作，包括重試邏輯和深度限制。
"""

from typing import List, Dict, Any
import logging
import random
from deap import gp, tools

from .base import EvolutionStrategy
from ..gp import pset

logger = logging.getLogger(__name__)

class MutationStrategy(EvolutionStrategy):
    """
    變異策略基類
    """
    
    def __init__(self, min_depth: int = 0, max_depth: int = 2, max_retries: int = 10, max_tree_depth: int = 17):
        """
        初始化變異策略
        
        Args:
            min_depth: 變異子樹最小深度
            max_depth: 變異子樹最大深度
            max_retries: 最大重試次數
            max_tree_depth: 最大樹深度限制
        """
        super().__init__()
        self.name = "mutation_strategy"
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.max_tree_depth = max_tree_depth
        self.retry_stats = {
            'total_mutations': 0,
            'successful_mutations': 0,
            'failed_mutations': 0
        }
    
    def mutate(self, individuals: List, data: Dict[str, Any],
               record_parents: bool = True, mutation_rate: float = 1.0) -> List:
        """
        執行變異操作

        Args:
            individuals: 要變異的個體列表
            data: 演化數據
            record_parents: 是否記錄父母資訊 (默認 True)
            mutation_rate: 個體級別的變異機率 (0.0~1.0)，每個個體以此機率
                          決定是否被 mutate，未被選中的保留原樣（僅 clone）。
                          默認 1.0 = 全部 mutate（向後兼容）。

        Returns:
            處理後的個體列表（mutated + 未變異的 clone）
        """
        result_individuals = []
        actually_mutated = 0
        kept_original = 0

        for individual in individuals:
            # 按機率決定是否 mutate
            if random.random() < mutation_rate:
                # --- 執行 mutation ---
                self.retry_stats['total_mutations'] += 1
                success = False

                for attempt in range(self.max_retries):
                    mutant = self._clone_individual(individual)
                    self._perform_mutation(mutant)

                    if self._check_depth_constraint(mutant):
                        self._invalidate_fitness(mutant)

                        if record_parents:
                            self._record_parents(mutant, [individual], operation='mutation')
                        else:
                            # 串聯模式：保留 crossover 記錄的 parents，但標記 operation
                            if hasattr(mutant, 'operation'):
                                mutant.operation = 'crossover+mutation'

                        result_individuals.append(mutant)
                        self.retry_stats['successful_mutations'] += 1
                        actually_mutated += 1
                        success = True
                        break

                if not success:
                    logger.warning(f"變異重試 {self.max_retries} 次失敗，生成隨機個體")
                    mutant = self._generate_random_individual()
                    result_individuals.append(mutant)
                    self.retry_stats['failed_mutations'] += 1
                    actually_mutated += 1
            else:
                # --- 不 mutate，保留原始 crossover 結果（clone 以避免 reference 問題）---
                clone = self._clone_individual(individual)
                # 保留 crossover 時設定的 fitness invalidation 和 parents
                self._invalidate_fitness(clone)
                result_individuals.append(clone)
                kept_original += 1

        logger.info(f"變異統計: {actually_mutated} mutated, {kept_original} kept original "
                    f"(rate={mutation_rate:.2f}, total={len(individuals)})")

        return result_individuals
    
    def _perform_mutation(self, individual):
        """執行具體的變異操作"""
        # 使用 DEAP 的均勻變異
        from ..gp import pset
        from functools import partial
        
        # 創建表達式生成函數
        expr_mut = partial(gp.genFull, min_=self.min_depth, max_=self.max_depth)
        gp.mutUniform(individual, expr=expr_mut, pset=pset)
    
    def _clone_individual(self, individual):
        """複製個體"""
        import copy
        import uuid
        cloned = copy.deepcopy(individual)
        # 為克隆個體分配新的 ID，避免緩存衝突
        if hasattr(cloned, 'id'):
            cloned.id = str(uuid.uuid4())
        return cloned
    
    def _check_depth_constraint(self, individual) -> bool:
        """檢查深度約束"""
        return individual.height <= self.max_tree_depth
    
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
        """獲取變異統計信息"""
        total = self.retry_stats['total_mutations']
        if total > 0:
            success_rate = self.retry_stats['successful_mutations'] / total
        else:
            success_rate = 0.0
        
        return {
            'total_mutations': total,
            'successful_mutations': self.retry_stats['successful_mutations'],
            'failed_mutations': self.retry_stats['failed_mutations'],
            'success_rate': success_rate
        }

class PointMutationStrategy(MutationStrategy):
    """
    點變異策略
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "point_mutation"
    
    def _perform_mutation(self, individual):
        """執行點變異"""
        # 使用 DEAP 的點變異
        gp.mutNodeReplacement(individual, pset=pset)

class SubtreeMutationStrategy(MutationStrategy):
    """
    子樹變異策略
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "subtree_mutation"
    
    def _perform_mutation(self, individual):
        """執行子樹變異"""
        # 使用 DEAP 的子樹變異
        expr_mut = gp.genGrow(pset, min_=self.min_depth, max_=self.max_depth)
        gp.mutUniform(individual, expr=expr_mut, pset=pset)
