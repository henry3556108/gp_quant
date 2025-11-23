"""
初始化策略模組

實現各種族群初始化策略，包括 Ramped Half-and-Half、Full、Grow 等方法。
"""

from typing import List, Dict, Any
import logging
import uuid
from deap import gp, tools, creator

from .base import EvolutionStrategy
from ..gp import pset
from ..individual import EvolutionIndividual

logger = logging.getLogger(__name__)

class InitializationStrategy(EvolutionStrategy):
    """
    初始化策略基類
    """
    
    def __init__(self, min_depth: int = 2, max_depth: int = 6):
        """
        初始化策略基類
        
        Args:
            min_depth: 最小樹深度
            max_depth: 最大樹深度
        """
        super().__init__()
        self.name = "initialization_strategy"
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def initialize(self, population_size: int, data: Dict[str, Any]) -> List:
        """
        初始化族群
        
        Args:
            population_size: 族群大小
            data: 演化數據
            
        Returns:
            初始化的族群
        """
        raise NotImplementedError("子類必須實現 initialize 方法")
    
    def create_individual(self) -> Any:
        """
        創建單個個體
        
        Returns:
            新創建的個體
        """
        raise NotImplementedError("子類必須實現 create_individual 方法")
    
    def _assign_individual_id(self, individual):
        """為個體分配唯一 ID"""
        # EvolutionIndividual 已經有 ID，但我們可以重新分配
        individual.id = str(uuid.uuid4())
        
        # 記錄初始化資訊
        individual.parents = []  # 初始個體沒有父母
        individual.operation = 'initialization'
        individual.generation = 0
        
        # 設置創建時間
        import time
        individual.creation_time = time.time()

class RampedHalfAndHalfStrategy(InitializationStrategy):
    """
    Ramped Half-and-Half 初始化策略
    
    這是 GP 中最常用的初始化方法，結合了 Full 和 Grow 方法：
    - 一半個體使用 Full 方法（所有葉子節點在相同深度）
    - 一半個體使用 Grow 方法（葉子節點深度可變）
    - 深度在 min_depth 到 max_depth 之間均勻分布
    """
    
    def __init__(self, min_depth: int = 2, max_depth: int = 6):
        super().__init__(min_depth, max_depth)
        self.name = "ramped_half_and_half"
    
    def initialize(self, population_size: int, data: Dict[str, Any]) -> List:
        """
        使用 Ramped Half-and-Half 方法初始化族群
        """
        population = []
        
        # 計算每個深度層級的個體數量
        depth_range = self.max_depth - self.min_depth + 1
        individuals_per_depth = population_size // depth_range
        remaining = population_size % depth_range
        
        logger.info(f"初始化族群: size={population_size}, depth_range=[{self.min_depth}, {self.max_depth}]")
        
        for depth in range(self.min_depth, self.max_depth + 1):
            # 計算當前深度的個體數量
            current_count = individuals_per_depth
            if depth - self.min_depth < remaining:
                current_count += 1
            
            # 一半使用 Full，一半使用 Grow
            full_count = current_count // 2
            grow_count = current_count - full_count
            
            # 生成 Full 個體
            for _ in range(full_count):
                expr = gp.genFull(pset, min_=depth, max_=depth)
                individual = EvolutionIndividual(expr)
                self._assign_individual_id(individual)
                population.append(individual)
            
            # 生成 Grow 個體
            for _ in range(grow_count):
                expr = gp.genGrow(pset, min_=self.min_depth, max_=depth)
                individual = EvolutionIndividual(expr)
                self._assign_individual_id(individual)
                population.append(individual)
            
            logger.debug(f"   深度 {depth}: {full_count} Full + {grow_count} Grow = {current_count} 個體")
        
        logger.info(f"✅ 族群初始化完成: {len(population)} 個個體")
        return population
    
    def create_individual(self) -> Any:
        """創建單個個體（使用 Half-and-Half）"""
        import random
        
        # 隨機選擇深度
        depth = random.randint(self.min_depth, self.max_depth)
        
        # 隨機選擇 Full 或 Grow
        if random.random() < 0.5:
            expr = gp.genFull(pset, min_=depth, max_=depth)
        else:
            expr = gp.genGrow(pset, min_=self.min_depth, max_=depth)
        
        individual = EvolutionIndividual(expr)
        self._assign_individual_id(individual)
        return individual

class FullStrategy(InitializationStrategy):
    """
    Full 初始化策略
    
    所有個體的葉子節點都在相同深度，創建完全平衡的樹。
    """
    
    def __init__(self, min_depth: int = 2, max_depth: int = 6):
        super().__init__(min_depth, max_depth)
        self.name = "full"
    
    def initialize(self, population_size: int, data: Dict[str, Any]) -> List:
        """使用 Full 方法初始化族群"""
        population = []
        
        # 計算每個深度層級的個體數量
        depth_range = self.max_depth - self.min_depth + 1
        individuals_per_depth = population_size // depth_range
        remaining = population_size % depth_range
        
        logger.info(f"Full 初始化族群: size={population_size}, depth_range=[{self.min_depth}, {self.max_depth}]")
        
        for depth in range(self.min_depth, self.max_depth + 1):
            current_count = individuals_per_depth
            if depth - self.min_depth < remaining:
                current_count += 1
            
            for _ in range(current_count):
                expr = gp.genFull(pset, min_=depth, max_=depth)
                individual = EvolutionIndividual(expr)
                self._assign_individual_id(individual)
                population.append(individual)
            
            logger.debug(f"   深度 {depth}: {current_count} 個 Full 個體")
        
        logger.info(f"✅ Full 族群初始化完成: {len(population)} 個個體")
        return population
    
    def create_individual(self) -> Any:
        """創建單個 Full 個體"""
        import random
        depth = random.randint(self.min_depth, self.max_depth)
        expr = gp.genFull(pset, min_=depth, max_=depth)
        individual = EvolutionIndividual(expr)
        self._assign_individual_id(individual)
        return individual

class GrowStrategy(InitializationStrategy):
    """
    Grow 初始化策略
    
    個體的葉子節點深度可變，創建不規則的樹結構。
    """
    
    def __init__(self, min_depth: int = 2, max_depth: int = 6):
        super().__init__(min_depth, max_depth)
        self.name = "grow"
    
    def initialize(self, population_size: int, data: Dict[str, Any]) -> List:
        """使用 Grow 方法初始化族群"""
        population = []
        
        logger.info(f"Grow 初始化族群: size={population_size}, depth_range=[{self.min_depth}, {self.max_depth}]")
        
        for _ in range(population_size):
            expr = gp.genGrow(pset, min_=self.min_depth, max_=self.max_depth)
            individual = EvolutionIndividual(expr)
            self._assign_individual_id(individual)
            population.append(individual)
        
        logger.info(f"✅ Grow 族群初始化完成: {len(population)} 個個體")
        return population
    
    def create_individual(self) -> Any:
        """創建單個 Grow 個體"""
        expr = gp.genGrow(pset, min_=self.min_depth, max_=self.max_depth)
        individual = EvolutionIndividual(expr)
        self._assign_individual_id(individual)
        return individual
