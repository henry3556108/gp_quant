"""
選擇策略模組

實現各種個體選擇策略，包括排名選擇、錦標賽選擇等。
"""

from typing import List, Dict, Any, Tuple
import logging
import random
from deap import tools

from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class SelectionStrategy(EvolutionStrategy):
    """
    選擇策略基類
    """
    
    def __init__(self):
        super().__init__()
        self.name = "selection_strategy"
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        選擇個體
        
        Args:
            population: 族群
            k: 選擇個體數量
            data: 演化數據
            
        Returns:
            選中的個體列表
        """
        raise NotImplementedError("子類必須實現 select_individuals 方法")
    
    def select_pairs(self, population: List, k: int, data: Dict[str, Any]) -> List[Tuple]:
        """
        選擇父母對
        
        Args:
            population: 族群
            k: 選擇對數
            data: 演化數據
            
        Returns:
            父母對列表 [(parent1, parent2), ...]
        """
        # 默認實現：隨機配對選中的個體
        selected = self.select_individuals(population, k * 2, data)
        pairs = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                pairs.append((selected[i], selected[i + 1]))
        return pairs

class RankedSUSStrategy(SelectionStrategy):
    """
    排名選擇 + 隨機通用抽樣 (SUS) 策略
    
    實現論文中描述的排名選擇方法：
    1. 根據原始適應度對個體排名
    2. 分配基於排名的新適應度值
    3. 使用隨機通用抽樣進行選擇
    """
    
    def __init__(self, max_rank_fitness: float = 1.8, min_rank_fitness: float = 0.2):
        """
        初始化排名選擇策略
        
        Args:
            max_rank_fitness: 最佳個體的排名適應度
            min_rank_fitness: 最差個體的排名適應度
        """
        super().__init__()
        self.name = "ranked_sus"
        self.max_rank_fitness = max_rank_fitness
        self.min_rank_fitness = min_rank_fitness
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        使用排名選擇 + SUS 選擇個體
        """
        if not population:
            return []
        
        if k <= 0:
            return []
        
        # 根據原始適應度排序 (降序)，只考慮有有效適應度的個體
        valid_individuals = [ind for ind in population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        if not valid_individuals:
            return []
        sorted_individuals = sorted(valid_individuals, key=lambda ind: ind.fitness.values[0], reverse=True)
        pop_size = len(sorted_individuals)
        
        # 分配基於排名的臨時適應度
        original_fitnesses = []
        for i, ind in enumerate(sorted_individuals):
            rank = i + 1
            # 保存原始適應度
            original_fitnesses.append(ind.fitness.values)
            
            # 計算排名適應度
            rank_fitness = self.max_rank_fitness - (
                (self.max_rank_fitness - self.min_rank_fitness) * (rank - 1) / (pop_size - 1)
            )
            
            # 臨時設置排名適應度
            ind.fitness.values = (rank_fitness,)
        
        # 使用隨機通用抽樣選擇
        try:
            logger.debug(f"調用 SUS 選擇，k={k}, tools類型={type(tools)}")
            chosen = tools.selStochasticUniversalSampling(sorted_individuals, k)
            logger.debug(f"SUS 選擇成功，選中 {len(chosen)} 個個體")
        except Exception as e:
            logger.warning(f"SUS 選擇失敗，回退到錦標賽選擇: {e}")
            logger.debug(f"調用錦標賽選擇，tools類型={type(tools)}")
            chosen = tools.selTournament(sorted_individuals, k, tournsize=3)
            logger.debug(f"錦標賽選擇成功，選中 {len(chosen)} 個個體")
        
        # 恢復原始適應度
        for ind, original_fitness in zip(sorted_individuals, original_fitnesses):
            ind.fitness.values = original_fitness
        
        # 為選中的個體記錄保留操作（如果需要）
        # 注意：這裡不直接修改個體，因為選擇策略可能被多次調用
        return chosen

class TournamentStrategy(SelectionStrategy):
    """
    錦標賽選擇策略
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        初始化錦標賽選擇策略
        
        Args:
            tournament_size: 錦標賽大小
        """
        super().__init__()
        self.name = "tournament"
        self.tournament_size = tournament_size
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        使用錦標賽選擇個體
        """
        if not population or k <= 0:
            return []
        
        return tools.selTournament(population, k, tournsize=self.tournament_size)

class RouletteStrategy(SelectionStrategy):
    """
    輪盤賭選擇策略，如果檢測到負適應度值，則回退到錦標賽選擇
    """
    
    def __init__(self):
        super().__init__()
        self.name = "roulette"
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        使用輪盤賭選擇個體
        """
        if not population or k <= 0:
            return []
        
        try:
            # 檢查是否有負適應度值
            min_fitness = min(ind.fitness.values[0] for ind in population)
            if min_fitness < 0:
                # 如果有負值，使用錦標賽選擇作為回退
                logger.warning("檢測到負適應度值，回退到錦標賽選擇")
                return tools.selTournament(population, k, tournsize=3)
            
            return tools.selRoulette(population, k)
        except Exception as e:
            logger.warning(f"輪盤賭選擇失敗，回退到錦標賽選擇: {e}")
            return tools.selTournament(population, k, tournsize=3)
