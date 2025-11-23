"""
替換策略模組

實現各種世代替換策略，包括世代替換、穩態替換、菁英保留等。
"""

from typing import List, Dict, Any
import logging
from deap import tools

from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class ReplacementStrategy(EvolutionStrategy):
    """
    替換策略基類
    """
    
    def __init__(self):
        super().__init__()
        self.name = "replacement_strategy"
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        執行世代替換
        
        Args:
            population: 當前族群
            offspring: 子代個體
            data: 演化數據
            
        Returns:
            新的族群
        """
        raise NotImplementedError("子類必須實現 replace 方法")

class GenerationalReplacementStrategy(ReplacementStrategy):
    """
    世代替換策略
    
    完全用子代替換父代，這是最簡單的替換策略。
    """
    
    def __init__(self):
        super().__init__()
        self.name = "generational"
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        完全世代替換：子代完全替換父代
        """
        logger.debug(f"   世代替換: {len(population)} → {len(offspring)}")
        return offspring[:len(population)]  # 確保族群大小一致

class ElitistStrategy(ReplacementStrategy):
    """
    菁英保留策略
    
    保留最佳的個體，其餘用子代替換。
    """
    
    def __init__(self, elite_size: int = 1):
        """
        初始化菁英保留策略
        
        Args:
            elite_size: 保留的菁英個體數量
        """
        super().__init__()
        self.name = "elitist"
        self.elite_size = elite_size
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        菁英保留替換：保留最佳個體 + 子代
        """
        if not population or not offspring:
            logger.warning("族群或子代為空，無法執行菁英保留")
            return offspring[:len(population)] if population else offspring
        
        population_size = len(population)
        
        # 1. 選擇菁英個體（按適應度排序）
        elite_count = min(self.elite_size, population_size, len(population))
        
        # 確保所有個體都有適應度值
        valid_population = [ind for ind in population if hasattr(ind, 'fitness') and ind.fitness.values]
        
        if not valid_population:
            logger.warning("沒有有效的適應度值，回退到世代替換")
            return offspring[:population_size]
        
        # 按適應度降序排序（假設是最大化問題）
        elite_individuals = sorted(valid_population, key=lambda x: x.fitness.values[0], reverse=True)[:elite_count]
        
        # 2. 選擇子代個體填補剩餘位置
        remaining_slots = population_size - elite_count
        selected_offspring = offspring[:remaining_slots]
        
        # 3. 合併菁英和子代
        new_population = elite_individuals + selected_offspring
        
        logger.debug(f"   菁英保留: {elite_count} 菁英 + {len(selected_offspring)} 子代 = {len(new_population)}")
        
        # 記錄菁英個體的保留資訊
        for elite in elite_individuals:
            if hasattr(elite, 'operation'):
                elite.operation = 'elite_preserved'
        
        return new_population

class SteadyStateStrategy(ReplacementStrategy):
    """
    穩態替換策略
    
    每次只替換一小部分個體，而不是整個世代。
    """
    
    def __init__(self, replacement_rate: float = 0.1):
        """
        初始化穩態替換策略
        
        Args:
            replacement_rate: 每次替換的個體比例 (0.0-1.0)
        """
        super().__init__()
        self.name = "steady_state"
        self.replacement_rate = max(0.0, min(1.0, replacement_rate))
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        穩態替換：只替換一部分最差的個體
        """
        if not population or not offspring:
            return offspring[:len(population)] if population else offspring
        
        population_size = len(population)
        replacement_count = max(1, int(population_size * self.replacement_rate))
        replacement_count = min(replacement_count, len(offspring))
        
        # 確保所有個體都有適應度值
        valid_population = [ind for ind in population if hasattr(ind, 'fitness') and ind.fitness.values]
        
        if not valid_population:
            logger.warning("沒有有效的適應度值，回退到世代替換")
            return offspring[:population_size]
        
        # 按適應度排序，選擇最佳的個體（只考慮有效適應度）
        valid_population = [ind for ind in population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        sorted_population = sorted(valid_population, key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # 保留較好的個體
        survivors = sorted_population[replacement_count:]
        
        # 添加新的子代
        new_individuals = offspring[:replacement_count]
        
        # 合併
        new_population = survivors + new_individuals
        
        # 如果數量不足，用原族群補充
        if len(new_population) < population_size:
            remaining = population_size - len(new_population)
            new_population.extend(sorted_population[:remaining])
        
        logger.debug(f"   穩態替換: 替換 {replacement_count} 個體 (rate={self.replacement_rate:.2f})")
        
        return new_population[:population_size]

class TournamentReplacementStrategy(ReplacementStrategy):
    """
    錦標賽替換策略
    
    通過錦標賽選擇決定哪些個體被替換。
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        初始化錦標賽替換策略
        
        Args:
            tournament_size: 錦標賽大小
        """
        super().__init__()
        self.name = "tournament_replacement"
        self.tournament_size = tournament_size
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        錦標賽替換：通過錦標賽選擇最佳個體組成新族群
        """
        if not population or not offspring:
            return offspring[:len(population)] if population else offspring
        
        population_size = len(population)
        
        # 合併父代和子代
        combined_population = population + offspring
        
        # 確保所有個體都有適應度值
        valid_individuals = [ind for ind in combined_population if hasattr(ind, 'fitness') and ind.fitness.values]
        
        if len(valid_individuals) < population_size:
            logger.warning(f"有效個體數量不足: {len(valid_individuals)} < {population_size}")
            return valid_individuals + offspring[:population_size - len(valid_individuals)]
        
        # 通過錦標賽選擇新族群
        new_population = []
        for _ in range(population_size):
            winner = tools.selTournament(valid_individuals, 1, tournsize=self.tournament_size)[0]
            new_population.append(winner)
        
        logger.debug(f"   錦標賽替換: 從 {len(combined_population)} 個體中選擇 {population_size} 個")
        
        return new_population
