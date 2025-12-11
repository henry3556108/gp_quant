"""
保留策略模組

實現各種個體保留 (Reproduction) 策略，包括標準機率性保留和菁英保留。
"""

from typing import List, Dict, Any
import logging
import copy
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class ReproductionStrategy(EvolutionStrategy):
    """
    保留策略基類
    """
    
    def __init__(self):
        super().__init__()
        self.name = "reproduction_strategy"
    
    def reproduce(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        執行保留操作
        
        Args:
            population: 當前族群
            k: 保留個體數量
            data: 演化數據
            
        Returns:
            保留的個體列表 (已複製)
        """
        raise NotImplementedError("子類必須實現 reproduce 方法")
    
    def _clone_and_record(self, individual):
        """
        複製個體並記錄保留操作的父母資訊
        """
        # 複製個體
        reproduced = copy.deepcopy(individual)
        
        # 記錄父母資訊
        if hasattr(reproduced, 'parents'):
            if hasattr(individual, 'id'):
                reproduced.parents = [individual.id]
            else:
                reproduced.parents = [id(individual)]
        
        # 記錄操作類型
        if hasattr(reproduced, 'operation'):
            reproduced.operation = 'reproduction'
        
        # 記錄世代信息
        if hasattr(reproduced, 'generation') and self.engine:
            reproduced.generation = getattr(self.engine, 'current_generation', 0) + 1
            
        return reproduced

class StandardReproduction(ReproductionStrategy):
    """
    標準保留策略 (機率性)
    
    使用演化引擎配置的選擇策略 (Selection Strategy) 來挑選保留個體。
    這是原本的預設行為。
    """
    
    def __init__(self):
        super().__init__()
        self.name = "standard"
    
    def reproduce(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        使用選擇策略挑選個體進行保留
        """
        if k <= 0:
            return []
            
        # 使用引擎的選擇策略
        selection_strategy = self.engine.strategies.get('selection')
        if not selection_strategy:
            logger.error("未找到選擇策略，無法執行標準保留")
            return []
            
        selected = selection_strategy.select_individuals(population, k, data)
        
        # 複製並記錄
        reproduced_individuals = [self._clone_and_record(ind) for ind in selected]
        
        logger.debug(f"   標準保留: 選中 {len(reproduced_individuals)} 個體")
        return reproduced_individuals

class ElitistReproduction(ReproductionStrategy):
    """
    菁英保留策略 (確定性)
    
    直接選擇適應度最高的 k 個個體進行保留。
    """
    
    def __init__(self, elite_ratio: float = 1.0):
        """
        初始化菁英保留策略
        
        Args:
            elite_ratio: 菁英比例 (0.0-1.0)。
                         如果 < 1.0，則剩餘名額使用標準選擇策略填補。
                         預設 1.0 表示全部名額都由菁英填補。
        """
        super().__init__()
        self.name = "elitist"
        self.elite_ratio = max(0.0, min(1.0, elite_ratio))
    
    def reproduce(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        選擇最佳個體進行保留
        """
        if k <= 0:
            return []
            
        # 確保個體有有效適應度
        valid_individuals = [ind for ind in population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        
        if not valid_individuals:
            logger.warning("沒有有效適應度的個體，回退到隨機選擇")
            import random
            selected = random.sample(population, min(k, len(population)))
            return [self._clone_and_record(ind) for ind in selected]
            
        # 排序 (假設適應度越高越好)
        sorted_individuals = sorted(valid_individuals, key=lambda ind: ind.fitness.values[0], reverse=True)
        
        # 計算菁英數量
        num_elites = int(k * self.elite_ratio)
        # 確保至少有一個菁英 (如果 k > 0 且 elite_ratio > 0)
        if k > 0 and self.elite_ratio > 0 and num_elites == 0:
            num_elites = 1
            
        num_elites = min(num_elites, k, len(sorted_individuals))
        
        # 選擇菁英
        elites = sorted_individuals[:num_elites]
        reproduced_individuals = [self._clone_and_record(ind) for ind in elites]
        
        # 標記為菁英
        for ind in reproduced_individuals:
            if hasattr(ind, 'operation'):
                ind.operation = 'reproduction_elite'
        
        # 如果還有剩餘名額，使用標準選擇策略填補
        remaining = k - len(reproduced_individuals)
        if remaining > 0:
            selection_strategy = self.engine.strategies.get('selection')
            if selection_strategy:
                others = selection_strategy.select_individuals(population, remaining, data)
                reproduced_others = [self._clone_and_record(ind) for ind in others]
                reproduced_individuals.extend(reproduced_others)
                
        logger.debug(f"   菁英保留: {num_elites} 菁英 + {remaining} 其他 = {len(reproduced_individuals)} 個體")
        return reproduced_individuals

class NicheElitistReproduction(ReproductionStrategy):
    """
    生態位菁英保留策略 (Niche Elitism)
    
    從每個生態位 (Cluster) 中保留最佳個體，確保跨世代的多樣性。
    使用 Round-Robin (輪詢) 方式從各個 Cluster 挑選菁英。
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "niche_elitist"
        # Store any extra parameters if needed, or just ignore them
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def reproduce(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        執行生態位菁英保留
        """
        if k <= 0:
            return []
            
        # 1. 嘗試從 Selection Strategy 獲取分群資訊
        selection_strategy = self.engine.strategies.get('selection')
        clusters = None
        
        if selection_strategy and hasattr(selection_strategy, '_get_or_compute_niching'):
            try:
                # 傳入 generation 參數 (有些實作需要)
                generation = data.get('generation', 0)
                # 呼叫 _get_or_compute_niching
                # 注意：不同實作的參數可能略有不同，這裡做一些兼容性處理
                import inspect
                sig = inspect.signature(selection_strategy._get_or_compute_niching)
                if 'generation' in sig.parameters:
                    clusters, _ = selection_strategy._get_or_compute_niching(population, generation)
                else:
                    clusters, _ = selection_strategy._get_or_compute_niching(population, data)
                    
            except Exception as e:
                logger.warning(f"獲取生態位分群失敗: {e}，回退到標準菁英保留")
        
        # 如果無法獲取分群，回退到標準菁英保留
        if not clusters:
            logger.info("無分群資訊，使用標準菁英保留")
            fallback = ElitistReproduction(elite_ratio=1.0)
            fallback.set_engine(self.engine)
            return fallback.reproduce(population, k, data)
            
        # 2. Round-Robin 選擇
        # 過濾掉空 cluster 並按 fitness 排序每個 cluster 的個體
        sorted_clusters = []
        for cluster in clusters:
            if not cluster:
                continue
            # 確保有 fitness
            valid_inds = [ind for ind in cluster if hasattr(ind.fitness, 'values') and ind.fitness.values]
            if valid_inds:
                # 降序排列
                sorted_inds = sorted(valid_inds, key=lambda ind: ind.fitness.values[0], reverse=True)
                sorted_clusters.append(sorted_inds)
        
        if not sorted_clusters:
            logger.warning("所有 Cluster 都沒有有效個體，回退到標準菁英保留")
            fallback = ElitistReproduction(elite_ratio=1.0)
            fallback.set_engine(self.engine)
            return fallback.reproduce(population, k, data)
            
        selected_individuals = []
        cluster_indices = [0] * len(sorted_clusters)  # 追蹤每個 cluster 取到第幾個
        active_clusters = list(range(len(sorted_clusters))) # 還有個體可取的 cluster index
        
        current_cluster_idx = 0 # 用於輪詢 active_clusters
        
        while len(selected_individuals) < k and active_clusters:
            # 輪詢
            real_cluster_idx = active_clusters[current_cluster_idx]
            ind_idx = cluster_indices[real_cluster_idx]
            
            # 取出個體
            individual = sorted_clusters[real_cluster_idx][ind_idx]
            selected_individuals.append(self._clone_and_record(individual))
            
            # 更新索引
            cluster_indices[real_cluster_idx] += 1
            
            # 檢查該 cluster 是否還有個體
            if cluster_indices[real_cluster_idx] >= len(sorted_clusters[real_cluster_idx]):
                # 該 cluster 已空，移除
                active_clusters.pop(current_cluster_idx)
                # 移除後，current_cluster_idx 指向的已經是下一個元素了，所以不需要 +1
                # 但如果移除的是最後一個元素，則需要回到 0
                if current_cluster_idx >= len(active_clusters):
                    current_cluster_idx = 0
            else:
                # 移動到下一個 cluster
                current_cluster_idx = (current_cluster_idx + 1) % len(active_clusters)
                
        # 標記為 Niche Elite
        for ind in selected_individuals:
            if hasattr(ind, 'operation'):
                ind.operation = 'reproduction_niche_elite'
                
        logger.debug(f"   Niche菁英保留: 從 {len(sorted_clusters)} 個 Cluster 中選出了 {len(selected_individuals)} 個體")
        
        # 如果還不夠 (理論上不應該發生，除非總個體數 < k)，用隨機填補
        if len(selected_individuals) < k:
            remaining = k - len(selected_individuals)
            logger.warning(f"個體不足，隨機填補 {remaining} 個")
            import random
            # 從整個 population 隨機選，排除已選的 (這裡簡化處理，直接選)
            others = random.sample(population, min(remaining, len(population)))
            selected_individuals.extend([self._clone_and_record(ind) for ind in others])
            
        return selected_individuals
