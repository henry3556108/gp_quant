"""
操作策略模組

定義演化操作的執行模式，包括並聯模式和串聯模式。
"""

from typing import List, Dict, Any
import random
import logging
# from ..individual import EvolutionIndividual  # 暫時註解，等待實作
from .base import EvolutionStrategy

logger = logging.getLogger(__name__)

class OperationStrategy(EvolutionStrategy):
    """操作策略基類"""
    
    def execute_operations(self, population: List, data: Dict[str, Any]) -> List:
        """
        執行演化操作
        
        Args:
            population: 當前族群
            data: 演化數據
            
        Returns:
            產生的子代列表
        """
        raise NotImplementedError("子類必須實現 execute_operations 方法")
    
    def _clone_and_record_reproduction(self, individual):
        """
        複製個體並記錄保留操作的父母資訊
        
        Args:
            individual: 要複製的個體
            
        Returns:
            複製並記錄父母資訊的個體
        """
        import copy
        
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
        
        # 保留適應度 (reproduction 不改變個體，不需要重新評估)
        # 適應度已經在 deepcopy 時複製，保持 valid 狀態
        
        return reproduced

class ParallelOperationStrategy(OperationStrategy):
    """
    並聯操作策略
    
    每個個體按機率選擇操作：交配 OR 變異 OR 保留
    機率總和必須為 1.0
    """
    
    def __init__(self):
        super().__init__()
        self.name = "parallel_operation"
    
    def execute_operations(self, population: List, data: Dict[str, Any]) -> List:
        """
        並聯操作：每個位置隨機選擇一種操作
        """
        config = self.engine.config
        population_size = len(population)
        
        # 獲取操作機率
        crossover_rate = config['crossover']['rate']
        mutation_rate = config['mutation']['rate'] 
        reproduction_rate = config.get('reproduction', {}).get('rate', 0.0)
        
        # 驗證機率總和
        total_rate = crossover_rate + mutation_rate + reproduction_rate
        if abs(total_rate - 1.0) > 0.001:
            logger.warning(f"操作機率總和不為1.0: {total_rate}, 將進行歸一化")
            crossover_rate /= total_rate
            mutation_rate /= total_rate
            reproduction_rate /= total_rate
        
        offspring = []
        crossover_count = mutation_count = reproduction_count = 0
        
        # 為每個位置決定操作類型
        for i in range(population_size):
            rand = random.random()
            
            if rand < crossover_rate:
                # 交配操作
                try:
                    parent_pairs = self.engine.strategies['selection'].select_pairs(population, 1, data)
                    if parent_pairs:
                        crossover_offspring = self.engine.strategies['crossover'].crossover(parent_pairs, data)
                        if isinstance(crossover_offspring, list):
                            offspring.extend(crossover_offspring[:1])  # 只取一個子代
                            crossover_count += 1
                        else:
                            logger.error(f"交配返回的不是列表: {type(crossover_offspring)}")
                except Exception as e:
                    logger.error(f"交配操作失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    
            elif rand < crossover_rate + mutation_rate:
                # 變異操作
                selected = self.engine.strategies['selection'].select_individuals(population, 1, data)
                if selected:
                    mutated = self.engine.strategies['mutation'].mutate(selected, data)
                    offspring.extend(mutated[:1])  # 只取一個變異個體
                    mutation_count += 1
                    
            else:
                # 保留操作 (直接複製)
                selected = self.engine.strategies['selection'].select_individuals(population, 1, data)
                if selected:
                    reproduced = self._clone_and_record_reproduction(selected[0])
                    offspring.append(reproduced)
                    reproduction_count += 1
        
        logger.debug(f"   並聯操作: 交配={crossover_count}, 變異={mutation_count}, 保留={reproduction_count}")
        return offspring

class SerialOperationStrategy(OperationStrategy):
    """
    串聯操作策略
    
    操作依序進行：選擇 → 交配 → 變異 → 保留 → 合併
    """
    
    def __init__(self):
        super().__init__()
        self.name = "serial_operation"
    
    def execute_operations(self, population: List, data: Dict[str, Any]) -> List:
        """
        串聯操作：操作依序進行
        """
        config = self.engine.config
        population_size = len(population)
        logger.debug(f"開始串聯操作，族群大小: {population_size}")
        
        # 1. 交配操作
        crossover_offspring = []
        crossover_rate = config['crossover']['rate']
        reproduction_rate = config.get('reproduction', {}).get('rate', 0.0)
        
        if crossover_rate > 0:
            # 計算需要的子代數量：總族群 - 複製保留的數量
            num_offspring_needed = int(population_size * (1 - reproduction_rate))
            # 每對父母產生 2 個子代，所以需要的父母對數是子代數的一半
            num_crossover = num_offspring_needed // 2
            
            print(f"[DEBUG] 交配: population_size={population_size}, crossover_rate={crossover_rate}, reproduction_rate={reproduction_rate}")
            print(f"[DEBUG] 交配: num_offspring_needed={num_offspring_needed}, num_crossover_pairs={num_crossover}")
            
            try:
                parent_pairs = self.engine.strategies['selection'].select_pairs(population, num_crossover, data)
                logger.debug(f"選擇了 {len(parent_pairs) if parent_pairs else 0} 對父母")
                if parent_pairs:
                    crossover_offspring = self.engine.strategies['crossover'].crossover(parent_pairs, data)
                    print(f"[DEBUG] 交配產生: {len(crossover_offspring)} 個子代")
                    logger.debug(f"   交配產生 {len(crossover_offspring)} 個子代")
            except Exception as e:
                logger.error(f"交配操作失敗: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. 變異操作
        mutation_offspring = []
        mutation_rate = config['mutation']['rate']
        mutation_mode = config['mutation'].get('apply_to', 'offspring')
        
        print(f"[DEBUG] 變異: rate={mutation_rate}, mode={mutation_mode}, crossover_offspring={len(crossover_offspring)}")
        
        if mutation_rate > 0:
            if mutation_mode == 'offspring' and crossover_offspring:
                # 對交配子代進行變異 (不記錄父母資訊，因為已在交配時記錄)
                mutation_offspring = self.engine.strategies['mutation'].mutate(
                    crossover_offspring, data, record_parents=False
                )
                print(f"[DEBUG] 變異offspring模式產生: {len(mutation_offspring)} 個個體")
                logger.debug(f"   對交配子代變異產生 {len(mutation_offspring)} 個個體")
            elif mutation_mode == 'population':
                # 對原族群進行變異
                num_mutation = int(population_size * mutation_rate)
                selected_for_mutation = self.engine.strategies['selection'].select_individuals(
                    population, num_mutation, data
                )
                mutation_only_offspring = self.engine.strategies['mutation'].mutate(selected_for_mutation, data)
                mutation_offspring = crossover_offspring + mutation_only_offspring
                print(f"[DEBUG] 變異population模式: crossover={len(crossover_offspring)}, mutation={len(mutation_only_offspring)}, total={len(mutation_offspring)}")
                logger.debug(f"   對族群變異產生 {len(mutation_only_offspring)} 個個體")
            else:
                mutation_offspring = crossover_offspring
        else:
            mutation_offspring = crossover_offspring
        
        # 3. 保留操作
        reproduction_offspring = []
        reproduction_rate = config.get('reproduction', {}).get('rate', 0.0)
        if reproduction_rate > 0:
            num_reproduce = int(population_size * reproduction_rate)
            print(f"[DEBUG] 複製: population_size={population_size}, rate={reproduction_rate}, num_reproduce={num_reproduce}")
            selected_for_reproduction = self.engine.strategies['selection'].select_individuals(
                population, num_reproduce, data
            )
            # 複製並記錄父母資訊
            reproduction_offspring = [
                self._clone_and_record_reproduction(ind) for ind in selected_for_reproduction
            ]
            print(f"[DEBUG] 複製產生: {len(reproduction_offspring)} 個優秀個體")
            logger.debug(f"   保留 {len(reproduction_offspring)} 個優秀個體")
        
        # 4. 合併所有子代
        all_offspring = mutation_offspring + reproduction_offspring
        
        print(f"[DEBUG] 合併: mutation_offspring={len(mutation_offspring)}, reproduction_offspring={len(reproduction_offspring)}, total={len(all_offspring)}")
        logger.debug(f"   串聯操作完成: 總共 {len(all_offspring)} 個子代")
        return all_offspring
