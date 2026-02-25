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


class DeterministicCrowdingReplacementStrategy(ReplacementStrategy):
    """
    擁擠度取代策略 (Deterministic Crowding, DC)
    
    基於 Mahfoud (1992, 1995) 的 DC 演算法，透過父子局部競爭維護族群多樣性。
    
    核心機制：
    1. Distance-Based Matching: 子代與「最相似」的父母配對
    2. Local Tournament: 配對後進行 fitness 比較，勝者保留
    
    重要：Distance metric 僅用於決定配對，不包含 fitness。
    Fitness 僅用於「競爭」階段，決定配對中誰存活。
    
    Distance Modes:
    - genotype: Tree Kernel Distance (結構相似度)
    - phenotype: Signal Overlap Distance (行為相似度)  
    - combined: 加權平均結合兩者
    """
    
    def __init__(self, 
                 distance_mode: str = 'genotype',
                 lambda_decay: float = 0.5,
                 genotype_weight: float = 0.5,
                 phenotype_weight: float = 0.5):
        """
        初始化 Deterministic Crowding 策略
        
        Args:
            distance_mode: 距離計算方式 ('genotype', 'phenotype', 'combined')
            lambda_decay: Tree kernel 的衰減因子 (genotype distance)
            genotype_weight: Genotype 權重 (combined mode only)
            phenotype_weight: Phenotype 權重 (combined mode only)
        """
        super().__init__()
        self.name = "deterministic_crowding"
        self.distance_mode = distance_mode
        self.lambda_decay = lambda_decay
        self.genotype_weight = genotype_weight
        self.phenotype_weight = phenotype_weight
        
        # Normalize weights
        total_weight = self.genotype_weight + self.phenotype_weight
        if total_weight > 0:
            self.genotype_weight /= total_weight
            self.phenotype_weight /= total_weight
        
        # Statistics
        self.stats = {
            'parent_wins': 0,
            'offspring_wins': 0,
            'total_competitions': 0,
            'phenotype_calls': 0,
            'phenotype_fallbacks': 0
        }
    
    def replace(self, population: List, offspring: List, data: Dict[str, Any]) -> List:
        """
        執行 Deterministic Crowding 替換
        
        Args:
            population: 當前族群 (父代)
            offspring: 子代個體 (應包含 parents 屬性記錄父母 ID)
            data: 演化數據
            
        Returns:
            新的族群
        """
        # Log DC entry for debugging
        print(f"🎯 DC Replacement: distance_mode={self.distance_mode}, "
              f"pop_size={len(population)}, offspring_size={len(offspring)}")
        logger.info(f"🎯 DC Replacement: distance_mode={self.distance_mode}, "
                   f"pop_size={len(population)}, offspring_size={len(offspring)}")
        
        if not population or not offspring:
            logger.warning("族群或子代為空，回退到世代替換")
            return offspring[:len(population)] if population else offspring
        
        population_size = len(population)
        
        # Build parent ID -> individual mapping
        parent_map = {}
        for ind in population:
            if hasattr(ind, 'id'):
                parent_map[ind.id] = ind
            else:
                parent_map[id(ind)] = ind
        
        # Group offspring by their parent pairs
        # offspring should have 'parents' attribute: [parent1_id, parent2_id]
        parent_pair_groups = {}  # (p1_id, p2_id) -> [offspring1, offspring2, ...]
        unmatched_offspring = []
        
        for child in offspring:
            if hasattr(child, 'parents') and child.parents and len(child.parents) >= 2:
                # Sort parent IDs to create consistent key
                p1_id, p2_id = child.parents[0], child.parents[1]
                key = (p1_id, p2_id) if str(p1_id) <= str(p2_id) else (p2_id, p1_id)
                
                if key not in parent_pair_groups:
                    parent_pair_groups[key] = []
                parent_pair_groups[key].append(child)
            else:
                unmatched_offspring.append(child)
        
        # Log parent matching stats
        logger.debug(f"   DC Parent Matching: {len(parent_pair_groups)} pairs matched, "
                    f"{len(unmatched_offspring)} unmatched offspring")
        
        # Process each parent pair and their offspring
        survivors = set()  # Use set to track surviving parent IDs
        new_population = []
        
        for (p1_id, p2_id), children in parent_pair_groups.items():
            # Get parent individuals
            p1 = parent_map.get(p1_id)
            p2 = parent_map.get(p2_id)
            
            if p1 is None or p2 is None:
                # Parents not found, children survive by default
                new_population.extend(children[:2])
                continue
            
            # We need exactly 2 children for standard DC
            if len(children) < 2:
                # Only 1 child: compete with closest parent
                c1 = children[0]
                d_p1_c1 = self._calculate_distance(p1, c1, data)
                d_p2_c1 = self._calculate_distance(p2, c1, data)
                
                if d_p1_c1 <= d_p2_c1:
                    # c1 competes with p1
                    winner = self._compete(p1, c1)
                    new_population.append(winner)
                    if p2_id not in survivors:
                        new_population.append(p2)
                        survivors.add(p2_id)
                else:
                    # c1 competes with p2
                    winner = self._compete(p2, c1)
                    new_population.append(winner)
                    if p1_id not in survivors:
                        new_population.append(p1)
                        survivors.add(p1_id)
                survivors.add(p1_id)
                survivors.add(p2_id)
                continue
            
            c1, c2 = children[0], children[1]
            
            # Calculate distances for pairing decision
            d_p1_c1 = self._calculate_distance(p1, c1, data)
            d_p2_c2 = self._calculate_distance(p2, c2, data)
            d_p1_c2 = self._calculate_distance(p1, c2, data)
            d_p2_c1 = self._calculate_distance(p2, c1, data)
            
            # Determine pairing (Mahfoud's rule)
            if d_p1_c1 + d_p2_c2 <= d_p1_c2 + d_p2_c1:
                # P1 vs C1, P2 vs C2
                winner1 = self._compete(p1, c1)
                winner2 = self._compete(p2, c2)
            else:
                # P1 vs C2, P2 vs C1
                winner1 = self._compete(p1, c2)
                winner2 = self._compete(p2, c1)
            
            new_population.append(winner1)
            new_population.append(winner2)
            survivors.add(p1_id)
            survivors.add(p2_id)
            
            # Handle extra children (if > 2)
            for extra_child in children[2:]:
                unmatched_offspring.append(extra_child)
        
        # Add parents that didn't participate in any competition
        for ind in population:
            ind_id = ind.id if hasattr(ind, 'id') else id(ind)
            if ind_id not in survivors:
                new_population.append(ind)
        
        # If we have unmatched offspring and need to fill population
        if len(new_population) < population_size and unmatched_offspring:
            # Add unmatched offspring
            slots_needed = population_size - len(new_population)
            new_population.extend(unmatched_offspring[:slots_needed])
        
        # Truncate or pad to exact population size
        if len(new_population) > population_size:
            # Sort by fitness and keep best
            valid_pop = [ind for ind in new_population 
                        if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and ind.fitness.values]
            invalid_pop = [ind for ind in new_population 
                          if not (hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and ind.fitness.values)]
            valid_pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
            new_population = (valid_pop + invalid_pop)[:population_size]
        elif len(new_population) < population_size:
            # Pad with best parents
            sorted_parents = sorted(
                [p for p in population if hasattr(p, 'fitness') and p.fitness.values],
                key=lambda x: x.fitness.values[0], reverse=True
            )
            for p in sorted_parents:
                if len(new_population) >= population_size:
                    break
                if p not in new_population:
                    new_population.append(p)
        
        # Log statistics
        total = self.stats['total_competitions']
        if total > 0:
            parent_pct = 100 * self.stats['parent_wins'] / total
            offspring_pct = 100 * self.stats['offspring_wins'] / total
            logger.info(f"DC 統計: 父代勝={parent_pct:.1f}%, 子代勝={offspring_pct:.1f}% (共{total}場)")

        # Log phenotype fallback statistics
        ph_calls = self.stats['phenotype_calls']
        ph_fallbacks = self.stats['phenotype_fallbacks']
        if ph_calls > 0:
            fb_pct = 100 * ph_fallbacks / ph_calls
            print(f"⚠️  Phenotype distance fallback: {ph_fallbacks}/{ph_calls} ({fb_pct:.1f}%)")
            logger.warning(f"Phenotype distance fallback: {ph_fallbacks}/{ph_calls} ({fb_pct:.1f}%)")
        
        logger.debug(f"   DC 替換: {len(population)} → {len(new_population)}")
        return new_population[:population_size]
    
    def _compete(self, parent, child) -> Any:
        """
        父子競爭，Fitness 較高者存活
        
        Args:
            parent: 父代個體
            child: 子代個體
            
        Returns:
            勝者個體
        """
        self.stats['total_competitions'] += 1
        
        # Get fitness values
        p_fitness = self._get_fitness(parent)
        c_fitness = self._get_fitness(child)
        
        if c_fitness > p_fitness:
            self.stats['offspring_wins'] += 1
            return child
        else:
            self.stats['parent_wins'] += 1
            return parent
    
    def _get_fitness(self, individual) -> float:
        """取得個體的適應度值"""
        if hasattr(individual, 'fitness') and hasattr(individual.fitness, 'values') and individual.fitness.values:
            return individual.fitness.values[0]
        return float('-inf')
    
    def _calculate_distance(self, ind1, ind2, data: Dict[str, Any]) -> float:
        """
        計算兩個個體之間的距離 (用於 parent-offspring matching)
        
        依據 Mahfoud (1995)，距離僅用於決定配對，不包含 fitness。
        
        Args:
            ind1: 個體 1
            ind2: 個體 2
            data: 演化數據 (可能包含 cached signals)
            
        Returns:
            距離值，範圍 [0.0, 1.0] (0 = identical, 1 = completely different)
        """
        import numpy as np
        
        genotype_dist = 0.0
        phenotype_dist = 0.0
        
        # Genotype distance (Tree Kernel)
        if self.distance_mode in ['genotype', 'combined']:
            genotype_dist = self._compute_genotype_distance(ind1, ind2)
        
        # Phenotype distance (Signal Overlap)
        if self.distance_mode in ['phenotype', 'combined']:
            phenotype_dist = self._compute_phenotype_distance(ind1, ind2, data)
        
        # Return based on mode
        if self.distance_mode == 'genotype':
            return genotype_dist
        elif self.distance_mode == 'phenotype':
            return phenotype_dist
        else:  # combined
            return self.genotype_weight * genotype_dist + self.phenotype_weight * phenotype_dist
    
    def _compute_genotype_distance(self, ind1, ind2) -> float:
        """
        計算 Genotype Distance (Tree Kernel)
        
        使用 SubtreeKernel 的 normalized distance: d = 1 - K(T1,T2) / sqrt(K(T1,T1)*K(T2,T2))
        
        Returns:
            距離值，範圍 [0.0, 1.0]
        """
        try:
            from gp_quant.similarity.tree_kernel import SubtreeKernel
            from gp_quant.similarity.tree_edit_distance import deap_to_tree_node
            
            kernel = SubtreeKernel(lambda_decay=self.lambda_decay)
            node1 = deap_to_tree_node(ind1)
            node2 = deap_to_tree_node(ind2)
            
            # Use normalized distance directly
            return kernel.compute_normalized_distance(node1, node2)
            
        except Exception as e:
            logger.warning(f"Genotype distance 計算失敗: {e}")
            return 0.5  # Default to neutral distance on failure
    
    def _compute_phenotype_distance(self, ind1, ind2, data: Dict[str, Any]) -> float:
        """
        計算 Phenotype Distance (Signal Overlap)
        
        比較兩個個體產生的交易訊號的相似度。
        使用 Jaccard distance: 1 - |S1 ∩ S2| / |S1 ∪ S2|
        
        注意：需要 data 中包含 cached signals 或 backtest engine
        
        Returns:
            距離值，範圍 [0.0, 1.0]
        """
        import numpy as np
        
        self.stats['phenotype_calls'] += 1

        try:
            # Try to get cached signals from individuals
            sig1 = self._get_individual_signal(ind1, data)
            sig2 = self._get_individual_signal(ind2, data)
            
            if sig1 is None or sig2 is None:
                self.stats['phenotype_fallbacks'] += 1
                return self._compute_genotype_distance(ind1, ind2)
            
            # Convert to numpy arrays
            sig1 = np.asarray(sig1, dtype=bool)
            sig2 = np.asarray(sig2, dtype=bool)
            
            # Ensure same length
            min_len = min(len(sig1), len(sig2))
            sig1 = sig1[:min_len]
            sig2 = sig2[:min_len]
            
            # Jaccard distance
            intersection = np.sum(sig1 & sig2)
            union = np.sum(sig1 | sig2)
            
            if union == 0:
                # Both signals are all zeros - identical behavior
                return 0.0
            
            jaccard_similarity = intersection / union
            return 1.0 - jaccard_similarity
            
        except Exception as e:
            logger.warning(f"Phenotype distance 計算失敗: {e}")
            return self._compute_genotype_distance(ind1, ind2)
    
    def _get_individual_signal(self, individual, data: Dict[str, Any]):
        """
        取得個體的交易訊號
        
        優先順序：
        1. 個體的 metadata 中的 cached signal
        2. data 中的 signal cache
        3. 重新計算 (需要 backtest engine)
        
        Returns:
            Signal array or None if not available
        """
        import numpy as np
        
        # 1. Check individual metadata
        if hasattr(individual, 'metadata') and 'signals' in individual.metadata:
            return individual.metadata['signals']
        
        if hasattr(individual, 'get_metadata'):
            cached_signal = individual.get_metadata('signals')
            if cached_signal is not None:
                return cached_signal
        
        # 2. Check data dict for signal cache
        ind_id = individual.id if hasattr(individual, 'id') else id(individual)
        signal_cache = data.get('signal_cache', {})
        if ind_id in signal_cache:
            return signal_cache[ind_id]
        
        # 3. Try to compute signal using backtest engine (if available)
        backtest_engine = data.get('backtest_engine')
        if backtest_engine is not None:
            try:
                # Generate signals and cache them
                all_signals = backtest_engine._generate_signals_for_all_stocks(individual)
                # Flatten to single signal array (combine all tickers)
                combined_signal = []
                for ticker_signals in all_signals.values():
                    combined_signal.extend(ticker_signals.values())
                signal_array = np.array(combined_signal, dtype=bool)
                
                # Cache for future use
                if 'signal_cache' not in data:
                    data['signal_cache'] = {}
                data['signal_cache'][ind_id] = signal_array
                
                return signal_array
            except Exception as e:
                logger.debug(f"Signal 計算失敗: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """取得 DC 統計資訊"""
        total = self.stats['total_competitions']
        return {
            'parent_wins': self.stats['parent_wins'],
            'offspring_wins': self.stats['offspring_wins'],
            'total_competitions': total,
            'parent_win_rate': self.stats['parent_wins'] / total if total > 0 else 0.0,
            'offspring_win_rate': self.stats['offspring_wins'] / total if total > 0 else 0.0
        }
