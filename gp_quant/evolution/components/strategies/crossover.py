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
        import uuid
        cloned = copy.deepcopy(individual)
        # 為克隆個體分配新的 ID，避免緩存衝突
        if hasattr(cloned, 'id'):
            cloned.id = str(uuid.uuid4())
        return cloned
    
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


class KNNNichingCrossoverStrategy(CrossoverStrategy):
    """
    KNN-based Niching Crossover Strategy
    
    基於 IFLGP 論文的想法，在 crossover 階段實現多樣性維護：
    - exploration_prob 機率選擇 FARTHEST 配偶 (全域探索)
    - (1 - exploration_prob) 機率選擇 NEAREST 配偶 (局部開發)
    
    距離計算使用 normalized fitness + tree kernel distance。
    """
    
    def __init__(self, 
                 neighborhood_ratio: float = 0.2,
                 exploration_prob: float = 0.7,
                 distance_mode: str = 'fitness_tree_kernel',
                 lambda_decay: float = 0.5,
                 save_distance_matrix: bool = False,
                 mate_selection_mode: str = 'random',
                 mate_tournament_size: int = 3,
                 **kwargs):
        """
        Initialize KNN Niching Crossover Strategy.
        
        Args:
            neighborhood_ratio: 候選池比例 (default 0.2 = 20%)
            exploration_prob: 選擇 farthest 的機率 (default 0.7 = 70% exploration)
            distance_mode: 距離計算方式 ('fitness_tree_kernel' or 'fitness_overlap')
            lambda_decay: Tree kernel 的衰減因子
            save_distance_matrix: 是否保存 distance matrix (default False)
            mate_selection_mode: 從候選池選擇配偶的方式 ('random' 或 'tournament')
            mate_tournament_size: 錦標賽選擇的大小 (僅在 mate_selection_mode='tournament' 時生效)
        """
        super().__init__(**kwargs)
        self.name = "knn_niching_crossover"
        self.neighborhood_ratio = neighborhood_ratio
        self.exploration_prob = exploration_prob
        self.distance_mode = distance_mode
        self.lambda_decay = lambda_decay
        self.save_distance_matrix = save_distance_matrix
        self.mate_selection_mode = mate_selection_mode
        self.mate_tournament_size = mate_tournament_size
        
        # Cache for distance matrix and neighbor indices
        self._cached_generation = -1
        self._distance_matrix = None
        self._nearest_indices = None
        self._farthest_indices = None
        
        # Statistics
        self.niching_stats = {
            'exploration_count': 0,
            'niching_count': 0
        }
    
    def crossover(self, parent_pairs: List[Tuple], data: Dict[str, Any]) -> List:
        """
        執行 KNN Niching Crossover
        
        Note: This method expects full population in data['population'] for 
        distance matrix computation. If not available, falls back to standard crossover.
        """
        population = data.get('population', None)
        
        # If no population data, fall back to standard crossover
        if population is None:
            logger.warning("KNN Niching: 無法取得 population，使用標準 crossover")
            return super().crossover(parent_pairs, data)
        
        # Build or update neighbor matrices
        current_gen = data.get('generation', -1)
        if current_gen != self._cached_generation or self._distance_matrix is None:
            self._build_neighbor_matrices(population, data)
            self._cached_generation = current_gen
        
        offspring = []
        
        for parent1, parent2_original in parent_pairs:
            self.retry_stats['total_crossovers'] += 1
            
            # Determine exploration vs niching
            r = random.random()
            
            if r < self.exploration_prob:
                # Exploration: select mate from FARTHEST
                parent2 = self._select_mate(parent1, population, mode='farthest')
                self.niching_stats['exploration_count'] += 1
            else:
                # Niching: select mate from NEAREST
                parent2 = self._select_mate(parent1, population, mode='nearest')
                self.niching_stats['niching_count'] += 1
            
            # Perform crossover with selected mate
            success = False
            for attempt in range(self.max_retries):
                child1 = self._clone_individual(parent1)
                child2 = self._clone_individual(parent2)
                
                self._perform_crossover(child1, child2)
                
                if self._check_depth_constraint(child1) and self._check_depth_constraint(child2):
                    self._invalidate_fitness(child1)
                    self._invalidate_fitness(child2)
                    self._record_parents(child1, [parent1, parent2], operation='crossover')
                    self._record_parents(child2, [parent1, parent2], operation='crossover')
                    offspring.extend([child1, child2])
                    self.retry_stats['successful_crossovers'] += 1
                    success = True
                    break
            
            if not success:
                logger.warning(f"KNN Niching 交配重試 {self.max_retries} 次失敗")
                child1 = self._generate_random_individual()
                child2 = self._generate_random_individual()
                offspring.extend([child1, child2])
                self.retry_stats['failed_crossovers'] += 1
        
        # Log statistics
        total = self.niching_stats['exploration_count'] + self.niching_stats['niching_count']
        if total > 0:
            exp_pct = 100 * self.niching_stats['exploration_count'] / total
            print(f"[KNN Niching] 配對統計: 探索={exp_pct:.1f}%, 開發={100-exp_pct:.1f}% (共{total}對)")
            logger.info(f"KNN Niching 配對: 探索={exp_pct:.1f}%, 開發={100-exp_pct:.1f}%")
        
        return offspring
    
    def _build_neighbor_matrices(self, population: List, data: Dict[str, Any] = None):
        """
        Build distance matrix and nearest/farthest neighbor indices.
        
        Args:
            population: List of individuals
            data: Evolution data (needed for signal overlap computation)
        """
        import numpy as np
        from gp_quant.similarity.tree_kernel import SubtreeKernel
        from gp_quant.similarity.tree_edit_distance import deap_to_tree_node
        
        n = len(population)
        S = max(1, int(n * self.neighborhood_ratio))  # 候選池大小
        
        logger.info(f"KNN Niching: 建立鄰居矩陣 (n={n}, S={S}, mode={self.distance_mode})...")
        print(f"[KNN Niching] 建立鄰居矩陣 (n={n}, S={S}, mode={self.distance_mode})...")
        
        # Collect fitness values and normalize
        fitness_values = []
        for ind in population:
            if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and ind.fitness.values:
                fitness_values.append(ind.fitness.values[0])
            else:
                fitness_values.append(0.0)
        
        fitness_arr = np.array(fitness_values)
        f_min, f_max = fitness_arr.min(), fitness_arr.max()
        if f_max - f_min > 1e-10:
            fitness_norm = (fitness_arr - f_min) / (f_max - f_min)
        else:
            fitness_norm = np.zeros_like(fitness_arr)
        
        # Compute tree kernel distances
        if self.distance_mode in ['fitness_tree_kernel', 'combined', 'tree_kernel']:
            kernel = SubtreeKernel(lambda_decay=self.lambda_decay)
            
            # Convert to TreeNode
            tree_nodes = []
            node_sizes = []
            for ind in population:
                try:
                    node = deap_to_tree_node(ind)
                    tree_nodes.append(node)
                    node_sizes.append(SubtreeKernel.count_nodes(node))
                except Exception as e:
                    tree_nodes.append(None)
                    node_sizes.append(0)
            
            # Pre-compute self-kernels
            self_kernels = []
            for node in tree_nodes:
                if node is not None:
                    self_kernels.append(kernel.compute(node, node))
                else:
                    self_kernels.append(1.0)
            
            # Parallel computation of kernel distances
            n_jobs = getattr(self.engine.config.get('evolution', {}), 'get', lambda k, d: self.engine.config.get('evolution', {}).get(k, d))('max_processors', 4)
            if isinstance(n_jobs, dict):
                n_jobs = self.engine.config.get('evolution', {}).get('max_processors', 4)
            
            # Generate all pairs (i, j) where i < j
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            if n > 100 and n_jobs > 1:
                # Use joblib for large populations
                from joblib import Parallel, delayed
                
                def compute_pair_distance(pair_idx, tree_nodes, self_kernels, lambda_decay):
                    i, j = pair_idx
                    if tree_nodes[i] is None or tree_nodes[j] is None:
                        return (i, j, 1.0)
                    
                    # Create new kernel for each worker (thread-safe)
                    local_kernel = SubtreeKernel(lambda_decay=lambda_decay)
                    k_ij = local_kernel.compute(tree_nodes[i], tree_nodes[j])
                    k_ii = self_kernels[i]
                    k_jj = self_kernels[j]
                    
                    dist_sq = max(0.0, k_ii + k_jj - 2 * k_ij)
                    dist = np.sqrt(dist_sq)
                    
                    max_dist = np.sqrt(k_ii + k_jj)
                    if max_dist > 0:
                        norm_dist = dist / max_dist
                    else:
                        norm_dist = 0.0
                    
                    return (i, j, norm_dist)
                
                print(f"[KNN Niching] 並行計算距離矩陣 ({len(pairs)} pairs, {n_jobs} workers)...")
                results = Parallel(n_jobs=n_jobs, prefer="threads")(
                    delayed(compute_pair_distance)(pair, tree_nodes, self_kernels, self.lambda_decay)
                    for pair in pairs
                )
                
                tree_dist_matrix = np.zeros((n, n))
                for i, j, dist in results:
                    tree_dist_matrix[i, j] = dist
                    tree_dist_matrix[j, i] = dist
            else:
                # Sequential for small populations
                tree_dist_matrix = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        if tree_nodes[i] is None or tree_nodes[j] is None:
                            tree_dist_matrix[i, j] = 1.0
                        else:
                            k_ij = kernel.compute(tree_nodes[i], tree_nodes[j])
                            k_ii = self_kernels[i]
                            k_jj = self_kernels[j]
                            
                            dist_sq = max(0.0, k_ii + k_jj - 2 * k_ij)
                            dist = np.sqrt(dist_sq)
                            
                            max_dist = np.sqrt(k_ii + k_jj)
                            if max_dist > 0:
                                tree_dist_matrix[i, j] = dist / max_dist
                            else:
                                tree_dist_matrix[i, j] = 0.0
                        
                        tree_dist_matrix[j, i] = tree_dist_matrix[i, j]
                
                kernel.clear_cache()
            
            # Log tree size stats
            valid_sizes = [s for s in node_sizes if s > 0]
            if valid_sizes:
                logger.info(f"樹大小統計: 平均={np.mean(valid_sizes):.1f}, 範圍=[{min(valid_sizes)}, {max(valid_sizes)}]")
        elif self.distance_mode == 'fitness_overlap':
            # Only use signal overlap, no tree kernel
            tree_dist_matrix = np.zeros((n, n))
        
        # Compute signal overlap if needed
        if self.distance_mode in ['fitness_overlap', 'combined', 'signal_overlap']:
            overlap_dist_matrix = self._compute_signal_overlap_distances(population, data, n)
        else:
            overlap_dist_matrix = np.zeros((n, n))
        
        # Compute combined distance matrix based on mode
        self._distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    self._distance_matrix[i, j] = 0.0
                else:
                    # Pure tree kernel mode (no fitness)
                    if self.distance_mode == 'tree_kernel':
                        self._distance_matrix[i, j] = tree_dist_matrix[i, j]
                    # Pure signal overlap mode (no fitness)
                    elif self.distance_mode == 'signal_overlap':
                        self._distance_matrix[i, j] = overlap_dist_matrix[i, j]
                    # Combined modes with fitness
                    else:
                        f_dist = abs(fitness_norm[i] - fitness_norm[j])
                        t_dist = tree_dist_matrix[i, j]
                        o_dist = overlap_dist_matrix[i, j]
                        self._distance_matrix[i, j] = np.sqrt(f_dist**2 + t_dist**2 + o_dist**2)
        
        # Build nearest and farthest indices for each individual
        self._nearest_indices = np.zeros((n, S), dtype=int)
        self._farthest_indices = np.zeros((n, S), dtype=int)
        
        for i in range(n):
            sorted_indices = np.argsort(self._distance_matrix[i])
            # Exclude self (index 0 after sort should be self with distance 0)
            others = [j for j in sorted_indices if j != i]
            self._nearest_indices[i] = others[:S]
            self._farthest_indices[i] = others[-S:][::-1]  # Reverse to have farthest first
        
        # Save distance matrix if enabled
        if self.save_distance_matrix and self.engine and hasattr(self.engine, 'config'):
            records_dir = self.engine.config.get('logging', {}).get('records_dir', None)
            if records_dir:
                import os
                gen = self._cached_generation
                matrix_path = os.path.join(records_dir, f'distance_matrix_gen{gen:03d}.npy')
                np.save(matrix_path, self._distance_matrix)
                print(f"[KNN Niching] Distance matrix 已保存: {matrix_path}")
        
        logger.info(f"KNN Niching: 鄰居矩陣建立完成")
    
    def _compute_signal_overlap_distances(self, population: List, data: Dict[str, Any], n: int) -> 'np.ndarray':
        """
        Compute signal overlap distance matrix.
        
        Distance = 1 - overlap_ratio, where overlap_ratio = (matching signals) / (total signals)
        
        Args:
            population: List of individuals
            data: Evolution data containing train_data and config
            n: Population size
            
        Returns:
            np.ndarray: Distance matrix (n x n) with values in [0, 1]
        """
        import numpy as np
        from tqdm import tqdm
        
        logger.info(f"[KNN Niching] 計算 Signal Overlap Matrix ({n} x {n})...")
        print(f"[KNN Niching] 計算 Signal Overlap Matrix ({n} x {n})...")
        
        # Step 1: Get backtesting engine
        engine = self._get_or_create_backtest_engine(data)
        if engine is None:
            logger.warning("無法建立 backtest engine，使用零距離矩陣")
            return np.zeros((n, n))
        
        # Step 2: Compute signals for all individuals
        signals = []
        for i, ind in enumerate(tqdm(population, desc="計算 Signals", ncols=100)):
            try:
                sig = engine.get_signals(ind)
                signals.append(sig)
            except Exception as e:
                logger.warning(f"Signal 計算失敗 (個體 {i}): {e}")
                # Use zero signals on failure
                if len(signals) > 0:
                    signals.append(np.zeros_like(signals[0]))
                else:
                    signals.append(np.array([]))
        
        # Check if signals are valid
        if len(signals) == 0 or len(signals[0]) == 0:
            logger.warning("Signal 計算失敗，使用零距離矩陣")
            return np.zeros((n, n))
        
        print(f"[KNN Niching] Signal shape: {signals[0].shape}")
        
        # Step 3: Calculate pairwise overlap distances
        overlap_dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if len(signals[i]) == len(signals[j]) and len(signals[i]) > 0:
                    overlap = np.sum(signals[i] == signals[j]) / len(signals[i])
                    dist = 1.0 - overlap
                else:
                    dist = 1.0  # Max distance if signals incompatible
                
                overlap_dist_matrix[i, j] = dist
                overlap_dist_matrix[j, i] = dist
        
        # Stats
        upper_tri = np.triu_indices(n, k=1)
        mean_dist = np.mean(overlap_dist_matrix[upper_tri])
        mean_overlap = 1.0 - mean_dist
        
        logger.info(f"Signal Overlap 統計: 平均重疊={mean_overlap:.4f}, 平均距離={mean_dist:.4f}")
        print(f"[KNN Niching] Signal Overlap: 平均重疊={mean_overlap:.4f}, 平均距離={mean_dist:.4f}")
        
        return overlap_dist_matrix
    
    def _get_or_create_backtest_engine(self, data: Dict[str, Any]):
        """
        Get or create backtesting engine for signal computation.
        
        Args:
            data: Evolution data
            
        Returns:
            PortfolioBacktestingEngine or None
        """
        # Check if already cached
        if hasattr(self, '_backtest_engine') and self._backtest_engine is not None:
            return self._backtest_engine
        
        # Try to get from data or engine
        if data is None:
            return None
        
        try:
            from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine
            
            # Get config from engine
            if not hasattr(self, 'engine') or self.engine is None:
                return None
            
            config = self.engine.config
            
            # Try to get train_data from data dict first (already loaded)
            train_data = data.get('train_data', None)
            
            # Convert train_data format if needed
            # train_data may be {ticker: {'data': df, 'backtest_start': ...}} or {ticker: df}
            processed_data = {}
            if train_data is not None:
                for ticker, ticker_data in train_data.items():
                    if isinstance(ticker_data, dict) and 'data' in ticker_data:
                        processed_data[ticker] = ticker_data['data']
                    else:
                        processed_data[ticker] = ticker_data
            else:
                # Load data if not available
                from gp_quant.data.loader import load_and_process_data
                import os
                
                tickers_dir = config.get('data', {}).get('tickers_dir', 'history')
                
                # Get list of tickers from directory
                if os.path.isdir(tickers_dir):
                    tickers = [f.replace('.csv', '') for f in os.listdir(tickers_dir) if f.endswith('.csv')]
                else:
                    tickers = ['SPY']
                
                processed_data = load_and_process_data(tickers_dir, tickers)
            
            # Import pset from operators
            from gp_quant.evolution.components.gp import operators
            
            # Create engine
            self._backtest_engine = PortfolioBacktestingEngine(
                data=processed_data,
                pset=operators.pset,
                backtest_start=config['data']['train_backtest_start'],
                backtest_end=config['data']['train_backtest_end'],
                initial_capital=100000.0
            )
            
            logger.info(f"[KNN Niching] Created backtest engine "
                       f"(period: {config['data']['train_backtest_start']} ~ "
                       f"{config['data']['train_backtest_end']})")
            
            return self._backtest_engine
            
        except Exception as e:
            logger.error(f"建立 backtest engine 失敗: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def _select_mate(self, parent, population: List, mode: str):
        """
        Select mate from nearest or farthest candidates.
        
        Args:
            parent: The parent individual
            population: Full population list
            mode: 'nearest' or 'farthest'
        
        Returns:
            Selected mate individual
        """
        # Find parent index in population
        parent_idx = None
        for i, ind in enumerate(population):
            if ind is parent or (hasattr(ind, 'id') and hasattr(parent, 'id') and ind.id == parent.id):
                parent_idx = i
                break
        
        if parent_idx is None:
            # Parent not found in population, fallback to random
            return random.choice(population)
        
        if mode == 'nearest':
            candidates = self._nearest_indices[parent_idx]
        else:
            candidates = self._farthest_indices[parent_idx]
        
        # Select mate based on mate_selection_mode
        if self.mate_selection_mode == 'tournament':
            # Tournament selection from candidates
            tournament_size = min(self.mate_tournament_size, len(candidates))
            tournament_indices = random.sample(list(candidates), tournament_size)
            # Select the one with highest fitness
            best_idx = None
            best_fitness = float('-inf')
            for idx in tournament_indices:
                ind = population[idx]
                if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and ind.fitness.values:
                    fitness = ind.fitness.values[0]
                else:
                    fitness = 0.0
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_idx = idx
            mate_idx = best_idx if best_idx is not None else random.choice(candidates)
        else:
            # Random selection (default)
            mate_idx = random.choice(candidates)
        
        return population[mate_idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crossover statistics including niching info."""
        stats = super().get_stats()
        stats['exploration_count'] = self.niching_stats['exploration_count']
        stats['niching_count'] = self.niching_stats['niching_count']
        total = stats['exploration_count'] + stats['niching_count']
        if total > 0:
            stats['exploration_ratio'] = stats['exploration_count'] / total
        else:
            stats['exploration_ratio'] = 0.0
        return stats
