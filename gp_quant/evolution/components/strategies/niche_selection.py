"""
Niche Selection Strategies

基於不同距離度量的生態位選擇策略：
1. TEDNicheSelectionStrategy: 基於 Tree Edit Distance（基因型多樣性）
2. PnLNicheSelectionStrategy: 基於 PnL Correlation（表現型多樣性）- 待實作

使用階層式分群（Complete Linkage）將族群分為多個生態位，
每個生態位保留 Top M 個體，形成 Elite Pool 用於親代選擇。
"""

from typing import List, Dict, Any, Tuple
import logging
import random
import numpy as np
from deap import tools
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed
from tqdm import tqdm

from .selection import SelectionStrategy
from ....similarity.tree_edit_distance import compute_ted

logger = logging.getLogger(__name__)


class TEDNicheSelectionStrategy(SelectionStrategy):
    """
    基於 TED 的生態位選擇策略（自動搜索最佳 K）
    
    工作流程：
    1. 計算 TED distance matrix（每個世代一次，快取）
    2. 使用階層式分群（Complete Linkage）建立聚類樹
    3. 自動搜索最佳 K 值（K = 2 到 max_k）
       - 條件：Elite Pool 達成率 = 100%
       - 選擇：CV 最小（或最大，根據 cv_criterion）
    4. 每個 cluster 保留 Top M 個體（按 fitness）→ Elite Pool
    5. Crossover & Mutation 從 Elite Pool 選擇
    6. Reproduction 從整個 Population 選擇（使用 Tournament）
    """
    
    def __init__(self, 
                 max_k: int = 5,
                 top_m_per_cluster: int = 50,
                 cross_group_ratio: float = 0.3,
                 tournament_size: int = 3,
                 max_rank_fitness: float = 1.8,
                 min_rank_fitness: float = 0.2,
                 cv_criterion: str = 'min',
                 n_jobs: int = 6):
        """
        初始化 TED Niche Selection Strategy
        
        Args:
            max_k: 最大分群數量（會搜索 K=2 到 max_k）
            top_m_per_cluster: 每個 cluster 保留的個體數 M
            cross_group_ratio: 跨群配對比例（0.0-1.0）
            tournament_size: Tournament selection 的大小
            max_rank_fitness: Ranked SUS 的最大排名適應度
            min_rank_fitness: Ranked SUS 的最小排名適應度
            cv_criterion: CV 選擇標準（'min' 或 'max'）
            n_jobs: 平行計算的 worker 數量
        """
        super().__init__()
        self.name = "ted_niche"
        self.max_k = max_k
        self.M = top_m_per_cluster
        self.cross_group_ratio = cross_group_ratio
        self.tournament_size = tournament_size
        self.max_rank_fitness = max_rank_fitness
        self.min_rank_fitness = min_rank_fitness
        self.cv_criterion = cv_criterion
        self.n_jobs = n_jobs
        
        # 快取（每個世代只計算一次）
        self._cached_generation = -1
        self._cached_clusters = None
        self._cached_elite_pool = None
        self._optimal_k = None
        
        logger.info(f"初始化 TED Niche Selection: max_k={self.max_k}, M={self.M}, "
                   f"cross_group_ratio={self.cross_group_ratio}, "
                   f"cv_criterion={self.cv_criterion}, "
                   f"tournament_size={self.tournament_size}")
    
    def _calculate_ted_for_pair(self, i: int, j: int, ind_i: Any, ind_j: Any) -> Tuple[int, int, float]:
        """
        計算一對個體的標準化 TED
        
        Args:
            i, j: 個體索引
            ind_i, ind_j: 個體
            
        Returns:
            (i, j, normalized_ted)
        """
        try:
            ted = compute_ted(ind_i, ind_j)
            max_size = max(len(ind_i), len(ind_j))
            norm_ted = ted / max_size if max_size > 0 else 0.0
            return i, j, norm_ted
        except Exception as e:
            logger.warning(f"TED 計算失敗 ({i}, {j}): {e}，使用最大距離")
            return i, j, 1.0
    
    def _calculate_ted_distance_matrix(self, population: List) -> np.ndarray:
        """
        計算標準化 TED distance matrix（平行化，帶進度條）
        
        Args:
            population: 族群列表
            
        Returns:
            Normalized TED distance matrix (n x n)
        """
        n = len(population)
        logger.info(f"計算 TED Distance Matrix ({n} x {n})...")
        
        # 初始化矩陣
        ted_matrix = np.zeros((n, n))
        
        # 生成所有需要計算的配對（上三角）
        pairs = [(i, j, population[i], population[j]) 
                 for i in range(n) for j in range(i + 1, n)]
        
        total_pairs = len(pairs)
        logger.info(f"平行計算 {total_pairs} 對 TED (n_jobs={self.n_jobs})...")
        
        # 使用分批處理來實現實時進度更新
        # 批次大小根據 worker 數量調整
        batch_size = max(100, total_pairs // (self.n_jobs * 10))
        
        results = []
        with tqdm(total=total_pairs, desc="計算 TED", ncols=100, 
                  unit="pairs", position=0, leave=True) as pbar:
            
            # 分批處理
            for batch_start in range(0, total_pairs, batch_size):
                batch_end = min(batch_start + batch_size, total_pairs)
                batch_pairs = pairs[batch_start:batch_end]
                
                # 平行計算當前批次（使用 threading backend 避免 DEAP creator 序列化問題）
                batch_results = Parallel(n_jobs=self.n_jobs, backend='threading', verbose=0)(
                    delayed(self._calculate_ted_for_pair)(i, j, ind_i, ind_j)
                    for i, j, ind_i, ind_j in batch_pairs
                )
                
                results.extend(batch_results)
                
                # 更新進度條
                pbar.update(len(batch_results))
        
        # 填充矩陣（對稱）
        for i, j, ted in results:
            ted_matrix[i, j] = ted
            ted_matrix[j, i] = ted
        
        # 對角線為 0
        np.fill_diagonal(ted_matrix, 0.0)
        
        # 統計信息
        upper_tri = np.triu_indices(n, k=1)
        mean_ted = np.mean(ted_matrix[upper_tri])
        std_ted = np.std(ted_matrix[upper_tri])
        
        logger.info(f"TED 統計: 平均={mean_ted:.4f} ± {std_ted:.4f}, "
                   f"範圍=[{np.min(ted_matrix[upper_tri]):.4f}, {np.max(ted_matrix[upper_tri]):.4f}]")
        
        return ted_matrix
    
    def _find_optimal_k_and_cluster(self, distance_matrix: np.ndarray, population: List) -> Tuple[int, np.ndarray]:
        """
        自動搜索最佳 K 值並執行聚類
        
        使用階層式聚類樹，從 K=2 到 max_k 搜索最佳 K：
        - 條件：Elite Pool 達成率 = 100%
        - 選擇：CV 最小（或最大，根據 cv_criterion）
        
        Args:
            distance_matrix: 距離矩陣
            population: 族群列表
            
        Returns:
            (optimal_k, cluster_labels)
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        logger.info(f"自動搜索最佳 K (K=2~{self.max_k}, 標準={self.cv_criterion})...")
        
        # 建立聚類樹（只建立一次）
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        best_k = 2
        best_cv = float('inf') if self.cv_criterion == 'min' else float('-inf')
        best_achievement_rate = 0.0
        
        # 搜索所有 K 值
        for k in range(2, self.max_k + 1):
            # 從聚類樹切割出 K 個 clusters
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # 計算 Elite Pool 達成率
            elite_pool_size = sum(min(count, self.M) for count in counts)
            expected_size = k * self.M
            achievement_rate = elite_pool_size / expected_size
            
            # 只考慮 100% 達成率的 K
            if achievement_rate >= 1.0:
                # 計算 CV
                cv = np.std(counts) / np.mean(counts)
                
                logger.debug(f"  K={k}: CV={cv:.4f}, 達成率={achievement_rate*100:.1f}%")
                
                # 根據標準選擇
                is_better = False
                if self.cv_criterion == 'min':
                    if cv < best_cv:
                        is_better = True
                else:  # 'max'
                    if cv > best_cv:
                        is_better = True
                
                if is_better:
                    best_cv = cv
                    best_k = k
                    best_achievement_rate = achievement_rate
        
        logger.info(f"✅ 最佳 K={best_k}, CV={best_cv:.4f}, 達成率={best_achievement_rate*100:.1f}%")
        
        # 使用最佳 K 進行聚類
        cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust') - 1
        
        # 統計每個 cluster 的大小
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        logger.info(f"分群完成: {len(unique_labels)} 個 clusters")
        for label, count in zip(unique_labels, counts):
            logger.debug(f"  Cluster {label}: {count} 個體")
        
        return best_k, cluster_labels
    
    def _extract_elite_pool(self, population: List, cluster_labels: np.ndarray) -> Tuple[List[List], List]:
        """
        從每個 cluster 提取 Top M 個體，形成 Elite Pool
        
        Args:
            population: 族群列表
            cluster_labels: cluster 標籤
            
        Returns:
            (clusters, elite_pool)
            clusters: List[List], 每個元素是一個 cluster 的 Top M 個體
            elite_pool: List, 所有 Elite 個體的扁平列表
        """
        logger.info(f"提取 Elite Pool (每個 cluster Top {self.M})...")
        
        clusters = []
        elite_pool = []
        
        # 獲取實際的 cluster 數量
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        for cluster_id in unique_labels:
            # 找到該 cluster 的所有個體
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_individuals = [population[i] for i in cluster_indices]
            
            if not cluster_individuals:
                logger.warning(f"Cluster {cluster_id} 為空！")
                clusters.append([])
                continue
            
            # 按 fitness 排序（降序）
            cluster_sorted = sorted(cluster_individuals, 
                                   key=lambda x: x.fitness.values[0], 
                                   reverse=True)
            
            # 保留 Top M（如果 cluster 小於 M，則全部保留）
            top_m = cluster_sorted[:min(self.M, len(cluster_sorted))]
            
            clusters.append(top_m)
            elite_pool.extend(top_m)
            
            logger.debug(f"  Cluster {cluster_id}: {len(cluster_individuals)} 個體 → Top {len(top_m)}")
        
        logger.info(f"Elite Pool 大小: {len(elite_pool)} 個體 (目標: {n_clusters * self.M})")
        
        return clusters, elite_pool
    
    def _get_or_compute_niching(self, population: List, data: Dict[str, Any]) -> Tuple[List[List], List]:
        """
        獲取或計算生態位分群（帶快取）
        
        Args:
            population: 族群列表
            data: 演化數據（包含 generation 信息）
            
        Returns:
            (clusters, elite_pool)
        """
        current_gen = data.get('generation', -1)
        
        # 如果是同一個世代且快取存在，直接返回
        if current_gen == self._cached_generation and self._cached_clusters is not None:
            logger.debug(f"使用快取的生態位分群 (generation {current_gen}, K={self._optimal_k})")
            return self._cached_clusters, self._cached_elite_pool
        
        # 否則重新計算
        logger.info(f"計算新的生態位分群 (generation {current_gen})...")
        
        # 1. 計算 TED distance matrix
        ted_matrix = self._calculate_ted_distance_matrix(population)
        
        # 2. 自動搜索最佳 K 並執行聚類
        optimal_k, cluster_labels = self._find_optimal_k_and_cluster(ted_matrix, population)
        self._optimal_k = optimal_k
        
        # 3. 提取 Elite Pool
        clusters, elite_pool = self._extract_elite_pool(population, cluster_labels)
        
        # 更新快取
        self._cached_generation = current_gen
        self._cached_clusters = clusters
        self._cached_elite_pool = elite_pool
        
        logger.info(f"生態位分群完成並快取 (generation {current_gen}, K={optimal_k})")
        
        return clusters, elite_pool
    
    def select_pairs(self, population: List, k: int, data: Dict[str, Any]) -> List[Tuple]:
        """
        選擇 k 對 parents（用於 Crossover）
        
        從 Elite Pool 選擇，支援同群/跨群配對。
        
        Args:
            population: 族群列表
            k: 需要的 parent pairs 數量（會產生 k*2 個 offspring）
            data: 演化數據
            
        Returns:
            k 對 parents: [(parent1, parent2), ...]
        """
        if k <= 0:
            return []
        
        # 獲取生態位分群
        clusters, elite_pool = self._get_or_compute_niching(population, data)
        
        # 過濾掉空的 clusters
        non_empty_clusters = [c for c in clusters if len(c) > 0]
        
        if len(non_empty_clusters) == 0:
            logger.error("所有 clusters 都為空！回退到標準 tournament selection")
            return self._fallback_select_pairs(population, k)
        
        pairs = []
        
        for _ in range(k):
            try:
                if random.random() < self.cross_group_ratio and len(non_empty_clusters) >= 2:
                    # 跨群配對
                    cluster_i, cluster_j = random.sample(range(len(non_empty_clusters)), 2)
                    
                    parent1 = tools.selTournament(non_empty_clusters[cluster_i], 1, 
                                                 tournsize=self.tournament_size)[0]
                    parent2 = tools.selTournament(non_empty_clusters[cluster_j], 1, 
                                                 tournsize=self.tournament_size)[0]
                else:
                    # 同群配對
                    cluster_i = random.choice(range(len(non_empty_clusters)))
                    cluster = non_empty_clusters[cluster_i]
                    
                    if len(cluster) >= 2:
                        # 從同一個 cluster 選 2 個不同的 parents
                        parents = tools.selTournament(cluster, 2, 
                                                     tournsize=self.tournament_size)
                        parent1, parent2 = parents[0], parents[1]
                    else:
                        # cluster 只有 1 個個體，回退到跨群
                        if len(non_empty_clusters) >= 2:
                            cluster_j = random.choice([c for c in range(len(non_empty_clusters)) if c != cluster_i])
                            parent1 = cluster[0]
                            parent2 = tools.selTournament(non_empty_clusters[cluster_j], 1,
                                                         tournsize=self.tournament_size)[0]
                        else:
                            # 只有一個 cluster 且只有一個個體，複製
                            parent1 = parent2 = cluster[0]
                
                pairs.append((parent1, parent2))
                
            except Exception as e:
                logger.error(f"選擇 parent pair 失敗: {e}，使用回退方案")
                # 回退：從 elite_pool 隨機選擇
                if len(elite_pool) >= 2:
                    parent1, parent2 = random.sample(elite_pool, 2)
                    pairs.append((parent1, parent2))
        
        logger.debug(f"選擇了 {len(pairs)} 對 parents (目標: {k})")
        
        return pairs
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        選擇 k 個個體（用於 Mutation）
        
        使用 Ranked SUS 從 Elite Pool 選擇。
        
        Args:
            population: 族群列表
            k: 需要選擇的個體數量
            data: 演化數據
            
        Returns:
            選中的 k 個個體
        """
        if k <= 0:
            return []
        
        # 獲取生態位分群
        clusters, elite_pool = self._get_or_compute_niching(population, data)
        
        if not elite_pool:
            logger.error("Elite Pool 為空！回退到標準 tournament selection")
            return tools.selTournament(population, k, tournsize=self.tournament_size)
        
        # 使用 Ranked SUS 從 elite_pool 選擇
        sorted_elite = sorted(elite_pool, 
                             key=lambda x: x.fitness.values[0], 
                             reverse=True)
        
        pop_size = len(sorted_elite)
        
        # 分配排名適應度
        original_fitnesses = []
        for i, ind in enumerate(sorted_elite):
            rank = i + 1
            original_fitnesses.append(ind.fitness.values)
            
            # 計算排名適應度
            if pop_size > 1:
                rank_fitness = self.max_rank_fitness - (
                    (self.max_rank_fitness - self.min_rank_fitness) * (rank - 1) / (pop_size - 1)
                )
            else:
                rank_fitness = self.max_rank_fitness
            
            # 臨時設置排名適應度
            ind.fitness.values = (rank_fitness,)
        
        # 使用 SUS 選擇
        try:
            chosen = tools.selStochasticUniversalSampling(sorted_elite, k)
            logger.debug(f"使用 Ranked SUS 選擇了 {len(chosen)} 個個體")
        except Exception as e:
            logger.warning(f"SUS 選擇失敗，回退到 tournament selection: {e}")
            chosen = tools.selTournament(sorted_elite, k, tournsize=self.tournament_size)
        
        # 恢復原始適應度
        for ind, original_fitness in zip(sorted_elite, original_fitnesses):
            ind.fitness.values = original_fitness
        
        return chosen
    
    def _fallback_select_pairs(self, population: List, k: int) -> List[Tuple]:
        """回退方案：使用標準 tournament selection"""
        logger.warning("使用回退方案選擇 parent pairs")
        selected = tools.selTournament(population, k * 2, tournsize=self.tournament_size)
        pairs = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                pairs.append((selected[i], selected[i + 1]))
        return pairs
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取策略統計信息"""
        return {
            'name': self.name,
            'max_k': self.max_k,
            'optimal_k': self._optimal_k,
            'top_m_per_cluster': self.M,
            'cross_group_ratio': self.cross_group_ratio,
            'cv_criterion': self.cv_criterion,
            'tournament_size': self.tournament_size,
            'cached_generation': self._cached_generation,
            'elite_pool_size': len(self._cached_elite_pool) if self._cached_elite_pool else 0
        }
