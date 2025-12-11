"""
Niche Selection Strategies

基於不同距離度量的生態位選擇策略：
1. TEDNicheSelectionStrategy: 基於 Tree Edit Distance（基因型多樣性）
2. PnLNicheSelectionStrategy: 基於 PnL Correlation（表現型多樣性）- 待實作

使用階層式分群（Complete Linkage）將族群分為多個生態位，
每個生態位保留 Top M 個體，形成 Elite Pool 用於親代選擇。

---
PnL Correlation 方法選擇建議：
- Pearson Correlation（預設）：
  * 衡量線性相關性
  * 對異常值敏感
  * 計算速度快
  * 適用於 PnL curve 呈線性關係的情況
  
- Spearman Correlation（備選）：
  * 衡量單調相關性（基於排序）
  * 對異常值更穩健
  * 計算速度較慢
  * 適用於 PnL curve 有非線性關係或異常值的情況
  
建議：先使用 Pearson，如果發現聚類結果不穩定或受異常值影響，再改用 Spearman。
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
       - 計算每個 K 的 Silhouette Score
       - 使用二階導數法找到 knee point
    4. 每個 cluster 保留 Top M 個體（按 fitness）→ Elite Pool
    5. Crossover & Mutation 從 Elite Pool 選擇
    6. Reproduction 從整個 Population 選擇（使用 Tournament）
    """
    
    def __init__(self, 
                 max_k: int = 5,
                 fixed_k: int = None,
                 top_m_per_cluster: int = 50,
                 cross_group_ratio: float = 0.3,
                 tournament_size: int = 3,
                 max_rank_fitness: float = 1.8,
                 min_rank_fitness: float = 0.2,
                 n_jobs: int = 6):
        """
        初始化 TED Niche Selection Strategy
        
        Args:
            max_k: 最大分群數量（會搜索 K=2 到 max_k）
            fixed_k: 固定的 K 值（如果設置，則不進行自動搜索）
            top_m_per_cluster: 每個 cluster 保留的個體數 M
            cross_group_ratio: 跨群配對比例（0.0-1.0）
            tournament_size: Tournament selection 的大小
            max_rank_fitness: Ranked SUS 的最大排名適應度
            min_rank_fitness: Ranked SUS 的最小排名適應度
            n_jobs: 平行計算的 worker 數量
        """
        super().__init__()
        self.name = "ted_niche"
        self.max_k = max_k
        self.fixed_k = fixed_k
        self.M = top_m_per_cluster
        self.cross_group_ratio = cross_group_ratio
        self.tournament_size = tournament_size
        self.max_rank_fitness = max_rank_fitness
        self.min_rank_fitness = min_rank_fitness
        self.n_jobs = n_jobs
        
        # 快取（每個世代只計算一次）
        self._cached_generation = -1
        self._cached_clusters = None
        self._cached_elite_pool = None
        self._optimal_k = None
        
        if self.fixed_k is not None:
            logger.info(f"初始化 TED Niche Selection: fixed_k={self.fixed_k}, M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
                       f"tournament_size={self.tournament_size}")
        else:
            logger.info(f"初始化 TED Niche Selection: max_k={self.max_k} (使用 Silhouette + 二階導數法), M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
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
        計算標準化 TED distance matrix（使用 ParallelSimilarityMatrix 真正並行化）
        
        Args:
            population: 族群列表
            
        Returns:
            Normalized TED distance matrix (n x n)
        """
        from gp_quant.similarity.parallel_calculator import ParallelSimilarityMatrix
        
        n = len(population)
        total_pairs = n * (n - 1) // 2
        logger.info(f"計算 TED Distance Matrix ({n} x {n}, {total_pairs} pairs)，使用 {self.n_jobs} workers...")
        
        # 使用 ParallelSimilarityMatrix（真正的 multiprocessing）
        calculator = ParallelSimilarityMatrix(
            population=population,
            n_workers=self.n_jobs
        )
        calculator.compute(show_progress=True)
        
        # 取得 raw distance matrix
        ted_matrix = calculator.distance_matrix.copy()
        
        # 正規化：每對除以 max(len(tree_i), len(tree_j))
        sizes = np.array([len(ind) for ind in population], dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                max_size = max(sizes[i], sizes[j])
                if max_size > 0:
                    norm_ted = ted_matrix[i, j] / max_size
                    ted_matrix[i, j] = norm_ted
                    ted_matrix[j, i] = norm_ted
        
        # 統計信息
        upper_tri = np.triu_indices(n, k=1)
        mean_ted = np.mean(ted_matrix[upper_tri])
        std_ted = np.std(ted_matrix[upper_tri])
        
        logger.info(f"TED 統計: 平均={mean_ted:.4f} ± {std_ted:.4f}, "
                   f"範圍=[{np.min(ted_matrix[upper_tri]):.4f}, {np.max(ted_matrix[upper_tri]):.4f}]")
        
        return ted_matrix
    
    def _find_optimal_k_and_cluster(self, distance_matrix: np.ndarray, population: List) -> Tuple[int, np.ndarray]:
        """
        自動搜索最佳 K 值並執行聚類（或使用固定 K）
        
        如果設置了 fixed_k，則直接使用該值；
        否則使用階層式聚類樹，從 K=2 到 max_k 搜索最佳 K：
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
        
        # 建立聚類樹（只建立一次）
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        # 如果設置了 fixed_k，直接使用
        if self.fixed_k is not None:
            logger.info(f"使用固定 K={self.fixed_k}")
            cluster_labels = fcluster(linkage_matrix, self.fixed_k, criterion='maxclust') - 1
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # 計算統計信息
            elite_pool_size = sum(min(count, self.M) for count in counts)
            expected_size = self.fixed_k * self.M
            achievement_rate = elite_pool_size / expected_size
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0.0
            
            logger.info(f"✅ 固定 K={self.fixed_k}, CV={cv:.4f}, 達成率={achievement_rate*100:.1f}%")
            logger.info(f"分群完成: {len(unique_labels)} 個 clusters")
            for label, count in zip(unique_labels, counts):
                logger.debug(f"  Cluster {label}: {count} 個體")
            
            return self.fixed_k, cluster_labels
        
        # 否則自動搜索最佳 K（使用 Silhouette Score + 二階導數法）
        from sklearn.metrics import silhouette_score
        
        logger.info(f"自動搜索最佳 K (K=2~{self.max_k}, 使用 Silhouette Score + 二階導數法)...")
        
        # 計算所有 K 值的 Silhouette Score
        k_values = []
        silhouette_scores = []
        cluster_labels_dict = {}
        
        for k in range(2, self.max_k + 1):
            # 從聚類樹切割出 K 個 clusters
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            cluster_labels_dict[k] = cluster_labels
            
            # 計算 Silhouette Score
            try:
                silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                k_values.append(k)
                silhouette_scores.append(silhouette_avg)
                logger.debug(f"  K={k}: Silhouette={silhouette_avg:.4f}")
            except Exception as e:
                logger.warning(f"  K={k}: Silhouette 計算失敗 ({e})")
        
        # 使用二階導數法找 knee point
        if len(silhouette_scores) < 3:
            # 如果少於 3 個有效的 K，直接使用最後一個
            best_k = k_values[-1] if k_values else 2
            best_silhouette = silhouette_scores[-1] if silhouette_scores else 0.0
            logger.warning(f"只有 {len(k_values)} 個有效 K 值，直接使用 K={best_k}")
        else:
            # 計算二階導數（曲率）
            second_derivatives = np.diff(silhouette_scores, n=2)
            
            # 找到曲率最大的點（絕對值最大）
            knee_idx = np.argmax(np.abs(second_derivatives))
            best_k = k_values[knee_idx + 2]  # +2 因為二階 diff 會減少兩個元素
            best_silhouette = silhouette_scores[knee_idx + 2]
            
            logger.info(f"二階導數分析（曲率）:")
            for i, (k, deriv) in enumerate(zip(k_values[:-2], second_derivatives)):
                marker = " <- Knee Point (最大曲率)" if i == knee_idx else ""
                logger.debug(f"  K={k}: 曲率={deriv:.6f}{marker}")
        
        logger.info(f"✅ 最佳 K={best_k}, Silhouette={best_silhouette:.4f}")
        
        # 使用最佳 K 的聚類結果
        cluster_labels = cluster_labels_dict.get(best_k)
        if cluster_labels is None:
            # 如果沒有快取，重新計算
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
            'tournament_size': self.tournament_size,
            'cached_generation': self._cached_generation,
            'elite_pool_size': len(self._cached_elite_pool) if self._cached_elite_pool else 0
        }


class PnLNicheSelectionStrategy(SelectionStrategy):
    """
    基於 PnL Correlation 的生態位選擇策略（自動搜索最佳 K）
    
    工作流程：
    1. 獲取每個個體的 PnL curve（從 individual.metadata 讀取）
    2. 計算 PnL Correlation Matrix 並轉換為 Distance Matrix
    3. 使用階層式分群（Complete Linkage）建立聚類樹
    4. 自動搜索最佳 K 值（K = 2 到 max_k）
       - 計算每個 K 的 Silhouette Score
       - 使用一階導數法找到 knee point
    5. 每個 cluster 保留 Top M 個體（按 fitness）→ Elite Pool
    6. Crossover & Mutation 從 Elite Pool 選擇
    7. Reproduction 從整個 Population 選擇（使用 Tournament）
    """
    
    def __init__(self, 
                 max_k: int = 5,
                 fixed_k: int = None,
                 top_m_per_cluster: int = 50,
                 cross_group_ratio: float = 0.3,
                 tournament_size: int = 3,
                 max_rank_fitness: float = 1.8,
                 min_rank_fitness: float = 0.2,
                 correlation_method: str = 'pearson',
                 n_jobs: int = 6):
        """
        初始化 PnL Niche Selection Strategy
        
        Args:
            max_k: 最大分群數量（會搜索 K=2 到 max_k）
            fixed_k: 固定的 K 值（如果設置，則不進行自動搜索）
            top_m_per_cluster: 每個 cluster 保留的個體數 M
            cross_group_ratio: 跨群配對比例（0.0-1.0）
            tournament_size: Tournament selection 的大小
            max_rank_fitness: Ranked SUS 的最大排名適應度
            min_rank_fitness: Ranked SUS 的最小排名適應度
            correlation_method: 相關性計算方法（'pearson', 'spearman', 'kendall'）
            n_jobs: 平行計算的 worker 數量（目前未使用，保留以便未來優化）
        """
        super().__init__()
        self.name = "pnl_niche"
        self.max_k = max_k
        self.fixed_k = fixed_k
        self.M = top_m_per_cluster
        self.cross_group_ratio = cross_group_ratio
        self.tournament_size = tournament_size
        self.max_rank_fitness = max_rank_fitness
        self.min_rank_fitness = min_rank_fitness
        self.correlation_method = correlation_method
        self.n_jobs = n_jobs
        
        # 快取（每個世代只計算一次）
        self._cached_generation = -1
        self._cached_clusters = None
        self._cached_elite_pool = None
        self._optimal_k = None
        
        if self.fixed_k is not None:
            logger.info(f"初始化 PnL Niche Selection: fixed_k={self.fixed_k}, M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
                       f"correlation_method={self.correlation_method}, "
                       f"tournament_size={self.tournament_size}")
        else:
            logger.info(f"初始化 PnL Niche Selection: max_k={self.max_k} (使用 Silhouette + 二階導數法), M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
                       f"correlation_method={self.correlation_method}, "
                       f"tournament_size={self.tournament_size}")
    
    def _get_pnl_curves(self, population: List) -> np.ndarray:
        """
        獲取每個個體的 PnL curve
        
        Args:
            population: 族群列表
            
        Returns:
            PnL curves 矩陣 (n_individuals, n_timepoints)
            
        Note:
            如果大部分個體的 PnL curves 是常數（例如初始族群中很多無效策略），
            可能導致聚類失敗（只產生1個cluster）。建議在演化幾代後再使用此策略。
        """
        pnl_curves = []
        
        for idx, ind in enumerate(population):
            # 從 metadata 讀取快取的 pnl_curve
            pnl_curve = ind.get_metadata('pnl_curve')
            
            if pnl_curve is None:
                
                raise ValueError(
                    f"Individual {idx} 沒有 pnl_curve metadata。"
                    f"請確保在 evaluation 時已將 pnl_curve 快取到 individual.metadata。"
                )
            
            # 轉換為 numpy array
            if hasattr(pnl_curve, 'values'):
                pnl_values = pnl_curve.values
            else:
                pnl_values = np.array(pnl_curve)
            
            pnl_curves.append(pnl_values)
        
        result = np.array(pnl_curves)
        
        # 統計有效個體數量（PnL 不是常數）
        stds = np.std(result, axis=1)
        n_valid = np.sum(stds > 1e-10)
        n_const = np.sum(stds <= 1e-10)
        logger.info(f"PnL 統計: 有效個體 {n_valid}/{len(result)} ({n_valid/len(result)*100:.1f}%), 常數個體 {n_const} ({n_const/len(result)*100:.1f}%)")
        
        if n_const > len(result) * 0.8:
            logger.warning(f"⚠️  警告：{n_const/len(result)*100:.1f}% 的個體 PnL 為常數，可能導致聚類失敗！建議增加族群大小。")
        
        return result
    
    def _calculate_pnl_correlation_matrix(self, population: List) -> np.ndarray:
        """
        計算 PnL Correlation Matrix 並轉換為 Distance Matrix
        
        Args:
            population: 族群列表
            
        Returns:
            距離矩陣 (n_individuals, n_individuals)
        """
        logger.info(f"計算 PnL Correlation Matrix ({self.correlation_method})...")
        
        # 獲取 PnL curves
        pnl_curves = self._get_pnl_curves(population)
        n = len(pnl_curves)
        
        logger.info(f"  PnL curves shape: {pnl_curves.shape}")
        
        # 計算 correlation matrix
        if self.correlation_method == 'pearson':
            # 使用 numpy 的 corrcoef（Pearson）
            corr_matrix = np.corrcoef(pnl_curves)
        elif self.correlation_method == 'spearman':
            from scipy.stats import spearmanr
            corr_matrix, _ = spearmanr(pnl_curves, axis=1)
        elif self.correlation_method == 'kendall':
            from scipy.stats import kendalltau
            # Kendall 需要逐對計算
            corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr, _ = kendalltau(pnl_curves[i], pnl_curves[j])
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
        else:
            raise ValueError(f"不支援的 correlation_method: {self.correlation_method}")
        
        nan_count = np.sum(np.isnan(corr_matrix))
        
        # 處理 NaN 值：智能填充
        # NaN 通常出現在常數序列（標準差為0），無法計算相關性
        # 策略：如果兩個常數序列值相同 → corr=1（完全相關）→ distance=0
        #      如果兩個常數序列值不同 → corr=0（無關）→ distance=1
        if nan_count > 0:
            logger.debug(f"處理 {nan_count} 個 NaN correlation 值...")
            # 計算每個 PnL curve 的標準差和均值
            stds = np.std(pnl_curves, axis=1)
            means = np.mean(pnl_curves, axis=1)
            
            for i in range(n):
                for j in range(i+1, n):
                    if np.isnan(corr_matrix[i, j]):
                        # 檢查是否為常數序列
                        is_i_const = stds[i] < 1e-10
                        is_j_const = stds[j] < 1e-10
                        
                        if is_i_const and is_j_const:
                            # 兩個都是常數，比較均值
                            if np.abs(means[i] - means[j]) < 1e-10:
                                corr_matrix[i, j] = corr_matrix[j, i] = 1.0  # 相同 → 完全相關
                            else:
                                corr_matrix[i, j] = corr_matrix[j, i] = 0.0  # 不同 → 無關
                        else:
                            # 至少一個不是常數 → 無關
                            corr_matrix[i, j] = corr_matrix[j, i] = 0.0
            
            logger.debug(f"NaN 處理完成")
        
        # 最後保險
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 轉換為距離矩陣: distance = 1 - abs(correlation)
        # 使用 abs() 是因為負相關也代表策略相關（只是方向相反）
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 確保對角線為 0（自己與自己的距離）
        np.fill_diagonal(distance_matrix, 0.0)
        
        # 確保對稱性
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        logger.info(f"  Distance matrix 統計: min={distance_matrix.min():.4f}, "
                   f"max={distance_matrix.max():.4f}, mean={distance_matrix.mean():.4f}")
        
        return distance_matrix
    
    def _find_optimal_k_and_cluster(self, distance_matrix: np.ndarray, population: List) -> Tuple[int, np.ndarray]:
        """
        自動搜索最佳 K 值並執行聚類（或使用固定 K）
        
        使用與 TED Niche Selection 相同的方法：
        - Silhouette Score + 二階導數法找 knee point
        
        Args:
            distance_matrix: 距離矩陣
            population: 族群列表
            
        Returns:
            (optimal_k, cluster_labels)
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # 建立聚類樹（只建立一次）
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        # 如果設置了 fixed_k，直接使用
        if self.fixed_k is not None:
            logger.info(f"使用固定 K={self.fixed_k}")
            cluster_labels = fcluster(linkage_matrix, self.fixed_k, criterion='maxclust') - 1
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            # 計算統計信息
            elite_pool_size = sum(min(count, self.M) for count in counts)
            expected_size = self.fixed_k * self.M
            achievement_rate = elite_pool_size / expected_size
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0.0
            
            logger.info(f"✅ 固定 K={self.fixed_k}, CV={cv:.4f}, 達成率={achievement_rate*100:.1f}%")
            logger.info(f"分群完成: {len(unique_labels)} 個 clusters")
            for label, count in zip(unique_labels, counts):
                logger.debug(f"  Cluster {label}: {count} 個體")
            
            return self.fixed_k, cluster_labels
        
        # 否則自動搜索最佳 K（使用 Silhouette Score + 二階導數法）
        from sklearn.metrics import silhouette_score
        
        logger.info(f"自動搜索最佳 K (K=2~{self.max_k}, 使用 Silhouette Score + 二階導數法)...")
        
        # 計算所有 K 值的 Silhouette Score
        k_values = []
        silhouette_scores = []
        cluster_labels_dict = {}
        
        for k in range(2, self.max_k + 1):
            # 從聚類樹切割出 K 個 clusters
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            cluster_labels_dict[k] = cluster_labels
            
            # 計算 Silhouette Score
            try:
                silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                k_values.append(k)
                silhouette_scores.append(silhouette_avg)
                logger.debug(f"  K={k}: Silhouette={silhouette_avg:.4f}")
            except Exception as e:
                logger.warning(f"  K={k}: Silhouette 計算失敗 ({e})")
        
        # 使用二階導數法找 knee point
        if len(silhouette_scores) < 3:
            # 如果少於 3 個有效的 K，直接使用最後一個
            best_k = k_values[-1] if k_values else 2
            best_silhouette = silhouette_scores[-1] if silhouette_scores else 0.0
            logger.warning(f"只有 {len(k_values)} 個有效 K 值，直接使用 K={best_k}")
        else:
            # 計算二階導數（曲率）
            second_derivatives = np.diff(silhouette_scores, n=2)
            
            # 找到曲率最大的點（絕對值最大）
            knee_idx = np.argmax(np.abs(second_derivatives))
            best_k = k_values[knee_idx + 2]  # +2 因為二階 diff 會減少兩個元素
            best_silhouette = silhouette_scores[knee_idx + 2]
            
            logger.info(f"二階導數分析（曲率）:")
            for i, (k, deriv) in enumerate(zip(k_values[:-2], second_derivatives)):
                marker = " <- Knee Point (最大曲率)" if i == knee_idx else ""
                logger.debug(f"  K={k}: 曲率={deriv:.6f}{marker}")
        
        logger.info(f"✅ 最佳 K={best_k}, Silhouette={best_silhouette:.4f}")
        
        # 使用最佳 K 的聚類結果
        cluster_labels = cluster_labels_dict.get(best_k)
        if cluster_labels is None:
            # 如果沒有快取，重新計算
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
        （與 TED Niche Selection 完全相同）
        
        Args:
            population: 族群列表
            cluster_labels: 聚類標籤
            
        Returns:
            (clusters, elite_pool)
        """
        # 按 cluster 分組
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(population[idx])
        
        # 從每個 cluster 選擇 Top M
        elite_pool = []
        actual_num_clusters = len(clusters)
        
        for label, cluster_individuals in clusters.items():
            # 按 fitness 排序（降序）
            sorted_individuals = sorted(
                cluster_individuals,
                key=lambda ind: ind.fitness.values[0] if ind.fitness.values else float('-inf'),
                reverse=True
            )
            
            # 選擇 Top M
            top_m = sorted_individuals[:self.M]
            elite_pool.extend(top_m)
            
            logger.debug(f"  Cluster {label}: {len(cluster_individuals)} 個體 -> 選擇 {len(top_m)} 個")
        
        # 計算達成率
        expected_size = actual_num_clusters * self.M
        achievement_rate = len(elite_pool) / expected_size if expected_size > 0 else 0.0
        
        logger.info(f"Elite Pool 構建完成: {len(elite_pool)} 個體 "
                   f"(預期 {expected_size}, 達成率 {achievement_rate*100:.1f}%)")
        
        return list(clusters.values()), elite_pool
    
    def _get_or_compute_niching(self, population: List, generation: int) -> Tuple[List[List], List]:
        """
        獲取或計算生態位分群和 Elite Pool（帶快取）
        
        Args:
            population: 族群列表
            generation: 當前世代
            
        Returns:
            (clusters, elite_pool)
        """
        # 檢查快取
        if generation == self._cached_generation and self._cached_clusters is not None:
            logger.debug(f"使用快取的分群結果 (generation {generation})")
            return self._cached_clusters, self._cached_elite_pool
        
        # 計算 PnL Correlation Distance Matrix
        distance_matrix = self._calculate_pnl_correlation_matrix(population)
        
        # 自動搜索最佳 K 並執行聚類
        optimal_k, cluster_labels = self._find_optimal_k_and_cluster(distance_matrix, population)
        self._optimal_k = optimal_k
        
        # 提取 Elite Pool
        clusters, elite_pool = self._extract_elite_pool(population, cluster_labels)
        
        # 更新快取
        self._cached_generation = generation
        self._cached_clusters = clusters
        self._cached_elite_pool = elite_pool
        
        return clusters, elite_pool
    
    def select_pairs(self, population: List, k: int, data: Dict[str, Any]) -> List[Tuple]:
        """
        選擇 k 對 parents 進行交配（從 Elite Pool 選擇）
        （與 TED Niche Selection 完全相同）
        
        Args:
            population: 族群列表
            k: 需要選擇的配對數量
            data: 額外數據（包含 generation）
            
        Returns:
            k 對 parents 的列表
        """
        generation = data.get('generation', 0)
        
        # 獲取分群和 Elite Pool
        clusters, elite_pool = self._get_or_compute_niching(population, generation)
        
        if not elite_pool:
            logger.warning("Elite Pool 為空，回退到從整個 population 選擇")
            elite_pool = population
        
        # 計算跨群和群內配對數量
        num_cross_group = int(k * self.cross_group_ratio)
        num_in_group = k - num_cross_group
        
        pairs = []
        
        # 跨群配對
        for _ in range(num_cross_group):
            parent1 = random.choice(elite_pool)
            parent2 = random.choice(elite_pool)
            pairs.append((parent1, parent2))
        
        # 群內配對
        for _ in range(num_in_group):
            if len(clusters) > 0:
                cluster = random.choice(clusters)
                if len(cluster) >= 2:
                    parent1, parent2 = random.sample(cluster, 2)
                else:
                    parent1 = parent2 = cluster[0]
                pairs.append((parent1, parent2))
            else:
                parent1 = random.choice(elite_pool)
                parent2 = random.choice(elite_pool)
                pairs.append((parent1, parent2))
        
        logger.debug(f"選擇了 {len(pairs)} 對 parents (跨群={num_cross_group}, 群內={num_in_group})")
        
        return pairs
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        選擇 k 個 individuals 進行變異（從 Elite Pool 使用 Ranked SUS）
        （與 TED Niche Selection 完全相同）
        
        Args:
            population: 族群列表
            k: 需要選擇的個體數量
            data: 額外數據（包含 generation）
            
        Returns:
            k 個 individuals 的列表
        """
        generation = data.get('generation', 0)
        
        # 獲取分群和 Elite Pool
        clusters, elite_pool = self._get_or_compute_niching(population, generation)
        
        if not elite_pool:
            logger.warning("Elite Pool 為空，回退到從整個 population 選擇")
            elite_pool = population
        
        # 使用 Ranked SUS 從 Elite Pool 選擇
        selected = self._ranked_sus_selection(elite_pool, k)
        
        logger.debug(f"使用 Ranked SUS 從 Elite Pool 選擇了 {len(selected)} 個 individuals")
        
        return selected
    
    def _ranked_sus_selection(self, population: List, k: int) -> List:
        """
        Ranked Stochastic Universal Sampling
        （與 TED Niche Selection 完全相同）
        
        Args:
            population: 族群列表
            k: 需要選擇的個體數量
            
        Returns:
            選中的個體列表
        """
        if not population:
            return []
        
        if k >= len(population):
            return population.copy()
        
        # 按 fitness 排序
        sorted_pop = sorted(
            population,
            key=lambda ind: ind.fitness.values[0] if ind.fitness.values else float('-inf'),
            reverse=True
        )
        
        # 計算排名適應度
        n = len(sorted_pop)
        rank_fitness = []
        for i in range(n):
            # 線性排名：最佳個體 = max_rank_fitness，最差個體 = min_rank_fitness
            fitness = self.max_rank_fitness - (self.max_rank_fitness - self.min_rank_fitness) * i / (n - 1)
            rank_fitness.append(fitness)
        
        # SUS 選擇
        total_fitness = sum(rank_fitness)
        distance = total_fitness / k
        start = random.uniform(0, distance)
        
        selected = []
        current_sum = 0
        pointer = start
        
        for i, ind in enumerate(sorted_pop):
            current_sum += rank_fitness[i]
            while pointer <= current_sum and len(selected) < k:
                selected.append(ind)
                pointer += distance
        
        return selected
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取策略統計信息"""
        return {
            'name': self.name,
            'max_k': self.max_k,
            'optimal_k': self._optimal_k,
            'top_m_per_cluster': self.M,
            'cross_group_ratio': self.cross_group_ratio,
            'correlation_method': self.correlation_method,
            'tournament_size': self.tournament_size,
            'cached_generation': self._cached_generation,
            'elite_pool_size': len(self._cached_elite_pool) if self._cached_elite_pool else 0
        }


class SignalNicheSelectionStrategy(SelectionStrategy):
    """
    Signal Overlap-based Niche Selection Strategy
    
    Uses signal overlap (percentage of matching trading decisions) as the
    distance metric for clustering strategies into niches. This approach
    doesn't require pnl_curve metadata, making it compatible with all evaluators.
    
    Workflow:
    1. Compute trading signals for all individuals using entire training data
    2. Calculate Signal Overlap Matrix (pairwise overlap ratios)
    3. Convert overlap to distance: distance = 1 - overlap
    4. Use hierarchical clustering (Complete Linkage) to form niches
    5. Auto-search optimal K (or use fixed K)
    6. Extract Top M individuals from each cluster -> Elite Pool
    7. Select parents from Elite Pool (cross-niche / within-niche)
    
    Note: Signal computation uses the FULL training period, independent of
          the fitness evaluator's segment/window strategy.
    """
    
    def __init__(self, 
                 max_k: int = 5,
                 fixed_k: int = None,
                 top_m_per_cluster: int = 50,
                 cross_group_ratio: float = 0.3,
                 tournament_size: int = 3,
                 max_rank_fitness: float = 1.8,
                 min_rank_fitness: float = 0.2,
                 n_jobs: int = 6):
        """
        Initialize Signal Niche Selection Strategy
        
        Args:
            max_k: Maximum number of clusters (searches K=2 to max_k)
            fixed_k: Fixed K value (if set, skips auto-search)
            top_m_per_cluster: Number of individuals to keep per cluster (M)
            cross_group_ratio: Ratio of cross-niche pairings (0.0-1.0)
            tournament_size: Tournament selection size
            max_rank_fitness: Max rank fitness for Ranked SUS
            min_rank_fitness: Min rank fitness for Ranked SUS
            n_jobs: Number of parallel workers (for future optimization)
        """
        super().__init__()
        self.name = "signal_niche"
        self.max_k = max_k
        self.fixed_k = fixed_k
        self.M = top_m_per_cluster
        self.cross_group_ratio = cross_group_ratio
        self.tournament_size = tournament_size
        self.max_rank_fitness = max_rank_fitness
        self.min_rank_fitness = min_rank_fitness
        self.n_jobs = n_jobs
        
        # Cache (computed once per generation)
        self._cached_generation = -1
        self._cached_clusters = None
        self._cached_elite_pool = None
        self._optimal_k = None
        self._backtest_engine = None  # Lazy initialization
        
        if self.fixed_k is not None:
            logger.info(f"初始化 Signal Niche Selection: fixed_k={self.fixed_k}, M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
                       f"tournament_size={self.tournament_size}")
        else:
            logger.info(f"初始化 Signal Niche Selection: max_k={self.max_k} (Silhouette + 一階導數法), M={self.M}, "
                       f"cross_group_ratio={self.cross_group_ratio}, "
                       f"tournament_size={self.tournament_size}")
    
    def _get_backtest_engine(self, data: Dict[str, Any]):
        """
        Get or create a PortfolioBacktestingEngine for signal computation.
        Uses engine config to match training data period.
        """
        if self._backtest_engine is None:
            from gp_quant.backtesting import PortfolioBacktestingEngine
            
            config = self.engine.config
            train_data = data.get('train_data')
            
            if train_data is None:
                raise ValueError("data 中沒有 'train_data'。Signal Niche 需要完整的 training data。")
            
            # Process train_data: extract DataFrame from dict structure
            # train_data structure: {ticker: {'data': DataFrame}} or {ticker: DataFrame}
            processed_data = {}
            for ticker, ticker_data in train_data.items():
                if isinstance(ticker_data, dict) and 'data' in ticker_data:
                    processed_data[ticker] = ticker_data['data']
                else:
                    processed_data[ticker] = ticker_data
            
            self._backtest_engine = PortfolioBacktestingEngine(
                data=processed_data,
                backtest_start=config['data']['train_backtest_start'],
                backtest_end=config['data']['train_backtest_end'],
                initial_capital=100000.0
            )
            # DEBUG: Log engine creation
            print(f"[DEBUG] SignalNiche: Created PortfolioBacktestingEngine "
                  f"(period: {config['data']['train_backtest_start']} ~ {config['data']['train_backtest_end']}, "
                  f"tickers: {list(processed_data.keys())})")
        
        return self._backtest_engine
    
    def _compute_signals_for_population(self, population: List, data: Dict[str, Any]) -> List[np.ndarray]:
        """
        Compute trading signals for all individuals.
        
        Args:
            population: List of individuals
            data: Evolution data (must contain 'train_data')
            
        Returns:
            List of signal arrays, one per individual
        """
        engine = self._get_backtest_engine(data)
        n = len(population)
        
        # DEBUG: Log signal computation start
        print(f"[DEBUG] SignalNiche: Computing signals for {n} individuals...")
        
        signals = []
        for i, ind in enumerate(tqdm(population, desc="計算 Signals", ncols=100)):
            try:
                sig = engine.get_signals(ind)
                signals.append(sig)
            except Exception as e:
                logger.warning(f"Signal 計算失敗 (個體 {i}): {e}，使用空信號")
                # Return zero signals on failure
                sample_sig = engine.get_signals(population[0]) if len(signals) == 0 else signals[0]
                signals.append(np.zeros_like(sample_sig))
        
        # DEBUG: Log signal computation complete
        print(f"[DEBUG] SignalNiche: Signal computation complete. Shape: {signals[0].shape if signals else 'N/A'}")
        
        return signals
    
    @staticmethod
    def _signal_overlap(signal_a: np.ndarray, signal_b: np.ndarray) -> float:
        """
        Compute overlap ratio between two signal arrays.
        
        Returns:
            Overlap ratio in [0, 1] where 1 = identical signals
        """
        if len(signal_a) != len(signal_b):
            raise ValueError(f"Signal length mismatch: {len(signal_a)} vs {len(signal_b)}")
        
        same = np.sum(signal_a == signal_b)
        return same / len(signal_a)
    
    def _calculate_signal_overlap_matrix(self, population: List, data: Dict[str, Any]) -> np.ndarray:
        """
        Calculate Signal Overlap Matrix and convert to Distance Matrix.
        
        Args:
            population: List of individuals
            data: Evolution data
            
        Returns:
            Distance matrix (n x n) where distance = 1 - overlap
        """
        n = len(population)
        logger.info(f"📊 計算 Signal Overlap Matrix ({n} x {n})...")
        
        # DEBUG: Print to console for visibility
        print(f"[DEBUG] SignalNiche: Calculating Signal Overlap Matrix ({n} x {n})...")
        
        # Step 1: Compute signals for all individuals
        signals = self._compute_signals_for_population(population, data)
        
        # Step 2: Calculate pairwise overlap
        overlap_matrix = np.zeros((n, n))
        
        total_pairs = n * (n - 1) // 2
        with tqdm(total=total_pairs, desc="計算 Overlap", ncols=100) as pbar:
            for i in range(n):
                overlap_matrix[i, i] = 1.0  # Self-overlap = 1
                for j in range(i + 1, n):
                    overlap = self._signal_overlap(signals[i], signals[j])
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
                    pbar.update(1)
        
        # Convert overlap to distance: high overlap -> low distance
        distance_matrix = 1.0 - overlap_matrix
        
        # Ensure diagonal is 0
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Stats
        upper_tri = np.triu_indices(n, k=1)
        mean_dist = np.mean(distance_matrix[upper_tri])
        std_dist = np.std(distance_matrix[upper_tri])
        mean_overlap = np.mean(overlap_matrix[upper_tri])
        
        logger.info(f"Signal Overlap 統計: 平均重疊={mean_overlap:.4f}, "
                   f"平均距離={mean_dist:.4f} ± {std_dist:.4f}")
        
        # DEBUG: Print stats
        print(f"[DEBUG] SignalNiche: Mean overlap={mean_overlap:.4f}, Mean distance={mean_dist:.4f}")
        
        return distance_matrix
    
    def _find_optimal_k_and_cluster(self, distance_matrix: np.ndarray, population: List) -> Tuple[int, np.ndarray]:
        """
        Auto-search best K and cluster (or use fixed K).
        Same logic as TEDNicheSelectionStrategy.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Build hierarchical clustering tree
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='complete')
        
        if self.fixed_k is not None:
            logger.info(f"使用固定 K={self.fixed_k}")
            cluster_labels = fcluster(linkage_matrix, self.fixed_k, criterion='maxclust') - 1
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            
            cv = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0.0
            logger.info(f"✅ 固定 K={self.fixed_k}, CV={cv:.4f}")
            print(f"[DEBUG] SignalNiche: Fixed K={self.fixed_k}, Clusters: {dict(zip(unique_labels, counts))}")
            
            return self.fixed_k, cluster_labels
        
        # Auto-search using Silhouette Score
        from sklearn.metrics import silhouette_score
        
        logger.info(f"自動搜索最佳 K (K=2~{self.max_k})...")
        
        k_values = []
        silhouette_scores = []
        cluster_labels_dict = {}
        
        for k in range(2, self.max_k + 1):
            cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
            cluster_labels_dict[k] = cluster_labels
            
            try:
                silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                k_values.append(k)
                silhouette_scores.append(silhouette_avg)
                logger.debug(f"  K={k}: Silhouette={silhouette_avg:.4f}")
            except Exception as e:
                logger.warning(f"  K={k}: Silhouette 計算失敗 ({e})")
        
        # Find knee point using second derivative
        if len(silhouette_scores) < 3:
            best_k = k_values[-1] if k_values else 2
            best_silhouette = silhouette_scores[-1] if silhouette_scores else 0.0
        else:
            second_derivatives = np.diff(silhouette_scores, n=2)
            knee_idx = np.argmax(np.abs(second_derivatives))
            best_k = k_values[knee_idx + 2]
            best_silhouette = silhouette_scores[knee_idx + 2]
        
        logger.info(f"✅ 最佳 K={best_k}, Silhouette={best_silhouette:.4f}")
        print(f"[DEBUG] SignalNiche: Best K={best_k}, Silhouette={best_silhouette:.4f}")
        
        cluster_labels = cluster_labels_dict.get(best_k)
        if cluster_labels is None:
            cluster_labels = fcluster(linkage_matrix, best_k, criterion='maxclust') - 1
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"[DEBUG] SignalNiche: Cluster distribution: {dict(zip(unique_labels, counts))}")
        
        return best_k, cluster_labels
    
    def _extract_elite_pool(self, population: List, cluster_labels: np.ndarray) -> Tuple[List[List], List]:
        """
        Extract Top M individuals from each cluster to form Elite Pool.
        Same logic as TEDNicheSelectionStrategy.
        """
        logger.info(f"提取 Elite Pool (每個 cluster Top {self.M})...")
        
        clusters = []
        elite_pool = []
        
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)
        
        for cluster_id in unique_labels:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_individuals = [population[i] for i in cluster_indices]
            
            if not cluster_individuals:
                clusters.append([])
                continue
            
            # Sort by fitness (descending)
            cluster_sorted = sorted(cluster_individuals, 
                                   key=lambda x: x.fitness.values[0], 
                                   reverse=True)
            
            top_m = cluster_sorted[:min(self.M, len(cluster_sorted))]
            
            clusters.append(top_m)
            elite_pool.extend(top_m)
            
            logger.debug(f"  Cluster {cluster_id}: {len(cluster_individuals)} 個體 → Top {len(top_m)}")
        
        logger.info(f"Elite Pool 大小: {len(elite_pool)} 個體 (目標: {n_clusters * self.M})")
        print(f"[DEBUG] SignalNiche: Elite Pool size={len(elite_pool)}")
        
        return clusters, elite_pool
    
    def _get_or_compute_niching(self, population: List, data: Dict[str, Any]) -> Tuple[List[List], List]:
        """
        Get or compute niche clustering (with cache).
        """
        current_gen = data.get('generation', -1)
        
        # Use cache if same generation
        if current_gen == self._cached_generation and self._cached_clusters is not None:
            logger.debug(f"使用快取的生態位分群 (generation {current_gen})")
            return self._cached_clusters, self._cached_elite_pool
        
        logger.info(f"計算新的生態位分群 (generation {current_gen})...")
        print(f"[DEBUG] SignalNiche: Computing niching for generation {current_gen}")
        
        # 1. Calculate Signal Overlap Matrix
        distance_matrix = self._calculate_signal_overlap_matrix(population, data)
        
        # 2. Auto-search optimal K and cluster
        optimal_k, cluster_labels = self._find_optimal_k_and_cluster(distance_matrix, population)
        self._optimal_k = optimal_k
        
        # 3. Extract Elite Pool
        clusters, elite_pool = self._extract_elite_pool(population, cluster_labels)
        
        # Update cache
        self._cached_generation = current_gen
        self._cached_clusters = clusters
        self._cached_elite_pool = elite_pool
        
        logger.info(f"生態位分群完成並快取 (generation {current_gen}, K={optimal_k})")
        
        return clusters, elite_pool
    
    def select_pairs(self, population: List, k: int, data: Dict[str, Any]) -> List[Tuple]:
        """
        Select k parent pairs for crossover.
        Same logic as TEDNicheSelectionStrategy.
        """
        if k <= 0:
            return []
        
        clusters, elite_pool = self._get_or_compute_niching(population, data)
        non_empty_clusters = [c for c in clusters if len(c) > 0]
        
        if len(non_empty_clusters) == 0:
            logger.error("所有 clusters 都為空！回退到標準 tournament selection")
            return self._fallback_select_pairs(population, k)
        
        pairs = []
        
        for _ in range(k):
            try:
                if random.random() < self.cross_group_ratio and len(non_empty_clusters) >= 2:
                    # Cross-niche pairing
                    cluster_i, cluster_j = random.sample(range(len(non_empty_clusters)), 2)
                    
                    parent1 = tools.selTournament(non_empty_clusters[cluster_i], 1, 
                                                 tournsize=self.tournament_size)[0]
                    parent2 = tools.selTournament(non_empty_clusters[cluster_j], 1, 
                                                 tournsize=self.tournament_size)[0]
                else:
                    # Within-niche pairing
                    cluster_i = random.choice(range(len(non_empty_clusters)))
                    cluster = non_empty_clusters[cluster_i]
                    
                    if len(cluster) >= 2:
                        parents = tools.selTournament(cluster, 2, tournsize=self.tournament_size)
                        parent1, parent2 = parents[0], parents[1]
                    else:
                        if len(non_empty_clusters) >= 2:
                            cluster_j = random.choice([c for c in range(len(non_empty_clusters)) if c != cluster_i])
                            parent1 = cluster[0]
                            parent2 = tools.selTournament(non_empty_clusters[cluster_j], 1,
                                                         tournsize=self.tournament_size)[0]
                        else:
                            parent1 = parent2 = cluster[0]
                
                pairs.append((parent1, parent2))
                
            except Exception as e:
                logger.error(f"選擇 parent pair 失敗: {e}")
                if len(elite_pool) >= 2:
                    parent1, parent2 = random.sample(elite_pool, 2)
                    pairs.append((parent1, parent2))
        
        logger.debug(f"選擇了 {len(pairs)} 對 parents (目標: {k})")
        
        return pairs
    
    def select_individuals(self, population: List, k: int, data: Dict[str, Any]) -> List:
        """
        Select k individuals for mutation.
        Uses Ranked SUS from Elite Pool.
        """
        if k <= 0:
            return []
        
        clusters, elite_pool = self._get_or_compute_niching(population, data)
        
        if not elite_pool:
            logger.error("Elite Pool 為空！回退到標準 tournament selection")
            return tools.selTournament(population, k, tournsize=self.tournament_size)
        
        # Ranked SUS selection
        sorted_elite = sorted(elite_pool, key=lambda x: x.fitness.values[0], reverse=True)
        pop_size = len(sorted_elite)
        
        original_fitnesses = []
        for i, ind in enumerate(sorted_elite):
            rank = i + 1
            original_fitnesses.append(ind.fitness.values)
            
            if pop_size > 1:
                rank_fitness = self.max_rank_fitness - (
                    (self.max_rank_fitness - self.min_rank_fitness) * (rank - 1) / (pop_size - 1)
                )
            else:
                rank_fitness = self.max_rank_fitness
            
            ind.fitness.values = (rank_fitness,)
        
        try:
            chosen = tools.selStochasticUniversalSampling(sorted_elite, k)
        except Exception as e:
            logger.warning(f"SUS 選擇失敗: {e}")
            chosen = tools.selTournament(sorted_elite, k, tournsize=self.tournament_size)
        
        # Restore original fitness
        for ind, original_fitness in zip(sorted_elite, original_fitnesses):
            ind.fitness.values = original_fitness
        
        return chosen
    
    def _fallback_select_pairs(self, population: List, k: int) -> List[Tuple]:
        """Fallback: standard tournament selection"""
        logger.warning("使用回退方案選擇 parent pairs")
        selected = tools.selTournament(population, k * 2, tournsize=self.tournament_size)
        pairs = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                pairs.append((selected[i], selected[i + 1]))
        return pairs
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'max_k': self.max_k,
            'optimal_k': self._optimal_k,
            'top_m_per_cluster': self.M,
            'cross_group_ratio': self.cross_group_ratio,
            'tournament_size': self.tournament_size,
            'cached_generation': self._cached_generation,
            'elite_pool_size': len(self._cached_elite_pool) if self._cached_elite_pool else 0
        }
