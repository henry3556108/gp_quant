"""
採樣相似度矩陣計算

使用採樣策略加速大族群的相似度矩陣計算：
1. 完整計算：隨機採樣 N 個代表性個體
2. 近似計算：其他個體只與代表性個體計算相似度
3. 矩陣補全：使用插值估算未計算的配對

這可以將計算量從 O(n²) 降到 O(n*k)，其中 k << n
"""

from typing import List, Tuple, Optional
import numpy as np
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from .tree_edit_distance import (
    TreeEditDistance,
    TreeNode,
    deap_to_tree_node,
    DEAP_AVAILABLE
)

if DEAP_AVAILABLE:
    from deap import gp


def _compute_distance_batch(
    pairs: List[Tuple[int, int]],
    trees: List[TreeNode],
    cost_insert: Optional[callable] = None,
    cost_delete: Optional[callable] = None,
    cost_rename: Optional[callable] = None
) -> List[Tuple[int, int, float]]:
    """計算一批配對的距離（worker 函數）"""
    ted = TreeEditDistance(
        cost_insert=cost_insert,
        cost_delete=cost_delete,
        cost_rename=cost_rename
    )
    
    results = []
    for i, j in pairs:
        distance = ted.compute(trees[i], trees[j])
        results.append((i, j, distance))
    
    return results


class SampledSimilarityMatrix:
    """
    採樣相似度矩陣計算器
    
    使用採樣策略加速計算：
    - 選擇 sample_size 個代表性個體
    - 所有個體只與代表性個體計算相似度
    - 使用 k-NN 插值估算未計算的配對
    """
    
    def __init__(self, 
                 population: List,
                 sample_size: int = 500,
                 n_workers: int = 8,
                 cost_insert: Optional[callable] = None,
                 cost_delete: Optional[callable] = None,
                 cost_rename: Optional[callable] = None):
        """
        初始化採樣相似度矩陣計算器
        
        Args:
            population: DEAP 族群
            sample_size: 採樣個體數量（建議 500-1000）
            n_workers: 並行 worker 數量
            cost_insert: 插入成本函數
            cost_delete: 刪除成本函數
            cost_rename: 重命名成本函數
        """
        self.population = population
        self.n = len(population)
        self.sample_size = min(sample_size, self.n)
        self.n_workers = n_workers
        
        self.cost_insert = cost_insert
        self.cost_delete = cost_delete
        self.cost_rename = cost_rename
        
        self.matrix = None
        self.distance_matrix = None
        self.sample_indices = None
        
    def _convert_population(self) -> List[TreeNode]:
        """轉換 DEAP population 為 TreeNode 列表"""
        if not DEAP_AVAILABLE:
            raise ImportError("需要安裝 DEAP 才能使用此功能")
        
        trees = []
        for ind in self.population:
            tree_node = deap_to_tree_node(ind)
            trees.append(tree_node)
        return trees
    
    def _select_samples(self) -> np.ndarray:
        """
        選擇代表性個體
        
        策略：
        1. 按 fitness 排序，選擇 top 20%
        2. 從剩餘個體中隨機選擇
        3. 確保覆蓋不同 fitness 範圍
        
        Returns:
            np.ndarray: 採樣個體的索引
        """
        # 獲取 fitness 值
        fitness_values = np.array([ind.fitness.values[0] for ind in self.population])
        
        # 策略 1: 選擇 top 20%
        n_top = max(1, int(self.sample_size * 0.2))
        top_indices = np.argsort(fitness_values)[-n_top:]
        
        # 策略 2: 從剩餘個體中分層採樣
        remaining_indices = np.setdiff1d(np.arange(self.n), top_indices)
        n_remaining = self.sample_size - n_top
        
        if n_remaining > 0:
            # 分層採樣：按 fitness 分成 5 層，每層均勻採樣
            n_strata = 5
            strata_indices = []
            
            remaining_fitness = fitness_values[remaining_indices]
            percentiles = np.linspace(0, 100, n_strata + 1)
            
            for i in range(n_strata):
                lower = np.percentile(remaining_fitness, percentiles[i])
                upper = np.percentile(remaining_fitness, percentiles[i + 1])
                
                stratum_mask = (remaining_fitness >= lower) & (remaining_fitness <= upper)
                stratum_indices_local = np.where(stratum_mask)[0]
                
                if len(stratum_indices_local) > 0:
                    # 從這一層採樣
                    n_sample_stratum = max(1, n_remaining // n_strata)
                    n_sample_stratum = min(n_sample_stratum, len(stratum_indices_local))
                    
                    sampled = np.random.choice(
                        stratum_indices_local,
                        size=n_sample_stratum,
                        replace=False
                    )
                    strata_indices.extend(remaining_indices[sampled])
            
            # 如果採樣不足，隨機補充
            if len(strata_indices) < n_remaining:
                remaining_pool = np.setdiff1d(remaining_indices, strata_indices)
                n_extra = n_remaining - len(strata_indices)
                n_extra = min(n_extra, len(remaining_pool))
                
                if n_extra > 0:
                    extra = np.random.choice(remaining_pool, size=n_extra, replace=False)
                    strata_indices.extend(extra)
            
            # 合併
            sample_indices = np.concatenate([top_indices, strata_indices[:n_remaining]])
        else:
            sample_indices = top_indices
        
        return sample_indices
    
    def compute(self, show_progress=True) -> np.ndarray:
        """
        計算採樣相似度矩陣
        
        Args:
            show_progress: 是否顯示進度條
            
        Returns:
            np.ndarray: 相似度矩陣 (n x n)
        """
        # 選擇代表性個體
        self.sample_indices = self._select_samples()
        
        if show_progress:
            print(f"📊 採樣策略：從 {self.n} 個個體中選擇 {len(self.sample_indices)} 個代表")
            print(f"   計算量：{self.n * len(self.sample_indices)} 對（vs 完整 {self.n * (self.n-1) // 2} 對）")
            print(f"   加速比：{(self.n * (self.n-1) // 2) / (self.n * len(self.sample_indices)):.1f}x")
        
        # 初始化矩陣
        self.distance_matrix = np.zeros((self.n, self.n))
        self.matrix = np.zeros((self.n, self.n))
        np.fill_diagonal(self.matrix, 1.0)
        
        # 轉換為 TreeNode
        trees = self._convert_population()
        
        # 生成配對：所有個體 vs 代表性個體
        pairs = []
        for i in range(self.n):
            for j in self.sample_indices:
                if i != j and i < j:  # 避免重複計算
                    pairs.append((i, j))
        
        total_pairs = len(pairs)
        
        if show_progress:
            print(f"   使用 {self.n_workers} 個 workers 並行計算...")
        
        # 分配配對到各 worker
        chunk_size = max(1, total_pairs // (self.n_workers * 4))
        pair_chunks = [pairs[i:i+chunk_size] for i in range(0, total_pairs, chunk_size)]
        
        # 創建 worker 函數
        worker_func = partial(
            _compute_distance_batch,
            trees=trees,
            cost_insert=self.cost_insert,
            cost_delete=self.cost_delete,
            cost_rename=self.cost_rename
        )
        
        # 並行計算
        if show_progress:
            pbar = tqdm(total=len(pair_chunks), desc="採樣計算", unit="batch")
        
        with Pool(processes=self.n_workers) as pool:
            results_iter = pool.imap_unordered(worker_func, pair_chunks)
            
            for batch_results in results_iter:
                for i, j, distance in batch_results:
                    self.distance_matrix[i][j] = distance
                    self.distance_matrix[j][i] = distance
                    
                    similarity = 1.0 / (1.0 + distance)
                    self.matrix[i][j] = similarity
                    self.matrix[j][i] = similarity
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        # 矩陣補全：使用 k-NN 插值估算未計算的配對
        if show_progress:
            print("   補全矩陣（k-NN 插值）...")
        
        self._complete_matrix()
        
        return self.matrix
    
    def _complete_matrix(self, k=5):
        """
        使用 k-NN 插值補全矩陣
        
        對於未直接計算的配對 (i, j)：
        1. 找到 i 和 j 的 k 個最近鄰（在代表性個體中）
        2. 使用這些鄰居的相似度加權平均估算 sim(i, j)
        """
        # 對於每個非代表性個體對
        for i in range(self.n):
            if i in self.sample_indices:
                continue
            
            for j in range(i + 1, self.n):
                if j in self.sample_indices:
                    continue
                
                # 如果已經計算過，跳過
                if self.matrix[i][j] > 0:
                    continue
                
                # 找到 i 和 j 與代表性個體的相似度
                i_sims = self.matrix[i, self.sample_indices]
                j_sims = self.matrix[j, self.sample_indices]
                
                # 使用 k 個最相似的代表性個體
                k_actual = min(k, len(self.sample_indices))
                i_top_k = np.argsort(i_sims)[-k_actual:]
                j_top_k = np.argsort(j_sims)[-k_actual:]
                
                # 計算加權平均
                # 如果 i 和 j 與相同的代表性個體相似，則它們可能相似
                common_neighbors = np.intersect1d(i_top_k, j_top_k)
                
                if len(common_neighbors) > 0:
                    # 使用共同鄰居的相似度
                    weights = i_sims[common_neighbors] * j_sims[common_neighbors]
                    estimated_sim = np.average(
                        i_sims[common_neighbors] * j_sims[common_neighbors],
                        weights=weights
                    )
                else:
                    # 使用所有 top-k 鄰居的平均
                    all_neighbors = np.union1d(i_top_k, j_top_k)
                    estimated_sim = np.mean(i_sims[all_neighbors] * j_sims[all_neighbors])
                
                self.matrix[i][j] = estimated_sim
                self.matrix[j][i] = estimated_sim
    
    def get_similarity(self, i: int, j: int) -> float:
        """獲取兩個個體之間的相似度"""
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        return self.matrix[i][j]
    
    def get_average_similarity(self) -> float:
        """計算平均相似度"""
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 排除對角線
        n = self.matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return self.matrix[mask].mean()
    
    def get_diversity_score(self) -> float:
        """計算族群多樣性分數（平均相異度）"""
        return 1.0 - self.get_average_similarity()
