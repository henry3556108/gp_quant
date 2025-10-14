"""
並行相似度矩陣計算

使用 multiprocessing 加速大族群的相似度矩陣計算。
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from multiprocessing import Pool, cpu_count
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
    cost_insert: Optional[Callable] = None,
    cost_delete: Optional[Callable] = None,
    cost_rename: Optional[Callable] = None
) -> List[Tuple[int, int, float]]:
    """
    計算一批配對的距離（worker 函數）
    
    Args:
        pairs: 配對列表 [(i, j), ...]
        trees: TreeNode 列表
        cost_insert: 插入成本函數
        cost_delete: 刪除成本函數
        cost_rename: 重命名成本函數
        
    Returns:
        List[Tuple[int, int, float]]: [(i, j, distance), ...]
    """
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


class ParallelSimilarityMatrix:
    """
    並行相似度矩陣計算器
    
    使用 multiprocessing 加速計算。適合大族群（n > 100）。
    
    Attributes:
        population: 族群（DEAP Individual 或 TreeNode 列表）
        ted: TreeEditDistance 計算器
        matrix: 相似度矩陣 (numpy array)
        distance_matrix: 距離矩陣 (numpy array)
        n_workers: 並行 worker 數量
    """
    
    def __init__(self, 
                 population: List,
                 cost_insert=None,
                 cost_delete=None,
                 cost_rename=None,
                 n_workers: Optional[int] = None):
        """
        初始化並行相似度矩陣計算器
        
        Args:
            population: 族群列表（DEAP Individual 或 TreeNode）
            cost_insert: 插入成本函數
            cost_delete: 刪除成本函數
            cost_rename: 重命名成本函數
            n_workers: 並行 worker 數量（None = cpu_count()）
        """
        self.population = population
        self.n = len(population)
        
        # 成本函數
        self.cost_insert = cost_insert
        self.cost_delete = cost_delete
        self.cost_rename = cost_rename
        
        # Worker 數量
        if n_workers is None:
            self.n_workers = cpu_count()
        else:
            self.n_workers = min(n_workers, cpu_count())
        
        # 初始化矩陣
        self.matrix = None
        self.distance_matrix = None
    
    def _convert_population(self) -> List[TreeNode]:
        """
        轉換 population 為 TreeNode 列表
        
        Returns:
            List[TreeNode]: TreeNode 列表
        """
        trees = []
        for ind in self.population:
            if isinstance(ind, TreeNode):
                trees.append(ind)
            elif DEAP_AVAILABLE and isinstance(ind, gp.PrimitiveTree):
                trees.append(deap_to_tree_node(ind))
            else:
                raise TypeError(f"不支援的類型: {type(ind)}")
        return trees
    
    def _generate_pairs(self) -> List[Tuple[int, int]]:
        """
        生成所有需要計算的配對（上三角）
        
        Returns:
            List[Tuple[int, int]]: 配對列表
        """
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j))
        return pairs
    
    def _split_pairs(self, pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        將配對分配到各 worker
        
        Args:
            pairs: 配對列表
            
        Returns:
            List[List[Tuple[int, int]]]: 分配後的配對列表
        """
        n_pairs = len(pairs)
        chunk_size = (n_pairs + self.n_workers - 1) // self.n_workers
        
        chunks = []
        for i in range(0, n_pairs, chunk_size):
            chunks.append(pairs[i:i + chunk_size])
        
        return chunks
    
    def compute(self, show_progress=True) -> np.ndarray:
        """
        並行計算相似度矩陣
        
        Args:
            show_progress: 是否顯示進度條
            
        Returns:
            np.ndarray: 相似度矩陣 (n x n)
        """
        # 初始化矩陣
        self.distance_matrix = np.zeros((self.n, self.n))
        self.matrix = np.zeros((self.n, self.n))
        
        # 對角線為 1（自己與自己完全相同）
        np.fill_diagonal(self.matrix, 1.0)
        
        # 轉換 population 為 TreeNode
        trees = self._convert_population()
        
        # 生成配對
        pairs = self._generate_pairs()
        total_pairs = len(pairs)
        
        if show_progress:
            print(f"計算 {self.n} 個個體的相似度矩陣（{total_pairs} 對）")
            print(f"使用 {self.n_workers} 個 workers 並行計算...")
        
        # 分配配對到各 worker
        pair_chunks = self._split_pairs(pairs)
        
        # 創建 worker 函數（綁定參數）
        worker_func = partial(
            _compute_distance_batch,
            trees=trees,
            cost_insert=self.cost_insert,
            cost_delete=self.cost_delete,
            cost_rename=self.cost_rename
        )
        
        # 並行計算
        if show_progress:
            pbar = tqdm(total=total_pairs, desc="並行計算")
        
        with Pool(processes=self.n_workers) as pool:
            # 使用 imap_unordered 以便即時更新進度
            results_iter = pool.imap_unordered(worker_func, pair_chunks)
            
            for batch_results in results_iter:
                # 填充矩陣
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
        
        return self.matrix
    
    def get_similarity(self, i: int, j: int) -> float:
        """
        獲取兩個個體之間的相似度
        
        Args:
            i: 個體 i 的索引
            j: 個體 j 的索引
            
        Returns:
            float: 相似度 [0, 1]
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        return self.matrix[i][j]
    
    def get_distance(self, i: int, j: int) -> float:
        """
        獲取兩個個體之間的距離
        
        Args:
            i: 個體 i 的索引
            j: 個體 j 的索引
            
        Returns:
            float: 距離
        """
        if self.distance_matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        return self.distance_matrix[i][j]
    
    def get_average_similarity(self) -> float:
        """
        獲取平均相似度（不包括對角線）
        
        Returns:
            float: 平均相似度
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 排除對角線
        mask = ~np.eye(self.n, dtype=bool)
        return np.mean(self.matrix[mask])
    
    def get_diversity_score(self) -> float:
        """
        獲取多樣性分數（1 - 平均相似度）
        
        Returns:
            float: 多樣性分數 [0, 1]
        """
        return 1.0 - self.get_average_similarity()
    
    def get_statistics(self) -> dict:
        """
        獲取相似度統計資訊
        
        Returns:
            dict: 統計資訊
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 排除對角線
        mask = ~np.eye(self.n, dtype=bool)
        similarities = self.matrix[mask]
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'diversity_score': float(self.get_diversity_score())
        }
    
    def get_most_similar_pairs(self, top_k=10) -> List[Tuple[int, int, float]]:
        """
        獲取最相似的 k 對個體
        
        Args:
            top_k: 返回前 k 對
            
        Returns:
            List[Tuple[int, int, float]]: [(i, j, similarity), ...]
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 只考慮上三角（排除對角線）
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j, self.matrix[i][j]))
        
        # 按相似度降序排序
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:top_k]
    
    def get_most_dissimilar_pairs(self, top_k=10) -> List[Tuple[int, int, float]]:
        """
        獲取最不相似的 k 對個體
        
        Args:
            top_k: 返回前 k 對
            
        Returns:
            List[Tuple[int, int, float]]: [(i, j, similarity), ...]
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 只考慮上三角（排除對角線）
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j, self.matrix[i][j]))
        
        # 按相似度升序排序
        pairs.sort(key=lambda x: x[2])
        
        return pairs[:top_k]
