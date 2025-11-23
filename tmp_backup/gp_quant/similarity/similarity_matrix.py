"""
Similarity Matrix 計算

計算族群中所有個體兩兩之間的相似度，生成相似度矩陣。
支援 DEAP Individual 和 TreeNode。
"""

from typing import List, Union
import numpy as np
from tqdm import tqdm

from .tree_edit_distance import (
    TreeEditDistance,
    TreeNode,
    deap_to_tree_node,
    DEAP_AVAILABLE
)

if DEAP_AVAILABLE:
    from deap import gp


class SimilarityMatrix:
    """
    相似度矩陣計算器
    
    計算族群中所有個體兩兩之間的相似度，生成對稱矩陣。
    相似度範圍 [0, 1]，其中 1 表示完全相同，0 表示完全不同。
    
    Attributes:
        population: 族群（DEAP Individual 或 TreeNode 列表）
        ted: TreeEditDistance 計算器
        matrix: 相似度矩陣 (numpy array)
        distance_matrix: 距離矩陣 (numpy array)
    """
    
    def __init__(self, 
                 population: List[Union[TreeNode, 'gp.PrimitiveTree']],
                 cost_insert=None,
                 cost_delete=None,
                 cost_rename=None):
        """
        初始化相似度矩陣計算器
        
        Args:
            population: 族群列表（DEAP Individual 或 TreeNode）
            cost_insert: 插入成本函數
            cost_delete: 刪除成本函數
            cost_rename: 重命名成本函數
        """
        self.population = population
        self.n = len(population)
        
        # 創建 TED 計算器
        self.ted = TreeEditDistance(
            cost_insert=cost_insert,
            cost_delete=cost_delete,
            cost_rename=cost_rename
        )
        
        # 初始化矩陣
        self.matrix = None
        self.distance_matrix = None
    
    def compute(self, show_progress=True) -> np.ndarray:
        """
        計算相似度矩陣
        
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
        
        # 轉換 population 為 TreeNode（如果需要）
        trees = self._convert_population()
        
        # 計算上三角矩陣（利用對稱性）
        total_pairs = self.n * (self.n - 1) // 2
        
        if show_progress:
            pbar = tqdm(total=total_pairs, desc="計算相似度矩陣")
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # 計算距離
                distance = self.ted.compute(trees[i], trees[j])
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance
                
                # 計算相似度
                similarity = 1.0 / (1.0 + distance)
                self.matrix[i][j] = similarity
                self.matrix[j][i] = similarity
                
                if show_progress:
                    pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        return self.matrix
    
    def _convert_population(self) -> List[TreeNode]:
        """
        將 population 轉換為 TreeNode 列表
        
        Returns:
            List[TreeNode]: 轉換後的樹列表
        """
        trees = []
        for ind in self.population:
            if isinstance(ind, TreeNode):
                trees.append(ind)
            else:
                # 假設是 DEAP Individual
                trees.append(deap_to_tree_node(ind))
        return trees
    
    def get_similarity(self, i: int, j: int) -> float:
        """
        獲取兩個個體之間的相似度
        
        Args:
            i: 第一個個體的索引
            j: 第二個個體的索引
            
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
            i: 第一個個體的索引
            j: 第二個個體的索引
            
        Returns:
            float: 編輯距離
        """
        if self.distance_matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        return self.distance_matrix[i][j]
    
    def get_most_similar_pairs(self, n: int = 5) -> List[tuple]:
        """
        獲取最相似的 n 對個體
        
        Args:
            n: 返回的配對數量
            
        Returns:
            List[tuple]: [(i, j, similarity), ...] 按相似度降序排列
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 獲取上三角矩陣的索引和值
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j, self.matrix[i][j]))
        
        # 按相似度降序排序
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs[:n]
    
    def get_least_similar_pairs(self, n: int = 5) -> List[tuple]:
        """
        獲取最不相似的 n 對個體
        
        Args:
            n: 返回的配對數量
            
        Returns:
            List[tuple]: [(i, j, similarity), ...] 按相似度升序排列
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 獲取上三角矩陣的索引和值
        pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                pairs.append((i, j, self.matrix[i][j]))
        
        # 按相似度升序排序
        pairs.sort(key=lambda x: x[2])
        
        return pairs[:n]
    
    def get_average_similarity(self) -> float:
        """
        獲取族群的平均相似度
        
        Returns:
            float: 平均相似度
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 只計算上三角矩陣（不包括對角線）
        total = 0.0
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                total += self.matrix[i][j]
                count += 1
        
        return total / count if count > 0 else 0.0
    
    def get_diversity_score(self) -> float:
        """
        獲取族群的多樣性分數
        
        多樣性 = 1 - 平均相似度
        範圍 [0, 1]，越大表示越多樣
        
        Returns:
            float: 多樣性分數
        """
        return 1.0 - self.get_average_similarity()
    
    def get_statistics(self) -> dict:
        """
        獲取相似度矩陣的統計資訊
        
        Returns:
            dict: 統計資訊
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 提取上三角矩陣的值（不包括對角線）
        similarities = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                similarities.append(self.matrix[i][j])
        
        similarities = np.array(similarities)
        
        return {
            'population_size': self.n,
            'total_pairs': len(similarities),
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'diversity_score': 1.0 - float(np.mean(similarities))
        }
    
    def print_matrix(self, precision=4):
        """
        打印相似度矩陣
        
        Args:
            precision: 小數點精度
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        # 打印表頭
        print(f"{'':>6}", end="")
        for j in range(self.n):
            print(f"{j:>{precision+3}}", end="")
        print()
        
        # 打印每一行
        for i in range(self.n):
            print(f"{i:>6}", end="")
            for j in range(self.n):
                print(f"{self.matrix[i][j]:>{precision+3}.{precision}f}", end="")
            print()
    
    def save(self, filepath: str):
        """
        儲存相似度矩陣到文件
        
        Args:
            filepath: 文件路徑
        """
        if self.matrix is None:
            raise ValueError("請先調用 compute() 計算相似度矩陣")
        
        np.save(filepath, self.matrix)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimilarityMatrix':
        """
        從文件載入相似度矩陣
        
        Args:
            filepath: 文件路徑
            
        Returns:
            SimilarityMatrix: 載入的相似度矩陣對象
        """
        matrix = np.load(filepath)
        
        # 創建一個空的 SimilarityMatrix 對象
        obj = cls([])
        obj.matrix = matrix
        obj.n = matrix.shape[0]
        
        return obj
