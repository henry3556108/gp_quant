"""
Niching Clustering Module

基於相似度矩陣進行聚類，將族群分成多個 niches。
支援 K-means 和 Hierarchical clustering。
"""

from typing import List, Union, Optional
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples


class NichingClusterer:
    """
    Niching 聚類器
    
    基於相似度矩陣將族群分成多個 niches (群)。
    支援多種聚類演算法。
    
    Attributes:
        n_clusters: 群數 (niches 數量)
        algorithm: 聚類演算法 ('kmeans' 或 'hierarchical')
        labels_: 每個個體的群標籤
        cluster_centers_: 群中心 (僅 kmeans)
        silhouette_score_: Silhouette 分數 (聚類品質指標)
    """
    
    def __init__(self, 
                 n_clusters: int = 5,
                 algorithm: str = 'kmeans',
                 random_state: int = 42):
        """
        初始化聚類器
        
        Args:
            n_clusters: 群數
            algorithm: 聚類演算法 ('kmeans' 或 'hierarchical')
            random_state: 隨機種子
        """
        self.n_clusters = n_clusters
        self.algorithm = algorithm.lower()
        self.random_state = random_state
        
        # 聚類結果
        self.labels_ = None
        self.cluster_centers_ = None
        self.silhouette_score_ = None
        self.silhouette_samples_ = None  # 每個樣本的 silhouette score
        self.per_cluster_silhouette_ = None  # 每個 cluster 的平均 silhouette score
        
        # 驗證參數
        if self.algorithm not in ['kmeans', 'hierarchical']:
            raise ValueError(f"不支援的演算法: {algorithm}. 請使用 'kmeans' 或 'hierarchical'")
    
    def fit(self, similarity_matrix: np.ndarray) -> 'NichingClusterer':
        """
        對相似度矩陣進行聚類
        
        Args:
            similarity_matrix: 相似度矩陣 (n x n)
            
        Returns:
            self: 聚類器本身
        """
        # 將相似度矩陣轉換為距離矩陣
        # distance = 1 - similarity
        distance_matrix = 1.0 - similarity_matrix
        
        # 確保對角線為 0
        np.fill_diagonal(distance_matrix, 0.0)
        
        # 執行聚類
        if self.algorithm == 'kmeans':
            self._fit_kmeans(distance_matrix)
        elif self.algorithm == 'hierarchical':
            self._fit_hierarchical(distance_matrix)
        
        # 計算 Silhouette 分數
        if len(np.unique(self.labels_)) > 1:
            # 整體 silhouette score
            self.silhouette_score_ = silhouette_score(
                distance_matrix, 
                self.labels_, 
                metric='precomputed'
            )
            
            # 每個樣本的 silhouette score
            self.silhouette_samples_ = silhouette_samples(
                distance_matrix,
                self.labels_,
                metric='precomputed'
            )
            
            # 計算每個 cluster 的平均 silhouette score
            self.per_cluster_silhouette_ = {}
            for cluster_id in np.unique(self.labels_):
                cluster_mask = self.labels_ == cluster_id
                cluster_scores = self.silhouette_samples_[cluster_mask]
                self.per_cluster_silhouette_[int(cluster_id)] = {
                    'mean': float(np.mean(cluster_scores)),
                    'std': float(np.std(cluster_scores)),
                    'min': float(np.min(cluster_scores)),
                    'max': float(np.max(cluster_scores)),
                    'size': int(np.sum(cluster_mask))
                }
        else:
            self.silhouette_score_ = 0.0
            self.silhouette_samples_ = np.zeros(len(self.labels_))
            self.per_cluster_silhouette_ = {}
        
        return self
    
    def _fit_kmeans(self, distance_matrix: np.ndarray):
        """使用 K-means 聚類"""
        # K-means 需要特徵向量，使用距離矩陣作為特徵
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.labels_ = kmeans.fit_predict(distance_matrix)
        self.cluster_centers_ = kmeans.cluster_centers_
    
    def _fit_hierarchical(self, distance_matrix: np.ndarray):
        """使用 Hierarchical clustering"""
        hierarchical = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric='precomputed',
            linkage='average'
        )
        self.labels_ = hierarchical.fit_predict(distance_matrix)
        self.cluster_centers_ = None  # Hierarchical 沒有中心點
    
    def fit_predict(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        聚類並返回標籤
        
        Args:
            similarity_matrix: 相似度矩陣 (n x n)
            
        Returns:
            np.ndarray: 每個個體的群標籤
        """
        self.fit(similarity_matrix)
        return self.labels_
    
    def get_niche_sizes(self) -> dict:
        """
        獲取每個 niche 的大小
        
        Returns:
            dict: {niche_id: size}
        """
        if self.labels_ is None:
            raise ValueError("請先調用 fit() 或 fit_predict()")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique, counts))
    
    def get_niche_members(self, niche_id: int) -> List[int]:
        """
        獲取指定 niche 的成員索引
        
        Args:
            niche_id: Niche ID
            
        Returns:
            List[int]: 成員索引列表
        """
        if self.labels_ is None:
            raise ValueError("請先調用 fit() 或 fit_predict()")
        
        return np.where(self.labels_ == niche_id)[0].tolist()
    
    def get_statistics(self) -> dict:
        """
        獲取聚類統計資訊
        
        Returns:
            dict: 統計資訊
        """
        if self.labels_ is None:
            raise ValueError("請先調用 fit() 或 fit_predict()")
        
        niche_sizes = self.get_niche_sizes()
        
        return {
            'n_clusters': self.n_clusters,
            'algorithm': self.algorithm,
            'niche_sizes': niche_sizes,
            'min_niche_size': min(niche_sizes.values()),
            'max_niche_size': max(niche_sizes.values()),
            'avg_niche_size': np.mean(list(niche_sizes.values())),
            'std_niche_size': np.std(list(niche_sizes.values())),
            'silhouette_score': self.silhouette_score_
        }
    
    def print_summary(self):
        """打印聚類摘要"""
        if self.labels_ is None:
            raise ValueError("請先調用 fit() 或 fit_predict()")
        
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"Niching 聚類摘要")
        print(f"{'='*60}")
        print(f"\n演算法: {stats['algorithm']}")
        print(f"群數: {stats['n_clusters']}")
        print(f"Silhouette 分數: {stats['silhouette_score']:.4f}")
        print(f"\n各 Niche 大小:")
        for niche_id, size in sorted(stats['niche_sizes'].items()):
            percentage = (size / sum(stats['niche_sizes'].values())) * 100
            bar = "█" * int(percentage / 2)
            print(f"  Niche {niche_id}: {size:>4} ({percentage:>5.1f}%) {bar}")
        
        print(f"\n統計:")
        print(f"  最小 Niche: {stats['min_niche_size']}")
        print(f"  最大 Niche: {stats['max_niche_size']}")
        print(f"  平均大小: {stats['avg_niche_size']:.1f}")
        print(f"  標準差: {stats['std_niche_size']:.1f}")
        print(f"{'='*60}\n")


def auto_select_k(similarity_matrix: np.ndarray, 
                  k_range: range = range(2, 11),
                  algorithm: str = 'kmeans') -> int:
    """
    自動選擇最佳的 k 值（群數）
    
    使用 Silhouette 分數來評估不同 k 值的聚類品質。
    
    Args:
        similarity_matrix: 相似度矩陣
        k_range: k 值範圍
        algorithm: 聚類演算法
        
    Returns:
        int: 最佳 k 值
    """
    best_k = k_range[0]
    best_score = -1.0
    
    for k in k_range:
        clusterer = NichingClusterer(n_clusters=k, algorithm=algorithm)
        clusterer.fit(similarity_matrix)
        
        if clusterer.silhouette_score_ > best_score:
            best_score = clusterer.silhouette_score_
            best_k = k
    
    return best_k
