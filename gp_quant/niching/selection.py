"""
Cross-Niche Parent Selection

實作跨群（Cross-Niche）親代選擇機制，用於 Niching 策略。
透過兩階段選擇過程，鼓勵不同 niche 之間的基因交流，同時保持群內競爭。

兩階段選擇機制：
1. Stage 1: Within-Niche Tournament Selection
   - 在每個 niche 內進行 tournament selection
   - 選出每個 niche 的優秀個體
   
2. Stage 2: Cross-Niche Pairing
   - 將選出的個體進行跨群配對
   - 可配置跨群比例（預設 80%）

參考文獻：
    Mahfoud, S. W. (1995). Niching methods for genetic algorithms.
"""

from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from collections import defaultdict


class CrossNicheSelector:
    """
    跨群親代選擇器
    
    實作兩階段選擇機制：
    1. Within-Niche Tournament Selection
    2. Cross-Niche Pairing
    
    Attributes:
        cross_niche_ratio: 跨群交配比例 [0, 1]
        tournament_size: Tournament selection 的大小
        statistics_: 選擇統計資訊
    """
    
    def __init__(self,
                 cross_niche_ratio: float = 0.8,
                 tournament_size: int = 3,
                 random_state: Optional[int] = None):
        """
        初始化跨群選擇器
        
        Args:
            cross_niche_ratio: 跨群交配比例，範圍 [0, 1]
                              0.8 表示 80% 跨群，20% 群內
            tournament_size: Tournament selection 的大小
            random_state: 隨機種子
        """
        if not 0 <= cross_niche_ratio <= 1:
            raise ValueError(f"cross_niche_ratio 必須在 [0, 1] 範圍內，得到: {cross_niche_ratio}")
        
        if tournament_size < 2:
            raise ValueError(f"tournament_size 必須 >= 2，得到: {tournament_size}")
        
        self.cross_niche_ratio = cross_niche_ratio
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # 統計資訊
        self.statistics_ = {
            'total_pairs': 0,
            'cross_niche_pairs': 0,
            'within_niche_pairs': 0,
            'niche_pair_counts': defaultdict(int)  # (niche_i, niche_j) -> count
        }
        
        if random_state is not None:
            random.seed(random_state)
    
    def select(self,
               population: List,
               niche_labels: np.ndarray,
               k: int) -> List:
        """
        選擇 k 個親代個體
        
        Args:
            population: 族群列表（DEAP Individual）
            niche_labels: 每個個體的 niche 標籤 (numpy array)
            k: 要選擇的個體數量（必須是偶數，用於配對）
            
        Returns:
            List: 選出的親代列表
        """
        if k % 2 != 0:
            raise ValueError(f"k 必須是偶數（用於配對），得到: {k}")
        
        if len(population) != len(niche_labels):
            raise ValueError(
                f"population 和 niche_labels 長度不一致: "
                f"{len(population)} vs {len(niche_labels)}"
            )
        
        # 重置統計資訊
        self.statistics_ = {
            'total_pairs': 0,
            'cross_niche_pairs': 0,
            'within_niche_pairs': 0,
            'niche_pair_counts': defaultdict(int)
        }
        
        # Stage 1: Within-Niche Tournament Selection
        niche_pools = self._stage1_within_niche_selection(population, niche_labels, k)
        
        # Stage 2: Cross-Niche Pairing
        selected = self._stage2_cross_niche_pairing(niche_pools, k)
        
        return selected
    
    def _stage1_within_niche_selection(self,
                                       population: List,
                                       niche_labels: np.ndarray,
                                       k: int) -> Dict[int, List]:
        """
        Stage 1: 在每個 niche 內進行 tournament selection
        
        Args:
            population: 族群列表
            niche_labels: Niche 標籤
            k: 總共要選擇的個體數量
            
        Returns:
            Dict[int, List]: {niche_id: [selected_individuals]}
        """
        # 將 population 按 niche 分組
        niches = defaultdict(list)
        for ind, label in zip(population, niche_labels):
            niches[label].append(ind)
        
        # 計算每個 niche 應該選出多少個體
        # 按 niche 大小比例分配
        niche_sizes = {niche_id: len(inds) for niche_id, inds in niches.items()}
        total_size = sum(niche_sizes.values())
        
        niche_quotas = {}
        allocated = 0
        for niche_id, size in niche_sizes.items():
            quota = int(k * size / total_size)
            niche_quotas[niche_id] = max(2, quota)  # 至少選 2 個（用於配對）
            allocated += niche_quotas[niche_id]
        
        # 調整配額以確保總數為 k
        while allocated < k:
            # 找最大的 niche 增加配額
            max_niche = max(niche_sizes.keys(), key=lambda x: niche_sizes[x])
            niche_quotas[max_niche] += 1
            allocated += 1
        
        while allocated > k:
            # 找配額最多的 niche 減少配額（但保持至少 2 個）
            max_quota_niche = max(niche_quotas.keys(), key=lambda x: niche_quotas[x])
            if niche_quotas[max_quota_niche] > 2:
                niche_quotas[max_quota_niche] -= 1
                allocated -= 1
            else:
                break
        
        # 在每個 niche 內進行 tournament selection
        niche_pools = {}
        for niche_id, individuals in niches.items():
            quota = niche_quotas[niche_id]
            selected = self._tournament_selection(individuals, quota)
            niche_pools[niche_id] = selected
        
        return niche_pools
    
    def _tournament_selection(self, individuals: List, k: int) -> List:
        """
        Tournament selection
        
        Args:
            individuals: 候選個體列表
            k: 要選擇的個體數量
            
        Returns:
            List: 選出的個體
        """
        selected = []
        for _ in range(k):
            # 隨機選擇 tournament_size 個個體
            tournament_size = min(self.tournament_size, len(individuals))
            tournament = random.sample(individuals, tournament_size)
            
            # 選擇 fitness 最高的
            winner = max(tournament, key=lambda ind: ind.fitness.values[0])
            selected.append(winner)
        
        return selected
    
    def _stage2_cross_niche_pairing(self,
                                    niche_pools: Dict[int, List],
                                    k: int) -> List:
        """
        Stage 2: 跨群配對
        
        Args:
            niche_pools: 每個 niche 的候選池
            k: 總共要選擇的個體數量
            
        Returns:
            List: 配對後的個體列表
        """
        num_pairs = k // 2
        num_cross_pairs = int(num_pairs * self.cross_niche_ratio)
        num_within_pairs = num_pairs - num_cross_pairs
        
        selected = []
        
        # 獲取所有 niche IDs
        niche_ids = list(niche_pools.keys())
        
        if len(niche_ids) < 2:
            # 只有一個 niche，全部都是群內配對
            num_within_pairs = num_pairs
            num_cross_pairs = 0
        
        # 1. 跨群配對
        for _ in range(num_cross_pairs):
            # 隨機選擇兩個不同的 niches
            niche_i, niche_j = random.sample(niche_ids, 2)
            
            # 從每個 niche 選一個個體
            ind_i = random.choice(niche_pools[niche_i])
            ind_j = random.choice(niche_pools[niche_j])
            
            selected.extend([ind_i, ind_j])
            
            # 記錄統計
            self.statistics_['cross_niche_pairs'] += 1
            pair_key = tuple(sorted([niche_i, niche_j]))
            self.statistics_['niche_pair_counts'][pair_key] += 1
        
        # 2. 群內配對
        for _ in range(num_within_pairs):
            # 隨機選擇一個 niche
            niche_id = random.choice(niche_ids)
            
            # 從同一個 niche 選兩個個體
            if len(niche_pools[niche_id]) >= 2:
                ind_i, ind_j = random.sample(niche_pools[niche_id], 2)
            else:
                # 如果 niche 只有一個個體，重複選擇
                ind_i = ind_j = random.choice(niche_pools[niche_id])
            
            selected.extend([ind_i, ind_j])
            
            # 記錄統計
            self.statistics_['within_niche_pairs'] += 1
            pair_key = (niche_id, niche_id)
            self.statistics_['niche_pair_counts'][pair_key] += 1
        
        # 更新總配對數
        self.statistics_['total_pairs'] = num_pairs
        
        return selected
    
    def get_statistics(self) -> Dict:
        """
        獲取選擇統計資訊
        
        Returns:
            Dict: 統計資訊
        """
        stats = dict(self.statistics_)
        
        # 計算比例
        if stats['total_pairs'] > 0:
            stats['cross_niche_ratio_actual'] = (
                stats['cross_niche_pairs'] / stats['total_pairs']
            )
            stats['within_niche_ratio_actual'] = (
                stats['within_niche_pairs'] / stats['total_pairs']
            )
        else:
            stats['cross_niche_ratio_actual'] = 0.0
            stats['within_niche_ratio_actual'] = 0.0
        
        # 轉換 defaultdict 為普通 dict
        stats['niche_pair_counts'] = dict(stats['niche_pair_counts'])
        
        return stats
    
    def print_statistics(self):
        """打印選擇統計資訊"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("跨群選擇統計資訊")
        print("=" * 60)
        print(f"總配對數: {stats['total_pairs']}")
        print(f"跨群配對數: {stats['cross_niche_pairs']} "
              f"({stats['cross_niche_ratio_actual']:.1%})")
        print(f"群內配對數: {stats['within_niche_pairs']} "
              f"({stats['within_niche_ratio_actual']:.1%})")
        print(f"\n配置的跨群比例: {self.cross_niche_ratio:.1%}")
        print(f"實際的跨群比例: {stats['cross_niche_ratio_actual']:.1%}")
        
        if stats['niche_pair_counts']:
            print("\n各 Niche 配對統計:")
            for (niche_i, niche_j), count in sorted(stats['niche_pair_counts'].items()):
                if niche_i == niche_j:
                    print(f"  Niche {niche_i} 內部配對: {count} 次")
                else:
                    print(f"  Niche {niche_i} ↔ Niche {niche_j}: {count} 次")
        
        print("=" * 60)
