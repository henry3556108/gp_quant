"""
動態 Niche 數量 (k) 選擇器

支持多種 k 值選擇策略：
1. 固定 k 值（向下兼容）
2. 動態選擇（基於 Silhouette Score）
3. 基於 ln(n) 的自適應上限
4. 階段性校準（前幾代動態，後續固定）
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from sklearn.metrics import silhouette_score
from .clustering import NichingClusterer


class DynamicKSelector:
    """動態 k 值選擇器"""
    
    def __init__(self,
                 mode: str = 'fixed',
                 fixed_k: Optional[int] = None,
                 k_min: int = 2,
                 k_max: Union[int, str] = 8,
                 calibration_generations: int = 3,
                 algorithm: str = 'kmeans',
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        初始化動態 k 值選擇器
        
        Args:
            mode: 選擇模式
                - 'fixed': 固定 k 值（向下兼容）
                - 'dynamic': 每次都動態選擇最佳 k
                - 'auto': 使用 ln(n) 作為 k_max，動態選擇
                - 'calibration': 前幾代動態選擇，之後使用校準期的最佳 k
            fixed_k: 固定 k 值（mode='fixed' 時使用）
            k_min: 最小 k 值（mode='dynamic'/'auto'/'calibration' 時使用）
            k_max: 最大 k 值，可以是整數或 'auto'（使用 ln(n)）
            calibration_generations: 校準期代數（mode='calibration' 時使用）
            algorithm: 聚類演算法 ('kmeans' 或 'hierarchical')
            random_state: 隨機種子
            verbose: 是否顯示詳細資訊
        """
        self.mode = mode
        self.fixed_k = fixed_k
        self.k_min = k_min
        self.k_max = k_max
        self.calibration_generations = calibration_generations
        self.algorithm = algorithm
        self.random_state = random_state
        self.verbose = verbose
        
        # 校準期記錄
        self.calibration_history = []
        self.calibrated_k = None
        self.generation_count = 0
        
        # 驗證參數
        self._validate_params()
    
    def _validate_params(self):
        """驗證參數"""
        valid_modes = ['fixed', 'dynamic', 'auto', 'calibration']
        if self.mode not in valid_modes:
            raise ValueError(f"mode 必須是 {valid_modes} 之一，得到: {self.mode}")
        
        if self.mode == 'fixed' and self.fixed_k is None:
            raise ValueError("mode='fixed' 時必須指定 fixed_k")
        
        if self.mode == 'fixed' and self.fixed_k < 2:
            raise ValueError(f"fixed_k 必須 >= 2，得到: {self.fixed_k}")
        
        if self.k_min < 2:
            raise ValueError(f"k_min 必須 >= 2，得到: {self.k_min}")
        
        if isinstance(self.k_max, int) and self.k_max < self.k_min:
            raise ValueError(f"k_max ({self.k_max}) 必須 >= k_min ({self.k_min})")
    
    def _compute_k_max(self, population_size: int) -> int:
        """
        計算 k 的上限
        
        Args:
            population_size: 族群大小
            
        Returns:
            k 的上限值
        """
        if self.k_max == 'auto':
            # 使用 ln(n) 作為上限
            k_max = int(np.log(population_size))
            # 確保至少為 k_min
            k_max = max(k_max, self.k_min)
            return k_max
        else:
            return self.k_max
    
    def _test_k_values(self, 
                       similarity_matrix: np.ndarray,
                       k_range: List[int]) -> Tuple[int, Dict[int, float]]:
        """
        測試不同 k 值的聚類效果
        
        Args:
            similarity_matrix: 相似度矩陣
            k_range: 要測試的 k 值列表
            
        Returns:
            (best_k, scores_dict)
        """
        scores = {}
        
        for k in k_range:
            try:
                clusterer = NichingClusterer(
                    n_clusters=k,
                    algorithm=self.algorithm,
                    random_state=self.random_state
                )
                clusterer.fit(similarity_matrix)
                scores[k] = clusterer.silhouette_score_
            except Exception as e:
                if self.verbose:
                    print(f"    ⚠️  k={k} 聚類失敗: {e}")
                scores[k] = -1.0
        
        # 選擇最佳 k
        best_k = max(scores.keys(), key=lambda k: scores[k])
        
        return best_k, scores
    
    def select_k(self, 
                 similarity_matrix: np.ndarray,
                 population_size: int,
                 generation: Optional[int] = None) -> Dict:
        """
        選擇最佳 k 值
        
        Args:
            similarity_matrix: 相似度矩陣
            population_size: 族群大小
            generation: 當前代數（用於 calibration 模式）
            
        Returns:
            包含選擇結果的字典：
            {
                'k': 選擇的 k 值,
                'mode': 使用的模式,
                'scores': k 值測試結果（如果有動態選擇）,
                'k_range': 測試的 k 值範圍（如果有動態選擇）
            }
        """
        self.generation_count += 1
        
        # ====================================================================
        # Mode 1: Fixed K（固定 k 值）
        # ====================================================================
        if self.mode == 'fixed':
            if self.verbose:
                print(f"  🎯 使用固定 k 值: {self.fixed_k}")
            
            return {
                'k': self.fixed_k,
                'mode': 'fixed',
                'scores': None,
                'k_range': None
            }
        
        # ====================================================================
        # Mode 2: Dynamic / Auto（動態選擇）
        # ====================================================================
        elif self.mode in ['dynamic', 'auto']:
            # 計算 k 範圍
            k_max = self._compute_k_max(population_size)
            k_range = list(range(self.k_min, k_max + 1))
            
            # 確保 k_range 不超過 population_size
            k_range = [k for k in k_range if k < population_size]
            
            if self.verbose:
                print(f"  🔍 動態選擇 k 值...")
                print(f"     測試範圍: k ∈ [{self.k_min}, {k_max}]")
                if self.mode == 'auto':
                    print(f"     k_max = ln({population_size}) = {k_max}")
            
            # 測試所有 k 值
            best_k, scores = self._test_k_values(similarity_matrix, k_range)
            
            if self.verbose:
                print(f"     最佳 k: {best_k} (Silhouette: {scores[best_k]:.4f})")
            
            return {
                'k': best_k,
                'mode': self.mode,
                'scores': scores,
                'k_range': k_range
            }
        
        # ====================================================================
        # Mode 3: Calibration（階段性校準）
        # ====================================================================
        elif self.mode == 'calibration':
            # 判斷是否還在校準期
            if generation is None:
                generation = self.generation_count
            
            in_calibration = generation <= self.calibration_generations
            
            if in_calibration:
                # 校準期：動態選擇
                k_max = self._compute_k_max(population_size)
                k_range = list(range(self.k_min, k_max + 1))
                k_range = [k for k in k_range if k < population_size]
                
                if self.verbose:
                    print(f"  🔬 校準期 ({generation}/{self.calibration_generations})")
                    print(f"     測試範圍: k ∈ [{self.k_min}, {k_max}]")
                
                best_k, scores = self._test_k_values(similarity_matrix, k_range)
                
                # 記錄校準結果
                self.calibration_history.append({
                    'generation': generation,
                    'best_k': best_k,
                    'score': scores[best_k],
                    'all_scores': scores
                })
                
                if self.verbose:
                    print(f"     最佳 k: {best_k} (Silhouette: {scores[best_k]:.4f})")
                
                # 如果是最後一代校準期，計算校準後的 k
                if generation == self.calibration_generations:
                    # 使用校準期最常出現的 k 值
                    k_values = [h['best_k'] for h in self.calibration_history]
                    self.calibrated_k = max(set(k_values), key=k_values.count)
                    
                    if self.verbose:
                        print(f"\n  ✅ 校準完成！")
                        print(f"     校準期 k 值: {k_values}")
                        print(f"     校準後固定 k: {self.calibrated_k}")
                
                return {
                    'k': best_k,
                    'mode': 'calibration_active',
                    'scores': scores,
                    'k_range': k_range,
                    'calibration_progress': f"{generation}/{self.calibration_generations}"
                }
            
            else:
                # 校準期後：使用校準後的 k
                if self.calibrated_k is None:
                    raise RuntimeError("校準期已結束但未設置 calibrated_k")
                
                if self.verbose:
                    print(f"  🎯 使用校準後的 k 值: {self.calibrated_k}")
                
                return {
                    'k': self.calibrated_k,
                    'mode': 'calibration_fixed',
                    'scores': None,
                    'k_range': None,
                    'calibration_history': self.calibration_history
                }
    
    def get_statistics(self) -> Dict:
        """
        獲取選擇器統計資訊
        
        Returns:
            統計資訊字典
        """
        stats = {
            'mode': self.mode,
            'generation_count': self.generation_count,
        }
        
        if self.mode == 'fixed':
            stats['fixed_k'] = self.fixed_k
        
        elif self.mode in ['dynamic', 'auto']:
            stats['k_min'] = self.k_min
            stats['k_max'] = self.k_max
        
        elif self.mode == 'calibration':
            stats['calibration_generations'] = self.calibration_generations
            stats['calibrated_k'] = self.calibrated_k
            stats['calibration_history'] = self.calibration_history
        
        return stats
    
    def reset(self):
        """重置選擇器狀態"""
        self.calibration_history = []
        self.calibrated_k = None
        self.generation_count = 0


def create_k_selector(config: Dict) -> DynamicKSelector:
    """
    從配置字典創建 k 選擇器（工廠函數）
    
    Args:
        config: 配置字典，支持以下格式：
        
        # 固定 k 值（向下兼容）
        {
            'niching_n_clusters': 3
        }
        
        # 動態選擇
        {
            'niching_k_selection': 'dynamic',
            'niching_k_min': 2,
            'niching_k_max': 8
        }
        
        # 自動上限（ln(n)）
        {
            'niching_k_selection': 'auto',
            'niching_k_min': 2,
            'niching_k_max': 'auto'
        }
        
        # 階段性校準
        {
            'niching_k_selection': 'calibration',
            'niching_k_min': 2,
            'niching_k_max': 'auto',
            'niching_k_calibration_gens': 3
        }
    
    Returns:
        DynamicKSelector 實例
    """
    # 檢查是否使用新的動態選擇配置
    if 'niching_k_selection' in config:
        mode = config['niching_k_selection']
        
        return DynamicKSelector(
            mode=mode,
            fixed_k=config.get('niching_n_clusters'),  # 用於 fixed 模式
            k_min=config.get('niching_k_min', 2),
            k_max=config.get('niching_k_max', 8),
            calibration_generations=config.get('niching_k_calibration_gens', 3),
            algorithm=config.get('niching_algorithm', 'kmeans'),
            random_state=config.get('random_state', None),
            verbose=config.get('verbose', True)
        )
    
    else:
        # 向下兼容：使用固定 k 值
        fixed_k = config.get('niching_n_clusters', 3)
        
        return DynamicKSelector(
            mode='fixed',
            fixed_k=fixed_k,
            algorithm=config.get('niching_algorithm', 'kmeans'),
            random_state=config.get('random_state', None),
            verbose=config.get('verbose', True)
        )
