"""
Generation Data Loader

提供向後相容的 generation pkl 載入功能。
支援新舊兩種格式：
- 舊格式：只有 population 和基本資訊
- 新格式：包含 cluster_labels 和 niching_info
"""
import dill
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def load_generation(file_path: Union[str, Path]) -> Dict:
    """
    載入 generation pkl 檔案，支援向後相容
    
    Args:
        file_path: generation pkl 檔案路徑
        
    Returns:
        包含以下 key 的字典：
        - generation: int - generation 編號
        - population: List - 族群個體列表
        - hall_of_fame: List - 名人堂
        - statistics: Dict - 統計資訊
        - timestamp: str - 時間戳記
        - cluster_labels: Optional[List[int]] - 每個個體的 niche ID（新格式）
        - niching_info: Optional[Dict] - niching 資訊（新格式）
            - n_clusters: int - niche 數量
            - algorithm: str - 聚類演算法
            - silhouette_score: float - silhouette 分數
        - early_stopped: Optional[bool] - 是否早停（僅最終 generation）
        - early_stopping_status: Optional[Dict] - 早停狀態（僅最終 generation）
        
    Examples:
        >>> gen_data = load_generation('generations/generation_001.pkl')
        >>> population = gen_data['population']
        >>> cluster_labels = gen_data.get('cluster_labels')  # 可能為 None（舊格式）
        >>> if cluster_labels is not None:
        ...     print(f"個體 0 屬於 niche {cluster_labels[0]}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Generation file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = dill.load(f)
    
    # 確保返回的資料包含所有必要的 key
    # 如果是舊格式，cluster_labels 和 niching_info 會是 None
    result = {
        'generation': data.get('generation'),
        'population': data.get('population'),
        'hall_of_fame': data.get('hall_of_fame'),
        'statistics': data.get('statistics'),
        'timestamp': data.get('timestamp'),
        'cluster_labels': data.get('cluster_labels'),  # 新格式才有
        'niching_info': data.get('niching_info'),      # 新格式才有
        'early_stopped': data.get('early_stopped'),    # 僅最終 generation
        'early_stopping_status': data.get('early_stopping_status')  # 僅最終 generation
    }
    
    return result


def has_niching_info(gen_data: Dict) -> bool:
    """
    檢查 generation 資料是否包含 niching 資訊
    
    Args:
        gen_data: load_generation() 返回的資料
        
    Returns:
        True 如果包含 niching 資訊，False 否則
    """
    return gen_data.get('cluster_labels') is not None and gen_data.get('niching_info') is not None


def get_niche_individuals(gen_data: Dict, niche_id: int) -> List:
    """
    獲取屬於特定 niche 的所有個體
    
    Args:
        gen_data: load_generation() 返回的資料
        niche_id: niche ID
        
    Returns:
        屬於該 niche 的個體列表
        
    Raises:
        ValueError: 如果沒有 niching 資訊
    """
    if not has_niching_info(gen_data):
        raise ValueError("Generation data does not contain niching information")
    
    population = gen_data['population']
    cluster_labels = gen_data['cluster_labels']
    
    return [ind for ind, label in zip(population, cluster_labels) if label == niche_id]


def get_niche_statistics(gen_data: Dict) -> Optional[Dict]:
    """
    獲取每個 niche 的統計資訊
    
    Args:
        gen_data: load_generation() 返回的資料
        
    Returns:
        包含每個 niche 統計資訊的字典，如果沒有 niching 資訊則返回 None
        格式：{
            niche_id: {
                'size': int,
                'individuals': List,
                'fitness_mean': float,
                'fitness_std': float,
                'fitness_min': float,
                'fitness_max': float
            }
        }
    """
    if not has_niching_info(gen_data):
        return None
    
    population = gen_data['population']
    cluster_labels = gen_data['cluster_labels']
    n_clusters = gen_data['niching_info']['n_clusters']
    
    stats = {}
    for niche_id in range(n_clusters):
        niche_inds = [ind for ind, label in zip(population, cluster_labels) if label == niche_id]
        
        if niche_inds:
            fitnesses = [ind.fitness.values[0] for ind in niche_inds]
            stats[niche_id] = {
                'size': len(niche_inds),
                'individuals': niche_inds,
                'fitness_mean': np.mean(fitnesses),
                'fitness_std': np.std(fitnesses),
                'fitness_min': np.min(fitnesses),
                'fitness_max': np.max(fitnesses)
            }
    
    return stats


def load_multiple_generations(directory: Union[str, Path], 
                              start_gen: int = 1, 
                              end_gen: Optional[int] = None) -> List[Dict]:
    """
    載入多個 generation 的資料
    
    Args:
        directory: generations 目錄路徑
        start_gen: 起始 generation（包含）
        end_gen: 結束 generation（包含），None 表示載入所有
        
    Returns:
        generation 資料列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # 找出所有 generation 檔案
    gen_files = sorted(directory.glob('generation_*.pkl'))
    
    results = []
    for gen_file in gen_files:
        # 解析 generation 編號
        try:
            gen_num = int(gen_file.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        
        # 檢查是否在範圍內
        if gen_num < start_gen:
            continue
        if end_gen is not None and gen_num > end_gen:
            break
        
        try:
            gen_data = load_generation(gen_file)
            results.append(gen_data)
        except Exception as e:
            print(f"Warning: Failed to load {gen_file.name}: {e}")
    
    return results
