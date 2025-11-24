"""
演化個體類

實現帶 ID 和譜系追蹤的個體類，擴展 DEAP 的 Individual。
"""

import uuid
from typing import List, Optional, Any, Dict
from deap import gp, creator

class EvolutionIndividual(gp.PrimitiveTree):
    """
    演化個體類
    
    擴展 DEAP 的 PrimitiveTree，添加 ID、譜系追蹤和其他元數據。
    """
    
    def __init__(self, content=None):
        """
        初始化演化個體
        
        Args:
            content: GP 樹內容（可選）
        """
        if content is not None:
            super().__init__(content)
        else:
            super().__init__()
        
        # 個體唯一標識
        self.id: str = str(uuid.uuid4())
        
        # 譜系信息
        self.parents: List[str] = []  # 父母個體的 ID 列表
        self.operation: str = 'unknown'  # 產生此個體的操作類型
        self.generation: int = 0  # 所屬世代
        
        # 適應度（使用 DEAP 的 fitness 系統）
        if not hasattr(self, 'fitness'):
            self.fitness = creator.FitnessMax()
        
        # 不需要 valid 屬性，直接使用 values 來判斷是否有效
        
        # 統計信息
        self.evaluation_count: int = 0  # 被評估的次數
        self.creation_time: Optional[float] = None  # 創建時間戳
        
        # 額外元數據
        self.metadata: Dict[str, Any] = {}
    
    @property
    def fitness_value(self) -> Optional[float]:
        """獲取適應度值（便利屬性）"""
        if hasattr(self.fitness, 'values') and self.fitness.values:
            return self.fitness.values[0]
        return None
    
    @fitness_value.setter
    def fitness_value(self, value: float):
        """設置適應度值"""
        self.fitness.values = (value,)
    
    def set_parents(self, parent_ids: List[str], operation: str):
        """
        設置父母信息
        
        Args:
            parent_ids: 父母個體的 ID 列表
            operation: 產生此個體的操作類型
        """
        self.parents = parent_ids.copy()
        self.operation = operation
    
    def add_metadata(self, key: str, value: Any):
        """添加元數據"""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default=None):
        """獲取元數據"""
        return self.metadata.get(key, default)
    
    def get_genealogy_info(self) -> Dict[str, Any]:
        """獲取譜系信息"""
        return {
            'id': self.id,
            'parents': self.parents,
            'operation': self.operation,
            'generation': self.generation,
            'fitness': self.fitness_value,
            'evaluation_count': self.evaluation_count,
            'tree_size': len(self),
            'tree_depth': self.height,
            'creation_time': self.creation_time
        }
    
    def clone(self):
        """創建個體的深拷貝"""
        import copy
        cloned = copy.deepcopy(self)
        
        # 為克隆個體分配新的 ID
        cloned.id = str(uuid.uuid4())
        
        # 保留其他屬性
        cloned.parents = self.parents.copy()
        cloned.operation = self.operation
        cloned.generation = self.generation
        cloned.evaluation_count = 0  # 重置評估次數
        cloned.metadata = self.metadata.copy()
        
        return cloned
    
    def get_expression(self) -> str:
        """獲取 GP 表達式字符串"""
        return gp.PrimitiveTree.__str__(self)
    
    def __str__(self) -> str:
        """字符串表示 - 返回 GP 表達式以支持編譯"""
        # 返回 GP 表達式而不是元數據，這樣才能被 gp.compile 正確處理
        return gp.PrimitiveTree.__str__(self)
    
    def __repr__(self) -> str:
        """詳細字符串表示 - 顯示元數據"""
        fitness_str = f"{self.fitness_value:.4f}" if self.fitness_value is not None else "N/A"
        return f"Individual({self.id[:8]}..., gen={self.generation}, fitness={fitness_str}, op={self.operation})"
    
    def info(self) -> str:
        """獲取個體信息摘要"""
        return (f"EvolutionIndividual(id='{self.id}', generation={self.generation}, "
                f"fitness={self.fitness_value}, operation='{self.operation}', "
                f"parents={len(self.parents)}, size={len(self)}, depth={self.height})")

# 為了兼容性，創建一個工廠函數
def create_individual(content=None) -> EvolutionIndividual:
    """
    創建演化個體的工廠函數
    
    Args:
        content: GP 樹內容（可選）
        
    Returns:
        新的演化個體
    """
    individual = EvolutionIndividual(content)
    
    # 設置創建時間
    import time
    individual.creation_time = time.time()
    
    return individual
