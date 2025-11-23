"""
適應度評估器基類

定義演化計算中適應度評估的統一接口。
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class FitnessEvaluator(ABC):
    """
    適應度評估器基類
    
    所有適應度評估器都必須繼承此類並實現評估方法。
    """
    
    def __init__(self):
        self.engine = None
        self.name = "base_evaluator"
    
    def set_engine(self, engine):
        """設置演化引擎引用"""
        self.engine = engine
    
    @abstractmethod
    def evaluate_individual(self, individual, data: Dict[str, Any]) -> float:
        """
        評估單個個體的適應度
        
        Args:
            individual: 要評估的個體
            data: 評估所需的數據
            
        Returns:
            適應度值
        """
        pass
    
    @abstractmethod
    def evaluate_population(self, population: List, data: Dict[str, Any]):
        """
        評估整個族群的適應度
        
        Args:
            population: 要評估的族群
            data: 評估所需的數據
        """
        pass
