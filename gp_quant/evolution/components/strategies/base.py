"""
演化策略基類

定義所有演化策略的統一接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EvolutionStrategy(ABC):
    """
    演化策略基類
    
    所有演化策略都必須繼承此類並實現相應的方法。
    """
    
    def __init__(self):
        self.engine = None
        self.name = "base_strategy"
    
    def set_engine(self, engine):
        """設置演化引擎引用"""
        self.engine = engine
