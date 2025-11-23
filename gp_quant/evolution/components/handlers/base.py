"""
事件處理器基類

定義演化過程中事件處理的基本接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class EventHandler(ABC):
    """
    事件處理器基類
    
    定義演化過程中各種事件的處理接口。
    """
    
    def __init__(self):
        self.name = "base_handler"
        self.engine = None
    
    def set_engine(self, engine):
        """設置演化引擎引用"""
        self.engine = engine
    
    @abstractmethod
    def handle_event(self, event_name: str, **kwargs):
        """
        處理事件
        
        Args:
            event_name: 事件名稱
            **kwargs: 事件參數
        """
        pass
    
    def on_evolution_start(self, **kwargs):
        """演化開始事件"""
        pass
    
    def on_generation_start(self, **kwargs):
        """世代開始事件"""
        pass
    
    def on_generation_complete(self, **kwargs):
        """世代完成事件"""
        pass
    
    def on_evolution_complete(self, **kwargs):
        """演化完成事件"""
        pass
    
    def on_evolution_error(self, **kwargs):
        """演化錯誤事件"""
        pass
