"""
Early Stopping for GP Evolution

提供早停機制，當連續 N 代無進步時終止演化。
"""

from typing import Optional, Dict, Any


class EarlyStopping:
    """
    早停機制類
    
    當連續 N 個 generation 的最佳 fitness 沒有顯著改進時，提前終止演化。
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')
        >>> for gen in range(max_generations):
        ...     # ... 演化邏輯 ...
        ...     current_best = hof[0].fitness.values[0]
        ...     if early_stopping.step(current_best):
        ...         print("Early stopping triggered!")
        ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        初始化早停機制
        
        Args:
            patience: 連續無進步的 generation 數量。達到此數量時觸發早停。
            min_delta: 最小改進閾值。只有當改進大於此值時才被視為有進步。
            mode: 'max' (fitness 越大越好) 或 'min' (fitness 越小越好)
        
        Raises:
            ValueError: 如果 patience < 1 或 mode 不是 'max'/'min'
        """
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        
        if mode not in ['max', 'min']:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        # 內部狀態
        self.counter = 0  # 連續無進步的計數器
        self.best_fitness: Optional[float] = None  # 歷史最佳 fitness
        self.should_stop = False  # 是否應該停止
        self.generation = 0  # 當前 generation 數
    
    def step(self, current_fitness: float) -> bool:
        """
        檢查是否應該停止演化
        
        Args:
            current_fitness: 當前 generation 的最佳 fitness
        
        Returns:
            bool: True 表示應該停止，False 表示繼續
        """
        self.generation += 1
        
        # 第一代，初始化最佳 fitness
        if self.best_fitness is None:
            self.best_fitness = current_fitness
            return False
        
        # 計算改進量
        if self.mode == 'max':
            improvement = current_fitness - self.best_fitness
        else:  # mode == 'min'
            improvement = self.best_fitness - current_fitness
        
        # 檢查是否有顯著進步
        if improvement > self.min_delta:
            # 有進步，更新最佳 fitness 並重置計數器
            self.best_fitness = current_fitness
            self.counter = 0
        else:
            # 無進步，計數器 +1
            self.counter += 1
        
        # 判斷是否應該停止
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        返回當前早停狀態
        
        Returns:
            Dict 包含以下鍵值：
                - counter: 連續無進步的代數
                - best_fitness: 歷史最佳 fitness
                - should_stop: 是否應該停止
                - generation: 當前 generation 數
                - patience: 設定的 patience
                - min_delta: 設定的 min_delta
        """
        return {
            'counter': self.counter,
            'best_fitness': self.best_fitness,
            'should_stop': self.should_stop,
            'generation': self.generation,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode
        }
    
    def reset(self):
        """重置早停狀態"""
        self.counter = 0
        self.best_fitness = None
        self.should_stop = False
        self.generation = 0
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (f"EarlyStopping(patience={self.patience}, min_delta={self.min_delta}, "
                f"mode='{self.mode}', counter={self.counter}, generation={self.generation})")
