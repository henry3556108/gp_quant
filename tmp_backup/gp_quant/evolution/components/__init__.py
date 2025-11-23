"""
組件化演化計算框架

這個模組實現了方案 C：組件化架構，將演化過程中的各個策略
（選擇、評估、變異、Niching）抽象成可插拔的組件。

主要組件:
- EvolutionEngine: 核心演化引擎
- EvolutionIndividual: 帶 ID 和譜系追蹤的個體
- 各種演化策略 (strategies/)
- 適應度評估器 (evaluators/)  
- 事件處理器 (handlers/)
"""

from .engine import EvolutionEngine
from .individual import EvolutionIndividual
from .result import EvolutionResult

# 工廠函數
def create_evolution_engine(config: dict) -> EvolutionEngine:
    """
    工廠函數：根據配置創建演化引擎
    
    Args:
        config: 配置字典，包含所有演化參數
        
    Returns:
        配置好的演化引擎實例
    """
    from .strategies.initialization import RampedHalfAndHalfStrategy
    from .strategies.selection import RankedSUSStrategy  
    from .strategies.crossover import CrossoverStrategy
    from .strategies.mutation import MutationStrategy
    from .strategies.replacement import GenerationalReplacementStrategy
    from .evaluators.portfolio_evaluator import PortfolioFitnessEvaluator
    from .handlers.logging_handler import LoggingHandler
    from .handlers.save_handler import SaveHandler
    from .handlers.early_stopping_handler import EarlyStoppingHandler
    
    engine = EvolutionEngine(config)
    
    # 添加演化策略
    engine.add_strategy('initialization', RampedHalfAndHalfStrategy())
    engine.add_strategy('selection', RankedSUSStrategy())
    engine.add_strategy('crossover', CrossoverStrategy())
    engine.add_strategy('mutation', MutationStrategy())
    engine.add_strategy('replacement', GenerationalReplacementStrategy())
    
    # 設置適應度評估器
    if config['data']['mode'] == 'portfolio':
        evaluator = PortfolioFitnessEvaluator()
        engine.set_evaluator(evaluator)
    
    # 添加事件處理器
    engine.add_handler(LoggingHandler())
    engine.add_handler(SaveHandler())
    
    if config['termination']['early_stopping']:
        engine.add_handler(EarlyStoppingHandler())
    
    return engine

__all__ = [
    'EvolutionEngine',
    'EvolutionIndividual', 
    'EvolutionResult',
    'create_evolution_engine'
]
