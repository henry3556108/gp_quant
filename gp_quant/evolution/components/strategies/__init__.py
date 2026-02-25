"""
演化策略模組

包含所有演化策略的實現：
- 初始化策略
- 選擇策略  
- 交配策略
- 變異策略
- 替換策略
- 操作策略
"""

from .base import EvolutionStrategy
from .initialization import *
from .selection import *
from .niche_selection import (
    TEDNicheSelectionStrategy, 
    PnLNicheSelectionStrategy, 
    SignalNicheSelectionStrategy,
    TreeKernelNicheSelectionStrategy
)
from .crossover import *
from .mutation import *
from .replacement import *
from .operation import *

__all__ = [
    'EvolutionStrategy',
    # 初始化策略
    'InitializationStrategy', 'RampedHalfAndHalfStrategy', 'FullStrategy', 'GrowStrategy',
    # 選擇策略
    'SelectionStrategy', 'RankedSUSStrategy', 'TournamentStrategy', 'RouletteStrategy', 
    'TEDNicheSelectionStrategy', 'PnLNicheSelectionStrategy', 'SignalNicheSelectionStrategy',
    'TreeKernelNicheSelectionStrategy',
    # 交配策略
    'CrossoverStrategy', 'OnePointCrossoverStrategy', 'UniformCrossoverStrategy', 'KNNNichingCrossoverStrategy',
    # 變異策略
    'MutationStrategy', 'PointMutationStrategy', 'SubtreeMutationStrategy',
    # 替換策略
    'ReplacementStrategy', 'GenerationalReplacementStrategy', 'SteadyStateStrategy', 'ElitistStrategy', 'TournamentReplacementStrategy', 'DeterministicCrowdingReplacementStrategy',
    # 操作策略
    'OperationStrategy', 'ParallelOperationStrategy', 'SerialOperationStrategy'
]

