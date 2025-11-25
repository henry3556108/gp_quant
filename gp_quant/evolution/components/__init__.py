"""
組件化演化計算框架

這個模組實現了方案 C：組件化架構，將演化過程中的各個策略
（選擇、評估、變異、Niching）抽象成可插拔的組件。
"""

from .engine import EvolutionEngine
from .individual import EvolutionIndividual
from .result import EvolutionResult
from .loader import EvolutionLoader

def _create_strategy(strategy_type: str, strategy_name: str, strategies_module, config: dict):
    """
    根據配置動態創建演化策略
    
    Args:
        strategy_type: 策略類型 ('initialization', 'selection', 'crossover', 'mutation', 'replacement')
        strategy_name: 策略名稱 (如 'ramped_half_and_half', 'ranked_sus')
        strategies_module: 策略模組
        config: 配置字典
        
    Returns:
        創建的策略實例
        
    Raises:
        ValueError: 如果策略不存在
    """
    # 策略名稱到類名的映射
    strategy_mappings = {
        'initialization': {
            'ramped_half_and_half': 'RampedHalfAndHalfStrategy',
            'full': 'FullStrategy',
            'grow': 'GrowStrategy'
        },
        'selection': {
            'ranked_sus': 'RankedSUSStrategy',
            'tournament': 'TournamentStrategy',
            'roulette': 'RouletteStrategy',
            'ted_niche': 'TEDNicheSelectionStrategy'
        },
        'crossover': {
            'one_point_leaf_biased': 'CrossoverStrategy',
            'one_point': 'OnePointCrossoverStrategy',
            'uniform': 'UniformCrossoverStrategy'
        },
        'mutation': {
            'uniform': 'MutationStrategy',
            'point': 'PointMutationStrategy',
            'subtree': 'SubtreeMutationStrategy'
        },
        'replacement': {
            'generational': 'GenerationalReplacementStrategy',
            'steady_state': 'SteadyStateStrategy',
            'elitist': 'ElitistStrategy',
            'tournament': 'TournamentReplacementStrategy'
        },
        'operation': {
            'parallel': 'ParallelOperationStrategy',
            'serial': 'SerialOperationStrategy'
        }
    }
    
    if strategy_type not in strategy_mappings:
        raise ValueError(f"不支持的策略類型: {strategy_type}")
    
    if strategy_name not in strategy_mappings[strategy_type]:
        available = list(strategy_mappings[strategy_type].keys())
        raise ValueError(f"不支持的{strategy_type}策略: {strategy_name}。可用策略: {available}")
    
    class_name = strategy_mappings[strategy_type][strategy_name]
    
    # 根據映射表動態導入和創建策略類
    try:
        # 根據策略類型導入對應模組
        if strategy_type == 'initialization':
            from .strategies import initialization as strategy_module
        elif strategy_type == 'selection':
            # 特殊處理：TEDNicheSelectionStrategy 在 niche_selection 模組中
            if strategy_name == 'ted_niche':
                from .strategies import niche_selection as strategy_module
            else:
                from .strategies import selection as strategy_module
        elif strategy_type == 'crossover':
            from .strategies import crossover as strategy_module
        elif strategy_type == 'mutation':
            from .strategies import mutation as strategy_module
        elif strategy_type == 'replacement':
            from .strategies import replacement as strategy_module
        elif strategy_type == 'operation':
            from .strategies import operation as strategy_module
        else:
            raise ValueError(f"未知的策略類型: {strategy_type}")
        
        # 使用映射表獲取類名，然後從模組中獲取類
        if not hasattr(strategy_module, class_name):
            raise ValueError(f"策略類 {class_name} 在 {strategy_type} 模組中不存在")
        
        strategy_class = getattr(strategy_module, class_name)
        
        # 獲取策略的配置參數
        strategy_config = config.get(strategy_type, {})
        strategy_params = strategy_config.get('parameters', {})
        
        # 創建策略實例，傳入配置參數
        try:
            return strategy_class(**strategy_params)
        except Exception as e:
            raise ValueError(f"創建策略 {strategy_type}.{strategy_name} 失敗: {e}. 參數: {strategy_params}")
        
    except ImportError as e:
        raise ImportError(f"無法導入策略類 {class_name}: {e}")
    except Exception as e:
        raise Exception(f"創建策略 {strategy_type}.{strategy_name} 時出錯: {e}")

def _create_evaluator(evaluator_type: str, evaluators_module, config: dict):
    """
    根據配置動態創建適應度評估器
    
    Args:
        evaluator_type: 評估器類型
        evaluators_module: 評估器模組
        config: 配置字典
        
    Returns:
        創建的評估器實例
    """
    evaluator_mappings = {
        'portfolio_backtest': 'PortfolioFitnessEvaluator',
        'single_backtest': 'SingleFitnessEvaluator'
    }
    
    if evaluator_type not in evaluator_mappings:
        available = list(evaluator_mappings.keys())
        raise ValueError(f"不支持的評估器類型: {evaluator_type}。可用評估器: {available}")
    
    class_name = evaluator_mappings[evaluator_type]
    
    try:
        # 導入評估器模組
        from .evaluators import portfolio_evaluator as evaluator_module
        
        # 使用映射表獲取類名，然後從模組中獲取類
        if not hasattr(evaluator_module, class_name):
            raise ValueError(f"評估器類 {class_name} 在評估器模組中不存在")
        
        evaluator_class = getattr(evaluator_module, class_name)
        
        # 獲取評估器的配置參數
        fitness_config = config.get('fitness', {})
        evaluator_params = fitness_config.get('parameters', {})
        
        # 創建評估器實例，傳入配置參數
        return evaluator_class(**evaluator_params)
        
    except ImportError as e:
        raise ImportError(f"無法導入評估器類 {class_name}: {e}")

def _create_handler(handler_type: str, handler_name: str, handlers_module, config: dict):
    """
    根據配置動態創建事件處理器
    
    Args:
        handler_type: 處理器類型
        handler_name: 處理器名稱
        handlers_module: 處理器模組
        config: 配置字典
        
    Returns:
        創建的處理器實例
    """
    handler_mappings = {
        'logging_handler': 'LoggingHandler',
        'save_handler': 'SaveHandler',
        'early_stopping_handler': 'EarlyStoppingHandler'
    }
    
    if handler_name not in handler_mappings:
        available = list(handler_mappings.keys())
        raise ValueError(f"不支持的處理器: {handler_name}。可用處理器: {available}")
    
    class_name = handler_mappings[handler_name]
    
    try:
        # 根據處理器名稱導入對應模組
        if handler_name == 'logging_handler':
            from .handlers import logging_handler as handler_module
        elif handler_name == 'save_handler':
            from .handlers import save_handler as handler_module
        elif handler_name == 'early_stopping_handler':
            from .handlers import early_stopping_handler as handler_module
        else:
            raise ValueError(f"未知的處理器: {handler_name}")
        
        # 使用映射表獲取類名，然後從模組中獲取類
        if not hasattr(handler_module, class_name):
            raise ValueError(f"處理器類 {class_name} 在 {handler_name} 模組中不存在")
        
        handler_class = getattr(handler_module, class_name)
        
        # 獲取處理器的配置參數
        handler_params = {}
        if handler_type == 'early_stopping':
            termination_config = config.get('termination', {})
            handler_params = termination_config.get('parameters', {})
        elif handler_type == 'logging':
            logging_config = config.get('logging', {})
            handler_params = {k: v for k, v in logging_config.items() if k != 'parameters'}
            handler_params.update(logging_config.get('parameters', {}))
        elif handler_type == 'save':
            logging_config = config.get('logging', {})
            handler_params = {
                'records_dir': logging_config.get('records_dir', 'evolution_records'),
                'save_populations': logging_config.get('save_populations', True),
                'save_genealogy': logging_config.get('save_genealogy', True),
                'save_format': logging_config.get('save_format', 'json')
            }
            handler_params.update(logging_config.get('parameters', {}))
        
        # 創建處理器實例，傳入配置參數
        return handler_class(**handler_params)
        
    except ImportError as e:
        raise ImportError(f"無法導入處理器類 {class_name}: {e}")

def create_evolution_engine(config: dict) -> EvolutionEngine:
    """
    工廠函數：根據配置創建演化引擎
    
    這個函數根據配置文件創建一個完全配置好的演化引擎，包括：
    - 演化策略 (初始化、選擇、交配、變異、替換)
    - 適應度評估器 (投資組合回測)
    - 事件處理器 (日誌、保存、早停)
    
    Args:
        config: 配置字典，包含所有演化參數
        
    Returns:
        配置好的演化引擎實例
        
    Raises:
        ValueError: 如果配置參數無效
        ImportError: 如果無法導入必要的組件
    """
    print(f"🏗️ 創建組件化演化引擎...")
    
    try:
        # 導入策略模組 (不是具體的策略類)
        from . import strategies
        from . import evaluators
        from . import handlers
        
        print(f"   ✅ 組件模組導入成功")
        
    except ImportError as e:
        raise ImportError(f"無法導入演化組件模組: {e}")
    
    # 驗證配置
    required_sections = ['experiment', 'data', 'evolution', 'fitness', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"配置文件缺少必要部分: {section}")
    
    print(f"   ✅ 配置驗證通過")
    
    # 創建演化引擎
    engine = EvolutionEngine(config)
    print(f"   ✅ 演化引擎核心創建完成")
    
    # 添加演化策略 (根據配置動態選擇)
    print(f"   🔧 添加演化策略...")
    
    # 1. 初始化策略
    init_strategy_name = config['initialization']['method']
    init_strategy = _create_strategy('initialization', init_strategy_name, strategies, config)
    engine.add_strategy('initialization', init_strategy)
    print(f"      ├─ 初始化策略: {init_strategy_name}")
    
    # 2. 選擇策略  
    selection_method = config['selection']['method']
    selection_strategy = _create_strategy('selection', selection_method, strategies, config)
    engine.add_strategy('selection', selection_strategy)
    print(f"      ├─ 選擇策略: {selection_method}")
    
    # 3. 交配策略
    crossover_strategy_name = config['crossover']['strategy']
    crossover_strategy = _create_strategy('crossover', crossover_strategy_name, strategies, config)
    engine.add_strategy('crossover', crossover_strategy)
    print(f"      ├─ 交配策略: {crossover_strategy_name} (rate={config['crossover']['rate']})")
    
    # 4. 變異策略
    mutation_strategy_name = config['mutation']['strategy']
    mutation_strategy = _create_strategy('mutation', mutation_strategy_name, strategies, config)
    engine.add_strategy('mutation', mutation_strategy)
    print(f"      ├─ 變異策略: {mutation_strategy_name} (rate={config['mutation']['rate']})")
    
    # 5. 替換策略
    replacement_method = config['replacement']['method']
    replacement_strategy = _create_strategy('replacement', replacement_method, strategies, config)
    engine.add_strategy('replacement', replacement_strategy)
    print(f"      ├─ 替換策略: {replacement_method}")
    
    # 6. 操作策略
    operation_mode = config.get('operation_mode', 'serial')
    operation_strategy = _create_strategy('operation', operation_mode, strategies, config)
    engine.add_strategy('operation', operation_strategy)
    print(f"      └─ 操作策略: {operation_mode}")
    
    # 設置適應度評估器 (根據配置動態選擇)
    print(f"   🎯 設置適應度評估器...")
    evaluator_type = config['fitness']['evaluator']
    evaluator = _create_evaluator(evaluator_type, evaluators, config)
    engine.set_evaluator(evaluator)
    print(f"      ✅ 評估器: {evaluator_type} ({config['fitness']['function']})")
    
    # 添加事件處理器 (根據配置動態選擇)
    print(f"   📝 添加事件處理器...")
    
    # 添加保存處理器
    if config.get('logging', {}).get('save_populations', False) or config.get('logging', {}).get('save_genealogy', False):
        try:
            save_handler = _create_handler('save', 'save_handler', handlers, config)
            engine.add_handler(save_handler)
            print(f"      ├─ 保存處理器: ✅ 已啟用")
        except Exception as e:
            print(f"      ├─ 保存處理器: ❌ 創建失敗 ({e})")
    else:
        print(f"      ├─ 保存處理器: ⏸️ 未啟用")

    
    print(f"✅ 演化引擎創建完成!")
    print(f"   📊 族群大小: {config['evolution']['population_size']}")
    print(f"   🔄 演化世代: {config['evolution']['generations']}")
    print(f"   🌳 最大深度: {config['evolution']['maximum_depth']}")
    print(f"   ⚡ 處理器數: {config['evolution']['max_processors']}")
    
    return engine

__all__ = ['EvolutionEngine', 'EvolutionIndividual', 'EvolutionResult', 'create_evolution_engine', 'EvolutionLoader']
