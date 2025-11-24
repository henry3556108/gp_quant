"""
演化引擎核心類

這個模組實現了組件化演化引擎的核心邏輯，負責協調各個演化策略、
適應度評估器和事件處理器，執行完整的演化過程。
"""

from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
import uuid

from .individual import EvolutionIndividual
from .result import EvolutionResult
from .strategies.base import EvolutionStrategy
from .evaluators.base import FitnessEvaluator
from .handlers.base import EventHandler

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """
    組件化演化引擎
    
    這個類是演化計算的核心，負責：
    1. 管理演化策略 (初始化、選擇、交配、變異、替換)
    2. 協調適應度評估器
    3. 處理事件和回調
    4. 執行完整的演化循環
    5. 收集和返回演化結果
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化演化引擎
        
        Args:
            config: 演化配置字典
        """
        self.config = config
        self.engine_id = str(uuid.uuid4())[:8]
        self.created_at = datetime.now()
        
        # 演化狀態
        self.current_generation = 0
        self.population: List[EvolutionIndividual] = []
        self.best_individual: Optional[EvolutionIndividual] = None
        self.fitness_history: List[Dict[str, float]] = []
        self.is_running = False
        self.should_stop = False
        
        # 組件容器
        self.strategies: Dict[str, EvolutionStrategy] = {}
        self.evaluator: Optional[FitnessEvaluator] = None
        self.handlers: List[EventHandler] = []
        
        # 演化參數
        self.population_size = config['evolution']['population_size']
        self.max_generations = config['evolution']['generations']
        self.max_depth = config['evolution']['maximum_depth']
        self.initial_depth = config['evolution']['initial_depth']
        
        logger.info(f"演化引擎已創建 (ID: {self.engine_id})")
        logger.info(f"配置: 族群={self.population_size}, 世代={self.max_generations}")
    
    def add_strategy(self, strategy_type: str, strategy: EvolutionStrategy):
        """
        添加演化策略
        
        Args:
            strategy_type: 策略類型 ('initialization', 'selection', 'crossover', 'mutation', 'replacement')
            strategy: 策略實例
        """
        if not isinstance(strategy, EvolutionStrategy):
            raise TypeError(f"策略必須繼承自 EvolutionStrategy: {type(strategy)}")
        
        self.strategies[strategy_type] = strategy
        strategy.set_engine(self)  # 設置引擎引用
        logger.debug(f"已添加 {strategy_type} 策略: {strategy.__class__.__name__}")
    
    def set_evaluator(self, evaluator: FitnessEvaluator):
        """
        設置適應度評估器
        
        Args:
            evaluator: 評估器實例
        """
        if not isinstance(evaluator, FitnessEvaluator):
            raise TypeError(f"評估器必須繼承自 FitnessEvaluator: {type(evaluator)}")
        
        self.evaluator = evaluator
        evaluator.set_engine(self)  # 設置引擎引用
        logger.debug(f"已設置評估器: {evaluator.__class__.__name__}")
    
    def add_handler(self, handler: EventHandler):
        """
        添加事件處理器
        
        Args:
            handler: 處理器實例
        """
        if not isinstance(handler, EventHandler):
            raise TypeError(f"處理器必須繼承自 EventHandler: {type(handler)}")
        
        self.handlers.append(handler)
        handler.set_engine(self)  # 設置引擎引用
        logger.debug(f"已添加事件處理器: {handler.__class__.__name__}")
    
    def _validate_components(self):
        """驗證所有必要組件是否已設置"""
        required_strategies = ['initialization', 'selection', 'crossover', 'mutation', 'replacement', 'operation']
        
        for strategy_type in required_strategies:
            if strategy_type not in self.strategies:
                raise ValueError(f"缺少必要的演化策略: {strategy_type}")
        
        if self.evaluator is None:
            raise ValueError("缺少適應度評估器")
        
        logger.debug("組件驗證通過")
    
    def _fire_event(self, event_name: str, **kwargs):
        """
        觸發事件，通知所有處理器
        
        Args:
            event_name: 事件名稱
            **kwargs: 事件參數
        """
        for handler in self.handlers:
            try:
                if hasattr(handler, f'on_{event_name}'):
                    getattr(handler, f'on_{event_name}')(**kwargs)
            except Exception as e:
                logger.error(f"事件處理器 {handler.__class__.__name__} 處理 {event_name} 事件時出錯: {e}")
    
    def evolve(self, data: Dict[str, Any]) -> EvolutionResult:
        """
        執行演化過程
        
        Args:
            data: 演化所需的數據 (訓練數據、測試數據等)
            
        Returns:
            演化結果
        """
        logger.info(f"🚀 開始演化過程 (引擎 ID: {self.engine_id})")
        
        try:
            # 1. 驗證組件
            self._validate_components()
            
            # 2. 初始化
            self.is_running = True
            self.should_stop = False
            self._fire_event('evolution_start', engine=self, data=data)
            
            # 3. 創建初始族群
            logger.info("🌱 創建初始族群...")
            self.population = self.strategies['initialization'].initialize(
                population_size=self.population_size,
                data=data
            )
            logger.info(f"   ✅ 初始族群創建完成: {len(self.population)} 個個體")
            
            # 4. 評估初始族群
            logger.info("🎯 評估初始族群適應度...")
            self.evaluator.evaluate_population(self.population, data)
            self._update_best_individual()
            self._record_generation_stats()
            
            self._fire_event('generation_complete', 
                           generation=0, 
                           population=self.population,
                           best_individual=self.best_individual,
                           engine=self)

            # 5. 演化循環
            for generation in range(1, self.max_generations + 1):
                if self.should_stop:
                    logger.info(f"⏹️ 演化在第 {generation} 世代提前停止")
                    break
                
                self.current_generation = generation
                logger.info(f"🔄 第 {generation}/{self.max_generations} 世代")
                
                # 5.1 使用操作策略產生子代
                all_offspring: List[EvolutionIndividual] = self.strategies['operation'].execute_operations(self.population, data)
                logger.debug(f"   總共產生 {len(all_offspring)} 個子代")
                print("all offspring len:", len(all_offspring))
                # 5.6 評估新產生的子代 (跳過已評估的保留個體)
                new_offspring: List[EvolutionIndividual] = [ind for ind in all_offspring if not ind.fitness.valid]
                if new_offspring:
                    self.evaluator.evaluate_population(new_offspring, data)
                    logger.debug(f"   評估了 {len(new_offspring)} 個新個體")
                print("new offspring len:", len(new_offspring))
                # 5.7 替換策略決定下一代族群
                print("after evaluate population:", self.population)
                self.population: List[EvolutionIndividual] = self.strategies['replacement'].replace(
                    self.population, all_offspring, data
                )
                # print(self.population)
                # 5.6 更新統計
                self._update_best_individual()
                self._record_generation_stats()
                
                # 5.7 觸發世代完成事件
                self._fire_event('generation_complete',
                               generation=generation,
                               population=self.population,
                               best_individual=self.best_individual,
                               engine=self)
                
                logger.info(f"   📊 最佳適應度: {self.best_individual.fitness.values[0]:.6f}")
            
            # 6. 演化完成
            self.is_running = False
            result = self._create_result()
            
            self._fire_event('evolution_complete', 
                           engine=self, 
                           result=result)
            
            logger.info(f"✅ 演化完成! 最終最佳適應度: {self.best_individual.fitness.values[0]:.6f}")
            return result
            
        except Exception as e:
            self.is_running = False
            self._fire_event('evolution_error', engine=self, error=e)
            logger.error(f"❌ 演化過程出錯: {e}")
            raise
    
    def _update_best_individual(self):
        """更新最佳個體"""
        if not self.population:
            return
        
        # 只考慮有有效適應度的個體
        valid_individuals = [ind for ind in self.population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        
        if not valid_individuals:
            return
        
        current_best = max(valid_individuals, key=lambda ind: ind.fitness.values[0])
        
        if self.best_individual is None or current_best.fitness.values[0] > self.best_individual.fitness.values[0]:
            self.best_individual = current_best
            logger.debug(f"在 generation {self.current_generation} 發現新的最佳個體: fitness={current_best.fitness.values[0]:.6f}")
    
    def _record_generation_stats(self):
        """記錄世代統計信息"""
        if not self.population:
            return
        
        # 只考慮有有效適應度的個體
        valid_individuals = [ind for ind in self.population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        
        if valid_individuals:
            fitness_values = [ind.fitness.values[0] for ind in valid_individuals]
            stats = {
                'generation': self.current_generation,
                'best_fitness': max(fitness_values),
                'avg_fitness': sum(fitness_values) / len(fitness_values),
                'worst_fitness': min(fitness_values),
                'population_size': len(self.population),
                'valid_individuals': len(valid_individuals)
            }
            self.fitness_history.append(stats)
    
    def _create_result(self) -> EvolutionResult:
        """創建演化結果"""
        # 創建名人堂（包含最佳個體）
        hall_of_fame = []
        if self.best_individual:
            hall_of_fame.append(self.best_individual)
        
        # 添加其他優秀個體到名人堂
        if self.population:
            sorted_pop = sorted(self.population, key=lambda ind: ind.fitness.values[0] if hasattr(ind.fitness, 'values') and ind.fitness.values else -float('inf'), reverse=True)
            for ind in sorted_pop[:min(10, len(sorted_pop))]:  # 最多10個
                if ind not in hall_of_fame:
                    hall_of_fame.append(ind)
        
        return EvolutionResult(
            engine_id=self.engine_id,
            config=self.config,
            best_individual=self.best_individual,
            final_population=self.population.copy() if self.population else [],
            fitness_history=self.fitness_history.copy(),
            generations_completed=self.current_generation,
            total_evaluations=self.current_generation * self.population_size,
            hall_of_fame=hall_of_fame,
            genealogy={}  # TODO: 實作譜系追蹤
        )
    
    def stop(self):
        """停止演化過程"""
        self.should_stop = True
        logger.info("收到停止信號")
    
    def get_status(self) -> Dict[str, Any]:
        """獲取引擎狀態"""
        return {
            'engine_id': self.engine_id,
            'is_running': self.is_running,
            'current_generation': self.current_generation,
            'max_generations': self.max_generations,
            'population_size': len(self.population),
            'best_fitness': self.best_individual.fitness if self.best_individual else None,
            'created_at': self.created_at.isoformat()
        }
