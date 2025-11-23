"""
演化結果類

封裝演化過程的結果，包括最佳個體、適應度歷史、統計信息等。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvolutionResult:
    """
    演化結果封裝類
    
    包含演化過程的完整結果信息，包括最佳個體、適應度歷史、
    統計信息和配置參數等。
    """
    
    # 基本信息
    engine_id: str
    config: Dict[str, Any]
    
    # 演化結果
    best_individual: Any  # 最佳個體
    final_population: List[Any]  # 最終族群
    
    # 統計信息
    fitness_history: List[Dict[str, Any]]  # 適應度歷史
    generations_completed: int  # 完成的世代數
    total_evaluations: int  # 總評估次數
    
    # 可選信息
    hall_of_fame: Optional[List[Any]] = None  # 名人堂
    genealogy: Optional[Dict[str, Any]] = None  # 譜系信息
    execution_time: Optional[float] = None  # 執行時間（秒）
    
    @property
    def final_generation(self) -> int:
        """最終世代數（兼容性屬性）"""
        return self.generations_completed
    
    @property
    def best_fitness(self) -> float:
        """最佳適應度值"""
        if self.best_individual and hasattr(self.best_individual, 'fitness'):
            return self.best_individual.fitness.values[0] if hasattr(self.best_individual.fitness, 'values') else self.best_individual.fitness
        return 0.0
    
    @property
    def convergence_generation(self) -> Optional[int]:
        """收斂世代（最佳適應度首次出現的世代）"""
        if not self.fitness_history:
            return None
        
        best_fitness = self.best_fitness
        for i, stats in enumerate(self.fitness_history):
            if abs(stats.get('best_fitness', 0) - best_fitness) < 1e-10:
                return stats.get('generation', i)
        
        return None
    
    @property
    def improvement_rate(self) -> float:
        """改進率（最終適應度相對於初始適應度的改進）"""
        if not self.fitness_history or len(self.fitness_history) < 2:
            return 0.0
        
        initial_fitness = self.fitness_history[0].get('best_fitness', 0)
        final_fitness = self.best_fitness
        
        if initial_fitness == 0:
            return float('inf') if final_fitness > 0 else 0.0
        
        return (final_fitness - initial_fitness) / abs(initial_fitness)
    
    def get_fitness_statistics(self) -> Dict[str, float]:
        """獲取適應度統計信息"""
        if not self.fitness_history:
            return {}
        
        final_stats = self.fitness_history[-1]
        initial_stats = self.fitness_history[0]
        
        return {
            'initial_best': initial_stats.get('best_fitness', 0),
            'initial_avg': initial_stats.get('avg_fitness', 0),
            'initial_worst': initial_stats.get('worst_fitness', 0),
            'final_best': final_stats.get('best_fitness', 0),
            'final_avg': final_stats.get('avg_fitness', 0),
            'final_worst': final_stats.get('worst_fitness', 0),
            'improvement_rate': self.improvement_rate,
            'convergence_generation': self.convergence_generation
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """獲取結果摘要"""
        return {
            'engine_id': self.engine_id,
            'experiment_name': self.config.get('experiment', {}).get('name', 'Unknown'),
            'generations_completed': self.generations_completed,
            'total_evaluations': self.total_evaluations,
            'population_size': len(self.final_population),
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'fitness_statistics': self.get_fitness_statistics(),
            'hall_of_fame_size': len(self.hall_of_fame) if self.hall_of_fame else 0,
            'genealogy_size': len(self.genealogy) if self.genealogy else 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式（用於序列化）"""
        return {
            'engine_id': self.engine_id,
            'config': self.config,
            'generations_completed': self.generations_completed,
            'total_evaluations': self.total_evaluations,
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'fitness_history': self.fitness_history,
            'fitness_statistics': self.get_fitness_statistics(),
            'summary': self.get_summary()
        }
