"""
演化狀態載入器 - 用於重新載入保存的演化狀態
"""
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .individual import EvolutionIndividual
from .engine import EvolutionEngine
from .result import EvolutionResult


class EvolutionLoader:
    """演化狀態載入器"""
    
    def __init__(self, records_dir: str):
        """
        初始化載入器
        
        Args:
            records_dir: 記錄目錄路徑
        """
        self.records_dir = Path(records_dir)
        if not self.records_dir.exists():
            raise FileNotFoundError(f"記錄目錄不存在: {records_dir}")
    
    def load_population(self, generation: int) -> List[EvolutionIndividual]:
        """
        載入指定世代的族群
        
        Args:
            generation: 世代號
            
        Returns:
            該世代的族群列表
        """
        pickle_file = self.records_dir / "populations" / f"generation_{generation:03d}.pkl"
        
        if not pickle_file.exists():
            raise FileNotFoundError(f"世代 {generation} 的族群文件不存在: {pickle_file}")
        
        with open(pickle_file, 'rb') as f:
            population = pickle.load(f)
        
        print(f"✅ 載入世代 {generation} 族群: {len(population)} 個個體")
        return population
    
    def load_engine_state(self) -> Dict[str, Any]:
        """
        載入完整的演化引擎狀態
        
        Returns:
            包含引擎狀態的字典
        """
        state_file = self.records_dir / "engine_state.pkl"
        
        if not state_file.exists():
            raise FileNotFoundError(f"演化引擎狀態文件不存在: {state_file}")
        
        with open(state_file, 'rb') as f:
            engine_state = pickle.load(f)
        
        print(f"✅ 載入演化引擎狀態")
        print(f"   🔄 當前世代: {engine_state['current_generation']}")
        print(f"   👥 族群大小: {len(engine_state['population'])}")
        print(f"   🏆 最佳適應度: {engine_state['best_individual'].fitness.values[0] if engine_state['best_individual'] else 'N/A'}")
        
        return engine_state
    
    def continue_evolution(self, additional_generations: int, data: Dict[str, Any]) -> EvolutionResult:
        """
        從保存的狀態繼續演化
        
        Args:
            additional_generations: 額外的演化世代數
            data: 演化數據
            
        Returns:
            演化結果
        """
        # 載入演化狀態
        engine_state = self.load_engine_state()
        engine = engine_state['engine']
        
        # 更新演化世代數
        original_generations = engine.max_generations
        engine.max_generations = engine.current_generation + additional_generations
        
        print(f"🔄 繼續演化:")
        print(f"   📊 原始世代: {original_generations}")
        print(f"   🔄 當前世代: {engine.current_generation}")
        print(f"   ➕ 額外世代: {additional_generations}")
        print(f"   🎯 目標世代: {engine.max_generations}")
        
        # 繼續演化
        result = engine.evolve(data)
        
        return result
    
    def get_available_generations(self) -> List[int]:
        """
        獲取可用的世代列表
        
        Returns:
            可用世代號列表
        """
        populations_dir = self.records_dir / "populations"
        if not populations_dir.exists():
            return []
        
        generations = []
        for pkl_file in populations_dir.glob("generation_*.pkl"):
            try:
                gen_num = int(pkl_file.stem.split('_')[1])
                generations.append(gen_num)
            except (ValueError, IndexError):
                continue
        
        return sorted(generations)
    
    def analyze_evolution_progress(self) -> Dict[str, Any]:
        """
        分析演化進度
        
        Returns:
            演化進度分析結果
        """
        available_gens = self.get_available_generations()
        if not available_gens:
            return {"error": "沒有可用的世代數據"}
        
        # 載入統計數據
        stats_file = self.records_dir / "generation_stats.json"
        if stats_file.exists():
            import json
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        else:
            stats = []
        
        analysis = {
            "total_generations": len(available_gens),
            "generation_range": f"{min(available_gens)} - {max(available_gens)}",
            "available_generations": available_gens,
            "has_statistics": len(stats) > 0,
            "statistics_count": len(stats)
        }
        
        if stats:
            fitness_values = [s['best_fitness'] for s in stats if s['best_fitness'] is not None]
            if fitness_values:
                analysis.update({
                    "best_fitness_overall": max(fitness_values),
                    "worst_fitness_overall": min(fitness_values),
                    "fitness_improvement": fitness_values[-1] - fitness_values[0] if len(fitness_values) > 1 else 0,
                    "convergence_detected": len(set(fitness_values[-5:])) == 1 if len(fitness_values) >= 5 else False
                })
        
        return analysis


def load_and_continue_evolution(records_dir: str, additional_generations: int, data: Dict[str, Any]) -> EvolutionResult:
    """
    便利函數：載入並繼續演化
    
    Args:
        records_dir: 記錄目錄
        additional_generations: 額外演化世代數
        data: 演化數據
        
    Returns:
        演化結果
    """
    loader = EvolutionLoader(records_dir)
    return loader.continue_evolution(additional_generations, data)


def analyze_saved_evolution(records_dir: str) -> Dict[str, Any]:
    """
    便利函數：分析保存的演化數據
    
    Args:
        records_dir: 記錄目錄
        
    Returns:
        分析結果
    """
    loader = EvolutionLoader(records_dir)
    return loader.analyze_evolution_progress()
