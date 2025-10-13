"""
Parallel Fitness Evaluator

This module implements parallel fitness evaluation for GP individuals
using multiprocessing to avoid GIL and race conditions.
"""

import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Any, Callable, Optional
import logging
from functools import partial
import dill

logger = logging.getLogger(__name__)


def _evaluate_individual_worker(individual: Any,
                                evaluation_func: Callable,
                                **kwargs) -> tuple:
    """
    Worker function to evaluate a single individual.
    
    This function runs in a separate process, so it's completely isolated
    from other workers (no race conditions).
    
    Args:
        individual: DEAP individual to evaluate
        evaluation_func: Function to evaluate fitness
        **kwargs: Additional arguments for evaluation_func
        
    Returns:
        Tuple of (individual_id, fitness_value)
    """
    try:
        fitness = evaluation_func(individual, **kwargs)
        return (id(individual), fitness)
    except Exception as e:
        logger.error(f"Error evaluating individual: {e}")
        return (id(individual), 0.0)  # Return 0 fitness on error


class ParallelFitnessEvaluator:
    """
    Parallel fitness evaluator using multiprocessing.
    
    Key Features:
    - Uses multiprocessing.Pool for true parallelism (no GIL)
    - Each worker is completely isolated (no shared state)
    - Automatic load balancing
    - Error handling and fallback to sequential mode
    
    Thread Safety:
    - This class is thread-safe
    - Each evaluation creates a new process pool
    - No shared mutable state
    
    Resource Management:
    - Uses all available CPU cores by default
    - Can be configured to use fewer cores
    - Automatically cleans up resources
    """
    
    def __init__(self, 
                 n_workers: Optional[int] = None,
                 enable_parallel: bool = True,
                 min_population_for_parallel: int = 50):
        """
        Initialize parallel fitness evaluator.
        
        Args:
            n_workers: Number of worker processes (default: CPU count)
            enable_parallel: Enable parallel evaluation
            min_population_for_parallel: Minimum population size to use parallel
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        self.n_workers = n_workers
        self.enable_parallel = enable_parallel
        self.min_population_for_parallel = min_population_for_parallel
        
        logger.info(f"ParallelFitnessEvaluator initialized with {n_workers} workers")
    
    def evaluate_population(self,
                          population: List[Any],
                          evaluation_func: Callable,
                          **kwargs) -> List[float]:
        """
        Evaluate fitness for entire population in parallel.
        
        Args:
            population: List of DEAP individuals
            evaluation_func: Function to evaluate fitness
                           Should accept (individual, **kwargs) and return float
            **kwargs: Additional arguments for evaluation_func
            
        Returns:
            List of fitness values (same order as population)
            
        Example:
            evaluator = ParallelFitnessEvaluator(n_workers=8)
            fitness_scores = evaluator.evaluate_population(
                population,
                portfolio_engine.get_fitness
            )
        """
        # Decide whether to use parallel or sequential
        if not self.enable_parallel or len(population) < self.min_population_for_parallel:
            logger.info("Using sequential evaluation")
            return self._evaluate_sequential(population, evaluation_func, **kwargs)
        
        try:
            logger.info(f"Using parallel evaluation with {self.n_workers} workers")
            return self._evaluate_parallel(population, evaluation_func, **kwargs)
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}, falling back to sequential")
            return self._evaluate_sequential(population, evaluation_func, **kwargs)
    
    def _evaluate_sequential(self,
                           population: List[Any],
                           evaluation_func: Callable,
                           **kwargs) -> List[float]:
        """Sequential evaluation (fallback)"""
        fitness_scores = []
        for individual in population:
            try:
                fitness = evaluation_func(individual, **kwargs)
                fitness_scores.append(fitness)
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                fitness_scores.append(0.0)
        
        return fitness_scores
    
    def _evaluate_parallel(self,
                         population: List[Any],
                         evaluation_func: Callable,
                         **kwargs) -> List[float]:
        """
        Parallel evaluation using multiprocessing.
        
        Implementation Notes:
        - Uses multiprocessing.Pool for process isolation
        - Each worker gets a batch of individuals
        - Results are collected and returned in original order
        - Automatic cleanup of resources
        """
        # Create a partial function with kwargs
        worker_func = partial(_evaluate_individual_worker,
                             evaluation_func=evaluation_func,
                             **kwargs)
        
        # Create process pool and evaluate
        with Pool(processes=self.n_workers) as pool:
            # Use map for automatic load balancing
            # map maintains order, so results align with population
            results = pool.map(worker_func, population)
        
        # Extract fitness values (results are tuples of (id, fitness))
        fitness_scores = [fitness for _, fitness in results]
        
        return fitness_scores
    
    def evaluate_population_with_progress(self,
                                        population: List[Any],
                                        evaluation_func: Callable,
                                        progress_callback: Optional[Callable] = None,
                                        **kwargs) -> List[float]:
        """
        Evaluate population with progress tracking.
        
        Args:
            population: List of DEAP individuals
            evaluation_func: Function to evaluate fitness
            progress_callback: Optional callback(current, total) for progress
            **kwargs: Additional arguments for evaluation_func
            
        Returns:
            List of fitness values
        """
        if not self.enable_parallel or len(population) < self.min_population_for_parallel:
            return self._evaluate_sequential_with_progress(
                population, evaluation_func, progress_callback, **kwargs
            )
        
        try:
            return self._evaluate_parallel_with_progress(
                population, evaluation_func, progress_callback, **kwargs
            )
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}, falling back to sequential")
            return self._evaluate_sequential_with_progress(
                population, evaluation_func, progress_callback, **kwargs
            )
    
    def _evaluate_sequential_with_progress(self,
                                         population: List[Any],
                                         evaluation_func: Callable,
                                         progress_callback: Optional[Callable],
                                         **kwargs) -> List[float]:
        """Sequential evaluation with progress tracking"""
        fitness_scores = []
        total = len(population)
        
        for i, individual in enumerate(population):
            try:
                fitness = evaluation_func(individual, **kwargs)
                fitness_scores.append(fitness)
            except Exception as e:
                logger.error(f"Error evaluating individual: {e}")
                fitness_scores.append(0.0)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return fitness_scores
    
    def _evaluate_parallel_with_progress(self,
                                       population: List[Any],
                                       evaluation_func: Callable,
                                       progress_callback: Optional[Callable],
                                       **kwargs) -> List[float]:
        """Parallel evaluation with progress tracking"""
        worker_func = partial(_evaluate_individual_worker,
                             evaluation_func=evaluation_func,
                             **kwargs)
        
        fitness_scores = []
        total = len(population)
        
        with Pool(processes=self.n_workers) as pool:
            # Use imap_unordered for progress tracking
            # Note: This doesn't maintain order, so we need to track IDs
            results_dict = {}
            
            for i, result in enumerate(pool.imap_unordered(worker_func, population)):
                ind_id, fitness = result
                results_dict[ind_id] = fitness
                
                if progress_callback:
                    progress_callback(i + 1, total)
            
            # Reconstruct in original order
            fitness_scores = [results_dict[id(ind)] for ind in population]
        
        return fitness_scores


class PortfolioFitnessEvaluator:
    """
    Specialized fitness evaluator for portfolio backtesting.
    
    This class wraps ParallelFitnessEvaluator and provides
    portfolio-specific evaluation logic.
    """
    
    def __init__(self,
                 portfolio_engine_factory: Callable,
                 n_workers: Optional[int] = None,
                 enable_parallel: bool = True):
        """
        Initialize portfolio fitness evaluator.
        
        Args:
            portfolio_engine_factory: Factory function to create PortfolioBacktestingEngine
                                     Should accept no arguments and return engine instance
            n_workers: Number of worker processes
            enable_parallel: Enable parallel evaluation
        """
        self.portfolio_engine_factory = portfolio_engine_factory
        self.parallel_evaluator = ParallelFitnessEvaluator(
            n_workers=n_workers,
            enable_parallel=enable_parallel
        )
    
    def evaluate_population(self, population: List[Any]) -> List[float]:
        """
        Evaluate population using portfolio backtesting.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            List of fitness values (excess returns)
        """
        # Create evaluation function that creates its own engine instance
        # This ensures each worker has its own engine (no shared state)
        def eval_func(individual):
            engine = self.portfolio_engine_factory()
            return engine.get_fitness(individual)
        
        return self.parallel_evaluator.evaluate_population(
            population,
            eval_func
        )
