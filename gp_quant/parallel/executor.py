"""
Parallel Executor

This module provides a general-purpose parallel execution framework
for managing concurrent tasks with proper resource allocation.
"""

import concurrent.futures
from typing import Callable, Any, Optional, List, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    General-purpose parallel executor for managing concurrent tasks.
    
    Features:
    - Support for both concurrent (parallel) and sequential execution
    - Automatic resource management
    - Error handling and timeout support
    - Progress tracking
    
    Use Cases:
    - Execute Fitness + Similarity calculations concurrently
    - Manage CPU core allocation
    - Provide fallback to sequential mode
    """
    
    def __init__(self, max_workers: int = 2):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of concurrent tasks
        """
        self.max_workers = max_workers
    
    def execute_concurrent(self,
                          tasks: List[Tuple[Callable, tuple, dict]],
                          timeout: Optional[float] = None) -> List[Any]:
        """
        Execute multiple tasks concurrently.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            timeout: Optional timeout in seconds for each task
            
        Returns:
            List of results in the same order as tasks
            
        Example:
            executor = ParallelExecutor(max_workers=2)
            tasks = [
                (evaluate_fitness, (population,), {}),
                (calculate_similarity, (population,), {})
            ]
            results = executor.execute_concurrent(tasks)
            fitness_scores, similarity_matrix = results
        """
        results = [None] * len(tasks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, (func, args, kwargs) in enumerate(tasks):
                future = executor.submit(func, *args, **kwargs)
                future_to_index[future] = i
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_index, timeout=timeout):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    results[index] = None
        
        return results
    
    def execute_sequential(self,
                          tasks: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """
        Execute tasks sequentially (fallback mode).
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            
        Returns:
            List of results
        """
        results = []
        for func, args, kwargs in tasks:
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
    
    def execute_with_timing(self,
                           tasks: List[Tuple[Callable, tuple, dict]],
                           mode: str = 'concurrent') -> Tuple[List[Any], Dict[str, float]]:
        """
        Execute tasks and return results with timing information.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            mode: 'concurrent' or 'sequential'
            
        Returns:
            Tuple of (results, timing_info)
        """
        start_time = time.time()
        
        if mode == 'concurrent':
            results = self.execute_concurrent(tasks)
        else:
            results = self.execute_sequential(tasks)
        
        end_time = time.time()
        
        timing_info = {
            'total_time': end_time - start_time,
            'mode': mode,
            'n_tasks': len(tasks)
        }
        
        return results, timing_info


class SyncPoint:
    """
    Synchronization point for coordinating parallel tasks.
    
    Use this when you need to ensure multiple tasks complete
    before proceeding.
    """
    
    def __init__(self, n_tasks: int):
        """
        Initialize sync point.
        
        Args:
            n_tasks: Number of tasks to wait for
        """
        self.n_tasks = n_tasks
        self.completed = 0
        self.results = {}
    
    def mark_complete(self, task_id: str, result: Any):
        """Mark a task as complete"""
        self.results[task_id] = result
        self.completed += 1
    
    def is_ready(self) -> bool:
        """Check if all tasks are complete"""
        return self.completed >= self.n_tasks
    
    def get_results(self) -> dict:
        """Get all results"""
        if not self.is_ready():
            raise RuntimeError("Not all tasks are complete")
        return self.results
