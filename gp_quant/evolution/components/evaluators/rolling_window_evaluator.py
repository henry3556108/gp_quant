"""
Rolling Window Fitness Evaluator

實現滾動視窗適應度評估機制：
- 將回測期間分割成多個連續的時間視窗
- 對每個個體在所有視窗上進行回測
- 將各視窗的適應度取平均（或其他聚合方式）作為最終適應度
- 這種方法可評估策略在不同市場環境下的穩健性
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from .portfolio_evaluator import PortfolioFitnessEvaluator, setup_deap_creator
from ..backtesting import PortfolioBacktestingEngine

logger = logging.getLogger(__name__)


def soft_min_score(values: List[float], alpha: float = 1.0) -> float:
    """
    Compute Negative Exponential Sum (Soft-Min) score.
    
    Formula: S(x) = -ln(sum(e^(-alpha * x_i)))
    
    This function heavily penalizes low values:
    - High values: e^(-alpha * x) -> 0 (small penalty)
    - Low/negative values: e^(-alpha * x) -> large (huge penalty)
    
    Args:
        values: List of fitness scores.
        alpha: Sensitivity parameter. Higher alpha = more aggressive penalty.
               Recommended: 0.5-2.0 for typical fitness ranges.
    
    Returns:
        Aggregated score (higher is better).
    """
    if not values:
        return -100000.0
    
    arr = np.array(values)
    
    # Numerical stability: shift values to avoid overflow
    # log(sum(e^x_i)) = max_x + log(sum(e^(x_i - max_x)))
    shifted = -alpha * arr
    max_shifted = np.max(shifted)
    
    # Compute log-sum-exp for numerical stability
    exp_sum = np.sum(np.exp(shifted - max_shifted))
    log_penalty = max_shifted + np.log(exp_sum)
    
    return -log_penalty


def _evaluate_rolling_window_worker(individual_data: tuple, engine_config: Dict[str, Any],
                                     fitness_config: Dict[str, Any]) -> tuple:
    """
    Worker function for parallel rolling window evaluation.
    Evaluates an individual across all windows and returns aggregated fitness.
    """
    setup_deap_creator()
    
    try:
        individual_id, individual_tree = individual_data
        
        # Import pset inside worker to avoid pickling issues
        from ..gp import pset
        from pathlib import Path
        
        windows = engine_config['windows']
        tickers_dir = engine_config['tickers_dir']
        initial_capital = engine_config['initial_capital']
        fitness_metric = fitness_config.get('function', 'sharpe_ratio')
        aggregation = fitness_config.get('aggregation', 'mean')
        no_trade_penalty = fitness_config.get('no_trade_penalty', -10.0)
        
        window_fitnesses = []
        
        for window in windows:
            # Load data for this window
            data = {}
            for ticker in engine_config['tickers']:
                csv_path = Path(tickers_dir) / f"{ticker}.csv"
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                
                # Convert window dates to Timestamps for proper slicing
                data_start = pd.Timestamp(window['data_start'])
                backtest_end = pd.Timestamp(window['backtest_end'])
                
                # Clamp to actual data range to avoid KeyError
                actual_start = max(data_start, df.index.min())
                actual_end = min(backtest_end, df.index.max())
                
                # Filter to window period (including warmup)
                df = df.loc[actual_start:actual_end]
                data[ticker] = df
            
            # Create backtest engine for this window
            engine = PortfolioBacktestingEngine(
                data=data,
                backtest_start=window['backtest_start'],
                backtest_end=window['backtest_end'],
                initial_capital=initial_capital,
                pset=None
            )
            engine.pset = pset
            
            # Run backtest
            try:
                result = engine.backtest(individual_tree)
                
                # Check for no trades - apply consistent penalty
                if result['transactions'].empty:
                    fitness_value = no_trade_penalty
                # Handle consistent_sharpe specially (not in metrics dict)
                elif fitness_metric == 'consistent_sharpe':
                    fitness_value = engine.calculate_consistent_sharpe(result['equity_curve'])
                elif fitness_metric == 'sterling_ratio':
                    from gp_quant.backtesting.metrics import PortfolioMetrics
                    fitness_value = PortfolioMetrics.calculate_sterling_ratio(
                        result['metrics']['total_return'],
                        result['metrics']['max_drawdown']
                    )
                else:
                    fitness_value = result['metrics'].get(fitness_metric, -100000.0)
                
                # Handle invalid values
                if np.isnan(fitness_value) or np.isinf(fitness_value):
                    fitness_value = -100000.0
                    
                window_fitnesses.append(fitness_value)
            except Exception as e:
                window_fitnesses.append(-100000.0)
        
        # Aggregate fitness across windows
        soft_min_alpha = fitness_config.get('soft_min_alpha', 1.0)
        
        if not window_fitnesses:
            aggregated_fitness = -100000.0
        elif aggregation == 'mean':
            aggregated_fitness = np.mean(window_fitnesses)
        elif aggregation == 'median':
            aggregated_fitness = np.median(window_fitnesses)
        elif aggregation == 'min':
            aggregated_fitness = np.min(window_fitnesses)
        elif aggregation == 'consistency':
            # Penalize high variance
            mean_fit = np.mean(window_fitnesses)
            std_fit = np.std(window_fitnesses)
            aggregated_fitness = mean_fit - 0.5 * std_fit
        elif aggregation == 'soft_min':
            # Negative Exponential Sum - heavily penalizes worst windows
            aggregated_fitness = soft_min_score(window_fitnesses, alpha=soft_min_alpha)
        else:
            aggregated_fitness = np.mean(window_fitnesses)
        
        return individual_id, aggregated_fitness, window_fitnesses
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return individual_data[0], -100000.0, []


class RollingWindowFitnessEvaluator(PortfolioFitnessEvaluator):
    """
    Rolling Window Fitness Evaluator.
    
    Evaluates each individual across multiple consecutive time windows
    and aggregates the fitness scores to assess robustness.
    
    Parameters:
        window_days: Length of each backtest window in days
        step_days: Rolling step size in days
        warmup_days: Days before window start for indicator warm-up
        aggregation: How to aggregate window fitnesses ('mean', 'median', 'min', 'consistency')
    """
    
    def __init__(self, max_processors: int = 1, cache_enabled: bool = True,
                 window_days: int = 365, step_days: int = 180, 
                 warmup_days: int = 250, aggregation: str = 'mean',
                 soft_min_alpha: float = 1.0, **kwargs):
        super().__init__(max_processors=max_processors, cache_enabled=cache_enabled, **kwargs)
        
        self.window_days = window_days
        self.step_days = step_days
        self.warmup_days = warmup_days
        self.aggregation = aggregation
        self.soft_min_alpha = soft_min_alpha
        self.windows = []  # Will be populated in set_data
        
        logger.info(f"Rolling Window Evaluator: window={window_days}d, step={step_days}d, "
                   f"warmup={warmup_days}d, aggregation={aggregation}, soft_min_alpha={soft_min_alpha}")
    
    def set_data(self, data: Dict[str, Any]):
        """
        Set data and generate rolling windows.
        """
        super().set_data(data)
        
        # Get date range from config
        data_config = self.engine.config.get('data', {})
        train_start = pd.Timestamp(data_config.get('train_backtest_start'))
        train_end = pd.Timestamp(data_config.get('train_backtest_end'))
        
        # Generate windows
        self.windows = self._generate_windows(train_start, train_end)
        
        logger.info(f"Rolling Window: Generated {len(self.windows)} windows "
                   f"from {train_start.date()} to {train_end.date()}")
        
        # Initialize persistent backtest engine for SaveHandler (use first window)
        if self.windows:
            try:
                from ..gp import pset
                
                train_data = data['train_data']
                processed_data = {}
                for ticker, ticker_data in train_data.items():
                    if isinstance(ticker_data, dict) and 'data' in ticker_data:
                        processed_data[ticker] = ticker_data['data']
                    else:
                        processed_data[ticker] = ticker_data
                
                self.backtest_engine = PortfolioBacktestingEngine(
                    data=processed_data,
                    backtest_start=self.engine_config['backtest_start'],
                    backtest_end=self.engine_config['backtest_end'],
                    initial_capital=self.engine_config['initial_capital'],
                    pset=pset
                )
                logger.info("Rolling Window: Backtest engine initialized for SaveHandler")
                
            except Exception as e:
                logger.error(f"Rolling Window: Failed to initialize backtest engine: {e}")
    
    def _generate_windows(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict]:
        """
        Generate rolling windows from start to end date.
        """
        windows = []
        current_start = start_date
        
        while True:
            window_end = current_start + timedelta(days=self.window_days)
            
            if window_end > end_date:
                break
            
            # Calculate data start (for indicator warmup)
            data_start = current_start - timedelta(days=self.warmup_days)
            
            windows.append({
                'data_start': data_start.strftime('%Y-%m-%d'),
                'backtest_start': current_start.strftime('%Y-%m-%d'),
                'backtest_end': window_end.strftime('%Y-%m-%d')
            })
            
            current_start += timedelta(days=self.step_days)
        
        return windows
    
    def evaluate_population(self, population: List, data: Dict[str, Any]):
        """
        Evaluate population using rolling window fitness.
        """
        # Filter individuals that need evaluation
        individuals_to_evaluate = [
            ind for ind in population 
            if ind.fitness is None or not hasattr(ind.fitness, 'values') or not ind.fitness.values
        ]
        
        if not individuals_to_evaluate:
            logger.debug("All individuals already have fitness, skipping evaluation")
            return
        
        logger.info(f"Rolling Window: Evaluating {len(individuals_to_evaluate)}/{len(population)} "
                   f"individuals across {len(self.windows)} windows")
        
        if len(individuals_to_evaluate) == 1 or self.max_processors == 1:
            self._evaluate_sequential(individuals_to_evaluate, data)
        else:
            self._evaluate_parallel(individuals_to_evaluate, data)
    
    def _evaluate_sequential(self, individuals: List, data: Dict[str, Any]):
        """Sequential evaluation across all windows."""
        from ..gp import pset
        
        fitness_metric = self.engine.config['fitness']['function']
        tickers = list(data['train_data'].keys())
        
        for individual in tqdm(individuals, desc="Rolling Window Sequential", unit="ind"):
            try:
                window_fitnesses = []
                
                for window in self.windows:
                    # Prepare data for this window
                    window_data = {}
                    for ticker, ticker_data in data['train_data'].items():
                        if isinstance(ticker_data, dict) and 'data' in ticker_data:
                            df = ticker_data['data']
                        else:
                            df = ticker_data
                        
                        # Convert window dates to Timestamps for proper slicing
                        data_start = pd.Timestamp(window['data_start'])
                        backtest_end = pd.Timestamp(window['backtest_end'])
                        
                        # Clamp to actual data range to avoid KeyError
                        actual_start = max(data_start, df.index.min())
                        actual_end = min(backtest_end, df.index.max())
                        
                        # Filter to window
                        df_window = df.loc[actual_start:actual_end]
                        window_data[ticker] = df_window
                    
                    # Create engine for this window
                    engine = PortfolioBacktestingEngine(
                        data=window_data,
                        backtest_start=window['backtest_start'],
                        backtest_end=window['backtest_end'],
                        initial_capital=100000.0,
                        pset=pset
                    )
                    
                    # Run backtest
                    result = engine.backtest(individual)
                    
                    # Check for no trades - apply consistent penalty
                    if result['transactions'].empty:
                        fitness_value = -10.0  # Consistent no_trade_penalty
                    # Handle consistent_sharpe specially (not in metrics dict)
                    elif fitness_metric == 'consistent_sharpe':
                        fitness_value = engine.calculate_consistent_sharpe(result['equity_curve'])
                    elif fitness_metric == 'sterling_ratio':
                        from gp_quant.backtesting.metrics import PortfolioMetrics
                        fitness_value = PortfolioMetrics.calculate_sterling_ratio(
                            result['metrics']['total_return'],
                            result['metrics']['max_drawdown']
                        )
                    else:
                        fitness_value = result['metrics'].get(fitness_metric, -100000.0)
                    
                    if np.isnan(fitness_value) or np.isinf(fitness_value):
                        fitness_value = -100000.0
                    
                    window_fitnesses.append(fitness_value)
                
                # Aggregate
                if self.aggregation == 'mean':
                    final_fitness = np.mean(window_fitnesses)
                elif self.aggregation == 'median':
                    final_fitness = np.median(window_fitnesses)
                elif self.aggregation == 'min':
                    final_fitness = np.min(window_fitnesses)
                elif self.aggregation == 'consistency':
                    mean_fit = np.mean(window_fitnesses)
                    std_fit = np.std(window_fitnesses)
                    final_fitness = mean_fit - 0.5 * std_fit
                elif self.aggregation == 'soft_min':
                    final_fitness = soft_min_score(window_fitnesses, alpha=self.soft_min_alpha)
                else:
                    final_fitness = np.mean(window_fitnesses)
                
                individual.fitness.values = (final_fitness,)
                individual.add_metadata('window_fitnesses', window_fitnesses)
                individual.add_metadata('aggregated_fitness', final_fitness)
                
                if self.cache_enabled:
                    self.fitness_cache[individual.id] = final_fitness
                
            except Exception as e:
                logger.error(f"Rolling Window: Error evaluating {individual.id}: {e}")
                individual.fitness.values = (-100000.0,)
    
    def _evaluate_parallel(self, individuals: List, data: Dict[str, Any]):
        """Parallel evaluation across all windows."""
        try:
            import sys
            
            tickers = list(data['train_data'].keys())
            
            engine_config = {
                'tickers_dir': self.engine.config['data']['tickers_dir'],
                'initial_capital': 100000.0,
                'windows': self.windows,
                'tickers': tickers
            }
            
            fitness_config = {
                'function': self.engine.config['fitness']['function'],
                'aggregation': self.aggregation,
                'soft_min_alpha': self.soft_min_alpha
            }
            
            individual_data = [(ind.id, ind) for ind in individuals]
            n_workers = min(self.max_processors, len(individuals))
            
            sys.stderr.write(f"DEBUG: Rolling Window Parallel with {n_workers} workers, "
                           f"{len(self.windows)} windows\n")
            sys.stderr.flush()
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                worker_func = partial(_evaluate_rolling_window_worker,
                                     engine_config=engine_config,
                                     fitness_config=fitness_config)
                
                future_to_individual = {
                    executor.submit(worker_func, ind_data): ind_data[0]
                    for ind_data in individual_data
                }
                
                results = {}
                with tqdm(total=len(individuals), desc="Rolling Window Parallel", unit="ind") as pbar:
                    for future in as_completed(future_to_individual):
                        try:
                            individual_id, aggregated_fitness, window_fitnesses = future.result()
                            results[individual_id] = (aggregated_fitness, window_fitnesses)
                            
                            if self.cache_enabled:
                                self.fitness_cache[individual_id] = aggregated_fitness
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            individual_id = future_to_individual[future]
                            logger.error(f"Rolling Window Parallel: Worker failed for {individual_id}: {e}")
                            results[individual_id] = (-100000.0, [])
                            pbar.update(1)
            
            # Assign fitness to individuals
            for individual in individuals:
                if individual.id in results:
                    aggregated_fitness, window_fitnesses = results[individual.id]
                    individual.fitness.values = (aggregated_fitness,)
                    individual.add_metadata('window_fitnesses', window_fitnesses)
                    individual.add_metadata('aggregated_fitness', aggregated_fitness)
                else:
                    individual.fitness.values = (-100000.0,)
            
            logger.info(f"Rolling Window Parallel: Completed {len(results)} individuals")
            
        except BaseException as e:
            import sys
            import traceback
            sys.stderr.write(f"CRITICAL: Rolling Window Parallel failed: {e}\n")
            sys.stderr.flush()
            traceback.print_exc()
            logger.error(f"Rolling Window Parallel failed: {e}")
            self._evaluate_sequential(individuals, data)
