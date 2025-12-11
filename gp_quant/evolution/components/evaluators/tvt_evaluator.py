"""
Train-Validate-Test Evaluator

實現三階段評估機制，用於防止過擬合：
- Train: 演化過程中的適應度計算
- Validate: 模型選擇，決定是否更新最佳個體
- Test: 最終 Out-of-Sample 評估
"""

import logging
import random
from typing import Dict, Any, List, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

from .portfolio_evaluator import PortfolioFitnessEvaluator, setup_deap_creator
from ..backtesting import PortfolioBacktestingEngine

logger = logging.getLogger(__name__)


def _evaluate_tvt_worker(individual_data: tuple, engine_config: Dict[str, Any], 
                         fitness_config: Dict[str, Any]) -> tuple:
    """
    Worker function for parallel TVT evaluation.
    Evaluates on both Train and Validate sets.
    """
    setup_deap_creator()
    
    try:
        individual_id, individual_tree = individual_data
        
        # Import pset inside worker to avoid pickling issues
        from ..gp import pset
        
        # Load data in worker process
        from pathlib import Path
        
        results = {}
        
        for period in ['train', 'validate']:
            period_config = engine_config.get(period)
            if period_config is None:
                continue
                
            # Load data for this period
            data = {}
            for ticker, csv_path in period_config['data_paths'].items():
                tickers_dir = engine_config.get('tickers_dir', 'TSE300_selected')
                if not Path(csv_path).is_absolute():
                    csv_path = Path(tickers_dir) / f"{ticker}.csv"
                
                df = pd.read_csv(csv_path, parse_dates=['Date'])
                df.set_index('Date', inplace=True)
                
                # Filter to period
                df = df.loc[period_config['data_start']:period_config['backtest_end']]
                data[ticker] = df
            
            # Create backtest engine for this period
            engine = PortfolioBacktestingEngine(
                data=data,
                backtest_start=period_config['backtest_start'],
                backtest_end=period_config['backtest_end'],
                initial_capital=engine_config['initial_capital'],
                pset=None
            )
            engine.pset = pset
            
            # Run backtest
            result = engine.backtest(individual_tree)
            
            # Calculate fitness
            fitness_metric = fitness_config.get('function', 'total_return')
            if fitness_metric == 'consistent_sharpe':
                fitness_params = fitness_config.get('parameters', {})
                fitness_value = engine.calculate_consistent_sharpe(
                    result['equity_curve'], **fitness_params)
            elif fitness_metric == 'sterling_ratio':
                from gp_quant.backtesting.metrics import PortfolioMetrics
                fitness_value = PortfolioMetrics.calculate_sterling_ratio(
                    result['metrics']['total_return'],
                    result['metrics']['max_drawdown']
                )
            else:
                fitness_value = result['metrics'].get(fitness_metric, -100000.0)
            
            results[period] = fitness_value
        
        return individual_id, results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return individual_data[0], {'train': -100000.0, 'validate': -100000.0}


class TrainValidateTestEvaluator(PortfolioFitnessEvaluator):
    """
    Train-Validate-Test Evaluator.
    
    Uses Train set for fitness calculation (evolution selection).
    Uses Validate set for model selection (best individual tracking).
    Uses Test set for final out-of-sample evaluation.
    """
    
    def __init__(self, max_processors: int = 1, cache_enabled: bool = True, **kwargs):
        super().__init__(max_processors=max_processors, cache_enabled=cache_enabled, **kwargs)
        self.validate_engine = None
        self.validate_cache = {}  # Separate cache for validate fitness
        
        logger.info(f"TVT Evaluator Initialized: max_processors={max_processors}")
    
    def set_data(self, data: Dict[str, Any]):
        """
        Set data and initialize engines for train, validate, and test periods.
        """
        super().set_data(data)
        
        # Initialize backtest_engine for SaveHandler (uses Train period)
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
            logger.info("TVT: Backtest engine initialized for SaveHandler")
            
        except Exception as e:
            logger.error(f"TVT: Failed to initialize backtest engine: {e}")
        
        # Check if validate data exists
        validate_data = data.get('validate_data')
        if validate_data is None:
            logger.warning("TVT Evaluator: No validate_data found, falling back to train-only mode")
            return
        
        # Initialize validate engine
        try:
            from ..gp import pset
            
            processed_data = {}
            for ticker, ticker_data in validate_data.items():
                if isinstance(ticker_data, dict) and 'data' in ticker_data:
                    processed_data[ticker] = ticker_data['data']
                else:
                    processed_data[ticker] = ticker_data
            
            # Get validate backtest dates from first ticker
            first_ticker = next(iter(validate_data.values()))
            if isinstance(first_ticker, dict):
                validate_start = first_ticker.get('backtest_start')
                validate_end = first_ticker.get('backtest_end')
            else:
                # Fallback: use engine config
                validate_start = self.engine.config['data'].get('validate_backtest_start')
                validate_end = self.engine.config['data'].get('validate_backtest_end')
            
            if validate_start and validate_end:
                self.validate_engine = PortfolioBacktestingEngine(
                    data=processed_data,
                    backtest_start=validate_start,
                    backtest_end=validate_end,
                    initial_capital=100000.0,
                    pset=pset
                )
                logger.info(f"TVT: Validate engine initialized ({validate_start} to {validate_end})")
            
        except Exception as e:
            logger.error(f"TVT: Failed to initialize validate engine: {e}")
    
    def evaluate_population(self, population: List, data: Dict[str, Any]):
        """
        Evaluate population on both Train and Validate sets.
        """
        # Filter individuals that need evaluation
        individuals_to_evaluate = [
            ind for ind in population 
            if ind.fitness is None or not hasattr(ind.fitness, 'values') or not ind.fitness.values
        ]
        
        if not individuals_to_evaluate:
            logger.debug("All individuals already have fitness, skipping evaluation")
            return
        
        logger.info(f"TVT: Evaluating {len(individuals_to_evaluate)}/{len(population)} individuals")
        
        # Check if we have validate data
        has_validate = self.validate_engine is not None
        
        if len(individuals_to_evaluate) == 1 or self.max_processors == 1:
            self._evaluate_sequential(individuals_to_evaluate, data, has_validate)
        else:
            self._evaluate_parallel(individuals_to_evaluate, data, has_validate)
    
    def _evaluate_sequential(self, individuals: List, data: Dict[str, Any], has_validate: bool = True):
        """Sequential evaluation for both train and validate."""
        from ..gp import pset
        
        # Use parent's backtest engine for train
        if self.backtest_engine is None:
            processed_data = {}
            for ticker, ticker_data in data['train_data'].items():
                if isinstance(ticker_data, dict) and 'data' in ticker_data:
                    processed_data[ticker] = ticker_data['data']
                else:
                    processed_data[ticker] = ticker_data
            
            self.backtest_engine = PortfolioBacktestingEngine(
                data=processed_data,
                backtest_start=self.engine.config['data']['train_backtest_start'],
                backtest_end=self.engine.config['data']['train_backtest_end'],
                initial_capital=100000.0,
                pset=pset
            )
        
        fitness_metric = self.engine.config['fitness']['function']
        
        for individual in tqdm(individuals, desc="TVT Sequential", unit="ind"):
            try:
                # Evaluate on Train
                train_result = self.backtest_engine.backtest(individual)
                
                if fitness_metric == 'consistent_sharpe':
                    fitness_params = self.engine.config['fitness'].get('parameters', {})
                    train_fitness = self.backtest_engine.calculate_consistent_sharpe(
                        train_result['equity_curve'], **fitness_params)
                elif fitness_metric == 'sterling_ratio':
                    from gp_quant.backtesting.metrics import PortfolioMetrics
                    train_fitness = PortfolioMetrics.calculate_sterling_ratio(
                        train_result['metrics']['total_return'],
                        train_result['metrics']['max_drawdown']
                    )
                else:
                    train_fitness = train_result['metrics'].get(fitness_metric, -100000.0)
                
                # Set train fitness as the main fitness
                individual.fitness.values = (train_fitness,)
                
                # Store PnL curve for niche selection
                equity_curve = train_result['equity_curve']
                pnl_curve = equity_curve - self.backtest_engine.initial_capital
                individual.add_metadata('pnl_curve', pnl_curve)
                individual.add_metadata('equity_curve', equity_curve)
                individual.add_metadata('train_fitness', train_fitness)
                
                # Evaluate on Validate if available
                if has_validate and self.validate_engine is not None:
                    validate_result = self.validate_engine.backtest(individual)
                    
                    if fitness_metric == 'consistent_sharpe':
                        validate_fitness = self.validate_engine.calculate_consistent_sharpe(
                            validate_result['equity_curve'], **fitness_params)
                    elif fitness_metric == 'sterling_ratio':
                        from gp_quant.backtesting.metrics import PortfolioMetrics
                        validate_fitness = PortfolioMetrics.calculate_sterling_ratio(
                            validate_result['metrics']['total_return'],
                            validate_result['metrics']['max_drawdown']
                        )
                    else:
                        validate_fitness = validate_result['metrics'].get(fitness_metric, -100000.0)
                    
                    individual.add_metadata('validate_fitness', validate_fitness)
                    
                    # Cache both
                    if self.cache_enabled:
                        self.fitness_cache[individual.id] = train_fitness
                        self.validate_cache[individual.id] = validate_fitness
                else:
                    if self.cache_enabled:
                        self.fitness_cache[individual.id] = train_fitness
                
            except Exception as e:
                logger.error(f"TVT: Error evaluating {individual.id}: {e}")
                individual.fitness.values = (-100000.0,)
    
    def _evaluate_parallel(self, individuals: List, data: Dict[str, Any], has_validate: bool = True):
        """Parallel evaluation for both train and validate."""
        try:
            import sys
            
            # Prepare data paths
            train_data = data['train_data']
            data_paths = {}
            for ticker in train_data.keys():
                data_paths[ticker] = f"{ticker}.csv"
            
            # Build engine config
            engine_config = {
                'tickers_dir': self.engine.config['data']['tickers_dir'],
                'initial_capital': 100000.0,
                'train': {
                    'data_paths': data_paths,
                    'data_start': self.engine.config['data']['train_data_start'],
                    'backtest_start': self.engine.config['data']['train_backtest_start'],
                    'backtest_end': self.engine.config['data']['train_backtest_end']
                }
            }
            
            # Add validate config if available
            if has_validate:
                engine_config['validate'] = {
                    'data_paths': data_paths,
                    'data_start': self.engine.config['data'].get('validate_data_start'),
                    'backtest_start': self.engine.config['data'].get('validate_backtest_start'),
                    'backtest_end': self.engine.config['data'].get('validate_backtest_end')
                }
            
            fitness_config = {
                'function': self.engine.config['fitness']['function'],
                'parameters': self.engine.config['fitness'].get('parameters', {})
            }
            
            individual_data = [(ind.id, ind) for ind in individuals]
            n_workers = min(self.max_processors, len(individuals))
            
            sys.stderr.write(f"DEBUG: TVT Parallel with {n_workers} workers\n")
            sys.stderr.flush()
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                worker_func = partial(_evaluate_tvt_worker,
                                     engine_config=engine_config,
                                     fitness_config=fitness_config)
                
                future_to_individual = {
                    executor.submit(worker_func, ind_data): ind_data[0]
                    for ind_data in individual_data
                }
                
                results = {}
                with tqdm(total=len(individuals), desc="TVT Parallel", unit="ind") as pbar:
                    for future in as_completed(future_to_individual):
                        try:
                            individual_id, fitness_dict = future.result()
                            results[individual_id] = fitness_dict
                            
                            if self.cache_enabled:
                                self.fitness_cache[individual_id] = fitness_dict.get('train', -100000.0)
                                if 'validate' in fitness_dict:
                                    self.validate_cache[individual_id] = fitness_dict['validate']
                            
                            pbar.update(1)
                            
                        except Exception as e:
                            individual_id = future_to_individual[future]
                            logger.error(f"TVT Parallel: Worker failed for {individual_id}: {e}")
                            results[individual_id] = {'train': -100000.0, 'validate': -100000.0}
                            pbar.update(1)
            
            # Assign fitness to individuals
            for individual in individuals:
                if individual.id in results:
                    fitness_dict = results[individual.id]
                    train_fitness = fitness_dict.get('train', -100000.0)
                    individual.fitness.values = (train_fitness,)
                    individual.add_metadata('train_fitness', train_fitness)
                    
                    if 'validate' in fitness_dict:
                        individual.add_metadata('validate_fitness', fitness_dict['validate'])
                else:
                    individual.fitness.values = (-100000.0,)
            
            logger.info(f"TVT Parallel: Completed {len(results)} individuals")
            
        except BaseException as e:
            import sys
            import traceback
            sys.stderr.write(f"CRITICAL: TVT Parallel failed: {e}\n")
            sys.stderr.flush()
            traceback.print_exc()
            logger.error(f"TVT Parallel failed: {e}")
            self._evaluate_sequential(individuals, data, has_validate)
    
    def get_validate_fitness(self, individual) -> Optional[float]:
        """Get validate fitness for an individual."""
        if hasattr(individual, 'metadata'):
            return individual.get_metadata('validate_fitness')
        return None
