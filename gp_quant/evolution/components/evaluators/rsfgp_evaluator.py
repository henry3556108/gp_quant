"""
RSFGP Portfolio Evaluator
"""

import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import timedelta

from .base import FitnessEvaluator
from .portfolio_evaluator import PortfolioFitnessEvaluator
from ..backtesting import PortfolioBacktestingEngine
from ....backtesting.metrics import PortfolioMetrics

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

logger = logging.getLogger(__name__)


def soft_min_score(values: List, alpha: float = 1.0) -> float:
    """
    Compute Negative Exponential Sum (Soft-Min) score.
    
    Formula: S(x) = -ln(sum(e^(-alpha * x_i)))
    
    This function heavily penalizes low values:
    - High values: e^(-alpha * x) -> 0 (small penalty)
    - Low/negative values: e^(-alpha * x) -> large (huge penalty)
    
    Args:
        values: List of fitness scores.
        alpha: Sensitivity parameter. Higher alpha = more aggressive penalty.
    
    Returns:
        Aggregated score (higher is better).
    """
    if not values:
        return -100000.0
    
    arr = np.array(values)
    
    # Numerical stability: shift values to avoid overflow
    shifted = -alpha * arr
    max_shifted = np.max(shifted)
    
    # Compute log-sum-exp for numerical stability
    exp_sum = np.sum(np.exp(shifted - max_shifted))
    log_penalty = max_shifted + np.log(exp_sum)
    
    return -log_penalty

def setup_deap_creator():
    """初始化 DEAP creator (在子進程中需要重新初始化)"""
    from deap import creator, base, gp
    
    # 檢查是否已經創建
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def _evaluate_rsfgp_worker(individual_data: tuple, engine_config: Dict[str, Any], fitness_config: Dict[str, Any]) -> tuple:
    """
    工作進程函數：評估單個個體 (RSFGP 模式)
    """
    # 確保 DEAP creator 已初始化
    setup_deap_creator()
    
    try:
        individual_id, individual_tree = individual_data
        
        # Load data
        import pandas as pd
        from pathlib import Path
        from datetime import timedelta
        import random
        
        data = {}
        tickers_dir = engine_config.get('tickers_dir', 'TSE300_selected')
        
        for ticker, csv_path in engine_config['data_paths'].items():
            if not Path(csv_path).is_absolute():
                csv_path = Path(tickers_dir) / f"{ticker}.csv"
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            data[ticker] = df
            
        # RSFGP Logic
        n_segments = fitness_config.get('n_segments', 100)
        segment_length = fitness_config.get('segment_length', 255)
        no_trade_penalty = fitness_config.get('no_trade_penalty', fitness_config.get('penalty_value', -10.0))
        fitness_metric = fitness_config.get('function', 'sterling_ratio')
        aggregation = fitness_config.get('aggregation', 'mean')
        soft_min_alpha = fitness_config.get('soft_min_alpha', 1.0)
        
        # Get data range
        first_ticker = list(data.keys())[0]
        df = data[first_ticker]
        full_start_date = df.index[0]
        full_end_date = df.index[-1]
        
        total_days = (full_end_date - full_start_date).days
        if total_days < segment_length:
            return individual_id, penalty_value

        fitness_scores = []
        
        from ..backtesting import PortfolioBacktestingEngine
        from ....backtesting.metrics import PortfolioMetrics
        
        for _ in range(n_segments):
            max_start_ordinal = full_end_date.toordinal() - segment_length
            min_start_ordinal = full_start_date.toordinal()
            
            random_start_ordinal = random.randint(min_start_ordinal, max_start_ordinal)
            segment_start = pd.Timestamp.fromordinal(random_start_ordinal)
            segment_end = segment_start + timedelta(days=segment_length)
            
            engine = PortfolioBacktestingEngine(
                data=data,
                backtest_start=segment_start.strftime('%Y-%m-%d'),
                backtest_end=segment_end.strftime('%Y-%m-%d'),
                initial_capital=engine_config['initial_capital'],
                pset=None
            )
            
            # Import pset inside worker to avoid pickling issues
            from ..gp import pset
            engine.pset = pset
            
            result = engine.backtest(individual_tree)
            
            if not result['transactions'].empty:
                if fitness_metric == 'sterling_ratio':
                     score = PortfolioMetrics.calculate_sterling_ratio(
                         result['metrics']['total_return'],
                         result['metrics']['max_drawdown']
                     )
                elif fitness_metric == 'consistent_sharpe':
                    score = engine.calculate_consistent_sharpe(result['equity_curve'])
                else:
                    score = result['metrics'].get(fitness_metric, -100.0)
            else:
                score = no_trade_penalty
            
            fitness_scores.append(score)
        
        # Aggregate fitness scores
        if aggregation == 'mean':
            final_fitness = sum(fitness_scores) / len(fitness_scores)
        elif aggregation == 'median':
            final_fitness = float(np.median(fitness_scores))
        elif aggregation == 'min':
            final_fitness = min(fitness_scores)
        elif aggregation == 'soft_min':
            final_fitness = soft_min_score(fitness_scores, alpha=soft_min_alpha)
        else:
            final_fitness = sum(fitness_scores) / len(fitness_scores)
        
        return individual_id, final_fitness
        
    except Exception as e:
        # logger.error(f"RSFGP Worker Error: {e}") # Avoid logging in worker if possible or use print
        return individual_data[0], -100000.0

class RSFGPPortfolioEvaluator(PortfolioFitnessEvaluator):
    """
    Random Subset Fitness GP (RSFGP) Evaluator.
    """
    
    def __init__(self, n_segments: int = 100, segment_length: int = 255, 
                 penalty_value: float = -9.99, aggregation: str = 'mean',
                 soft_min_alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_segments = n_segments
        self.segment_length = segment_length
        self.penalty_value = penalty_value
        self.aggregation = aggregation
        self.soft_min_alpha = soft_min_alpha
        
        logger.info(f"RSFGP Evaluator Initialized: n_segments={n_segments}, "
                    f"length={segment_length}, penalty={penalty_value}, "
                    f"aggregation={aggregation}, soft_min_alpha={soft_min_alpha}")

    def set_data(self, data: Dict[str, Any]):
        """
        Set data and initialize the full backtest engine for SaveHandler.
        """
        super().set_data(data)
        
        # Initialize a persistent backtest engine for the full period
        # This is used by SaveHandler to generate best signals
        try:
            from ..gp import pset
            from ..backtesting import PortfolioBacktestingEngine
            
            # Prepare data
            train_data = data['train_data']
            processed_data = {}
            for ticker, ticker_data in train_data.items():
                if isinstance(ticker_data, dict) and 'data' in ticker_data:
                    processed_data[ticker] = ticker_data['data']
                else:
                    processed_data[ticker] = ticker_data
            
            # Get config from engine if available, otherwise use defaults/attributes
            # Note: super().set_data() sets self.engine_config
            
            self.backtest_engine = PortfolioBacktestingEngine(
                data=processed_data,
                backtest_start=self.engine_config['backtest_start'],
                backtest_end=self.engine_config['backtest_end'],
                initial_capital=self.engine_config['initial_capital'],
                pset=pset
            )
            logger.info("RSFGP: Persistent backtest engine initialized for SaveHandler")
            
        except Exception as e:
            logger.error(f"RSFGP: Failed to initialize persistent backtest engine: {e}")

    def _evaluate_parallel(self, individuals: List, data: Dict[str, Any]):
        """Override parallel evaluation to use RSFGP worker"""
        try:
            from ..gp import pset
            
            data_paths = self._get_data_paths(data)
            
            engine_config = {
                'data_paths': data_paths,
                'tickers_dir': self.engine.config['data']['tickers_dir'],
                'initial_capital': 100000.0,
                'initial_capital': 100000.0,
                # 'pset': pset  # Removed to avoid pickling issues
            }
            
            fitness_config = {
                'function': self.engine.config['fitness']['function'],
                'n_segments': self.n_segments,
                'segment_length': self.segment_length,
                'penalty_value': self.penalty_value,
                'aggregation': self.aggregation,
                'soft_min_alpha': self.soft_min_alpha
            }
            
            individual_data = [(ind.id, ind) for ind in individuals]
            n_workers = min(self.max_processors, len(individuals))
            
            logger.info(f"使用 {n_workers} 個進程並行評估")
            import sys
            sys.stderr.write(f"DEBUG: Starting parallel evaluation with {n_workers} workers\n")
            sys.stderr.flush()
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                worker_func = partial(_evaluate_rsfgp_worker, 
                                    engine_config=engine_config, 
                                    fitness_config=fitness_config)
                
                future_to_individual = {
                    executor.submit(worker_func, ind_data): ind_data[0] 
                    for ind_data in individual_data
                }
                
                results = {}
                with tqdm(total=len(individuals), desc="RSFGP Parallel", unit="ind") as pbar:
                    for future in as_completed(future_to_individual):
                        try:
                            individual_id, fitness_value = future.result()
                            results[individual_id] = fitness_value
                            if self.cache_enabled:
                                self.fitness_cache[individual_id] = fitness_value
                            pbar.update(1)
                        except Exception as e:
                            individual_id = future_to_individual[future]
                            logger.error(f"Worker failed for {individual_id}: {e}")
                            results[individual_id] = -100000.0
                            pbar.update(1)
            
            for individual in individuals:
                if individual.id in results:
                    individual.fitness.values = (results[individual.id],)
                else:
                    individual.fitness.values = (-100000.0,)
                    
        except BaseException as e:
            import sys
            sys.stderr.write(f"CRITICAL ERROR: RSFGP Parallel failed: {e}\n")
            sys.stderr.flush()
            import traceback
            traceback.print_exc()
            logger.error(f"RSFGP Parallel failed: {e}")
            self._evaluate_sequential(individuals, data)

    def _evaluate_sequential(self, individuals: List, data: Dict[str, Any]):
        """
        Override sequential evaluation to ensure RSFGP logic is used.
        The base class implementation sets up a single engine for the full period,
        which is incorrect for RSFGP. We must call evaluate_individual for each.
        """
        logger.info("RSFGP Sequential Evaluation")
        for individual in tqdm(individuals, desc="RSFGP Sequential", unit="ind"):
            try:
                fitness_value = self.evaluate_individual(individual, data)
                individual.fitness.values = (fitness_value,)
            except Exception as e:
                logger.error(f"Sequential evaluation failed for {individual.id}: {e}")
                individual.fitness.values = (-100000.0,)

    def evaluate_individual(self, individual, data: Dict[str, Any]) -> float:
        """
        Evaluate individual using RSFGP method.
        
        Args:
            individual: The individual to evaluate.
            data: Data dictionary containing 'train_data'.
            
        Returns:
            Average fitness over random segments.
        """
        # Check cache
        if self.cache_enabled and individual.id in self.fitness_cache:
            return self.fitness_cache[individual.id]
            
        try:
            # Get training data range
            train_data_dict = data['train_data']
            # Assuming all tickers have roughly same date range, pick one to determine bounds
            first_ticker = list(train_data_dict.keys())[0]
            df = train_data_dict[first_ticker]['data']
            
            full_start_date = df.index[0]
            full_end_date = df.index[-1]
            
            # Ensure we have enough data
            total_days = (full_end_date - full_start_date).days
            if total_days < self.segment_length:
                logger.warning(f"Data length ({total_days} days) is shorter than segment length ({self.segment_length}). Using full data.")
                return super().evaluate_individual(individual, data)

            fitness_scores = []
            
            # Use the configured fitness function name
            fitness_metric_name = self.engine.config['fitness']['function']
            
            # Pre-create engine to reuse if possible (though dates change)
            # Actually, PortfolioBacktestingEngine takes backtest_start/end in init, 
            # so we might need to recreate it or make it adjustable.
            # For now, we recreate it to be safe and simple.
            
            from ..gp import pset
            
            for _ in range(self.n_segments):
                # Randomly select start date
                # Valid start range: [full_start, full_end - segment_length]
                max_start_ordinal = full_end_date.toordinal() - self.segment_length
                min_start_ordinal = full_start_date.toordinal()
                
                random_start_ordinal = random.randint(min_start_ordinal, max_start_ordinal)
                segment_start = pd.Timestamp.fromordinal(random_start_ordinal)
                segment_end = segment_start + timedelta(days=self.segment_length)
                
                # Create engine for this segment
                # Note: We pass the FULL data, but specify the backtest range
                engine = PortfolioBacktestingEngine(
                    data={t: d['data'] for t, d in train_data_dict.items()},
                    backtest_start=segment_start.strftime('%Y-%m-%d'),
                    backtest_end=segment_end.strftime('%Y-%m-%d'),
                    initial_capital=100000.0,
                    pset=pset
                )
                
                # Run backtest
                result = engine.backtest(individual)
                
                # Check for trades
                if not result['transactions'].empty:
                    # Calculate specific metric
                    if fitness_metric_name == 'sterling_ratio':
                         # Calculate Sterling Ratio manually if not in standard metrics dict yet
                         # or use the one we added to PortfolioMetrics
                         score = PortfolioMetrics.calculate_sterling_ratio(
                             result['metrics']['total_return'],
                             result['metrics']['max_drawdown']
                         )
                    elif fitness_metric_name == 'consistent_sharpe':
                        score = engine.calculate_consistent_sharpe(result['equity_curve'])
                    else:
                        # Fallback to standard metric
                        score = result['metrics'].get(fitness_metric_name, -100.0)
                else:
                    score = self.penalty_value
                
                fitness_scores.append(score)
            
            # Aggregate fitness based on configured method
            if self.aggregation == 'mean':
                final_fitness = sum(fitness_scores) / len(fitness_scores)
            elif self.aggregation == 'median':
                final_fitness = float(np.median(fitness_scores))
            elif self.aggregation == 'min':
                final_fitness = min(fitness_scores)
            elif self.aggregation == 'soft_min':
                final_fitness = soft_min_score(fitness_scores, alpha=self.soft_min_alpha)
            else:
                final_fitness = sum(fitness_scores) / len(fitness_scores)
            
            # Cache result
            if self.cache_enabled:
                self.fitness_cache[individual.id] = final_fitness
                
            return final_fitness

        except Exception as e:
            logger.error(f"RSFGP Evaluation failed for individual {individual.id}: {e}")
            import traceback
            traceback.print_exc()
            return -100000.0
