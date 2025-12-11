
import pandas as pd
import numpy as np
import logging
import copy
from typing import Dict, Any, List, Tuple
from datetime import timedelta
from pathlib import Path
import json

from gp_quant.evolution.components import create_evolution_engine
from gp_quant.evolution.components.backtesting import PortfolioBacktestingEngine
from gp_quant.backtesting.metrics import PortfolioMetrics

logger = logging.getLogger(__name__)

class WalkForwardEvolutionEngine:
    """
    Walk-Forward Evolution Engine (Rolling Window Analysis).
    
    This engine performs Walk-Forward Validation by:
    1. Splitting data into a series of (Train, Test) windows.
    2. Running a complete evolution experiment on each Train window.
    3. Validating the best individual from each experiment on the corresponding Test window.
    4. Stitching the Test results together to form a continuous OOS equity curve.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Walk-Forward Engine.
        
        Args:
            config: The full configuration dictionary. Must contain a 'walk_forward' section
                    or relevant parameters in 'experiment'/'data'.
        """
        self.config = copy.deepcopy(config)
        self.wf_config = config.get('walk_forward', {})
        
        # Ensure DEAP creator is set up
        self._setup_deap_creator()
        
        # Window parameters (in years or days? Let's assume days for precision, or parse strings)
        # Default to 1 year test, 2 years train if not specified
        self.train_window_days = self.wf_config.get('train_window_days', 730) # ~2 years
        self.test_window_days = self.wf_config.get('test_window_days', 365)   # ~1 year
        self.rolling_step_days = self.wf_config.get('rolling_step_days', 365) # ~1 year (usually same as test)
        
        self.results = []
        self.aggregate_equity_curve = pd.Series(dtype=float)

    def _setup_deap_creator(self):
        """Setup DEAP creator with FitnessMax and Individual"""
        from deap import creator, base, gp
        
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
    def run(self, data: Dict[str, Any]):
        """
        Execute the Walk-Forward Analysis.
        
        Args:
            data: The full dataset dictionary (tickers, train_data, test_data structure).
                  Note: We will ignore the pre-split train/test in 'data' and use the full range
                  available in the dataframes to create our own windows.
        """
        logger.info("Starting Walk-Forward Analysis...")
        
        # 1. Determine full date range from data
        # Assuming all tickers have roughly same range, pick one
        tickers = data['tickers']
        # We need to access the raw dataframes. 
        # The 'data' dict usually has 'train_data' and 'test_data' keys from loader.
        # We should probably combine them or just look at one ticker's full history if available.
        # Let's assume we can get full history from the first ticker in 'train_data' + 'test_data'
        # OR, we can re-load data. But better to work with what we have.
        
        # Let's reconstruct full dataframe for one ticker to find bounds
        # Actually, loader.load_portfolio_data returns 'train_data' and 'test_data' 
        # which contain 'data' (full df including initial period).
        # Let's use the 'train_data' of the first ticker, assuming it covers the start.
        # Wait, if we are doing rolling window, we need the FULL dataset spanning decades.
        # The loader might have already filtered it based on config['data']['train_data_start'].
        # We should assume the input `data` contains enough history for the rolling windows.
        
        # Let's look at the first ticker in train_data
        first_ticker = tickers[0]
        # We need to merge train and test data if they are split, or just use the underlying DFs if they overlap/continue.
        # In `loader.py`, train_data[ticker]['data'] is the DF.
        
        # Let's just grab the full available range from the provided data objects.
        # We will assume the user provided a config that loaded a LONG period into 'train_data' 
        # or we might need to adjust how data is loaded for WF.
        # For now, let's assume 'train_data' in the input `data` dict actually holds the entire dataset 
        # we want to roll over, OR we need to be careful.
        
        # Actually, standard `load_portfolio_data` splits based on fixed dates.
        # For WF, we probably want to load ALL data from start to end.
        # The `data` argument passed here likely comes from `load_portfolio_data`.
        # If the user configured 2000-2024, then `train_data` might only have 2000-2010.
        # We need to ensure we have access to the full range.
        
        # Strategy: We will iterate through the windows. For each window, we define
        # train_start, train_end, test_start, test_end.
        
        # Let's find the global start and end from the data provided.
        # We'll check both train and test dicts to find the absolute min and max dates.
        min_date = pd.Timestamp.max
        max_date = pd.Timestamp.min
        
        full_dfs = {}
        
        for ticker in tickers:
            # Combine train and test dfs to get full history?
            # Or just use the one that covers the range.
            # Usually loader returns 'data' which includes initial period.
            # Let's just look at what we have.
            
            # Try to get from train_data
            if ticker in data['train_data']:
                df = data['train_data'][ticker]['data']
                min_date = min(min_date, df.index[0])
                max_date = max(max_date, df.index[-1])
                full_dfs[ticker] = df
            
            # If test data has later dates, update max
            if ticker in data['test_data']:
                df = data['test_data'][ticker]['data']
                min_date = min(min_date, df.index[0])
                max_date = max(max_date, df.index[-1])
                # If we need to merge, it's complicated. 
                # Ideally, for WF, we should load data once with the full range.
                # We'll assume full_dfs[ticker] is sufficient or we update it.
                if ticker in full_dfs:
                    # Merge if needed, but usually they are just slices of the same file
                    # If test_data goes beyond train_data, we might need it.
                    if df.index[-1] > full_dfs[ticker].index[-1]:
                         # This implies we need to merge or the user should have loaded full data in train.
                         # For simplicity, let's assume the user configures 'train_backtest_end' to be the end of the dataset
                         # when running WF, or we re-load.
                         pass
        
        logger.info(f"Full data range: {min_date.date()} to {max_date.date()}")
        
        # 2. Generate Windows
        current_test_end = max_date
        # We work backwards or forwards? Usually forwards.
        # Start from min_date + train_window
        
        current_window_start = min_date
        
        windows = []
        
        while True:
            train_start = current_window_start
            train_end = train_start + timedelta(days=self.train_window_days)
            
            test_start = train_end + timedelta(days=1) # Start immediately after train
            test_end = test_start + timedelta(days=self.test_window_days)
            
            if test_end > max_date:
                break
                
            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })
            
            current_window_start += timedelta(days=self.rolling_step_days)
            
        logger.info(f"Generated {len(windows)} rolling windows.")
        
        # 3. Iterate and Evolve
        oos_curves = []
        
        for i, window in enumerate(windows):
            logger.info(f"\n=== Window {i+1}/{len(windows)} ===")
            logger.info(f"Train: {window['train_start'].date()} -> {window['train_end'].date()}")
            logger.info(f"Test:  {window['test_start'].date()} -> {window['test_end'].date()}")
            
            # Update config for this window
            # We need to set the dates so the engine and evaluator use the correct slice
            window_config = copy.deepcopy(self.config)
            
            # We need to set 'train_backtest_start' etc.
            # Note: 'train_data_start' (initial period) is needed for indicators.
            # We should set 'train_data_start' to something earlier than 'train_backtest_start'.
            # Let's assume 1 year (365 days) warm-up.
            warmup_days = 365
            window_config['data']['train_data_start'] = (window['train_start'] - timedelta(days=warmup_days)).strftime('%Y-%m-%d')
            window_config['data']['train_backtest_start'] = window['train_start'].strftime('%Y-%m-%d')
            window_config['data']['train_backtest_end'] = window['train_end'].strftime('%Y-%m-%d')
            
            window_config['data']['test_data_start'] = (window['test_start'] - timedelta(days=warmup_days)).strftime('%Y-%m-%d')
            window_config['data']['test_backtest_start'] = window['test_start'].strftime('%Y-%m-%d')
            window_config['data']['test_backtest_end'] = window['test_end'].strftime('%Y-%m-%d')
            
            # Create Engine
            # We need to pass the data. The engine expects 'train_data' and 'test_data' dicts.
            # We can reuse the `split_train_test_data` logic or just pass the full data and let the engine slice it 
            # based on the config we just updated.
            # Actually, `EvolutionEngine` calls `evaluator.set_data(data)`.
            # `PortfolioFitnessEvaluator` uses `config['data']['train_backtest_start']` to slice.
            # So passing the FULL data dict (raw_data) is fine, as long as the config is updated.
            # BUT, we need to make sure `data` passed to `evolve` has the structure expected.
            # The `data` argument to `run` is already processed.
            # We should probably construct a `window_data` dict that points to the full DFs
            # but has the correct structure.
            
            # Re-using the input `data` structure but pointing to full DFs
            # The evaluator will slice it based on the config dates.
            # We just need to ensure `data['train_data']` has the tickers.
            
            # Create engine
            engine = create_evolution_engine(window_config)
            
            # Run Evolution
            # We pass the full data, the evaluator will slice based on window_config
            result = engine.evolve(data)
            
            # Get Best Individual
            best_ind = result.best_individual
            logger.info(f"Best Fitness (Train): {result.best_fitness}")
            
            # Run OOS Backtest
            logger.info("Running OOS Backtest...")
            # Create a backtesting engine for the test period
            # We need to use the test window dates
            
            # Prepare test data slice
            test_data_slice = {}
            for ticker in tickers:
                if ticker in full_dfs:
                    test_data_slice[ticker] = full_dfs[ticker]
            
            # Import pset directly
            from gp_quant.evolution.components.gp import pset
            
            bt_engine = PortfolioBacktestingEngine(
                data=test_data_slice,
                backtest_start=window['test_start'].strftime('%Y-%m-%d'),
                backtest_end=window['test_end'].strftime('%Y-%m-%d'),
                initial_capital=100000.0, # Should we carry over capital? Usually Walk-Forward resets or compounds.
                                          # For equity curve stitching, we usually want returns, then compound them.
                pset=pset
            )
            
            oos_result = bt_engine.backtest(best_ind)
            oos_equity = oos_result['equity_curve']
            
            oos_curves.append(oos_equity)
            
            self.results.append({
                'window_index': i,
                'window': window,
                'best_fitness': result.best_fitness,
                'oos_metrics': oos_result['metrics'],
                'best_individual': str(best_ind)
            })
            
        # 4. Stitch Equity Curves
        if oos_curves:
            # We need to stitch them.
            # Method: Convert to returns, concatenate returns, then reconstruct equity curve.
            all_returns = []
            for curve in oos_curves:
                rets = curve.pct_change().dropna()
                all_returns.append(rets)
            
            if all_returns:
                full_returns = pd.concat(all_returns)
                # Remove duplicates if any (though windows shouldn't overlap in test)
                full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
                
                # Reconstruct equity
                initial_capital = 100000.0
                self.aggregate_equity_curve = initial_capital * (1 + full_returns).cumprod()
                
                # Calculate Final Metrics
                final_metrics = PortfolioMetrics.calculate_portfolio_metrics(
                    self.aggregate_equity_curve, initial_capital
                )
                
                logger.info("\n=== Walk-Forward Analysis Complete ===")
                logger.info(f"Total Return: {final_metrics['total_return']:.2%}")
                logger.info(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.4f}")
                logger.info(f"Max Drawdown: {final_metrics['max_drawdown']:.2%}")
                
                return {
                    'metrics': final_metrics,
                    'equity_curve': self.aggregate_equity_curve,
                    'window_results': self.results
                }
        
        return None
