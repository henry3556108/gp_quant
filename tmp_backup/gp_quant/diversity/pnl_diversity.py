"""
PnL Correlation-based Diversity Metrics

Calculates population diversity based on the correlation of PnL curves between individuals.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from deap import gp

from gp_quant.backtesting.engine import BacktestingEngine


class PnLDiversityMetrics:
    """Calculate diversity metrics based on PnL curve correlations."""
    
    @staticmethod
    def calculate_pnl_correlation_diversity(
        population: List[Any],
        data: pd.DataFrame,
        backtest_start: str = None,
        backtest_end: str = None,
        initial_capital: float = 100000.0,
        sample_size: int = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Calculate diversity based on PnL curve correlations between individuals.
        
        Args:
            population: List of DEAP individuals
            data: Market data DataFrame for backtesting
            backtest_start: Start date for backtest period
            backtest_end: End date for backtest period
            initial_capital: Initial capital for simulation
            sample_size: If provided, randomly sample this many individuals (for performance)
            verbose: Print progress information
            
        Returns:
            Dictionary containing:
                - pnl_corr_mean: Mean correlation coefficient
                - pnl_corr_std: Standard deviation of correlations
                - pnl_corr_min: Minimum correlation
                - pnl_corr_max: Maximum correlation
                - pnl_corr_median: Median correlation
                - valid_individuals: Number of individuals with valid PnL curves
                - total_pairs: Number of correlation pairs calculated
        """
        # Sample population if requested
        if sample_size and sample_size < len(population):
            import random
            sampled_population = random.sample(population, sample_size)
            if verbose:
                print(f"  Sampling {sample_size} individuals from population of {len(population)}")
        else:
            sampled_population = population
        
        # Initialize backtesting engine
        engine = BacktestingEngine(
            data=data,
            initial_capital=initial_capital,
            backtest_start=backtest_start,
            backtest_end=backtest_end
        )
        
        # Generate PnL curves for all individuals
        pnl_curves = []
        valid_indices = []
        
        if verbose:
            print(f"  Generating PnL curves for {len(sampled_population)} individuals...")
        
        for idx, individual in enumerate(sampled_population):
            try:
                pnl_curve = engine.get_pnl_curve(individual)
                
                # Check if PnL curve is valid (not empty and has variance)
                if len(pnl_curve) > 0 and pnl_curve.std() > 0:
                    pnl_curves.append(pnl_curve.values)
                    valid_indices.append(idx)
                    
            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed to generate PnL for individual {idx}: {e}")
                continue
        
        if verbose:
            print(f"  Valid PnL curves: {len(pnl_curves)} / {len(sampled_population)}")
        
        # Need at least 2 valid curves to calculate correlation
        if len(pnl_curves) < 2:
            return {
                'pnl_corr_mean': 0.0,
                'pnl_corr_std': 0.0,
                'pnl_corr_min': 0.0,
                'pnl_corr_max': 0.0,
                'pnl_corr_median': 0.0,
                'valid_individuals': len(pnl_curves),
                'total_pairs': 0
            }
        
        # Convert to numpy array for efficient correlation calculation
        pnl_matrix = np.array(pnl_curves)
        
        # Calculate correlation matrix
        if verbose:
            print(f"  Calculating correlation matrix ({len(pnl_curves)} x {len(pnl_curves)})...")
        
        try:
            corr_matrix = np.corrcoef(pnl_matrix)
            
            # Check if correlation matrix is valid
            if corr_matrix.shape[0] < 2:
                if verbose:
                    print(f"  Warning: Correlation matrix too small: {corr_matrix.shape}")
                correlations = np.array([])
            else:
                # Extract upper triangle (excluding diagonal)
                # This gives us all unique pairs
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix[upper_tri_indices]
                
                # Remove any NaN or inf values
                correlations = correlations[np.isfinite(correlations)]
                
                if verbose:
                    print(f"  Valid correlations: {len(correlations)}")
                    if len(correlations) > 0:
                        print(f"  Correlation range: [{np.min(correlations):.4f}, {np.max(correlations):.4f}]")
        except Exception as e:
            if verbose:
                print(f"  Error calculating correlation matrix: {e}")
            correlations = np.array([])
        
        if len(correlations) == 0:
            return {
                'pnl_corr_mean': 0.0,
                'pnl_corr_std': 0.0,
                'pnl_corr_min': 0.0,
                'pnl_corr_max': 0.0,
                'pnl_corr_median': 0.0,
                'valid_individuals': len(pnl_curves),
                'total_pairs': 0
            }
        
        # Calculate statistics
        return {
            'pnl_corr_mean': float(np.mean(correlations)),
            'pnl_corr_std': float(np.std(correlations)),
            'pnl_corr_min': float(np.min(correlations)),
            'pnl_corr_max': float(np.max(correlations)),
            'pnl_corr_median': float(np.median(correlations)),
            'valid_individuals': len(pnl_curves),
            'total_pairs': len(correlations)
        }
