"""
Diversity Analyzer

Main class for analyzing population diversity across generations.
"""

import os
import dill
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from deap import creator, base, gp

from .metrics import DiversityMetrics
from .pnl_diversity import PnLDiversityMetrics


class DiversityAnalyzer:
    """Analyze population diversity across generations from saved population records."""
    
    def __init__(self, records_dir: str):
        """
        Initialize the diversity analyzer.
        
        Args:
            records_dir: Path to the individual_records directory
                        (e.g., "experiments_results/ABX_TO/individual_records_long_run01")
        """
        self.records_dir = Path(records_dir)
        if not self.records_dir.exists():
            raise ValueError(f"Records directory does not exist: {records_dir}")
        
        self.populations = {}
        self.diversity_data = None
        self._setup_deap_creator()
    
    def _setup_deap_creator(self):
        """Setup DEAP creator classes (required for unpickling)."""
        # Check if already created
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    def load_populations(self, verbose: bool = True) -> Dict[int, List[Any]]:
        """
        Load all populations from the records directory.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary mapping generation number to population list
        """
        generation_dirs = sorted([
            d for d in self.records_dir.iterdir() 
            if d.is_dir() and d.name.startswith('generation_')
        ])
        
        if verbose:
            print(f"Found {len(generation_dirs)} generations in {self.records_dir}")
        
        for gen_dir in generation_dirs:
            # Extract generation number
            gen_num = int(gen_dir.name.split('_')[1])
            
            # Load population
            pop_file = gen_dir / 'population.pkl'
            if pop_file.exists():
                try:
                    with open(pop_file, 'rb') as f:
                        population = dill.load(f)
                    self.populations[gen_num] = population
                    
                    if verbose and gen_num % 10 == 0:
                        print(f"  Loaded generation {gen_num}: {len(population)} individuals")
                        
                except Exception as e:
                    print(f"  Warning: Failed to load generation {gen_num}: {e}")
            else:
                print(f"  Warning: No population.pkl found in {gen_dir}")
        
        if verbose:
            print(f"Successfully loaded {len(self.populations)} generations")
        
        return self.populations
    
    def calculate_diversity_trends(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate diversity metrics for all loaded generations.
        
        Args:
            metrics: List of metric categories to calculate. 
                    Options: ['structural', 'genotypic', 'fitness', 'phenotypic']
                    If None, calculates all metrics.
        
        Returns:
            DataFrame with diversity metrics across generations
        """
        if not self.populations:
            raise ValueError("No populations loaded. Call load_populations() first.")
        
        results = []
        
        for gen_num in sorted(self.populations.keys()):
            population = self.populations[gen_num]
            
            # Calculate all metrics
            all_metrics = DiversityMetrics.calculate_all(population)
            
            # Flatten the nested dictionary
            row = {'generation': gen_num}
            
            for category, metrics_dict in all_metrics.items():
                if metrics is None or category in metrics:
                    for metric_name, value in metrics_dict.items():
                        # Skip non-numeric values (like usage dictionaries)
                        if isinstance(value, (int, float)):
                            row[f'{category}_{metric_name}'] = value
            
            results.append(row)
        
        self.diversity_data = pd.DataFrame(results)
        return self.diversity_data
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of diversity trends.
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.diversity_data is None:
            raise ValueError("No diversity data calculated. Call calculate_diversity_trends() first.")
        
        summary = {
            'total_generations': len(self.diversity_data),
            'metrics': {}
        }
        
        # Calculate trends for key metrics
        for col in self.diversity_data.columns:
            if col != 'generation':
                values = self.diversity_data[col]
                summary['metrics'][col] = {
                    'initial': float(values.iloc[0]),
                    'final': float(values.iloc[-1]),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'trend': 'decreasing' if values.iloc[-1] < values.iloc[0] else 'increasing'
                }
        
        return summary
    
    def save_results(self, output_path: str):
        """
        Save diversity analysis results to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        if self.diversity_data is None:
            raise ValueError("No diversity data calculated. Call calculate_diversity_trends() first.")
        
        self.diversity_data.to_csv(output_path, index=False)
        print(f"Diversity data saved to: {output_path}")
    
    def calculate_pnl_diversity_trends(
        self,
        data: pd.DataFrame,
        backtest_start: str = None,
        backtest_end: str = None,
        initial_capital: float = 100000.0,
        sample_size: int = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Calculate PnL correlation-based diversity for all loaded generations.
        
        Args:
            data: Market data DataFrame for backtesting
            backtest_start: Start date for backtest period
            backtest_end: End date for backtest period
            initial_capital: Initial capital for simulation
            sample_size: If provided, randomly sample this many individuals per generation
            verbose: Print progress information
        
        Returns:
            DataFrame with PnL diversity metrics across generations
        """
        if not self.populations:
            raise ValueError("No populations loaded. Call load_populations() first.")
        
        results = []
        
        for gen_num in sorted(self.populations.keys()):
            if verbose:
                print(f"Calculating PnL diversity for generation {gen_num}...")
            
            population = self.populations[gen_num]
            
            # Calculate PnL correlation diversity
            metrics = PnLDiversityMetrics.calculate_pnl_correlation_diversity(
                population=population,
                data=data,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                initial_capital=initial_capital,
                sample_size=sample_size,
                verbose=verbose
            )
            
            # Add generation number
            row = {'generation': gen_num, **metrics}
            results.append(row)
        
        pnl_diversity_data = pd.DataFrame(results)
        
        if verbose:
            print(f"\nPnL diversity calculation complete for {len(results)} generations")
        
        return pnl_diversity_data
    
    @classmethod
    def from_experiment_result(cls, experiment_dir: str, period: str, run_number: int):
        """
        Create analyzer from experiment result directory.
        
        Args:
            experiment_dir: Base experiment directory (e.g., "experiments_results/ABX_TO")
            period: 'short' or 'long'
            run_number: Run number (1-10)
            
        Returns:
            DiversityAnalyzer instance
        """
        records_dir = f"{experiment_dir}/individual_records_{period}_run{run_number:02d}"
        return cls(records_dir)
