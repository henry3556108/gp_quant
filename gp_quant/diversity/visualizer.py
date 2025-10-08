"""
Diversity Visualizer

Provides visualization tools for diversity analysis results.
"""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, List, Tuple
import numpy as np


class DiversityVisualizer:
    """Visualize diversity trends across generations."""
    
    @staticmethod
    def plot_diversity_trends(
        diversity_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot diversity trends across generations.
        
        Args:
            diversity_data: DataFrame from DiversityAnalyzer.calculate_diversity_trends()
            metrics: List of specific metrics to plot. If None, plots one key metric from each category.
            figsize: Figure size (width, height)
            save_path: Path to save the figure. If None, doesn't save.
            show: Whether to display the plot
        """
        if metrics is None:
            # Default: one representative metric from each of the 4 categories
            metrics = [
                'structural_height_std',      # Structural Diversity
                'genotypic_unique_ratio',     # Genotypic Diversity
                'fitness_cv',                 # Fitness Diversity
                'phenotypic_unique_primitives'  # Phenotypic Diversity
            ]
        
        # Filter metrics that exist in the data
        available_metrics = [m for m in metrics if m in diversity_data.columns]
        
        if not available_metrics:
            raise ValueError(f"None of the specified metrics found in data. Available: {list(diversity_data.columns)}")
        
        n_metrics = len(available_metrics)
        
        # Use 2x2 grid for 4 metrics (one from each category)
        if n_metrics == 4:
            n_rows, n_cols = 2, 2
        else:
            n_cols = 2
            n_rows = (n_metrics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        # Category labels for the 4 default metrics
        category_labels = {
            'structural_height_std': 'Structural Diversity\n(Tree Height Std)',
            'genotypic_unique_ratio': 'Genotypic Diversity\n(Unique Ratio)',
            'fitness_cv': 'Fitness Diversity\n(Coefficient of Variation)',
            'phenotypic_unique_primitives': 'Phenotypic Diversity\n(Unique Primitives)'
        }
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Plot the metric
            ax.plot(diversity_data['generation'], diversity_data[metric], 
                   linewidth=2.5, marker='o', markersize=5, alpha=0.8, color='steelblue')
            
            # Formatting
            if metric in category_labels:
                title = category_labels[metric]
            else:
                title = metric.replace('_', ' ').title()
            
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add trend line
            z = np.polyfit(diversity_data['generation'], diversity_data[metric], 1)
            p = np.poly1d(z)
            trend_direction = '↗' if z[0] > 0 else '↘'
            ax.plot(diversity_data['generation'], p(diversity_data['generation']), 
                   "r--", alpha=0.6, linewidth=2, label=f'Trend {trend_direction}')
            ax.legend(fontsize=9, loc='best')
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Population Diversity Analysis: Four Key Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_single_metric(
        diversity_data: pd.DataFrame,
        metric: str,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot a single diversity metric.
        
        Args:
            diversity_data: DataFrame from DiversityAnalyzer
            metric: Name of the metric column to plot
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        if metric not in diversity_data.columns:
            raise ValueError(f"Metric '{metric}' not found in data. Available: {list(diversity_data.columns)}")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the metric
        ax.plot(diversity_data['generation'], diversity_data[metric], 
               linewidth=2.5, marker='o', markersize=6, alpha=0.8, color='steelblue')
        
        # Add trend line
        z = np.polyfit(diversity_data['generation'], diversity_data[metric], 1)
        p = np.poly1d(z)
        ax.plot(diversity_data['generation'], p(diversity_data['generation']), 
               "r--", alpha=0.6, linewidth=2, label=f'Trend (slope={z[0]:.4f})')
        
        # Formatting
        metric_name = metric.replace('_', ' ').title()
        ax.set_title(f'{metric_name} Across Generations', fontsize=14, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_all_categories(
        diversity_data: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot all diversity categories in separate subplots.
        
        Args:
            diversity_data: DataFrame from DiversityAnalyzer
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        # Group metrics by category
        categories = {
            'Structural Diversity': [col for col in diversity_data.columns if col.startswith('structural_')],
            'Genotypic Diversity': [col for col in diversity_data.columns if col.startswith('genotypic_')],
            'Fitness Diversity': [col for col in diversity_data.columns if col.startswith('fitness_')],
            'Phenotypic Diversity': [col for col in diversity_data.columns if col.startswith('phenotypic_')]
        }
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, (category_name, metrics) in enumerate(categories.items()):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            for metric in metrics:
                # Normalize values for comparison
                values = diversity_data[metric]
                if values.max() != values.min():
                    normalized = (values - values.min()) / (values.max() - values.min())
                else:
                    normalized = values
                
                label = metric.replace(category_name.lower().replace(' ', '_') + '_', '').replace('_', ' ').title()
                ax.plot(diversity_data['generation'], normalized, 
                       linewidth=2, marker='o', markersize=3, alpha=0.7, label=label)
            
            ax.set_title(category_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Generation', fontsize=10)
            ax.set_ylabel('Normalized Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
        
        # Hide unused subplots
        for idx in range(len(categories), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Population Diversity Analysis by Category', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
