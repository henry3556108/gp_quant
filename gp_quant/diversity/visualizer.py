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
        Plot diversity trends across generations showing all 4 diversity categories.
        
        Args:
            diversity_data: DataFrame from DiversityAnalyzer.calculate_diversity_trends()
            metrics: Not used in default mode (kept for backward compatibility)
            figsize: Figure size (width, height)
            save_path: Path to save the figure. If None, doesn't save.
            show: Whether to display the plot
        """
        # Create 2x2 grid for the 4 diversity categories
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Define the 4 plots with their configurations
        plot_configs = [
            {
                'title': '1. Structural Diversity',
                'metrics': ['structural_height_std', 'structural_length_std'],
                'labels': ['Tree Height Std', 'Tree Length Std'],
                'colors': ['steelblue', 'darkorange'],
                'ylabel': 'Standard Deviation'
            },
            {
                'title': '2. Genotypic Diversity',
                'metrics': ['genotypic_unique_ratio'],
                'labels': ['Unique Individuals Ratio'],
                'colors': ['forestgreen'],
                'ylabel': 'Ratio'
            },
            {
                'title': '3. Fitness Diversity',
                'metrics': ['fitness_fitness_cv'],
                'labels': ['Coefficient of Variation'],
                'colors': ['crimson'],
                'ylabel': 'CV (std/mean)'
            },
            {
                'title': '4. Phenotypic Diversity',
                'metrics': ['phenotypic_unique_primitives'],
                'labels': ['Unique Primitives Count'],
                'colors': ['purple'],
                'ylabel': 'Count'
            }
        ]
        
        for idx, config in enumerate(plot_configs):
            ax = axes[idx]
            
            # Plot each metric in this category
            for metric, label, color in zip(config['metrics'], config['labels'], config['colors']):
                if metric in diversity_data.columns:
                    ax.plot(diversity_data['generation'], diversity_data[metric], 
                           linewidth=2.5, marker='o', markersize=4, alpha=0.8, 
                           color=color, label=label)
                    
                    # Add trend line
                    z = np.polyfit(diversity_data['generation'], diversity_data[metric], 1)
                    p = np.poly1d(z)
                    trend_direction = '↗' if z[0] > 0 else '↘'
                    ax.plot(diversity_data['generation'], p(diversity_data['generation']), 
                           linestyle='--', alpha=0.5, linewidth=1.5, color=color)
            
            # Formatting
            ax.set_title(config['title'], fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel(config['ylabel'], fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=9, loc='best')
        
        plt.suptitle('Population Diversity Analysis: Four Categories', 
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
