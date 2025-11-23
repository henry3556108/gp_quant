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
    def plot_pnl_diversity(
        pnl_diversity_data: pd.DataFrame,
        figsize: Tuple[int, int] = (16, 6),
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot PnL correlation diversity trends with boxplot.
        
        Args:
            pnl_diversity_data: DataFrame from DiversityAnalyzer.calculate_pnl_diversity_trends()
            figsize: Figure size
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Mean correlation
        ax1.plot(pnl_diversity_data['generation'], pnl_diversity_data['pnl_corr_mean'],
                linewidth=2.5, marker='o', markersize=5, alpha=0.8, color='steelblue',
                label='Mean Correlation')
        
        # Add trend line
        z = np.polyfit(pnl_diversity_data['generation'], pnl_diversity_data['pnl_corr_mean'], 1)
        p = np.poly1d(z)
        trend_direction = '↗' if z[0] > 0 else '↘'
        ax1.plot(pnl_diversity_data['generation'], p(pnl_diversity_data['generation']),
                linestyle='--', alpha=0.6, linewidth=2, color='red', label=f'Trend {trend_direction}')
        
        ax1.set_title('PnL Correlation Mean', fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Mean Correlation', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(fontsize=9, loc='best')
        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        # Plot 2: Std deviation of correlation
        ax2.plot(pnl_diversity_data['generation'], pnl_diversity_data['pnl_corr_std'],
                linewidth=2.5, marker='o', markersize=5, alpha=0.8, color='darkorange',
                label='Correlation Std Dev')
        
        # Add trend line
        z2 = np.polyfit(pnl_diversity_data['generation'], pnl_diversity_data['pnl_corr_std'], 1)
        p2 = np.poly1d(z2)
        trend_direction2 = '↗' if z2[0] > 0 else '↘'
        ax2.plot(pnl_diversity_data['generation'], p2(pnl_diversity_data['generation']),
                linestyle='--', alpha=0.6, linewidth=2, color='red', label=f'Trend {trend_direction2}')
        
        ax2.set_title('PnL Correlation Std Dev', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Std Deviation', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=9, loc='best')
        
        # Plot 3: Boxplot showing distribution across generations
        # Sample key generations for boxplot
        key_generations = [0, 10, 20, 30, 40, 50]
        available_gens = [g for g in key_generations if g in pnl_diversity_data['generation'].values]
        
        boxplot_data = []
        labels = []
        for gen in available_gens:
            row = pnl_diversity_data[pnl_diversity_data['generation'] == gen].iloc[0]
            # Create distribution from mean, std, min, max
            mean = row['pnl_corr_mean']
            std = row['pnl_corr_std']
            min_val = row['pnl_corr_min']
            max_val = row['pnl_corr_max']
            median = row['pnl_corr_median']
            
            # Approximate distribution for boxplot
            # Use quartiles based on mean and std
            boxplot_data.append({
                'med': median,
                'q1': max(min_val, mean - 0.675 * std),  # ~25th percentile
                'q3': min(max_val, mean + 0.675 * std),  # ~75th percentile
                'whislo': min_val,
                'whishi': max_val,
                'mean': mean,
                'fliers': []  # No outliers for now
            })
            labels.append(f'Gen {gen}')
        
        # Create boxplot
        bp = ax3.bxp([boxplot_data[i] for i in range(len(boxplot_data))], 
                     positions=range(len(available_gens)),
                     widths=0.6,
                     patch_artist=True,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(available_gens)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Correlation Distribution\n(Key Generations)', fontsize=13, fontweight='bold', pad=10)
        ax3.set_xlabel('Generation', fontsize=11)
        ax3.set_ylabel('Correlation Coefficient', fontsize=11)
        ax3.set_xticks(range(len(available_gens)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        plt.suptitle('PnL Correlation-based Diversity Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
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
                'metrics': ['fitness_fitness_cv', 'fitness_fitness_mean'],
                'labels': ['CV (Std/Mean)', 'Mean Fitness'],
                'colors': ['crimson', 'darkred'],
                'ylabel': 'CV',
                'ylabel2': 'Mean Fitness',
                'use_twin_axis': True  # Use secondary y-axis for different scales
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
            
            # Check if we need twin axis (for metrics with very different scales)
            use_twin = config.get('use_twin_axis', False) and len(config['metrics']) > 1
            
            # Plot each metric in this category
            for metric_idx, (metric, label, color) in enumerate(zip(config['metrics'], config['labels'], config['colors'])):
                if metric in diversity_data.columns:
                    # Use secondary axis for second metric if needed
                    if use_twin and metric_idx == 1:
                        ax2 = ax.twinx()
                        current_ax = ax2
                        ax2.set_ylabel(label, fontsize=10, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                    else:
                        current_ax = ax
                    
                    current_ax.plot(diversity_data['generation'], diversity_data[metric], 
                           linewidth=2.5, marker='o', markersize=4, alpha=0.8, 
                           color=color, label=label)
                    
                    # Add trend line
                    z = np.polyfit(diversity_data['generation'], diversity_data[metric], 1)
                    p = np.poly1d(z)
                    current_ax.plot(diversity_data['generation'], p(diversity_data['generation']), 
                           linestyle='--', alpha=0.5, linewidth=1.5, color=color)
                    
                    if use_twin and metric_idx == 1:
                        ax2.tick_params(axis='y', labelcolor=color)
            
            # Formatting
            ax.set_title(config['title'], fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Generation', fontsize=11)
            
            if use_twin and 'ylabel2' in config:
                # Set different y-labels for twin axes
                ax.set_ylabel(config['ylabel'], fontsize=11, color=config['colors'][0])
                ax.tick_params(axis='y', labelcolor=config['colors'][0])
            else:
                ax.set_ylabel(config['ylabel'], fontsize=11)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend (combine both axes if using twin)
            if use_twin:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='best')
            else:
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
