"""
Diversity Analysis CLI Tool

Command-line interface for analyzing population diversity from saved records.

Usage:
    python -m gp_quant.scripts.analyze_diversity \\
        --records_dir experiments_results/ABX_TO/individual_records_long_run01 \\
        --output diversity_analysis.png \\
        --csv diversity_data.csv
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Analyze population diversity from saved GP records',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single experiment
  python -m gp_quant.scripts.analyze_diversity \\
      --records_dir experiments_results/ABX_TO/individual_records_long_run01 \\
      --output diversity_plot.png

  # Save data to CSV
  python -m gp_quant.scripts.analyze_diversity \\
      --records_dir experiments_results/ABX_TO/individual_records_short_run05 \\
      --csv diversity_data.csv \\
      --output diversity_plot.png

  # Plot specific metrics
  python -m gp_quant.scripts.analyze_diversity \\
      --records_dir experiments_results/ABX_TO/individual_records_long_run01 \\
      --metrics genotypic_unique_ratio fitness_std \\
      --output custom_plot.png
        """
    )
    
    parser.add_argument(
        '--records_dir',
        type=str,
        required=True,
        help='Path to individual_records directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the diversity plot (PNG/PDF)'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Output path for diversity data CSV'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        help='Specific metrics to plot (default: key metrics)'
    )
    
    parser.add_argument(
        '--plot_type',
        type=str,
        choices=['trends', 'categories', 'single'],
        default='trends',
        help='Type of plot to generate'
    )
    
    parser.add_argument(
        '--no_show',
        action='store_true',
        help='Do not display the plot (only save)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate records directory
    if not Path(args.records_dir).exists():
        print(f"Error: Records directory does not exist: {args.records_dir}")
        sys.exit(1)
    
    print("="*80)
    print("🔬 Population Diversity Analysis")
    print("="*80)
    print(f"Records directory: {args.records_dir}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = DiversityAnalyzer(args.records_dir)
        
        # Load populations
        print("📂 Loading populations...")
        analyzer.load_populations(verbose=args.verbose)
        print()
        
        # Calculate diversity metrics
        print("📊 Calculating diversity metrics...")
        diversity_data = analyzer.calculate_diversity_trends()
        print(f"   Calculated metrics for {len(diversity_data)} generations")
        print()
        
        # Display summary
        if args.verbose:
            print("📈 Summary Statistics:")
            summary = analyzer.get_summary_statistics()
            print(f"   Total generations: {summary['total_generations']}")
            
            # Show key metrics
            key_metrics = ['genotypic_unique_ratio', 'fitness_std', 'structural_height_std']
            for metric in key_metrics:
                if metric in summary['metrics']:
                    stats = summary['metrics'][metric]
                    print(f"   {metric}:")
                    print(f"      Initial: {stats['initial']:.4f}")
                    print(f"      Final: {stats['final']:.4f}")
                    print(f"      Trend: {stats['trend']}")
            print()
        
        # Save CSV if requested
        if args.csv:
            analyzer.save_results(args.csv)
            print()
        
        # Generate plot
        if args.output or not args.no_show:
            print("🎨 Generating visualization...")
            
            if args.plot_type == 'trends':
                DiversityVisualizer.plot_diversity_trends(
                    diversity_data,
                    metrics=args.metrics,
                    save_path=args.output,
                    show=not args.no_show
                )
            elif args.plot_type == 'categories':
                DiversityVisualizer.plot_all_categories(
                    diversity_data,
                    save_path=args.output,
                    show=not args.no_show
                )
            elif args.plot_type == 'single' and args.metrics:
                for metric in args.metrics:
                    output_path = args.output.replace('.png', f'_{metric}.png') if args.output else None
                    DiversityVisualizer.plot_single_metric(
                        diversity_data,
                        metric,
                        save_path=output_path,
                        show=not args.no_show
                    )
        
        print()
        print("="*80)
        print("✅ Analysis complete!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
