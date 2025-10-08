"""
Test script for diversity analysis module

This script demonstrates how to use the diversity analysis module.
"""

from gp_quant.diversity import DiversityAnalyzer
from gp_quant.diversity.visualizer import DiversityVisualizer

# Path to the individual records directory
records_dir = "experiments_results/ABX_TO/individual_records_long_run01"

print("="*80)
print("🧪 Testing Diversity Analysis Module")
print("="*80)
print()

# Initialize analyzer
print("1. Initializing analyzer...")
analyzer = DiversityAnalyzer(records_dir)
print(f"   ✓ Analyzer created for: {records_dir}")
print()

# Load populations
print("2. Loading populations...")
populations = analyzer.load_populations(verbose=True)
print(f"   ✓ Loaded {len(populations)} generations")
print()

# Calculate diversity metrics
print("3. Calculating diversity metrics...")
diversity_data = analyzer.calculate_diversity_trends()
print(f"   ✓ Calculated metrics for {len(diversity_data)} generations")
print()

# Display first few rows
print("4. Sample data:")
print(diversity_data.head())
print()

# Get summary statistics
print("5. Summary statistics:")
summary = analyzer.get_summary_statistics()
print(f"   Total generations: {summary['total_generations']}")
print()

# Show key metrics
print("6. Key metrics trends:")
key_metrics = ['genotypic_unique_ratio', 'fitness_std', 'structural_height_std']
for metric in key_metrics:
    if metric in summary['metrics']:
        stats = summary['metrics'][metric]
        print(f"   {metric}:")
        print(f"      Initial: {stats['initial']:.4f}")
        print(f"      Final: {stats['final']:.4f}")
        print(f"      Mean: {stats['mean']:.4f}")
        print(f"      Trend: {stats['trend']}")
print()

# Save results
print("7. Saving results...")
analyzer.save_results("diversity_data_test.csv")
print()

# Generate visualization
print("8. Generating visualization...")
DiversityVisualizer.plot_diversity_trends(
    diversity_data,
    save_path="diversity_trends_test.png",
    show=False
)
print("   ✓ Plot saved to: diversity_trends_test.png")
print()

print("="*80)
print("✅ All tests passed!")
print("="*80)
