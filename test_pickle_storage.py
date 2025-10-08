"""
Small-scale test for pickle population storage
Tests the storage mechanism with reduced parameters to estimate space requirements
"""
import subprocess
import os
import json
from datetime import datetime

def get_directory_size(path):
    """Calculate total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_size(bytes_size):
    """Format bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def run_small_test():
    """
    Run a small-scale test with:
    - 1 ticker (ABX.TO)
    - 1 training period (short)
    - Reduced generations (10 instead of 50)
    - Reduced population (100 instead of 500)
    """
    
    ticker = 'ABX.TO'
    
    # Short training period configuration
    train_data_start = '1997-06-25'
    train_backtest_start = '1998-06-22'
    train_backtest_end = '1999-06-25'
    test_data_start = '1998-07-07'
    test_backtest_start = '1999-06-28'
    test_backtest_end = '2000-06-30'
    
    # Small-scale parameters
    generations = 10
    population = 100
    
    print("\n" + "="*80)
    print("🧪 Small-Scale Pickle Storage Test")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Generations: {generations}")
    print(f"Population: {population}")
    print(f"Training Period: {train_backtest_start} to {train_backtest_end}")
    print("="*80 + "\n")
    
    # Clean up any existing results for this ticker
    ticker_clean = ticker.replace('.', '_')
    ticker_dir = f"experiments_results/{ticker_clean}"
    individual_records_dir = os.path.join(ticker_dir, "individual_records")
    
    if os.path.exists(individual_records_dir):
        import shutil
        print(f"Cleaning up existing records at {individual_records_dir}...")
        shutil.rmtree(individual_records_dir)
    
    start_time = datetime.now()
    
    # Run the experiment
    result = subprocess.run([
        'python', 'main.py',
        '--tickers', ticker,
        '--mode', 'portfolio',
        '--generations', str(generations),
        '--population', str(population),
        '--train_data_start', train_data_start,
        '--train_backtest_start', train_backtest_start,
        '--train_backtest_end', train_backtest_end,
        '--test_data_start', test_data_start,
        '--test_backtest_start', test_backtest_start,
        '--test_backtest_end', test_backtest_end
    ], capture_output=True, text=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("📊 Test Results")
    print("="*80)
    print(f"Execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Check if individual_records directory was created
    if os.path.exists(individual_records_dir):
        # Calculate storage size
        total_size = get_directory_size(individual_records_dir)
        
        # Count generation directories
        gen_dirs = [d for d in os.listdir(individual_records_dir) 
                   if os.path.isdir(os.path.join(individual_records_dir, d))]
        num_generations = len(gen_dirs)
        
        print(f"\n✅ Individual records saved successfully!")
        print(f"Location: {individual_records_dir}")
        print(f"Number of generations saved: {num_generations}")
        print(f"Total storage used: {format_size(total_size)}")
        
        if num_generations > 0:
            avg_size_per_gen = total_size / num_generations
            print(f"Average size per generation: {format_size(avg_size_per_gen)}")
            
            # Extrapolate to full-scale experiment
            print("\n" + "="*80)
            print("📈 Extrapolation to Full-Scale Experiment")
            print("="*80)
            
            # Full scale: 4 tickers × 2 periods × 10 runs × 50 generations
            full_scale_tickers = 4
            full_scale_periods = 2
            full_scale_runs = 10
            full_scale_generations = 50
            full_scale_population = 500
            
            # Estimate based on population size scaling
            size_scaling_factor = (full_scale_population / population)
            estimated_size_per_gen_full = avg_size_per_gen * size_scaling_factor
            
            total_generations_full = (full_scale_tickers * full_scale_periods * 
                                     full_scale_runs * full_scale_generations)
            
            estimated_total_size = estimated_size_per_gen_full * total_generations_full
            
            print(f"Full-scale configuration:")
            print(f"  - Tickers: {full_scale_tickers}")
            print(f"  - Periods: {full_scale_periods}")
            print(f"  - Runs per period: {full_scale_runs}")
            print(f"  - Generations per run: {full_scale_generations}")
            print(f"  - Population size: {full_scale_population}")
            print(f"  - Total generations: {total_generations_full}")
            print(f"\nEstimated storage requirements:")
            print(f"  - Per generation (pop={full_scale_population}): {format_size(estimated_size_per_gen_full)}")
            print(f"  - Total estimated size: {format_size(estimated_total_size)}")
            
            # Show directory structure
            print("\n" + "="*80)
            print("📁 Directory Structure")
            print("="*80)
            print(f"{ticker_dir}/")
            print(f"  └── individual_records/")
            for i, gen_dir in enumerate(sorted(gen_dirs)[:5]):  # Show first 5
                gen_path = os.path.join(individual_records_dir, gen_dir)
                gen_size = get_directory_size(gen_path)
                print(f"      ├── {gen_dir}/ ({format_size(gen_size)})")
                # List files in this generation
                files = os.listdir(gen_path)
                for f in files:
                    file_path = os.path.join(gen_path, f)
                    file_size = os.path.getsize(file_path)
                    print(f"      │   └── {f} ({format_size(file_size)})")
            if len(gen_dirs) > 5:
                print(f"      └── ... ({len(gen_dirs) - 5} more generations)")
        
        # Save summary report
        report = {
            'test_config': {
                'ticker': ticker,
                'generations': generations,
                'population': population,
                'duration_seconds': duration
            },
            'storage_stats': {
                'total_size_bytes': total_size,
                'total_size_formatted': format_size(total_size),
                'num_generations': num_generations,
                'avg_size_per_gen_bytes': avg_size_per_gen if num_generations > 0 else 0,
                'avg_size_per_gen_formatted': format_size(avg_size_per_gen) if num_generations > 0 else "N/A"
            },
            'full_scale_estimate': {
                'estimated_total_size_bytes': estimated_total_size if num_generations > 0 else 0,
                'estimated_total_size_formatted': format_size(estimated_total_size) if num_generations > 0 else "N/A",
                'total_generations': total_generations_full
            }
        }
        
        report_path = os.path.join(ticker_dir, "pickle_storage_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved to: {report_path}")
        
    else:
        print(f"\n❌ Error: Individual records directory not created!")
        print(f"Expected location: {individual_records_dir}")
        print("\nCommand output:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
    
    print("\n" + "="*80)
    print("✅ Test Complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_small_test()
