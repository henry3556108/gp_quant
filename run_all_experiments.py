"""
Run comprehensive experiments for all tickers
Each ticker will be tested 10 times with both short and long training periods
Supports parallel execution using multiprocessing
"""
import subprocess
import json
import re
from datetime import datetime
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# modify_main_py() function removed - now using command-line arguments instead

def extract_results(output):
    """Extract key results from output"""
    results = {
        'train_gp_return': None,
        'train_bh_return': None,
        'train_excess_return': None,
        'test_gp_return': None,
        'test_bh_return': None,
        'test_excess_return': None,
        'best_fitness': None
    }
    
    # Extract GP returns
    gp_matches = re.findall(r'Total GP Return: \$([0-9,.-]+)', output)
    if len(gp_matches) >= 1:
        results['train_gp_return'] = float(gp_matches[0].replace(',', ''))
    if len(gp_matches) >= 2:
        results['test_gp_return'] = float(gp_matches[1].replace(',', ''))
    
    # Extract B&H returns
    bh_matches = re.findall(r'Total Buy-and-Hold Return: \$([0-9,.-]+)', output)
    if len(bh_matches) >= 1:
        results['train_bh_return'] = float(bh_matches[0].replace(',', ''))
    if len(bh_matches) >= 2:
        results['test_bh_return'] = float(bh_matches[1].replace(',', ''))
    
    # Extract excess returns
    excess_matches = re.findall(r'Total Excess Return: \$([0-9,.-]+)', output)
    if len(excess_matches) >= 1:
        results['train_excess_return'] = float(excess_matches[0].replace(',', ''))
    if len(excess_matches) >= 2:
        results['test_excess_return'] = float(excess_matches[1].replace(',', ''))
    
    # Extract best fitness
    fitness_match = re.search(r'Best Individual Fitness \(Total Excess Return\): \$([0-9,.-]+)', output)
    if fitness_match:
        results['best_fitness'] = float(fitness_match.group(1).replace(',', ''))
    
    return results

def run_single_experiment(ticker, period_name, 
                         train_data_start, train_backtest_start, train_backtest_end,
                         test_data_start, test_backtest_start, test_backtest_end,
                         run_number):
    """Run a single experiment (parallel-safe)"""
    # Reduced output for parallel execution
    print(f"🔬 開始: {ticker} | {period_name} | Run {run_number}")
    
    # Create directory for this ticker if it doesn't exist
    ticker_dir = f"experiments_results/{ticker.replace('.', '_')}"
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Run the experiment with date parameters
    start_time = datetime.now()
    
    result = subprocess.run([
        'python', 'main.py',
        '--tickers', ticker,
        '--mode', 'portfolio',
        '--generations', '50',
        '--population', '500',
        '--train_data_start', train_data_start,
        '--train_backtest_start', train_backtest_start,
        '--train_backtest_end', train_backtest_end,
        '--test_data_start', test_data_start,
        '--test_backtest_start', test_backtest_start,
        '--test_backtest_end', test_backtest_end
    ], capture_output=True, text=True)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Extract results
    results = extract_results(result.stdout)
    results['duration'] = duration
    results['ticker'] = ticker
    results['period'] = period_name
    results['run_number'] = run_number
    results['timestamp'] = datetime.now().isoformat()
    
    # Save trade files to ticker-specific directory
    period_short = 'short' if period_name == '短訓練期' else 'long'
    
    # Move/copy trade CSV files
    train_trades = f"portfolio_train_{ticker}_trades.csv"
    test_trades = f"portfolio_test_{ticker}_trades.csv"
    
    if os.path.exists(train_trades):
        new_train_name = f"{ticker_dir}/{period_short}_run{run_number:02d}_train_trades.csv"
        os.rename(train_trades, new_train_name)
        results['train_trades_file'] = new_train_name
    
    if os.path.exists(test_trades):
        new_test_name = f"{ticker_dir}/{period_short}_run{run_number:02d}_test_trades.csv"
        os.rename(test_trades, new_test_name)
        results['test_trades_file'] = new_test_name
    
    # Move individual_records directory if it exists
    # Look for temporary individual_records directories (created with unique names)
    import glob
    individual_records_pattern = f"{ticker_dir}/individual_records_tmp_*"
    individual_records_dirs = glob.glob(individual_records_pattern)
    
    if individual_records_dirs:
        # Should only be one, but take the most recent if multiple
        individual_records_src = sorted(individual_records_dirs)[-1]
        individual_records_dst = f"{ticker_dir}/individual_records_{period_short}_run{run_number:02d}"
        
        if os.path.exists(individual_records_dst):
            import shutil
            shutil.rmtree(individual_records_dst)
        
        os.rename(individual_records_src, individual_records_dst)
        results['individual_records_dir'] = individual_records_dst
        
        # Calculate storage size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(individual_records_dst):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        results['individual_records_size_bytes'] = total_size
    
    # Save individual result JSON
    result_json_file = f"{ticker_dir}/{period_short}_run{run_number:02d}_result.json"
    with open(result_json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full output log
    log_file = f"{ticker_dir}/{period_short}_run{run_number:02d}_output.log"
    with open(log_file, 'w') as f:
        f.write(result.stdout)
    
    # Print summary (compact for parallel execution)
    if results['test_excess_return'] is not None:
        status = "✅" if results['test_excess_return'] > 0 else "❌"
        print(f"✓ 完成: {ticker} | {period_name} | Run {run_number} | "
              f"超額: ${results['test_excess_return']:,.0f} {status} | {duration:.1f}s")
    
    return results

def run_all_experiments(max_workers=8):
    """
    Run all experiments for all tickers with parallel execution
    
    Args:
        max_workers: Maximum number of parallel workers (default: 8)
    """
    
    # Configuration
    tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
    n_runs = 10
    
    experiments = [
        {
            'name': '短訓練期',
            'train_data_start': '1997-06-25',
            'train_backtest_start': '1998-06-22',
            'train_backtest_end': '1999-06-25',
            'test_data_start': '1998-07-07',
            'test_backtest_start': '1999-06-28',
            'test_backtest_end': '2000-06-30'
        },
        {
            'name': '長訓練期',
            'train_data_start': '1992-06-30',
            'train_backtest_start': '1993-07-02',
            'train_backtest_end': '1999-06-25',
            'test_data_start': '1998-07-07',
            'test_backtest_start': '1999-06-28',
            'test_backtest_end': '2000-06-30'
        }
    ]
    
    # Build list of all experiment tasks
    tasks = []
    for ticker in tickers:
        for exp in experiments:
            for run in range(1, n_runs + 1):
                tasks.append({
                    'ticker': ticker,
                    'period_name': exp['name'],
                    'train_data_start': exp['train_data_start'],
                    'train_backtest_start': exp['train_backtest_start'],
                    'train_backtest_end': exp['train_backtest_end'],
                    'test_data_start': exp['test_data_start'],
                    'test_backtest_start': exp['test_backtest_start'],
                    'test_backtest_end': exp['test_backtest_end'],
                    'run_number': run
                })
    
    total_experiments = len(tasks)
    all_results = []
    
    print("\n" + "🚀"*50)
    print(f"開始大規模實驗（並行執行）")
    print(f"股票數量: {len(tickers)}")
    print(f"訓練期類型: {len(experiments)}")
    print(f"每個配置運行次數: {n_runs}")
    print(f"總實驗數: {total_experiments}")
    print(f"並行工作數: {max_workers}")
    print("🚀"*50 + "\n")
    
    start_time_all = datetime.now()
    
    # Execute experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_experiment,
                ticker=task['ticker'],
                period_name=task['period_name'],
                train_data_start=task['train_data_start'],
                train_backtest_start=task['train_backtest_start'],
                train_backtest_end=task['train_backtest_end'],
                test_data_start=task['test_data_start'],
                test_backtest_start=task['test_backtest_start'],
                test_backtest_end=task['test_backtest_end'],
                run_number=task['run_number']
            ): task for task in tasks
        }
        
        # Process completed tasks as they finish
        completed = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                
                # Progress update
                progress = (completed / total_experiments) * 100
                print(f"\n📊 總進度: {completed}/{total_experiments} ({progress:.1f}%) | "
                      f"剛完成: {task['ticker']} {task['period_name']} Run {task['run_number']}")
                
            except Exception as e:
                print(f"❌ 錯誤 ({task['ticker']} {task['period_name']} Run {task['run_number']}): {e}")
                continue
    
    end_time_all = datetime.now()
    total_duration = (end_time_all - start_time_all).total_seconds()
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('all_experiments_results.csv', index=False)
    
    with open('all_experiments_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary statistics
    generate_summary(results_df, total_duration)
    
    return results_df

def generate_summary(df, total_duration):
    """Generate summary statistics"""
    print("\n" + "="*100)
    print("📊 實驗總結")
    print("="*100)
    
    print(f"\n總執行時間: {total_duration/60:.2f} 分鐘 ({total_duration:.2f} 秒)")
    print(f"總實驗數: {len(df)}")
    
    # Summary by ticker and period
    summary = df.groupby(['ticker', 'period']).agg({
        'test_excess_return': ['mean', 'std', 'min', 'max'],
        'train_excess_return': ['mean', 'std'],
        'duration': 'mean'
    }).round(2)
    
    print("\n" + "="*100)
    print("各股票各訓練期的統計摘要 (10次運行)")
    print("="*100)
    print(summary)
    
    # Win rate (beating buy-and-hold)
    print("\n" + "="*100)
    print("樣本外勝率 (超越 Buy-and-Hold 的比例)")
    print("="*100)
    
    for ticker in df['ticker'].unique():
        print(f"\n{ticker}:")
        for period in df['period'].unique():
            subset = df[(df['ticker'] == ticker) & (df['period'] == period)]
            wins = (subset['test_excess_return'] > 0).sum()
            total = len(subset)
            win_rate = (wins / total) * 100
            
            avg_excess = subset['test_excess_return'].mean()
            status = "✅" if win_rate > 50 else "❌"
            
            print(f"  {period}: {wins}/{total} ({win_rate:.0f}%) {status} | 平均超額: ${avg_excess:,.2f}")
    
    # Best and worst performers
    print("\n" + "="*100)
    print("最佳與最差表現")
    print("="*100)
    
    best_idx = df['test_excess_return'].idxmax()
    worst_idx = df['test_excess_return'].idxmin()
    
    best = df.loc[best_idx]
    worst = df.loc[worst_idx]
    
    print(f"\n最佳表現:")
    print(f"  股票: {best['ticker']}")
    print(f"  訓練期: {best['period']}")
    print(f"  第 {best['run_number']} 次運行")
    print(f"  樣本外超額報酬: ${best['test_excess_return']:,.2f}")
    
    print(f"\n最差表現:")
    print(f"  股票: {worst['ticker']}")
    print(f"  訓練期: {worst['period']}")
    print(f"  第 {worst['run_number']} 次運行")
    print(f"  樣本外超額報酬: ${worst['test_excess_return']:,.2f}")
    
    # Overall conclusion
    print("\n" + "="*100)
    print("🎯 總體結論")
    print("="*100)
    
    short_period = df[df['period'] == '短訓練期']
    long_period = df[df['period'] == '長訓練期']
    
    short_win_rate = (short_period['test_excess_return'] > 0).sum() / len(short_period) * 100
    long_win_rate = (long_period['test_excess_return'] > 0).sum() / len(long_period) * 100
    
    print(f"\n短訓練期總體勝率: {short_win_rate:.1f}%")
    print(f"長訓練期總體勝率: {long_win_rate:.1f}%")
    
    if long_win_rate > short_win_rate:
        print(f"\n✅ 長訓練期明顯優於短訓練期 (勝率高 {long_win_rate - short_win_rate:.1f}%)")
    else:
        print(f"\n⚠️ 短訓練期表現優於長訓練期 (勝率高 {short_win_rate - long_win_rate:.1f}%)")
    
    # Individual records storage statistics
    if 'individual_records_size_bytes' in df.columns:
        total_storage = df['individual_records_size_bytes'].sum()
        avg_storage = df['individual_records_size_bytes'].mean()
        
        def format_bytes(bytes_size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.2f} TB"
        
        print("\n" + "="*100)
        print("💾 Individual Records 儲存統計")
        print("="*100)
        print(f"總儲存空間: {format_bytes(total_storage)}")
        print(f"平均每次運行: {format_bytes(avg_storage)}")
        print(f"實驗總數: {len(df)}")
    
    print("\n✅ 所有結果已儲存至:")
    print("   - all_experiments_results.csv (匯總表格)")
    print("   - all_experiments_results.json (匯總JSON)")
    print("\n📁 各股票詳細文件結構:")
    print("   experiments_results/")
    for ticker in df['ticker'].unique():
        ticker_clean = ticker.replace('.', '_')
        print(f"   ├── {ticker_clean}/")
        print(f"   │   ├── short_run01_train_trades.csv")
        print(f"   │   ├── short_run01_test_trades.csv")
        print(f"   │   ├── short_run01_result.json")
        print(f"   │   ├── short_run01_output.log")
        print(f"   │   ├── individual_records_short_run01/ (族群快照)")
        print(f"   │   ├── ... (run02 到 run10)")
        print(f"   │   ├── long_run01_train_trades.csv")
        print(f"   │   ├── long_run01_test_trades.csv")
        print(f"   │   ├── long_run01_result.json")
        print(f"   │   ├── long_run01_output.log")
        print(f"   │   ├── individual_records_long_run01/ (族群快照)")
        print(f"   │   └── ... (run02 到 run10)")
    print("="*100 + "\n")

if __name__ == "__main__":
    results_df = run_all_experiments()
    
    print("\n" + "🎉"*50)
    print("所有實驗完成！")
    print("🎉"*50 + "\n")
