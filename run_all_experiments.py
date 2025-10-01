"""
Run comprehensive experiments for all tickers
Each ticker will be tested 10 times with both short and long training periods
"""
import subprocess
import json
import re
from datetime import datetime
import pandas as pd
import os

def modify_main_py(train_start, train_end, test_start, test_end):
    """Modify main.py with new date ranges"""
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Replace the date strings
    content = re.sub(
        r"train_start = '[0-9]{4}-[0-9]{2}-[0-9]{2}'",
        f"train_start = '{train_start}'",
        content
    )
    content = re.sub(
        r"train_end = '[0-9]{4}-[0-9]{2}-[0-9]{2}'",
        f"train_end = '{train_end}'",
        content
    )
    content = re.sub(
        r"test_start = '[0-9]{4}-[0-9]{2}-[0-9]{2}'",
        f"test_start = '{test_start}'",
        content
    )
    content = re.sub(
        r"test_end = '[0-9]{4}-[0-9]{2}-[0-9]{2}'",
        f"test_end = '{test_end}'",
        content
    )
    
    with open('main.py', 'w') as f:
        f.write(content)

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

def run_single_experiment(ticker, period_name, train_start, train_end, test_start, test_end, run_number):
    """Run a single experiment"""
    print(f"\n{'='*100}")
    print(f"🔬 運行: {ticker} | {period_name} | 第 {run_number}/10 次")
    print(f"{'='*100}")
    
    # Create directory for this ticker if it doesn't exist
    ticker_dir = f"experiments_results/{ticker.replace('.', '_')}"
    os.makedirs(ticker_dir, exist_ok=True)
    
    # Modify main.py
    modify_main_py(train_start, train_end, test_start, test_end)
    
    # Run the experiment
    start_time = datetime.now()
    
    result = subprocess.run(
        ['python', 'main.py', '--tickers', ticker, '--mode', 'portfolio', 
         '--generations', '50', '--population', '500'],
        capture_output=True,
        text=True,
        cwd='/Users/hongyicheng/Desktop/code/研究/gp_paper'
    )
    
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
    
    # Save individual result JSON
    result_json_file = f"{ticker_dir}/{period_short}_run{run_number:02d}_result.json"
    with open(result_json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full output log
    log_file = f"{ticker_dir}/{period_short}_run{run_number:02d}_output.log"
    with open(log_file, 'w') as f:
        f.write(result.stdout)
    
    # Print summary
    if results['test_excess_return'] is not None:
        status = "✅ 盈利" if results['test_excess_return'] > 0 else "❌ 虧損"
        print(f"樣本外超額報酬: ${results['test_excess_return']:,.2f} {status}")
    print(f"執行時間: {duration:.2f} 秒")
    print(f"📁 文件已保存至: {ticker_dir}/")
    
    return results

def run_all_experiments():
    """Run all experiments for all tickers"""
    
    # Configuration
    tickers = ['ABX.TO', 'BBD-B.TO', 'RY.TO', 'TRP.TO']
    n_runs = 10
    
    experiments = [
        {
            'name': '短訓練期',
            'train_start': '1998-06-22',
            'train_end': '1999-06-25',
            'test_start': '1999-06-28',
            'test_end': '2000-06-30'
        },
        {
            'name': '長訓練期',
            'train_start': '1993-07-02',
            'train_end': '1999-06-25',
            'test_start': '1999-06-28',
            'test_end': '2000-06-30'
        }
    ]
    
    all_results = []
    total_experiments = len(tickers) * len(experiments) * n_runs
    completed = 0
    
    print("\n" + "🚀"*50)
    print(f"開始大規模實驗")
    print(f"股票數量: {len(tickers)}")
    print(f"訓練期類型: {len(experiments)}")
    print(f"每個配置運行次數: {n_runs}")
    print(f"總實驗數: {total_experiments}")
    print("🚀"*50 + "\n")
    
    start_time_all = datetime.now()
    
    for ticker in tickers:
        print(f"\n{'#'*100}")
        print(f"# 開始處理股票: {ticker}")
        print(f"{'#'*100}")
        
        for exp in experiments:
            print(f"\n{'='*100}")
            print(f"配置: {exp['name']}")
            print(f"訓練期: {exp['train_start']} 至 {exp['train_end']}")
            print(f"測試期: {exp['test_start']} 至 {exp['test_end']}")
            print(f"{'='*100}")
            
            for run in range(1, n_runs + 1):
                try:
                    result = run_single_experiment(
                        ticker=ticker,
                        period_name=exp['name'],
                        train_start=exp['train_start'],
                        train_end=exp['train_end'],
                        test_start=exp['test_start'],
                        test_end=exp['test_end'],
                        run_number=run
                    )
                    all_results.append(result)
                    completed += 1
                    
                    # Progress update
                    progress = (completed / total_experiments) * 100
                    print(f"\n📊 總進度: {completed}/{total_experiments} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"❌ 錯誤: {e}")
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
        print(f"   │   ├── ... (run02 到 run10)")
        print(f"   │   ├── long_run01_train_trades.csv")
        print(f"   │   ├── long_run01_test_trades.csv")
        print(f"   │   ├── long_run01_result.json")
        print(f"   │   ├── long_run01_output.log")
        print(f"   │   └── ... (run02 到 run10)")
    print("="*100 + "\n")

if __name__ == "__main__":
    results_df = run_all_experiments()
    
    print("\n" + "🎉"*50)
    print("所有實驗完成！")
    print("🎉"*50 + "\n")
