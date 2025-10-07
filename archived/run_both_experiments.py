"""
Automated script to run both training period experiments for BBD-B.TO
"""
import subprocess
import json
import re
from datetime import datetime

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
    
    # Extract training results
    train_match = re.search(r'Total GP Return: \$([0-9,.-]+)', output)
    if train_match:
        results['train_gp_return'] = float(train_match.group(1).replace(',', ''))
    
    bh_matches = re.findall(r'Total Buy-and-Hold Return: \$([0-9,.-]+)', output)
    if len(bh_matches) >= 1:
        results['train_bh_return'] = float(bh_matches[0].replace(',', ''))
    
    excess_matches = re.findall(r'Total Excess Return: \$([0-9,.-]+)', output)
    if len(excess_matches) >= 1:
        results['train_excess_return'] = float(excess_matches[0].replace(',', ''))
    
    # Extract testing results
    if len(bh_matches) >= 2:
        results['test_bh_return'] = float(bh_matches[1].replace(',', ''))
    
    if len(excess_matches) >= 2:
        results['test_excess_return'] = float(excess_matches[1].replace(',', ''))
    
    # Extract GP return from testing (need to find the second occurrence)
    gp_matches = re.findall(r'Total GP Return: \$([0-9,.-]+)', output)
    if len(gp_matches) >= 2:
        results['test_gp_return'] = float(gp_matches[1].replace(',', ''))
    
    # Extract best fitness
    fitness_match = re.search(r'Best Individual Fitness \(Total Excess Return\): \$([0-9,.-]+)', output)
    if fitness_match:
        results['best_fitness'] = float(fitness_match.group(1).replace(',', ''))
    
    return results

def run_experiment(name, ticker, train_start, train_end, test_start, test_end):
    """Run a single experiment"""
    print("\n" + "="*100)
    print(f"🚀 開始實驗: {name}")
    print("="*100)
    print(f"📊 股票代碼: {ticker}")
    print(f"📅 訓練期: {train_start} 至 {train_end}")
    print(f"📅 測試期: {test_start} 至 {test_end}")
    print("="*100 + "\n")
    
    # Modify main.py
    modify_main_py(train_start, train_end, test_start, test_end)
    print("✅ main.py 已更新為新的日期範圍\n")
    
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
    
    print(result.stdout)
    
    # Save full log
    log_filename = f"experiment_{name.replace(' ', '_').lower()}.log"
    with open(log_filename, 'w') as f:
        f.write(f"實驗: {name}\n")
        f.write(f"訓練期: {train_start} 至 {train_end}\n")
        f.write(f"測試期: {test_start} 至 {test_end}\n")
        f.write(f"執行時間: {duration:.2f} 秒\n")
        f.write("="*100 + "\n\n")
        f.write(result.stdout)
    
    print(f"\n✅ 完整日誌已儲存至: {log_filename}")
    print(f"⏱️  執行時間: {duration:.2f} 秒\n")
    
    # Extract and return results
    results = extract_results(result.stdout)
    results['duration'] = duration
    results['log_file'] = log_filename
    
    return results

def print_summary(exp1_results, exp2_results):
    """Print comparison summary"""
    print("\n" + "="*100)
    print("📊 實驗結果總結")
    print("="*100)
    
    print("\n" + "─"*100)
    print("實驗 1: 短訓練期 (Short Training Period)")
    print("─"*100)
    print(f"訓練期: 1998-06-22 至 1999-06-25 (約 256 天)")
    print(f"測試期: 1999-06-28 至 2000-06-30 (約 256 天)")
    print()
    print("樣本內表現 (In-Sample):")
    print(f"  GP 總報酬:        ${exp1_results['train_gp_return']:>15,.2f}")
    print(f"  Buy-and-Hold:     ${exp1_results['train_bh_return']:>15,.2f}")
    print(f"  超額報酬:         ${exp1_results['train_excess_return']:>15,.2f}")
    print()
    print("樣本外表現 (Out-of-Sample):")
    print(f"  GP 總報酬:        ${exp1_results['test_gp_return']:>15,.2f}")
    print(f"  Buy-and-Hold:     ${exp1_results['test_bh_return']:>15,.2f}")
    print(f"  超額報酬:         ${exp1_results['test_excess_return']:>15,.2f}")
    print(f"\n執行時間: {exp1_results['duration']:.2f} 秒")
    
    print("\n" + "─"*100)
    print("實驗 2: 長訓練期 (Long Training Period)")
    print("─"*100)
    print(f"訓練期: 1993-07-02 至 1999-06-25 (約 1498 天)")
    print(f"測試期: 1999-06-28 至 2000-06-30 (約 256 天)")
    print()
    print("樣本內表現 (In-Sample):")
    print(f"  GP 總報酬:        ${exp2_results['train_gp_return']:>15,.2f}")
    print(f"  Buy-and-Hold:     ${exp2_results['train_bh_return']:>15,.2f}")
    print(f"  超額報酬:         ${exp2_results['train_excess_return']:>15,.2f}")
    print()
    print("樣本外表現 (Out-of-Sample):")
    print(f"  GP 總報酬:        ${exp2_results['test_gp_return']:>15,.2f}")
    print(f"  Buy-and-Hold:     ${exp2_results['test_bh_return']:>15,.2f}")
    print(f"  超額報酬:         ${exp2_results['test_excess_return']:>15,.2f}")
    print(f"\n執行時間: {exp2_results['duration']:.2f} 秒")
    
    print("\n" + "="*100)
    print("📈 比較分析")
    print("="*100)
    
    # Calculate differences
    train_diff = exp2_results['train_excess_return'] - exp1_results['train_excess_return']
    test_diff = exp2_results['test_excess_return'] - exp1_results['test_excess_return']
    
    print(f"\n樣本內超額報酬差異 (長訓練期 - 短訓練期): ${train_diff:,.2f}")
    print(f"樣本外超額報酬差異 (長訓練期 - 短訓練期): ${test_diff:,.2f}")
    
    # Performance analysis - The key metric
    print("\n" + "="*100)
    print("🎯 關鍵指標：樣本外表現分析")
    print("="*100)
    
    # Check if strategies beat buy-and-hold in out-of-sample
    exp1_beats_bh = exp1_results['test_excess_return'] > 0
    exp2_beats_bh = exp2_results['test_excess_return'] > 0
    
    print(f"\n短訓練期樣本外表現:")
    print(f"  超額報酬: ${exp1_results['test_excess_return']:,.2f}")
    if exp1_beats_bh:
        print(f"  ✅ 超越 Buy-and-Hold")
    else:
        print(f"  ❌ 輸給 Buy-and-Hold")
    
    print(f"\n長訓練期樣本外表現:")
    print(f"  超額報酬: ${exp2_results['test_excess_return']:,.2f}")
    if exp2_beats_bh:
        print(f"  ✅ 超越 Buy-and-Hold")
    else:
        print(f"  ❌ 輸給 Buy-and-Hold")
    
    # Determine which is better
    print(f"\n{'='*100}")
    print("📊 最終結論")
    print("="*100)
    
    if exp2_beats_bh and not exp1_beats_bh:
        print(f"✅ 長訓練期明顯優於短訓練期")
        print(f"   - 長訓練期在樣本外盈利 (${exp2_results['test_excess_return']:,.2f})")
        print(f"   - 短訓練期在樣本外虧損 (${exp1_results['test_excess_return']:,.2f})")
    elif exp1_beats_bh and not exp2_beats_bh:
        print(f"✅ 短訓練期明顯優於長訓練期")
        print(f"   - 短訓練期在樣本外盈利 (${exp1_results['test_excess_return']:,.2f})")
        print(f"   - 長訓練期在樣本外虧損 (${exp2_results['test_excess_return']:,.2f})")
    elif exp2_beats_bh and exp1_beats_bh:
        if exp2_results['test_excess_return'] > exp1_results['test_excess_return']:
            print(f"✅ 長訓練期優於短訓練期")
            print(f"   - 兩者都超越 Buy-and-Hold")
            print(f"   - 長訓練期樣本外超額報酬更高 (${test_diff:,.2f})")
        else:
            print(f"✅ 短訓練期優於長訓練期")
            print(f"   - 兩者都超越 Buy-and-Hold")
            print(f"   - 短訓練期樣本外超額報酬更高 (${-test_diff:,.2f})")
    else:
        print(f"⚠️ 兩者都無法超越 Buy-and-Hold")
        if exp2_results['test_excess_return'] > exp1_results['test_excess_return']:
            print(f"   - 長訓練期虧損較少")
        else:
            print(f"   - 短訓練期虧損較少")
    
    # Overfitting analysis (secondary metric)
    print(f"\n補充資訊 - 過度擬合程度:")
    exp1_overfit = exp1_results['train_excess_return'] - exp1_results['test_excess_return']
    exp2_overfit = exp2_results['train_excess_return'] - exp2_results['test_excess_return']
    
    print(f"  短訓練期: ${exp1_overfit:,.2f} (訓練期 - 測試期)")
    print(f"  長訓練期: ${exp2_overfit:,.2f} (訓練期 - 測試期)")
    print(f"  註：過度擬合程度僅供參考，關鍵是樣本外是否盈利")
    
    # Save summary
    summary = {
        'experiment_1_short': exp1_results,
        'experiment_2_long': exp2_results,
        'comparison': {
            'train_excess_diff': train_diff,
            'test_excess_diff': test_diff,
            'exp1_overfitting': exp1_overfit,
            'exp2_overfitting': exp2_overfit
        }
    }
    
    with open('experiments_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ 完整總結已儲存至: experiments_summary.json")
    print("="*100 + "\n")

if __name__ == "__main__":
    # Configuration
    TICKER = 'RY.TO'  # Change this to test different stocks
    
    print("\n" + "🎯"*50)
    print(f"{TICKER} 自動化實驗系統")
    print("🎯"*50 + "\n")
    
    # Experiment 1: Short Training Period
    exp1_results = run_experiment(
        name="短訓練期",
        ticker=TICKER,
        train_start='1997-06-25',
        train_end='1999-06-25',
        test_start='1999-07-07',
        test_end='2000-06-30'
    )
    
    # Experiment 2: Long Training Period
    exp2_results = run_experiment(
        name="長訓練期",
        ticker=TICKER,
        train_start='1992-06-30',
        train_end='1999-06-25',
        test_start='1998-07-07',
        test_end='2000-06-30'
    )
    
    # Print summary
    print_summary(exp1_results, exp2_results)
    
    print("\n" + "🎉"*50)
    print("所有實驗完成！")
    print("🎉"*50 + "\n")
