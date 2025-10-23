"""
比較實驗結果與論文表格數據
計算訓練期和測試期的報酬率（相對於 B&H）
"""
import pandas as pd
import numpy as np

# 讀取資料
df = pd.read_csv('all_experiments_results.csv')

# Ticker 對應表（移除 .TO 後綴以匹配論文）
ticker_mapping = {
    'ABX.TO': 'ABX',
    'BBD-B.TO': 'BBD',
    'RY.TO': 'RY',
    'TRP.TO': 'TRP'
}

# 論文數據（用於比較）
paper_short = {
    'ABX': {'training': 202.16, 'testing': 38.12},
    'BBD': {'training': 123.04, 'testing': -34.92},
    'RY': {'training': 125.68, 'testing': 3.25},
    'TRP': {'training': 85.78, 'testing': 36.04}
}

paper_long = {
    'ABX': {'training': 34.27, 'testing': 7.69},
    'BBD': {'training': 18.30, 'testing': -79.46},
    'RY': {'training': 19.59, 'testing': 8.13},
    'TRP': {'training': 16.57, 'testing': 21.09}
}

print("="*120)
print("實驗結果與論文比較 - 訓練期與測試期報酬率")
print("="*120)

# 儲存結果用於最後的表格
results_summary = []

for ticker_full in sorted(df['ticker'].unique()):
    ticker_short = ticker_mapping.get(ticker_full, ticker_full)
    
    print(f"\n{'='*120}")
    print(f"📊 {ticker_short} ({ticker_full})")
    print(f"{'='*120}")
    
    for period_ch, period_en in [('短訓練期', 'Short Training'), ('長訓練期', 'Long Training')]:
        subset = df[(df['ticker'] == ticker_full) & (df['period'] == period_ch)]
        
        if len(subset) == 0:
            continue
        
        # 計算超額報酬率（相對於初始資金的百分比）
        # 超額報酬率 = 超額報酬 / 初始資金 * 100
        # 假設初始資金為 $100,000（根據論文設定）
        initial_capital = 100000
        subset = subset.copy()
        subset['train_excess_return_pct'] = (subset['train_excess_return'] / initial_capital) * 100
        subset['test_excess_return_pct'] = (subset['test_excess_return'] / initial_capital) * 100
        
        # 找出訓練期表現最好的那一筆
        best_idx = subset['train_excess_return_pct'].idxmax()
        best = subset.loc[best_idx]
        
        print(f"\n  【{period_en}】")
        print(f"  實驗次數: {len(subset)}")
        
        # 顯示最佳表現
        print(f"\n  ✅ 訓練期表現最佳 (Run {best['run_number']}):")
        print(f"     訓練期超額報酬率: {best['train_excess_return_pct']:.2f}%")
        print(f"     測試期超額報酬率: {best['test_excess_return_pct']:.2f}%")
        print(f"     訓練期超額報酬: ${best['train_excess_return']:,.2f}")
        print(f"     測試期超額報酬: ${best['test_excess_return']:,.2f}")
        print(f"     訓練期 GP 報酬: ${best['train_gp_return']:,.2f}")
        print(f"     訓練期 B&H 報酬: ${best['train_bh_return']:,.2f}")
        print(f"     測試期 GP 報酬: ${best['test_gp_return']:,.2f}")
        print(f"     測試期 B&H 報酬: ${best['test_bh_return']:,.2f}")
        
        # 顯示平均表現
        avg_train_pct = subset['train_excess_return_pct'].mean()
        avg_test_pct = subset['test_excess_return_pct'].mean()
        
        print(f"\n  📊 平均表現 (10次實驗):")
        print(f"     訓練期平均超額報酬率: {avg_train_pct:.2f}%")
        print(f"     測試期平均超額報酬率: {avg_test_pct:.2f}%")
        
        # 與論文比較
        paper_data = paper_short if period_ch == '短訓練期' else paper_long
        if ticker_short in paper_data:
            paper_train = paper_data[ticker_short]['training']
            paper_test = paper_data[ticker_short]['testing']
            
            print(f"\n  📄 論文數據:")
            print(f"     訓練期超額報酬率: {paper_train:.2f}%")
            print(f"     測試期超額報酬率: {paper_test:.2f}%")
            
            print(f"\n  🔍 差異分析 (最佳表現 vs 論文):")
            train_diff = best['train_excess_return_pct'] - paper_train
            test_diff = best['test_excess_return_pct'] - paper_test
            print(f"     訓練期差異: {train_diff:+.2f}% {'✅' if abs(train_diff) < 50 else '⚠️'}")
            print(f"     測試期差異: {test_diff:+.2f}% {'✅' if abs(test_diff) < 50 else '⚠️'}")
            
            print(f"\n  🔍 差異分析 (平均表現 vs 論文):")
            avg_train_diff = avg_train_pct - paper_train
            avg_test_diff = avg_test_pct - paper_test
            print(f"     訓練期差異: {avg_train_diff:+.2f}% {'✅' if abs(avg_train_diff) < 50 else '⚠️'}")
            print(f"     測試期差異: {avg_test_diff:+.2f}% {'✅' if abs(avg_test_diff) < 50 else '⚠️'}")
        
        # 儲存結果
        results_summary.append({
            'Symbol': ticker_short,
            'Period': period_en,
            'Best_Train': best['train_excess_return_pct'],
            'Best_Test': best['test_excess_return_pct'],
            'Avg_Train': avg_train_pct,
            'Avg_Test': avg_test_pct,
            'Paper_Train': paper_data[ticker_short]['training'] if ticker_short in paper_data else None,
            'Paper_Test': paper_data[ticker_short]['testing'] if ticker_short in paper_data else None
        })

# 生成比較表格
print("\n\n" + "="*120)
print("📋 綜合比較表格")
print("="*120)

results_df = pd.DataFrame(results_summary)

print("\n【Short Training Period】")
short_df = results_df[results_df['Period'] == 'Short Training']
print(f"\n{'Symbol':<10} {'Best Train':<12} {'Avg Train':<12} {'Paper Train':<12} {'Best Test':<12} {'Avg Test':<12} {'Paper Test':<12}")
print("-" * 120)
for _, row in short_df.iterrows():
    print(f"{row['Symbol']:<10} {row['Best_Train']:>10.2f}% {row['Avg_Train']:>10.2f}% {row['Paper_Train']:>10.2f}% "
          f"{row['Best_Test']:>10.2f}% {row['Avg_Test']:>10.2f}% {row['Paper_Test']:>10.2f}%")

print("\n【Long Training Period】")
long_df = results_df[results_df['Period'] == 'Long Training']
print(f"\n{'Symbol':<10} {'Best Train':<12} {'Avg Train':<12} {'Paper Train':<12} {'Best Test':<12} {'Avg Test':<12} {'Paper Test':<12}")
print("-" * 120)
for _, row in long_df.iterrows():
    print(f"{row['Symbol']:<10} {row['Best_Train']:>10.2f}% {row['Avg_Train']:>10.2f}% {row['Paper_Train']:>10.2f}% "
          f"{row['Best_Test']:>10.2f}% {row['Avg_Test']:>10.2f}% {row['Paper_Test']:>10.2f}%")

# 儲存為 CSV
results_df.to_csv('comparison_with_paper.csv', index=False)
print("\n✅ 比較結果已儲存至: comparison_with_paper.csv")

print("\n" + "="*120)
print("分析完成！")
print("="*120)
