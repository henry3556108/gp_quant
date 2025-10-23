"""
分析實驗結果
Analyze all experimental results from experiments_results folder
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 設定圖表樣式（圖表使用英文，不需要中文字體）
plt.rcParams['axes.unicode_minus'] = False

def load_all_results():
    """Load all experimental results"""
    results = []
    base_path = Path('experiments_results')
    
    # 遍歷所有ticker目錄
    for ticker_dir in base_path.iterdir():
        if not ticker_dir.is_dir() or ticker_dir.name.startswith('.'):
            continue
            
        # 找所有result.json檔案
        for json_file in ticker_dir.glob('*_result.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"讀取失敗: {json_file}, 錯誤: {e}")
    
    return pd.DataFrame(results)

def analyze_results(df):
    """分析實驗結果"""
    print("\n" + "="*100)
    print("📊 實驗結果分析")
    print("="*100)
    
    # 基本資訊
    print(f"\n總實驗數: {len(df)}")
    print(f"股票數量: {df['ticker'].nunique()}")
    print(f"股票列表: {', '.join(df['ticker'].unique())}")
    print(f"訓練期類型: {df['period'].unique()}")
    
    # 1. 按股票和訓練期分組統計
    print("\n" + "="*100)
    print("📈 各股票各訓練期的表現統計 (10次實驗)")
    print("="*100)
    
    summary = df.groupby(['ticker', 'period']).agg({
        'test_excess_return': ['mean', 'std', 'min', 'max', 'median'],
        'train_excess_return': ['mean', 'std'],
        'test_gp_return': ['mean'],
        'test_bh_return': ['mean'],
        'duration': ['mean']
    }).round(2)
    
    print(summary)
    
    # 2. 勝率分析（超越Buy-and-Hold）
    print("\n" + "="*100)
    print("🎯 樣本外勝率分析（超越Buy-and-Hold的比例）")
    print("="*100)
    
    for ticker in sorted(df['ticker'].unique()):
        print(f"\n【{ticker}】")
        for period in ['短訓練期', '長訓練期']:
            subset = df[(df['ticker'] == ticker) & (df['period'] == period)]
            if len(subset) == 0:
                continue
                
            wins = (subset['test_excess_return'] > 0).sum()
            total = len(subset)
            win_rate = (wins / total) * 100
            
            avg_excess = subset['test_excess_return'].mean()
            median_excess = subset['test_excess_return'].median()
            std_excess = subset['test_excess_return'].std()
            
            status = "✅" if win_rate >= 50 else "❌"
            
            print(f"  {period}: {wins}/{total} ({win_rate:.0f}%) {status}")
            print(f"    平均超額報酬: ${avg_excess:,.2f}")
            print(f"    中位數超額報酬: ${median_excess:,.2f}")
            print(f"    標準差: ${std_excess:,.2f}")
    
    # 3. 短期 vs 長期訓練比較
    print("\n" + "="*100)
    print("⚖️  短訓練期 vs 長訓練期 整體比較")
    print("="*100)
    
    short_df = df[df['period'] == '短訓練期']
    long_df = df[df['period'] == '長訓練期']
    
    print(f"\n短訓練期 (n={len(short_df)}):")
    print(f"  平均超額報酬: ${short_df['test_excess_return'].mean():,.2f}")
    print(f"  中位數超額報酬: ${short_df['test_excess_return'].median():,.2f}")
    print(f"  標準差: ${short_df['test_excess_return'].std():,.2f}")
    print(f"  勝率: {(short_df['test_excess_return'] > 0).sum()}/{len(short_df)} ({(short_df['test_excess_return'] > 0).mean()*100:.1f}%)")
    
    print(f"\n長訓練期 (n={len(long_df)}):")
    print(f"  平均超額報酬: ${long_df['test_excess_return'].mean():,.2f}")
    print(f"  中位數超額報酬: ${long_df['test_excess_return'].median():,.2f}")
    print(f"  標準差: ${long_df['test_excess_return'].std():,.2f}")
    print(f"  勝率: {(long_df['test_excess_return'] > 0).sum()}/{len(long_df)} ({(long_df['test_excess_return'] > 0).mean()*100:.1f}%)")
    
    # 統計檢定
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(short_df['test_excess_return'], long_df['test_excess_return'])
    print(f"\nT檢定結果:")
    print(f"  t統計量: {t_stat:.4f}")
    print(f"  p值: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  結論: 兩組有顯著差異 (p < 0.05)")
    else:
        print(f"  結論: 兩組無顯著差異 (p >= 0.05)")
    
    # 4. 最佳與最差表現
    print("\n" + "="*100)
    print("🏆 最佳與最差表現")
    print("="*100)
    
    best_idx = df['test_excess_return'].idxmax()
    worst_idx = df['test_excess_return'].idxmin()
    
    best = df.loc[best_idx]
    worst = df.loc[worst_idx]
    
    print(f"\n最佳表現:")
    print(f"  股票: {best['ticker']}")
    print(f"  訓練期: {best['period']}")
    print(f"  Run: {best['run_number']}")
    print(f"  樣本外超額報酬: ${best['test_excess_return']:,.2f}")
    print(f"  樣本外GP報酬: ${best['test_gp_return']:,.2f}")
    print(f"  樣本外B&H報酬: ${best['test_bh_return']:,.2f}")
    
    print(f"\n最差表現:")
    print(f"  股票: {worst['ticker']}")
    print(f"  訓練期: {worst['period']}")
    print(f"  Run: {worst['run_number']}")
    print(f"  樣本外超額報酬: ${worst['test_excess_return']:,.2f}")
    print(f"  樣本外GP報酬: ${worst['test_gp_return']:,.2f}")
    print(f"  樣本外B&H報酬: ${worst['test_bh_return']:,.2f}")
    
    # 5. 訓練期表現 vs 測試期表現相關性
    print("\n" + "="*100)
    print("🔗 訓練期表現與測試期表現的相關性")
    print("="*100)
    
    correlation = df['train_excess_return'].corr(df['test_excess_return'])
    print(f"\n整體相關係數: {correlation:.4f}")
    
    for period in ['短訓練期', '長訓練期']:
        subset = df[df['period'] == period]
        corr = subset['train_excess_return'].corr(subset['test_excess_return'])
        print(f"{period}相關係數: {corr:.4f}")
    
    # 6. 執行時間統計
    print("\n" + "="*100)
    print("⏱️  執行時間統計")
    print("="*100)
    
    print(f"\n平均執行時間: {df['duration'].mean():.2f} 秒")
    print(f"總執行時間: {df['duration'].sum()/60:.2f} 分鐘")
    
    for period in ['短訓練期', '長訓練期']:
        subset = df[df['period'] == period]
        print(f"{period}平均執行時間: {subset['duration'].mean():.2f} 秒")
    
    return df

def create_visualizations(df):
    """創建視覺化圖表"""
    print("\n" + "="*100)
    print("📊 生成視覺化圖表...")
    print("="*100)
    
    # 設定圖表風格
    sns.set_style("whitegrid")
    
    # 1. 箱型圖：各股票各訓練期的超額報酬分布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experimental Results Analysis', fontsize=16, fontweight='bold')
    
    # 1.1 測試期超額報酬箱型圖
    ax1 = axes[0, 0]
    df_plot = df.copy()
    # 將中文訓練期轉換為英文
    df_plot['period_en'] = df_plot['period'].map({'短訓練期': 'Short Training', '長訓練期': 'Long Training'})
    sns.boxplot(data=df_plot, x='ticker', y='test_excess_return', hue='period_en', ax=ax1)
    ax1.set_title('Test Excess Return Distribution by Ticker and Training Period')
    ax1.set_ylabel('Excess Return ($)')
    ax1.set_xlabel('Ticker')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.legend(title='Training Period')
    
    # 1.2 勝率比較
    ax2 = axes[0, 1]
    win_rates = []
    labels = []
    for ticker in sorted(df['ticker'].unique()):
        for period in ['短訓練期', '長訓練期']:
            subset = df[(df['ticker'] == ticker) & (df['period'] == period)]
            if len(subset) > 0:
                win_rate = (subset['test_excess_return'] > 0).mean() * 100
                win_rates.append(win_rate)
                period_en = 'Short' if period == '短訓練期' else 'Long'
                labels.append(f"{ticker}\n{period_en}")
    
    x_pos = np.arange(len(labels))
    colors = ['skyblue' if 'Short' in label else 'lightcoral' for label in labels]
    ax2.bar(x_pos, win_rates, color=colors)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Ticker and Training Period (vs Buy-and-Hold)')
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% Baseline')
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    # 1.3 訓練期 vs 測試期超額報酬散點圖
    ax3 = axes[1, 0]
    for period, color, label in [('短訓練期', 'blue', 'Short Training'), ('長訓練期', 'red', 'Long Training')]:
        subset = df[df['period'] == period]
        ax3.scatter(subset['train_excess_return'], subset['test_excess_return'], 
                   alpha=0.6, label=label, color=color)
    ax3.set_xlabel('Training Excess Return ($)')
    ax3.set_ylabel('Test Excess Return ($)')
    ax3.set_title('Training vs Test Excess Return Correlation')
    ax3.legend()
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    
    # 1.4 平均超額報酬比較
    ax4 = axes[1, 1]
    summary_data = df.groupby(['ticker', 'period'])['test_excess_return'].mean().unstack()
    # 重新命名欄位為英文
    summary_data.columns = ['Long Training', 'Short Training']
    summary_data.plot(kind='bar', ax=ax4, color=['lightcoral', 'skyblue'])
    ax4.set_title('Average Test Excess Return by Ticker')
    ax4.set_ylabel('Average Excess Return ($)')
    ax4.set_xlabel('Ticker')
    ax4.legend(title='Training Period')
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('experiments_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已儲存: experiments_analysis.png")
    
    # 2. 詳細的分布圖
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Detailed Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 2.1 測試期超額報酬直方圖
    ax1 = axes2[0, 0]
    df[df['period'] == '短訓練期']['test_excess_return'].hist(ax=ax1, bins=20, alpha=0.7, label='Short Training', color='blue')
    df[df['period'] == '長訓練期']['test_excess_return'].hist(ax=ax1, bins=20, alpha=0.7, label='Long Training', color='red')
    ax1.set_xlabel('Test Excess Return ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Test Excess Return Distribution')
    ax1.legend()
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 2.2 各股票的表現一致性
    ax2 = axes2[0, 1]
    consistency_data = []
    for ticker in sorted(df['ticker'].unique()):
        for period in ['短訓練期', '長訓練期']:
            subset = df[(df['ticker'] == ticker) & (df['period'] == period)]
            if len(subset) > 0:
                std = subset['test_excess_return'].std()
                mean = subset['test_excess_return'].mean()
                period_en = 'Short Training' if period == '短訓練期' else 'Long Training'
                consistency_data.append({
                    'ticker': ticker,
                    'period': period_en,
                    'cv': std / abs(mean) if mean != 0 else np.inf  # 變異係數
                })
    
    consistency_df = pd.DataFrame(consistency_data)
    consistency_pivot = consistency_df.pivot(index='ticker', columns='period', values='cv')
    consistency_pivot.plot(kind='bar', ax=ax2, color=['lightcoral', 'skyblue'])
    ax2.set_title('Performance Consistency (CV, lower is more stable)')
    ax2.set_ylabel('Coefficient of Variation (CV)')
    ax2.set_xlabel('Ticker')
    ax2.legend(title='Training Period')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2.3 執行時間比較
    ax3 = axes2[1, 0]
    df_plot = df.copy()
    df_plot['period_en'] = df_plot['period'].map({'短訓練期': 'Short Training', '長訓練期': 'Long Training'})
    df_plot.boxplot(column='duration', by='period_en', ax=ax3)
    ax3.set_title('Execution Time Distribution')
    ax3.set_ylabel('Duration (seconds)')
    ax3.set_xlabel('Training Period')
    plt.suptitle('')  # 移除自動標題
    
    # 2.4 累積勝率
    ax4 = axes2[1, 1]
    for ticker in sorted(df['ticker'].unique()):
        for period, style, label_suffix in [('短訓練期', '-', 'Short'), ('長訓練期', '--', 'Long')]:
            subset = df[(df['ticker'] == ticker) & (df['period'] == period)].sort_values('run_number')
            if len(subset) > 0:
                cumulative_wins = (subset['test_excess_return'] > 0).cumsum()
                cumulative_rate = cumulative_wins / subset['run_number'] * 100
                ax4.plot(subset['run_number'], cumulative_rate, 
                        label=f"{ticker} {label_suffix}", linestyle=style)
    
    ax4.set_xlabel('Run Number')
    ax4.set_ylabel('Cumulative Win Rate (%)')
    ax4.set_title('Cumulative Win Rate Trend')
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('experiments_analysis_detailed.png', dpi=300, bbox_inches='tight')
    print("✅ 詳細圖表已儲存: experiments_analysis_detailed.png")

def main():
    """主函數"""
    print("\n" + "🚀"*50)
    print("開始分析實驗結果")
    print("🚀"*50)
    
    # 載入資料
    df = load_all_results()
    
    if len(df) == 0:
        print("❌ 沒有找到實驗結果！")
        return
    
    # 分析結果
    df = analyze_results(df)
    
    # 創建視覺化
    try:
        create_visualizations(df)
    except Exception as e:
        print(f"⚠️  視覺化生成失敗: {e}")
    
    # 儲存詳細結果
    df.to_csv('experiments_analysis_detailed.csv', index=False)
    print("\n✅ 詳細分析結果已儲存: experiments_analysis_detailed.csv")
    
    print("\n" + "🎉"*50)
    print("分析完成！")
    print("🎉"*50 + "\n")

if __name__ == "__main__":
    main()
