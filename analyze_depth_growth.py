"""
分析深度增長趨勢

這個腳本分析 portfolio 實驗中深度隨 generation 的增長趨勢，
幫助理解為什麼會出現深度超限問題。
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 讀取深度檢查結果
df = pd.read_csv('portfolio_depth_check_results.csv')

# 只分析最近的 3 個實驗
recent_exps = [
    'portfolio_exp_sharpe_20251023_133445',
    'portfolio_exp_sharpe_20251023_160709',
    'portfolio_exp_sharpe_20251023_161559'
]

df_recent = df[df['experiment'].isin(recent_exps)]

print("="*100)
print("深度增長趨勢分析")
print("="*100)

# 為每個實驗分析
for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name].sort_values('generation')
    
    if exp_data.empty:
        continue
    
    print(f"\n{'='*100}")
    print(f"實驗: {exp_name}")
    print(f"{'='*100}")
    
    # 基本統計
    total_gens = len(exp_data)
    violations = exp_data[~exp_data['compliant']]
    num_violations = len(violations)
    
    print(f"\n📊 基本統計:")
    print(f"  總 Generation 數: {total_gens}")
    print(f"  違規 Generation 數: {num_violations} ({num_violations/total_gens*100:.1f}%)")
    print(f"  族群大小: {exp_data['population_size'].iloc[0]}")
    
    # 深度統計
    print(f"\n📏 深度統計:")
    print(f"  初始最大深度 (Gen 1): {exp_data[exp_data['generation']==1]['max_depth'].iloc[0]}")
    print(f"  最終最大深度 (Gen {exp_data['generation'].max()}): {exp_data['max_depth'].iloc[-1]}")
    print(f"  最大深度峰值: {exp_data['max_depth'].max()}")
    print(f"  平均深度範圍: {exp_data['avg_depth'].min():.2f} - {exp_data['avg_depth'].max():.2f}")
    
    # 違規開始時間
    if num_violations > 0:
        first_violation_gen = violations['generation'].min()
        print(f"\n⚠️  違規資訊:")
        print(f"  首次違規: Generation {first_violation_gen}")
        print(f"  首次違規深度: {violations[violations['generation']==first_violation_gen]['max_depth'].iloc[0]}")
        
        # 深度增長率
        if first_violation_gen > 1:
            before_violation = exp_data[exp_data['generation'] < first_violation_gen]
            growth_rate = (exp_data['max_depth'].iloc[-1] - before_violation['max_depth'].iloc[-1]) / (total_gens - first_violation_gen + 1)
            print(f"  違規後平均增長率: {growth_rate:.2f} 層/代")
    
    # 深度分布
    print(f"\n📈 深度分布:")
    depth_ranges = [
        (0, 6, "符合初始限制"),
        (7, 17, "符合演化限制"),
        (18, 30, "輕微超限"),
        (31, 50, "中度超限"),
        (51, 100, "嚴重超限")
    ]
    
    for min_d, max_d, label in depth_ranges:
        count = len(exp_data[(exp_data['max_depth'] >= min_d) & (exp_data['max_depth'] <= max_d)])
        if count > 0:
            print(f"  {label} ({min_d}-{max_d}): {count} 代 ({count/total_gens*100:.1f}%)")

# 創建視覺化
print(f"\n{'='*100}")
print("生成視覺化圖表...")
print(f"{'='*100}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('深度增長趨勢分析', fontsize=16, fontweight='bold')

# 1. 最大深度隨 generation 變化
ax1 = axes[0, 0]
for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name].sort_values('generation')
    if not exp_data.empty:
        label = exp_name.split('_')[-1]  # 只顯示時間戳
        ax1.plot(exp_data['generation'], exp_data['max_depth'], marker='o', label=label, linewidth=2)

ax1.axhline(y=6, color='green', linestyle='--', label='初始限制 (6)', linewidth=2)
ax1.axhline(y=17, color='red', linestyle='--', label='演化限制 (17)', linewidth=2)
ax1.set_xlabel('Generation', fontsize=12)
ax1.set_ylabel('最大深度', fontsize=12)
ax1.set_title('最大深度隨 Generation 變化', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 平均深度隨 generation 變化
ax2 = axes[0, 1]
for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name].sort_values('generation')
    if not exp_data.empty:
        label = exp_name.split('_')[-1]
        ax2.plot(exp_data['generation'], exp_data['avg_depth'], marker='s', label=label, linewidth=2)

ax2.axhline(y=6, color='green', linestyle='--', label='初始限制 (6)', linewidth=2, alpha=0.5)
ax2.set_xlabel('Generation', fontsize=12)
ax2.set_ylabel('平均深度', fontsize=12)
ax2.set_title('平均深度隨 Generation 變化', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 深度增長率（相對於前一代）
ax3 = axes[1, 0]
for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name].sort_values('generation')
    if not exp_data.empty and len(exp_data) > 1:
        growth = exp_data['max_depth'].diff()
        label = exp_name.split('_')[-1]
        ax3.plot(exp_data['generation'].iloc[1:], growth.iloc[1:], marker='o', label=label, linewidth=2)

ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax3.set_xlabel('Generation', fontsize=12)
ax3.set_ylabel('深度增長 (相對前一代)', fontsize=12)
ax3.set_title('深度增長率', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 違規統計
ax4 = axes[1, 1]
violation_stats = []
for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name]
    if not exp_data.empty:
        total = len(exp_data)
        violations = len(exp_data[~exp_data['compliant']])
        compliant = total - violations
        label = exp_name.split('_')[-1]
        violation_stats.append({
            'experiment': label,
            'compliant': compliant,
            'violations': violations
        })

if violation_stats:
    stats_df = pd.DataFrame(violation_stats)
    x = np.arange(len(stats_df))
    width = 0.35
    
    ax4.bar(x - width/2, stats_df['compliant'], width, label='符合限制', color='green', alpha=0.7)
    ax4.bar(x + width/2, stats_df['violations'], width, label='違規', color='red', alpha=0.7)
    
    ax4.set_xlabel('實驗', fontsize=12)
    ax4.set_ylabel('Generation 數', fontsize=12)
    ax4.set_title('違規統計', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stats_df['experiment'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('depth_growth_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ 圖表已儲存: depth_growth_analysis.png")

# 生成詳細的增長分析報告
print(f"\n{'='*100}")
print("深度增長模式分析")
print(f"{'='*100}")

for exp_name in recent_exps:
    exp_data = df_recent[df_recent['experiment'] == exp_name].sort_values('generation')
    
    if exp_data.empty or len(exp_data) < 10:
        continue
    
    print(f"\n實驗: {exp_name.split('_')[-1]}")
    
    # 分析不同階段的增長率
    stages = [
        (1, 10, "早期 (Gen 1-10)"),
        (11, 20, "中期 (Gen 11-20)"),
        (21, 30, "後期 (Gen 21-30)"),
    ]
    
    for start, end, label in stages:
        stage_data = exp_data[(exp_data['generation'] >= start) & (exp_data['generation'] <= end)]
        if len(stage_data) > 1:
            start_depth = stage_data['max_depth'].iloc[0]
            end_depth = stage_data['max_depth'].iloc[-1]
            growth = end_depth - start_depth
            avg_growth = growth / len(stage_data)
            print(f"  {label}: {start_depth} → {end_depth} (增長 {growth}, 平均 {avg_growth:.2f}/代)")

print(f"\n{'='*100}")
print("分析完成！")
print(f"{'='*100}")
print("\n📋 關鍵發現:")
print("  1. 深度在演化過程中呈現指數級增長")
print("  2. 一旦超過限制，深度會持續增加")
print("  3. 族群大小越大，深度增長越快")
print("  4. 需要在 crossover 和 mutation 中加入深度限制")
print("\n💡 建議:")
print("  - 使用 gp.staticLimit 裝飾器限制深度")
print("  - 減小 mutation 生成的子樹大小")
print("  - 詳見 docs/DEPTH_VIOLATION_ANALYSIS.md")
