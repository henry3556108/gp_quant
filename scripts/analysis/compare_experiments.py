#!/usr/bin/env python3
"""
比較多個實驗的多樣性演化

使用方式：
    python scripts/analysis/compare_experiments.py \
        --exp_dirs exp1 exp2 exp3 \
        --labels "Exp1" "Exp2" "Exp3"
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compare_experiments(
    exp_dirs: list,
    labels: list = None,
    save_path: str = None,
    figsize: tuple = (14, 10),
    dpi: int = 300
):
    """
    比較多個實驗的多樣性演化
    
    Args:
        exp_dirs: 實驗目錄列表
        labels: 實驗標籤列表
        save_path: 儲存路徑
        figsize: 圖表大小
        dpi: 圖片解析度
    """
    # 讀取所有實驗的數據
    all_data = []
    
    for i, exp_dir in enumerate(exp_dirs):
        exp_path = Path(exp_dir)
        diversity_file = exp_path / 'diversity_metrics.json'
        
        if not diversity_file.exists():
            print(f"⚠️  跳過 {exp_dir}（找不到 diversity_metrics.json）")
            continue
        
        with open(diversity_file, 'r') as f:
            data = json.load(f)
        
        # 使用標籤或目錄名
        if labels and i < len(labels):
            label = labels[i]
        else:
            label = exp_path.name
        
        all_data.append({
            'label': label,
            'data': data,
            'df': pd.DataFrame(data['metrics'])
        })
    
    if not all_data:
        print("✗ 沒有可用的實驗數據")
        return 1
    
    # 創建圖表
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 顏色列表
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # 子圖 1: 多樣性分數
    ax = axes[0, 0]
    for i, exp in enumerate(all_data):
        df = exp['df']
        color = colors[i % len(colors)]
        ax.plot(df['generation'], df['diversity_score'], 
                linewidth=2, marker='o', markersize=3,
                color=color, label=exp['label'], alpha=0.8)
    
    ax.set_xlabel('世代', fontsize=11)
    ax.set_ylabel('多樣性分數', fontsize=11)
    ax.set_title('多樣性分數演化', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 子圖 2: 平均相似度
    ax = axes[0, 1]
    for i, exp in enumerate(all_data):
        df = exp['df']
        color = colors[i % len(colors)]
        ax.plot(df['generation'], df['avg_similarity'], 
                linewidth=2, marker='s', markersize=3,
                color=color, label=exp['label'], alpha=0.8)
    
    ax.set_xlabel('世代', fontsize=11)
    ax.set_ylabel('平均相似度', fontsize=11)
    ax.set_title('平均相似度演化', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 子圖 3: 標準差
    ax = axes[1, 0]
    for i, exp in enumerate(all_data):
        df = exp['df']
        color = colors[i % len(colors)]
        ax.plot(df['generation'], df['std_similarity'], 
                linewidth=2, marker='^', markersize=3,
                color=color, label=exp['label'], alpha=0.8)
    
    ax.set_xlabel('世代', fontsize=11)
    ax.set_ylabel('相似度標準差', fontsize=11)
    ax.set_title('相似度標準差演化', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 子圖 4: 統計比較表
    ax = axes[1, 1]
    ax.axis('off')
    
    # 構建比較表
    table_data = []
    headers = ['實驗', '初始多樣性', '最終多樣性', '變化', '變化率']
    
    for exp in all_data:
        df = exp['df']
        first_div = df['diversity_score'].iloc[0]
        last_div = df['diversity_score'].iloc[-1]
        change = last_div - first_div
        change_rate = (change / first_div) * 100 if first_div != 0 else 0
        
        table_data.append([
            exp['label'],
            f'{first_div:.4f}',
            f'{last_div:.4f}',
            f'{change:+.4f}',
            f'{change_rate:+.1f}%'
        ])
    
    # 繪製表格
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 設置表頭樣式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 設置交替行顏色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.suptitle('實驗多樣性比較', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 儲存或顯示
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ 圖表已儲存: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 輸出統計比較
    print()
    print("="*80)
    print("📊 統計比較")
    print("="*80)
    print()
    
    for exp in all_data:
        df = exp['df']
        print(f"{exp['label']}:")
        print(f"  初始多樣性: {df['diversity_score'].iloc[0]:.4f}")
        print(f"  最終多樣性: {df['diversity_score'].iloc[-1]:.4f}")
        print(f"  變化: {df['diversity_score'].iloc[-1] - df['diversity_score'].iloc[0]:+.4f}")
        print(f"  平均多樣性: {df['diversity_score'].mean():.4f}")
        print(f"  多樣性標準差: {df['diversity_score'].std():.4f}")
        print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='比較多個實驗的多樣性演化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 比較兩個實驗
  python scripts/analysis/compare_experiments.py \\
      --exp_dirs exp1 exp2 \\
      --labels "With Niching" "Without Niching" \\
      --output comparison.png
  
  # 比較多個實驗（自動使用目錄名作為標籤）
  python scripts/analysis/compare_experiments.py \\
      --exp_dirs exp1 exp2 exp3
        """
    )
    
    parser.add_argument(
        '--exp_dirs',
        type=str,
        nargs='+',
        required=True,
        help='實驗目錄列表'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='實驗標籤列表（可選）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='diversity_comparison.png',
        help='輸出文件路徑（預設: diversity_comparison.png）'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("📊 比較實驗多樣性")
    print("="*80)
    print()
    print(f"實驗數量: {len(args.exp_dirs)}")
    for i, exp_dir in enumerate(args.exp_dirs):
        label = args.labels[i] if args.labels and i < len(args.labels) else Path(exp_dir).name
        print(f"  {i+1}. {label}: {exp_dir}")
    print()
    
    # 執行比較
    return compare_experiments(
        args.exp_dirs,
        labels=args.labels,
        save_path=args.output
    )


if __name__ == '__main__':
    exit(main())
