#!/usr/bin/env python3
"""
綜合比較兩個實驗的多樣性和樹結構

同時分析：
1. 多樣性指標演化
2. 樹結構演化
3. 兩者之間的關聯
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import pandas as pd

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_experiment_data(exp_dir: Path):
    """載入實驗的所有數據"""
    diversity_file = exp_dir / 'diversity_metrics.json'
    tree_stats_file = exp_dir / 'tree_structure_stats.json'
    
    data = {}
    
    # 載入多樣性數據
    if diversity_file.exists():
        with open(diversity_file, 'r') as f:
            data['diversity'] = json.load(f)
    else:
        print(f"⚠️  找不到 diversity_metrics.json: {diversity_file}")
        data['diversity'] = None
    
    # 載入樹結構數據
    if tree_stats_file.exists():
        with open(tree_stats_file, 'r') as f:
            data['tree_stats'] = json.load(f)
    else:
        print(f"⚠️  找不到 tree_structure_stats.json: {tree_stats_file}")
        data['tree_stats'] = None
    
    return data


def create_comprehensive_comparison(exp1_dir: Path, exp2_dir: Path, 
                                   exp1_label: str, exp2_label: str,
                                   output_file: Path):
    """創建綜合對比圖表"""
    
    # 載入數據
    print("載入實驗數據...")
    exp1_data = load_experiment_data(exp1_dir)
    exp2_data = load_experiment_data(exp2_dir)
    
    # 創建 3x2 子圖
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'Comprehensive Comparison: {exp1_label} vs {exp2_label}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 顏色設置
    color1 = '#2E86AB'  # 藍色
    color2 = '#A23B72'  # 紫紅色
    
    # ===== 第一行：多樣性指標 =====
    
    # 子圖 1: 多樣性分數演化
    ax1 = axes[0, 0]
    if exp1_data['diversity'] and exp2_data['diversity']:
        div1 = exp1_data['diversity']['metrics']
        div2 = exp2_data['diversity']['metrics']
        
        gens1 = [g['generation'] for g in div1]
        diversity1 = [g['diversity_score'] for g in div1]
        
        gens2 = [g['generation'] for g in div2]
        diversity2 = [g['diversity_score'] for g in div2]
        
        ax1.plot(gens1, diversity1, 'o-', color=color1, linewidth=2, 
                markersize=4, label=exp1_label, alpha=0.8)
        ax1.plot(gens2, diversity2, 's-', color=color2, linewidth=2, 
                markersize=4, label=exp2_label, alpha=0.8)
        ax1.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Diversity Score', fontsize=11, fontweight='bold')
        ax1.set_title('(A) Diversity Score Evolution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 添加平均線
        ax1.axhline(y=np.mean(diversity1), color=color1, linestyle=':', 
                   linewidth=1.5, alpha=0.5, label=f'{exp1_label} avg')
        ax1.axhline(y=np.mean(diversity2), color=color2, linestyle=':', 
                   linewidth=1.5, alpha=0.5, label=f'{exp2_label} avg')
    
    # 子圖 2: 平均相似度演化
    ax2 = axes[0, 1]
    if exp1_data['diversity'] and exp2_data['diversity']:
        similarity1 = [g['avg_similarity'] for g in div1]
        similarity2 = [g['avg_similarity'] for g in div2]
        
        ax2.plot(gens1, similarity1, 'o-', color=color1, linewidth=2, 
                markersize=4, label=exp1_label, alpha=0.8)
        ax2.plot(gens2, similarity2, 's-', color=color2, linewidth=2, 
                markersize=4, label=exp2_label, alpha=0.8)
        ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Average Similarity', fontsize=11, fontweight='bold')
        ax2.set_title('(B) Average Similarity Evolution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ===== 第二行：樹結構指標 =====
    
    # 子圖 3: 平均節點數
    ax3 = axes[1, 0]
    if exp1_data['tree_stats'] and exp2_data['tree_stats']:
        tree1 = exp1_data['tree_stats']['statistics']
        tree2 = exp2_data['tree_stats']['statistics']
        
        gens1_tree = [s['generation'] for s in tree1]
        nodes1 = [s['nodes']['mean'] for s in tree1]
        
        gens2_tree = [s['generation'] for s in tree2]
        nodes2 = [s['nodes']['mean'] for s in tree2]
        
        ax3.plot(gens1_tree, nodes1, 'o-', color=color1, linewidth=2, 
                markersize=4, label=exp1_label, alpha=0.8)
        ax3.plot(gens2_tree, nodes2, 's-', color=color2, linewidth=2, 
                markersize=4, label=exp2_label, alpha=0.8)
        ax3.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Average Number of Nodes', fontsize=11, fontweight='bold')
        ax3.set_title('(C) Average Tree Size Evolution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 子圖 4: 平均樹深度
    ax4 = axes[1, 1]
    if exp1_data['tree_stats'] and exp2_data['tree_stats']:
        depth1 = [s['depth']['mean'] for s in tree1]
        depth2 = [s['depth']['mean'] for s in tree2]
        
        ax4.plot(gens1_tree, depth1, 'o-', color=color1, linewidth=2, 
                markersize=4, label=exp1_label, alpha=0.8)
        ax4.plot(gens2_tree, depth2, 's-', color=color2, linewidth=2, 
                markersize=4, label=exp2_label, alpha=0.8)
        ax4.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Average Tree Depth', fontsize=11, fontweight='bold')
        ax4.set_title('(D) Average Tree Depth Evolution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
    
    # ===== 第三行：關聯分析 =====
    
    # 子圖 5: 多樣性 vs 樹大小
    ax5 = axes[2, 0]
    if exp1_data['diversity'] and exp1_data['tree_stats']:
        # 確保世代對齊
        common_gens1 = set(gens1) & set(gens1_tree)
        if common_gens1:
            div_dict1 = {g['generation']: g['diversity_score'] for g in div1}
            nodes_dict1 = {s['generation']: s['nodes']['mean'] for s in tree1}
            
            common_gens1_sorted = sorted(common_gens1)
            div_values1 = [div_dict1[g] for g in common_gens1_sorted]
            nodes_values1 = [nodes_dict1[g] for g in common_gens1_sorted]
            
            ax5.scatter(nodes_values1, div_values1, c=color1, s=50, 
                       alpha=0.6, label=exp1_label, edgecolors='white', linewidth=0.5)
    
    if exp2_data['diversity'] and exp2_data['tree_stats']:
        common_gens2 = set(gens2) & set(gens2_tree)
        if common_gens2:
            div_dict2 = {g['generation']: g['diversity_score'] for g in div2}
            nodes_dict2 = {s['generation']: s['nodes']['mean'] for s in tree2}
            
            common_gens2_sorted = sorted(common_gens2)
            div_values2 = [div_dict2[g] for g in common_gens2_sorted]
            nodes_values2 = [nodes_dict2[g] for g in common_gens2_sorted]
            
            ax5.scatter(nodes_values2, div_values2, c=color2, s=50, marker='s',
                       alpha=0.6, label=exp2_label, edgecolors='white', linewidth=0.5)
    
    ax5.set_xlabel('Average Tree Size (nodes)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Diversity Score', fontsize=11, fontweight='bold')
    ax5.set_title('(E) Diversity vs Tree Size', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 子圖 6: 計算時間對比
    ax6 = axes[2, 1]
    if exp1_data['diversity'] and exp2_data['diversity']:
        time1 = [g['computation_time'] for g in div1]
        time2 = [g['computation_time'] for g in div2]
        
        ax6.plot(gens1, time1, 'o-', color=color1, linewidth=2, 
                markersize=4, label=exp1_label, alpha=0.8)
        ax6.plot(gens2, time2, 's-', color=color2, linewidth=2, 
                markersize=4, label=exp2_label, alpha=0.8)
        ax6.set_xlabel('Generation', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Computation Time (seconds)', fontsize=11, fontweight='bold')
        ax6.set_title('(F) Computation Time per Generation', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        # 添加趨勢線
        if len(time2) > 5:
            z = np.polyfit(gens2, time2, 2)
            p = np.poly1d(z)
            ax6.plot(gens2, p(gens2), "--", color=color2, alpha=0.5, linewidth=2)
    
    # 調整布局
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # 保存圖表
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: {output_file}")
    
    return fig


def generate_summary_report(exp1_dir: Path, exp2_dir: Path,
                           exp1_label: str, exp2_label: str,
                           output_file: Path):
    """生成文字摘要報告"""
    
    exp1_data = load_experiment_data(exp1_dir)
    exp2_data = load_experiment_data(exp2_dir)
    
    report = []
    report.append("=" * 80)
    report.append("📊 實驗綜合比較報告")
    report.append("=" * 80)
    report.append("")
    report.append(f"實驗 1: {exp1_label}")
    report.append(f"  路徑: {exp1_dir}")
    report.append("")
    report.append(f"實驗 2: {exp2_label}")
    report.append(f"  路徑: {exp2_dir}")
    report.append("")
    
    # 多樣性比較
    if exp1_data['diversity'] and exp2_data['diversity']:
        report.append("=" * 80)
        report.append("🔬 多樣性指標比較")
        report.append("=" * 80)
        report.append("")
        
        div1 = exp1_data['diversity']['metrics']
        div2 = exp2_data['diversity']['metrics']
        
        # 計算平均值
        avg_div1 = np.mean([g['diversity_score'] for g in div1])
        avg_div2 = np.mean([g['diversity_score'] for g in div2])
        
        avg_sim1 = np.mean([g['avg_similarity'] for g in div1])
        avg_sim2 = np.mean([g['avg_similarity'] for g in div2])
        
        report.append(f"平均多樣性分數:")
        report.append(f"  {exp1_label}: {avg_div1:.4f}")
        report.append(f"  {exp2_label}: {avg_div2:.4f}")
        report.append(f"  差異: {abs(avg_div1 - avg_div2):.4f} ({((avg_div2/avg_div1 - 1) * 100):+.2f}%)")
        report.append("")
        
        report.append(f"平均相似度:")
        report.append(f"  {exp1_label}: {avg_sim1:.4f}")
        report.append(f"  {exp2_label}: {avg_sim2:.4f}")
        report.append(f"  差異: {abs(avg_sim1 - avg_sim2):.4f}")
        report.append("")
        
        # 多樣性穩定性
        std_div1 = np.std([g['diversity_score'] for g in div1])
        std_div2 = np.std([g['diversity_score'] for g in div2])
        
        report.append(f"多樣性穩定性 (標準差):")
        report.append(f"  {exp1_label}: {std_div1:.4f} {'(更穩定)' if std_div1 < std_div2 else ''}")
        report.append(f"  {exp2_label}: {std_div2:.4f} {'(更穩定)' if std_div2 < std_div1 else ''}")
        report.append("")
    
    # 樹結構比較
    if exp1_data['tree_stats'] and exp2_data['tree_stats']:
        report.append("=" * 80)
        report.append("🌲 樹結構比較")
        report.append("=" * 80)
        report.append("")
        
        tree1 = exp1_data['tree_stats']['statistics']
        tree2 = exp2_data['tree_stats']['statistics']
        
        # 初始和最終樹大小
        nodes1_start = tree1[0]['nodes']['mean']
        nodes1_end = tree1[-1]['nodes']['mean']
        nodes1_growth = (nodes1_end / nodes1_start - 1) * 100
        
        nodes2_start = tree2[0]['nodes']['mean']
        nodes2_end = tree2[-1]['nodes']['mean']
        nodes2_growth = (nodes2_end / nodes2_start - 1) * 100
        
        report.append(f"平均節點數:")
        report.append(f"  {exp1_label}:")
        report.append(f"    初始 (Gen 1): {nodes1_start:.2f}")
        report.append(f"    最終 (Gen {tree1[-1]['generation']}): {nodes1_end:.2f}")
        report.append(f"    增長: {nodes1_growth:+.1f}%")
        report.append("")
        report.append(f"  {exp2_label}:")
        report.append(f"    初始 (Gen 1): {nodes2_start:.2f}")
        report.append(f"    最終 (Gen {tree2[-1]['generation']}): {nodes2_end:.2f}")
        report.append(f"    增長: {nodes2_growth:+.1f}%")
        report.append("")
        
        # 樹深度
        depth1_start = tree1[0]['depth']['mean']
        depth1_end = tree1[-1]['depth']['mean']
        depth1_growth = (depth1_end / depth1_start - 1) * 100
        
        depth2_start = tree2[0]['depth']['mean']
        depth2_end = tree2[-1]['depth']['mean']
        depth2_growth = (depth2_end / depth2_start - 1) * 100
        
        report.append(f"平均樹深度:")
        report.append(f"  {exp1_label}:")
        report.append(f"    初始: {depth1_start:.2f}")
        report.append(f"    最終: {depth1_end:.2f}")
        report.append(f"    增長: {depth1_growth:+.1f}%")
        report.append("")
        report.append(f"  {exp2_label}:")
        report.append(f"    初始: {depth2_start:.2f}")
        report.append(f"    最終: {depth2_end:.2f}")
        report.append(f"    增長: {depth2_growth:+.1f}%")
        report.append("")
        
        # Bloat 控制效果
        report.append("💡 Bloat 控制效果:")
        if nodes1_growth < nodes2_growth:
            report.append(f"  {exp1_label} 更好地控制了樹膨脹")
            report.append(f"  節點數增長差異: {abs(nodes1_growth - nodes2_growth):.1f} 百分點")
        else:
            report.append(f"  {exp2_label} 更好地控制了樹膨脹")
            report.append(f"  節點數增長差異: {abs(nodes1_growth - nodes2_growth):.1f} 百分點")
        report.append("")
    
    # 計算效率比較
    if exp1_data['diversity'] and exp2_data['diversity']:
        report.append("=" * 80)
        report.append("⚡ 計算效率比較")
        report.append("=" * 80)
        report.append("")
        
        time1 = [g['computation_time'] for g in div1]
        time2 = [g['computation_time'] for g in div2]
        
        avg_time1 = np.mean(time1)
        avg_time2 = np.mean(time2)
        
        total_time1 = sum(time1)
        total_time2 = sum(time2)
        
        report.append(f"平均計算時間 (每世代):")
        report.append(f"  {exp1_label}: {avg_time1:.2f} 秒 ({avg_time1/60:.2f} 分鐘)")
        report.append(f"  {exp2_label}: {avg_time2:.2f} 秒 ({avg_time2/60:.2f} 分鐘)")
        report.append(f"  差異: {abs(avg_time1 - avg_time2):.2f} 秒")
        report.append("")
        
        report.append(f"總計算時間:")
        report.append(f"  {exp1_label}: {total_time1:.2f} 秒 ({total_time1/3600:.2f} 小時)")
        report.append(f"  {exp2_label}: {total_time2:.2f} 秒 ({total_time2/3600:.2f} 小時)")
        report.append("")
        
        # 計算時間增長趨勢
        time1_growth = (time1[-1] / time1[0] - 1) * 100
        time2_growth = (time2[-1] / time2[0] - 1) * 100
        
        report.append(f"計算時間增長:")
        report.append(f"  {exp1_label}: {time1_growth:+.1f}%")
        report.append(f"  {exp2_label}: {time2_growth:+.1f}%")
        report.append("")
    
    # 關鍵結論
    report.append("=" * 80)
    report.append("🎯 關鍵結論")
    report.append("=" * 80)
    report.append("")
    
    if exp1_data['diversity'] and exp2_data['diversity'] and \
       exp1_data['tree_stats'] and exp2_data['tree_stats']:
        
        # 判斷哪個實驗更好
        conclusions = []
        
        if avg_div1 > avg_div2:
            conclusions.append(f"✓ {exp1_label} 維持了更高的多樣性")
        else:
            conclusions.append(f"✓ {exp2_label} 維持了更高的多樣性")
        
        if std_div1 < std_div2:
            conclusions.append(f"✓ {exp1_label} 的多樣性更穩定")
        else:
            conclusions.append(f"✓ {exp2_label} 的多樣性更穩定")
        
        if nodes1_growth < nodes2_growth:
            conclusions.append(f"✓ {exp1_label} 更好地控制了樹膨脹")
        else:
            conclusions.append(f"✓ {exp2_label} 更好地控制了樹膨脹")
        
        if avg_time1 < avg_time2:
            conclusions.append(f"✓ {exp1_label} 的計算效率更高")
        else:
            conclusions.append(f"✓ {exp2_label} 的計算效率更高")
        
        for conclusion in conclusions:
            report.append(conclusion)
        report.append("")
    
    report.append("=" * 80)
    
    # 輸出到文件和終端
    report_text = "\n".join(report)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print()
    print(f"✅ 報告已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='綜合比較兩個實驗')
    parser.add_argument('--exp1', type=str, required=True,
                       help='實驗 1 目錄')
    parser.add_argument('--exp2', type=str, required=True,
                       help='實驗 2 目錄')
    parser.add_argument('--label1', type=str, default='Experiment 1',
                       help='實驗 1 標籤')
    parser.add_argument('--label2', type=str, default='Experiment 2',
                       help='實驗 2 標籤')
    parser.add_argument('--output_plot', type=str, 
                       default='comprehensive_comparison.png',
                       help='輸出圖表路徑')
    parser.add_argument('--output_report', type=str,
                       default='comprehensive_comparison_report.txt',
                       help='輸出報告路徑')
    
    args = parser.parse_args()
    
    exp1_dir = Path(args.exp1)
    exp2_dir = Path(args.exp2)
    
    if not exp1_dir.exists():
        print(f"✗ 找不到實驗 1 目錄: {exp1_dir}")
        return 1
    
    if not exp2_dir.exists():
        print(f"✗ 找不到實驗 2 目錄: {exp2_dir}")
        return 1
    
    print("=" * 80)
    print("📊 綜合實驗比較分析")
    print("=" * 80)
    print()
    print(f"實驗 1: {args.label1}")
    print(f"  {exp1_dir}")
    print()
    print(f"實驗 2: {args.label2}")
    print(f"  {exp2_dir}")
    print()
    
    # 生成圖表
    print("生成對比圖表...")
    create_comprehensive_comparison(exp1_dir, exp2_dir, 
                                   args.label1, args.label2,
                                   Path(args.output_plot))
    print()
    
    # 生成報告
    print("生成摘要報告...")
    generate_summary_report(exp1_dir, exp2_dir,
                          args.label1, args.label2,
                          Path(args.output_report))
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
