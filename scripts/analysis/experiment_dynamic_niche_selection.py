"""
動態 Niche 數量選擇實驗

測試在不同 generation 上，使用 Silhouette Score 自動選擇最佳 niche 數量的效果。

實驗設計：
1. 載入已保存的 generation 資料
2. 對每個 generation，測試 k=2 到 k=8 的聚類效果
3. 使用 Silhouette Score 評估每個 k 值的聚類品質
4. 記錄最佳 k 值和計算時間
5. 生成視覺化報告

使用方式：
    python scripts/analysis/experiment_dynamic_niche_selection.py
"""

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# 添加專案路徑
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# 設置 DEAP creator（載入 pickle 前必須先設置）
from deap import base, creator, gp
import operator

# 創建 fitness 和 individual 類別
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

from gp_quant.similarity import SimilarityMatrix, ParallelSimilarityMatrix
from gp_quant.niching.clustering import NichingClusterer


class DynamicNicheExperiment:
    """動態 Niche 數量選擇實驗"""
    
    def __init__(self, 
                 generations_dir: str,
                 k_range: range = range(2, 9),
                 algorithm: str = 'kmeans'):
        """
        初始化實驗
        
        Args:
            generations_dir: generations 資料夾路徑
            k_range: 要測試的 k 值範圍（預設 2-8）
            algorithm: 聚類演算法（'kmeans' 或 'hierarchical'）
        """
        self.generations_dir = Path(generations_dir)
        self.k_range = k_range
        self.algorithm = algorithm
        
        # 實驗結果
        self.results = []
        
    def load_generation(self, gen_file: Path) -> List:
        """載入 generation 資料"""
        print(f"  📂 載入: {gen_file.name}")
        with open(gen_file, 'rb') as f:
            data = pickle.load(f)
        
        # 檢查資料格式
        if isinstance(data, dict) and 'population' in data:
            population = data['population']
            print(f"     載入完整資料 (dict 格式)")
        elif isinstance(data, list):
            population = data
            print(f"     載入 population (list 格式)")
        else:
            raise ValueError(f"未知的資料格式: {type(data)}")
        
        print(f"     Population 大小: {len(population)}")
        return population
    
    def compute_similarity_matrix(self, population: List) -> Tuple[np.ndarray, float]:
        """
        計算相似度矩陣
        
        Returns:
            (similarity_matrix, computation_time)
        """
        n = len(population)
        total_pairs = n * (n - 1) // 2
        
        print(f"  🔬 計算相似度矩陣...")
        print(f"     Population 大小: {n}")
        print(f"     需要計算: {total_pairs:,} 對相似度")
        print(f"     預估時間: {total_pairs / 50000:.1f}-{total_pairs / 30000:.1f} 分鐘")
        
        start_time = time.time()
        
        # 計算相似度矩陣（會顯示 tqdm 進度條）
        # 根據族群大小選擇計算方式
        if n >= 200:
            # 大族群使用並行計算（8 核心）
            print(f"     使用並行計算（8 核心）")
            sim_matrix = ParallelSimilarityMatrix(population, n_workers=8)
            similarity_matrix = sim_matrix.compute(show_progress=True)
        else:
            # 小族群使用序列計算
            print(f"     使用序列計算")
            sim_matrix = SimilarityMatrix(population)
            sim_matrix.compute(show_progress=True)
        
        elapsed = time.time() - start_time
        print(f"     ✓ 完成！耗時: {elapsed:.2f}s ({elapsed/60:.2f} 分鐘)")
        print(f"     平均相似度: {sim_matrix.get_average_similarity():.4f}")
        print(f"     多樣性分數: {sim_matrix.get_diversity_score():.4f}")
        
        return sim_matrix.matrix, elapsed
    
    def test_k_values(self, 
                      similarity_matrix: np.ndarray,
                      gen_name: str,
                      population_size: int) -> Dict:
        """
        測試不同 k 值的聚類效果
        
        Returns:
            包含所有 k 值測試結果的字典
        """
        # 調整 k_range 以不超過 population_size
        max_k = min(max(self.k_range), population_size - 1)
        adjusted_k_range = [k for k in self.k_range if k <= max_k and k >= 2]
        
        if not adjusted_k_range:
            print(f"  ⚠️  Population 太小 ({population_size})，無法進行聚類測試")
            return {
                'generation': gen_name,
                'k_results': [],
                'best_k': None,
                'best_silhouette': None,
                'total_test_time': 0.0
            }
        
        print(f"  🎯 測試 k 值範圍: {adjusted_k_range} (原始: {list(self.k_range)})")
        print(f"     共需測試 {len(adjusted_k_range)} 個 k 值")
        
        k_results = []
        total_start = time.time()
        
        for idx, k in enumerate(adjusted_k_range, 1):
            print(f"\n    [{idx}/{len(adjusted_k_range)}] 測試 k={k}...")
            k_start = time.time()
            
            # 執行聚類
            clusterer = NichingClusterer(
                n_clusters=k,
                algorithm=self.algorithm,
                random_state=42
            )
            clusterer.fit(similarity_matrix)
            
            k_elapsed = time.time() - k_start
            
            # 記錄結果
            stats = clusterer.get_statistics()
            k_results.append({
                'k': k,
                'silhouette_score': clusterer.silhouette_score_,
                'time': k_elapsed,
                'niche_sizes': stats['niche_sizes'],
                'min_size': stats['min_niche_size'],
                'max_size': stats['max_niche_size'],
                'avg_size': stats['avg_niche_size'],
                'std_size': stats['std_niche_size']
            })
            
            print(f"       Silhouette Score: {clusterer.silhouette_score_:.4f}")
            print(f"       時間: {k_elapsed:.3f}s")
            print(f"       Niche 大小: min={stats['min_niche_size']}, "
                  f"max={stats['max_niche_size']}, "
                  f"avg={stats['avg_niche_size']:.1f}")
        
        total_elapsed = time.time() - total_start
        
        # 找出最佳 k
        if k_results:
            best_result = max(k_results, key=lambda x: x['silhouette_score'])
            
            print(f"\n  ✨ 最佳 k 值: {best_result['k']}")
            print(f"     Silhouette Score: {best_result['silhouette_score']:.4f}")
            print(f"     總測試時間: {total_elapsed:.2f}s")
            
            return {
                'generation': gen_name,
                'k_results': k_results,
                'best_k': best_result['k'],
                'best_silhouette': best_result['silhouette_score'],
                'total_test_time': total_elapsed
            }
        else:
            return {
                'generation': gen_name,
                'k_results': [],
                'best_k': None,
                'best_silhouette': None,
                'total_test_time': total_elapsed
            }
    
    def run_experiment(self):
        """執行完整實驗"""
        print("=" * 80)
        print("動態 Niche 數量選擇實驗")
        print("=" * 80)
        print(f"資料夾: {self.generations_dir}")
        print(f"K 值範圍: {list(self.k_range)}")
        print(f"聚類演算法: {self.algorithm}")
        print("=" * 80)
        
        # 獲取所有 generation 檔案
        gen_files = sorted(self.generations_dir.glob("generation_*.pkl"))
        print(f"\n找到 {len(gen_files)} 個 generation 檔案\n")
        
        # 對每個 generation 進行實驗
        overall_start = time.time()
        
        for i, gen_file in enumerate(gen_files, 1):
            gen_start = time.time()
            
            print(f"\n{'='*80}")
            print(f"📊 Generation {i}/{len(gen_files)}: {gen_file.name}")
            print(f"{'='*80}")
            
            # 載入 population
            population = self.load_generation(gen_file)
            
            # 計算相似度矩陣
            similarity_matrix, sim_time = self.compute_similarity_matrix(population)
            
            # 測試不同 k 值
            result = self.test_k_values(similarity_matrix, gen_file.stem, len(population))
            result['similarity_time'] = sim_time
            result['population_size'] = len(population)
            
            self.results.append(result)
            
            # 顯示本 generation 的總結
            gen_elapsed = time.time() - gen_start
            overall_elapsed = time.time() - overall_start
            avg_time_per_gen = overall_elapsed / i
            remaining_gens = len(gen_files) - i
            eta = avg_time_per_gen * remaining_gens
            
            print(f"\n  ⏱️  本 Generation 耗時: {gen_elapsed:.2f}s ({gen_elapsed/60:.2f} 分鐘)")
            print(f"  📈 總進度: {i}/{len(gen_files)} ({i/len(gen_files)*100:.1f}%)")
            print(f"  ⏰ 已用時間: {overall_elapsed/60:.2f} 分鐘")
            print(f"  🔮 預估剩餘: {eta/60:.2f} 分鐘 (平均 {avg_time_per_gen/60:.2f} 分鐘/generation)")
        
        print(f"\n{'='*80}")
        print("實驗完成！")
        print(f"{'='*80}\n")
    
    def generate_summary(self) -> pd.DataFrame:
        """生成實驗摘要表格"""
        summary_data = []
        
        for result in self.results:
            summary_data.append({
                'Generation': result['generation'],
                'Population Size': result['population_size'],
                'Best K': result['best_k'] if result['best_k'] is not None else 'N/A',
                'Best Silhouette': f"{result['best_silhouette']:.4f}" if result['best_silhouette'] is not None else 'N/A',
                'Similarity Time (s)': f"{result['similarity_time']:.2f}",
                'K Testing Time (s)': f"{result['total_test_time']:.2f}",
                'Total Time (s)': f"{result['similarity_time'] + result['total_test_time']:.2f}"
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_results(self, save_path: str = None):
        """繪製實驗結果圖表"""
        if not self.results:
            print("沒有實驗結果可繪製")
            return
        
        # 過濾掉沒有有效結果的 generation
        valid_results = [r for r in self.results if r['best_k'] is not None]
        
        if not valid_results:
            print("沒有有效的實驗結果可繪製")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('動態 Niche 數量選擇實驗結果', fontsize=16, fontweight='bold')
        
        # 1. 每個 generation 的最佳 k 值
        ax1 = axes[0, 0]
        generations = [r['generation'] for r in valid_results]
        best_ks = [r['best_k'] for r in valid_results]
        best_silhouettes = [r['best_silhouette'] for r in valid_results]
        
        ax1.plot(range(len(generations)), best_ks, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Generation Index', fontsize=12)
        ax1.set_ylabel('Best K Value', fontsize=12)
        ax1.set_title('最佳 Niche 數量隨 Generation 變化', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(generations)))
        ax1.set_xticklabels([g.replace('generation_', 'Gen ') for g in generations], 
                           rotation=45, ha='right')
        
        # 2. Silhouette Score 隨 k 值變化（所有 generations）
        ax2 = axes[0, 1]
        for result in valid_results:
            k_values = [kr['k'] for kr in result['k_results']]
            silhouette_scores = [kr['silhouette_score'] for kr in result['k_results']]
            ax2.plot(k_values, silhouette_scores, 'o-', 
                    label=result['generation'].replace('generation_', 'Gen '),
                    alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('K Value (Number of Niches)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Score vs K Value', fontsize=14)
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 最佳 Silhouette Score 隨 generation 變化
        ax3 = axes[1, 0]
        ax3.plot(range(len(generations)), best_silhouettes, 's-', 
                linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Generation Index', fontsize=12)
        ax3.set_ylabel('Best Silhouette Score', fontsize=12)
        ax3.set_title('最佳 Silhouette Score 隨 Generation 變化', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(len(generations)))
        ax3.set_xticklabels([g.replace('generation_', 'Gen ') for g in generations], 
                           rotation=45, ha='right')
        
        # 4. 計算時間分析
        ax4 = axes[1, 1]
        sim_times = [r['similarity_time'] for r in valid_results]
        test_times = [r['total_test_time'] for r in valid_results]
        
        x = np.arange(len(generations))
        width = 0.35
        
        ax4.bar(x - width/2, sim_times, width, label='Similarity Matrix', alpha=0.8)
        ax4.bar(x + width/2, test_times, width, label='K Testing', alpha=0.8)
        
        ax4.set_xlabel('Generation', fontsize=12)
        ax4.set_ylabel('Time (seconds)', fontsize=12)
        ax4.set_title('計算時間分析', fontsize=14)
        ax4.set_xticks(x)
        ax4.set_xticklabels([g.replace('generation_', 'Gen ') for g in generations], 
                           rotation=45, ha='right')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 圖表已保存至: {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, save_path: str):
        """保存詳細實驗結果"""
        import json
        
        # 準備可序列化的結果
        serializable_results = []
        for result in self.results:
            serializable_result = {
                'generation': result['generation'],
                'population_size': result['population_size'],
                'similarity_time': result['similarity_time'],
                'total_test_time': result['total_test_time'],
                'best_k': result['best_k'],
                'best_silhouette': result['best_silhouette'],
                'k_results': []
            }
            
            for kr in result['k_results']:
                serializable_result['k_results'].append({
                    'k': int(kr['k']),
                    'silhouette_score': float(kr['silhouette_score']),
                    'time': float(kr['time']),
                    'niche_sizes': {int(k): int(v) for k, v in kr['niche_sizes'].items()},
                    'min_size': int(kr['min_size']),
                    'max_size': int(kr['max_size']),
                    'avg_size': float(kr['avg_size']),
                    'std_size': float(kr['std_size'])
                })
            
            serializable_results.append(serializable_result)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 詳細結果已保存至: {save_path}")


def main():
    """主函數"""
    # 設定實驗參數
    GENERATIONS_DIR = "/Users/hongyicheng/Desktop/code/研究/gp_paper/portfolio_experiment_results/portfolio_exp_sharpe_20251017_122243/generations"
    K_RANGE = range(2, 9)  # 測試 k=2 到 k=8
    ALGORITHM = 'kmeans'
    
    # 輸出目錄
    output_dir = Path(__file__).parent.parent.parent / "experiment_results" / "dynamic_niche"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建實驗
    experiment = DynamicNicheExperiment(
        generations_dir=GENERATIONS_DIR,
        k_range=K_RANGE,
        algorithm=ALGORITHM
    )
    
    # 執行實驗
    experiment.run_experiment()
    
    # 生成摘要
    print("\n" + "=" * 80)
    print("實驗摘要")
    print("=" * 80)
    summary_df = experiment.generate_summary()
    print(summary_df.to_string(index=False))
    print()
    
    # 保存摘要
    summary_path = output_dir / "dynamic_niche_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ 摘要已保存至: {summary_path}")
    
    # 保存詳細結果
    detailed_path = output_dir / "dynamic_niche_detailed_results.json"
    experiment.save_detailed_results(str(detailed_path))
    
    # 繪製圖表
    plot_path = output_dir / "dynamic_niche_results.png"
    experiment.plot_results(save_path=str(plot_path))
    
    print("\n" + "=" * 80)
    print("所有結果已保存至:")
    print(f"  - 摘要表格: {summary_path}")
    print(f"  - 詳細結果: {detailed_path}")
    print(f"  - 視覺化圖表: {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
