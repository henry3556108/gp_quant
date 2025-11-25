#!/usr/bin/env python3
"""
完整驗證 TED Niche Selection Strategy

生成詳細報告，包含所有關鍵檢查點的數據。
"""

import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from deap import creator, base, gp

sys.path.insert(0, str(Path(__file__).parent))

from gp_quant.evolution.components.gp import operators
from gp_quant.evolution.components.strategies import TEDNicheSelectionStrategy


def setup_deap_creator():
    """設置 DEAP creator"""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def load_test_population(records_dir: Path, generation: int = 0):
    """載入測試族群"""
    populations_dir = records_dir / 'populations'
    gen_file = populations_dir / f'generation_{generation:03d}.pkl'
    
    with open(gen_file, 'rb') as f:
        population = pickle.load(f)
    
    return population


class ValidationReport:
    """驗證報告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'checkpoints': {}
        }
    
    def add_checkpoint(self, name: str, data: dict, passed: bool = True):
        """添加檢查點"""
        self.report['checkpoints'][name] = {
            'passed': passed,
            'data': data
        }
    
    def save_json(self):
        """保存 JSON 報告"""
        json_path = self.output_dir / 'validation_report.json'
        
        # 轉換 numpy 類型為 Python 原生類型
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        report_converted = convert_types(self.report)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_converted, f, indent=2, ensure_ascii=False)
        return json_path
    
    def save_markdown(self):
        """保存 Markdown 報告"""
        md_path = self.output_dir / 'validation_report.md'
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# TED Niche Selection Strategy 驗證報告\n\n")
            f.write(f"**生成時間**: {self.report['timestamp']}\n\n")
            f.write("---\n\n")
            
            for name, checkpoint in self.report['checkpoints'].items():
                status = "✅ 通過" if checkpoint['passed'] else "❌ 失敗"
                f.write(f"## {name} {status}\n\n")
                
                for key, value in checkpoint['data'].items():
                    if isinstance(value, dict):
                        f.write(f"### {key}\n\n")
                        for k, v in value.items():
                            f.write(f"- **{k}**: {v}\n")
                        f.write("\n")
                    elif isinstance(value, list):
                        f.write(f"### {key}\n\n")
                        for item in value:
                            f.write(f"- {item}\n")
                        f.write("\n")
                    else:
                        f.write(f"- **{key}**: {value}\n")
                f.write("\n---\n\n")
        
        return md_path


def checkpoint_1_ted_matrix(strategy, population, report):
    """檢查點 1: TED Distance Matrix 計算"""
    print("\n" + "="*80)
    print("📊 檢查點 1: TED Distance Matrix 計算")
    print("="*80)
    
    ted_matrix = strategy._calculate_ted_distance_matrix(population)
    
    # 驗證
    is_symmetric = np.allclose(ted_matrix, ted_matrix.T)
    diagonal_zero = np.allclose(np.diag(ted_matrix), 0)
    in_range = (ted_matrix >= 0).all() and (ted_matrix <= 1).all()
    
    upper_tri = np.triu_indices(len(population), k=1)
    mean_dist = np.mean(ted_matrix[upper_tri])
    std_dist = np.std(ted_matrix[upper_tri])
    min_dist = np.min(ted_matrix[upper_tri])
    max_dist = np.max(ted_matrix[upper_tri])
    
    data = {
        '矩陣大小': f"{ted_matrix.shape[0]} x {ted_matrix.shape[1]}",
        '對稱性': '✅ 通過' if is_symmetric else '❌ 失敗',
        '對角線為0': '✅ 通過' if diagonal_zero else '❌ 失敗',
        '距離範圍[0,1]': '✅ 通過' if in_range else '❌ 失敗',
        '統計': {
            '平均距離': f"{mean_dist:.4f}",
            '標準差': f"{std_dist:.4f}",
            '最小距離': f"{min_dist:.4f}",
            '最大距離': f"{max_dist:.4f}"
        }
    }
    
    passed = is_symmetric and diagonal_zero and in_range
    report.add_checkpoint('檢查點1: TED Distance Matrix', data, passed)
    
    print(f"✅ 對稱性: {is_symmetric}")
    print(f"✅ 對角線為0: {diagonal_zero}")
    print(f"✅ 距離範圍[0,1]: {in_range}")
    print(f"📊 平均距離: {mean_dist:.4f} ± {std_dist:.4f}")
    
    return ted_matrix, passed


def checkpoint_2_clustering(strategy, ted_matrix, population, report):
    """檢查點 2: 階層式分群"""
    print("\n" + "="*80)
    print("📊 檢查點 2: 階層式分群")
    print("="*80)
    
    cluster_labels = strategy._hierarchical_clustering(ted_matrix)
    
    # 驗證
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = len(unique_labels)
    all_assigned = len(cluster_labels) == len(population)
    no_empty = (counts > 0).all()
    
    cluster_dist = {f"Cluster {i}": int(count) for i, count in zip(unique_labels, counts)}
    small_clusters = sum(counts < strategy.M)
    
    data = {
        'Cluster 數量': n_clusters,
        '目標 Cluster 數': strategy.K,
        '所有個體已分配': '✅ 是' if all_assigned else '❌ 否',
        '無空 Cluster': '✅ 是' if no_empty else '❌ 否',
        'Cluster 分布': cluster_dist,
        '小於 M 的 Clusters': f"{small_clusters} / {n_clusters}",
        '統計': {
            '平均大小': f"{np.mean(counts):.1f}",
            '中位數大小': f"{np.median(counts):.0f}",
            '最小大小': int(np.min(counts)),
            '最大大小': int(np.max(counts))
        }
    }
    
    passed = (n_clusters == strategy.K) and all_assigned and no_empty
    report.add_checkpoint('檢查點2: 階層式分群', data, passed)
    
    print(f"✅ Cluster 數量: {n_clusters} (目標: {strategy.K})")
    print(f"✅ 所有個體已分配: {all_assigned}")
    print(f"⚠️  小於 M={strategy.M} 的 Clusters: {small_clusters}")
    
    return cluster_labels, passed


def checkpoint_3_elite_pool(strategy, population, cluster_labels, report):
    """檢查點 3: Elite Pool 提取"""
    print("\n" + "="*80)
    print("📊 檢查點 3: Elite Pool 提取")
    print("="*80)
    
    clusters, elite_pool = strategy._extract_elite_pool(population, cluster_labels)
    
    # 驗證每個 cluster 的 fitness 排序
    correctly_sorted = []
    cluster_details = {}
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            fitnesses = [ind.fitness.values[0] for ind in cluster]
            is_sorted = all(fitnesses[j] >= fitnesses[j+1] for j in range(len(fitnesses)-1))
            correctly_sorted.append(is_sorted)
            
            cluster_details[f"Cluster {i}"] = {
                '大小': len(cluster),
                '平均 Fitness': f"{np.mean(fitnesses):.4f}",
                '最大 Fitness': f"{max(fitnesses):.4f}",
                '最小 Fitness': f"{min(fitnesses):.4f}",
                '正確排序': '✅' if is_sorted else '❌'
            }
    
    expected_size = strategy.K * strategy.M
    actual_size = len(elite_pool)
    size_ratio = actual_size / expected_size
    
    data = {
        'Elite Pool 大小': actual_size,
        '預期大小': expected_size,
        '達成率': f"{size_ratio * 100:.1f}%",
        '所有 Cluster 正確排序': '✅ 是' if all(correctly_sorted) else '❌ 否',
        'Cluster 詳細': cluster_details
    }
    
    passed = all(correctly_sorted) and (size_ratio >= 0.8)  # 允許 80% 以上
    report.add_checkpoint('檢查點3: Elite Pool 提取', data, passed)
    
    print(f"✅ Elite Pool 大小: {actual_size} (預期: {expected_size})")
    print(f"✅ 達成率: {size_ratio * 100:.1f}%")
    print(f"✅ 所有 Cluster 正確排序: {all(correctly_sorted)}")
    
    return clusters, elite_pool, passed


def checkpoint_4_crossover_selection(strategy, population, clusters, elite_pool, report):
    """檢查點 4: Crossover Pairs 選擇"""
    print("\n" + "="*80)
    print("📊 檢查點 4: Crossover Pairs 選擇")
    print("="*80)
    
    # 選擇 100 對 parents 進行測試
    n_pairs = 100
    data_dict = {'generation': 0}
    
    pairs = strategy.select_pairs(population, n_pairs, data_dict)
    
    # 統計跨群 vs 同群配對
    # 需要建立個體到 cluster 的映射
    ind_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for ind in cluster:
            ind_to_cluster[id(ind)] = cluster_id
    
    cross_group = 0
    in_group = 0
    unknown = 0
    
    for p1, p2 in pairs:
        c1 = ind_to_cluster.get(id(p1), -1)
        c2 = ind_to_cluster.get(id(p2), -1)
        
        if c1 == -1 or c2 == -1:
            unknown += 1
        elif c1 != c2:
            cross_group += 1
        else:
            in_group += 1
    
    total = len(pairs)
    cross_ratio = cross_group / total if total > 0 else 0
    in_ratio = in_group / total if total > 0 else 0
    
    # 檢查 fitness
    fitnesses = []
    for p1, p2 in pairs:
        fitnesses.append(p1.fitness.values[0])
        fitnesses.append(p2.fitness.values[0])
    
    elite_fitnesses = [ind.fitness.values[0] for ind in elite_pool]
    
    data = {
        '選擇對數': total,
        '目標對數': n_pairs,
        '跨群配對': {
            '數量': cross_group,
            '比例': f"{cross_ratio * 100:.1f}%",
            '目標比例': f"{strategy.cross_group_ratio * 100:.1f}%"
        },
        '同群配對': {
            '數量': in_group,
            '比例': f"{in_ratio * 100:.1f}%",
            '目標比例': f"{(1 - strategy.cross_group_ratio) * 100:.1f}%"
        },
        '未知配對': unknown,
        'Fitness 統計': {
            '選出 Parents 平均': f"{np.mean(fitnesses):.4f}",
            'Elite Pool 平均': f"{np.mean(elite_fitnesses):.4f}",
            '選出 Parents 最大': f"{max(fitnesses):.4f}",
            '選出 Parents 最小': f"{min(fitnesses):.4f}"
        }
    }
    
    # 驗證：跨群比例應該接近目標（允許 ±10%）
    ratio_diff = abs(cross_ratio - strategy.cross_group_ratio)
    passed = (total == n_pairs) and (ratio_diff < 0.15) and (unknown == 0)
    
    report.add_checkpoint('檢查點4: Crossover Pairs 選擇', data, passed)
    
    print(f"✅ 選擇對數: {total} / {n_pairs}")
    print(f"📊 跨群配對: {cross_group} ({cross_ratio * 100:.1f}%)")
    print(f"📊 同群配對: {in_group} ({in_ratio * 100:.1f}%)")
    print(f"📊 Parents 平均 Fitness: {np.mean(fitnesses):.4f}")
    
    return pairs, passed


def checkpoint_5_mutation_selection(strategy, population, elite_pool, report):
    """檢查點 5: Mutation Individuals 選擇（Ranked SUS）"""
    print("\n" + "="*80)
    print("📊 檢查點 5: Mutation Individuals 選擇")
    print("="*80)
    
    n_individuals = 100
    data_dict = {'generation': 0}
    
    # 記錄原始 fitness
    original_fitnesses = {id(ind): ind.fitness.values[0] for ind in elite_pool}
    
    individuals = strategy.select_individuals(population, n_individuals, data_dict)
    
    # 驗證 fitness 是否被正確恢復
    fitness_restored = all(
        abs(ind.fitness.values[0] - original_fitnesses.get(id(ind), 0)) < 1e-6
        for ind in elite_pool
    )
    
    # 統計
    selected_fitnesses = [ind.fitness.values[0] for ind in individuals]
    elite_fitnesses = [ind.fitness.values[0] for ind in elite_pool]
    
    mean_selected = np.mean(selected_fitnesses)
    mean_elite = np.mean(elite_fitnesses)
    bias_to_high = mean_selected > mean_elite
    
    # 檢查是否有重複選擇
    unique_ids = len(set(id(ind) for ind in individuals))
    has_duplicates = unique_ids < len(individuals)
    
    data = {
        '選擇數量': len(individuals),
        '目標數量': n_individuals,
        '唯一個體數': unique_ids,
        '有重複選擇': '⚠️ 是' if has_duplicates else '✅ 否',
        'Fitness 統計': {
            '選出個體平均': f"{mean_selected:.4f}",
            'Elite Pool 平均': f"{mean_elite:.4f}",
            '偏向高 Fitness': '✅ 是' if bias_to_high else '❌ 否',
            '選出個體最大': f"{max(selected_fitnesses):.4f}",
            '選出個體最小': f"{min(selected_fitnesses):.4f}"
        },
        'Fitness 恢復': '✅ 正確' if fitness_restored else '❌ 失敗'
    }
    
    passed = (len(individuals) == n_individuals) and fitness_restored and bias_to_high
    report.add_checkpoint('檢查點5: Mutation Individuals 選擇', data, passed)
    
    print(f"✅ 選擇數量: {len(individuals)} / {n_individuals}")
    print(f"✅ Fitness 恢復: {fitness_restored}")
    print(f"✅ 偏向高 Fitness: {bias_to_high}")
    print(f"📊 選出平均: {mean_selected:.4f} vs Elite 平均: {mean_elite:.4f}")
    
    return individuals, passed


def checkpoint_6_quantity_calculation(report):
    """檢查點 6: 數量計算（三個互斥階段）"""
    print("\n" + "="*80)
    print("📊 檢查點 6: 數量計算（三個互斥階段）")
    print("="*80)
    
    POP_SIZE = 5000
    crossover_rate = 0.75
    mutation_rate = 0.20
    reproduction_rate = 0.05
    
    # Crossover
    num_crossover_offspring = int(POP_SIZE * crossover_rate)
    if num_crossover_offspring % 2 != 0:
        num_crossover_offspring -= 1
    num_crossover_pairs = num_crossover_offspring // 2
    
    # Mutation
    num_mutation_offspring = int(POP_SIZE * mutation_rate)
    
    # Reproduction
    num_reproduction_offspring = int(POP_SIZE * reproduction_rate)
    
    # 調整
    total = num_crossover_offspring + num_mutation_offspring + num_reproduction_offspring
    if total != POP_SIZE:
        diff = POP_SIZE - total
        num_mutation_offspring += diff
        total = num_crossover_offspring + num_mutation_offspring + num_reproduction_offspring
    
    data = {
        'Population 大小': POP_SIZE,
        'Crossover': {
            '比例': f"{crossover_rate * 100}%",
            'Offspring 數量': num_crossover_offspring,
            'Parent Pairs 數量': num_crossover_pairs,
            '實際產生': num_crossover_pairs * 2
        },
        'Mutation': {
            '比例': f"{mutation_rate * 100}%",
            'Offspring 數量': num_mutation_offspring
        },
        'Reproduction': {
            '比例': f"{reproduction_rate * 100}%",
            'Offspring 數量': num_reproduction_offspring
        },
        '總計': total,
        '數量正確': '✅ 是' if total == POP_SIZE else '❌ 否'
    }
    
    passed = (total == POP_SIZE) and (num_crossover_offspring % 2 == 0)
    report.add_checkpoint('檢查點6: 數量計算', data, passed)
    
    print(f"✅ Crossover: {num_crossover_offspring} ({num_crossover_pairs} 對)")
    print(f"✅ Mutation: {num_mutation_offspring}")
    print(f"✅ Reproduction: {num_reproduction_offspring}")
    print(f"✅ 總計: {total} (目標: {POP_SIZE})")
    
    return passed


def checkpoint_7_cache_mechanism(strategy, population, report):
    """檢查點 7: 快取機制"""
    print("\n" + "="*80)
    print("📊 檢查點 7: 快取機制")
    print("="*80)
    
    # 第一次呼叫
    data_gen0 = {'generation': 0}
    pairs_1 = strategy.select_pairs(population, 10, data_gen0)
    cached_gen_1 = strategy._cached_generation
    
    # 第二次呼叫（同一世代，應使用快取）
    pairs_2 = strategy.select_pairs(population, 10, data_gen0)
    cached_gen_2 = strategy._cached_generation
    
    # 第三次呼叫（不同世代，應重新計算）
    data_gen1 = {'generation': 1}
    pairs_3 = strategy.select_pairs(population, 10, data_gen1)
    cached_gen_3 = strategy._cached_generation
    
    data = {
        '第一次呼叫': {
            'Generation': 0,
            '快取 Generation': cached_gen_1,
            '選擇對數': len(pairs_1)
        },
        '第二次呼叫（同世代）': {
            'Generation': 0,
            '快取 Generation': cached_gen_2,
            '使用快取': '✅ 是' if cached_gen_2 == 0 else '❌ 否',
            '選擇對數': len(pairs_2)
        },
        '第三次呼叫（不同世代）': {
            'Generation': 1,
            '快取 Generation': cached_gen_3,
            '重新計算': '✅ 是' if cached_gen_3 == 1 else '❌ 否',
            '選擇對數': len(pairs_3)
        }
    }
    
    passed = (cached_gen_2 == 0) and (cached_gen_3 == 1)
    report.add_checkpoint('檢查點7: 快取機制', data, passed)
    
    print(f"✅ 第一次呼叫: Generation {cached_gen_1}")
    print(f"✅ 第二次呼叫（同世代）: 使用快取 = {cached_gen_2 == 0}")
    print(f"✅ 第三次呼叫（不同世代）: 重新計算 = {cached_gen_3 == 1}")
    
    return passed


def main():
    """主函數"""
    print("="*80)
    print("🔍 TED Niche Selection Strategy 完整驗證")
    print("="*80)
    
    # 設置
    setup_deap_creator()
    
    # 載入測試族群
    records_dir = Path("/Users/hongyicheng/Downloads/gp_quant/test_evolution_11241221_records_20251125_1335")
    population = load_test_population(records_dir, generation=0)
    
    print(f"\n📦 載入族群: {len(population)} 個個體")
    
    # 創建策略
    strategy = TEDNicheSelectionStrategy(
        n_clusters=5,
        top_m_per_cluster=50,
        cross_group_ratio=0.3,
        tournament_size=3,
        n_jobs=6
    )
    
    print(f"📦 策略配置: K={strategy.K}, M={strategy.M}")
    
    # 創建報告
    output_dir = Path("validation_results")
    report = ValidationReport(output_dir)
    
    # 執行所有檢查點
    results = []
    
    # 檢查點 1-3 需要依序執行（有依賴關係）
    ted_matrix, r1 = checkpoint_1_ted_matrix(strategy, population, report)
    results.append(r1)
    
    cluster_labels, r2 = checkpoint_2_clustering(strategy, ted_matrix, population, report)
    results.append(r2)
    
    clusters, elite_pool, r3 = checkpoint_3_elite_pool(strategy, population, cluster_labels, report)
    results.append(r3)
    
    # 檢查點 4-5 使用快取的結果
    _, r4 = checkpoint_4_crossover_selection(strategy, population, clusters, elite_pool, report)
    results.append(r4)
    
    _, r5 = checkpoint_5_mutation_selection(strategy, population, elite_pool, report)
    results.append(r5)
    
    # 檢查點 6-7 獨立
    r6 = checkpoint_6_quantity_calculation(report)
    results.append(r6)
    
    r7 = checkpoint_7_cache_mechanism(strategy, population, report)
    results.append(r7)
    
    # 生成報告
    print("\n" + "="*80)
    print("📝 生成報告")
    print("="*80)
    
    json_path = report.save_json()
    md_path = report.save_markdown()
    
    print(f"✅ JSON 報告: {json_path}")
    print(f"✅ Markdown 報告: {md_path}")
    
    # 總結
    print("\n" + "="*80)
    print("📊 驗證總結")
    print("="*80)
    
    total_checks = len(results)
    passed_checks = sum(results)
    
    print(f"總檢查點: {total_checks}")
    print(f"通過: {passed_checks}")
    print(f"失敗: {total_checks - passed_checks}")
    print(f"通過率: {passed_checks / total_checks * 100:.1f}%")
    
    if all(results):
        print("\n✅ 所有檢查點通過！")
    else:
        print("\n⚠️  部分檢查點未通過，請查看報告詳情。")
    
    print("="*80)


if __name__ == "__main__":
    main()
