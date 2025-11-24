#!/usr/bin/env python3
"""
Experiment evaluation script.

- Computes in-sample / out-of-sample performance per generation
  for full population and train-fitness-based top-3.
- Computes diversity trends (PnL correlation, tree edit distance)
  for both full population and top-3, using 6 processes for
  heavy calculations.

Usage (example for a records dir):

    python -m gp_quant.evaluation.evaluate_experiment \
        --records-dir large_scale_records_20251124_1738 \
        --generations all \
        --n-workers 6

This script assumes records_dir has the structure produced by
main_evolution.py, e.g.:

    records_dir/
      config.json
      populations/generation_000.pkl
      populations/generation_001.pkl
      ...

It will create an `evaluation/` subdirectory under records_dir
containing CSV summaries and PNG plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool

from gp_quant.data.loader import load_and_process_data, split_train_test_data
from gp_quant.evolution.components.backtesting.portfolio_engine import PortfolioBacktestingEngine
from gp_quant.similarity.parallel_calculator import ParallelSimilarityMatrix


def _parse_generations_arg(arg: str, max_gen: int) -> List[int]:
    """Parse generations argument.

    Examples:
        "all" -> [0..max_gen]
        "0-10" -> [0..10]
        "0,5,10" -> [0,5,10]
    """
    if arg == "all":
        return list(range(0, max_gen + 1))

    gens: List[int] = []
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            gens.extend(range(start, end + 1))
        else:
            gens.append(int(part))
    # Remove duplicates and sort
    gens = sorted(set(g for g in gens if 0 <= g <= max_gen))
    return gens


def _load_populations(pop_dir: Path) -> Dict[int, List[Any]]:
    """Load populations from populations/*.pkl.

    Returns:
        {generation: [individuals...]}
    """
    populations: Dict[int, List[Any]] = {}

    for pkl_path in sorted(pop_dir.glob("generation_*.pkl")):
        gen_str = pkl_path.stem.split("_")[1]
        gen = int(gen_str)
        with open(pkl_path, "rb") as f:
            pop = dill.load(f)
        populations[gen] = pop

    return populations


def _build_portfolio_engine_from_data(
    data_dict: Dict[str, Dict[str, Any]],
    backtest_start: str,
    backtest_end: str,
    initial_capital: float = 100000.0,
) -> PortfolioBacktestingEngine:
    """Create a PortfolioBacktestingEngine from split_train_test_data output.

    data_dict is expected to be the train_data or test_data dict from
    split_train_test_data, i.e.:

        data_dict[ticker] = {
            "data": DataFrame,
            "backtest_start": str,
            "backtest_end": str,
        }
    """
    data_frames: Dict[str, pd.DataFrame] = {}
    for ticker, info in data_dict.items():
        df = info["data"] if isinstance(info, dict) else info
        data_frames[ticker] = df

    engine = PortfolioBacktestingEngine(
        data=data_frames,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        initial_capital=initial_capital,
    )
    return engine


def _evaluate_individual_oos(args: Tuple[Any, PortfolioBacktestingEngine, str]) -> float:
    """Worker to evaluate a single individual out-of-sample.

    Args:
        args: (individual, engine, fitness_metric)
    """
    individual, engine, metric = args
    try:
        return engine.get_fitness(individual, fitness_metric=metric)
    except Exception:
        return -100000.0


def _compute_oos_performance(
    population: List[Any],
    test_engine: PortfolioBacktestingEngine,
    fitness_metric: str = "excess_return",
    n_workers: int = 1,
) -> List[float]:
    """Compute out-of-sample fitness for a population.

    Returns:
        List of OOS fitness values aligned with population order.
    """
    if n_workers <= 1:
        scores: List[float] = []
        for ind in population:
            try:
                val = test_engine.get_fitness(ind, fitness_metric=fitness_metric)
            except Exception:
                val = -100000.0
            scores.append(val)
        return scores

    # Multiprocessing version
    # NOTE: PortfolioBacktestingEngine is not strictly process-safe, so for
    # true correctness we would need one engine per worker. For small-scale
    # experiments this shared-engine approach is acceptable; if needed we
    # can later refactor to pass raw data and recreate engines in workers.
    with Pool(processes=n_workers) as pool:
        args_list = [(ind, test_engine, fitness_metric) for ind in population]
        scores = pool.map(_evaluate_individual_oos, args_list)
    return scores


def _select_top_k_by_train_fitness(population: List[Any], k: int) -> List[Any]:
    """Select top-k individuals by in-sample fitness (training fitness)."""
    fitness_values = []
    for idx, ind in enumerate(population):
        val = None
        if hasattr(ind, "fitness") and getattr(ind.fitness, "values", None):
            try:
                val = float(ind.fitness.values[0])
            except Exception:
                val = None
        if val is None:
            val = -100000.0
        fitness_values.append((idx, val))

    fitness_values.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in fitness_values[:k]]
    return [population[i] for i in top_indices]


def _compute_pnl_correlation_diversity(
    population: List[Any],
    engine: PortfolioBacktestingEngine,
    n_workers: int,
) -> Dict[str, float]:
    """Compute PnL-correlation-based diversity.

    Uses portfolio-level cumulative PnL curves from PortfolioBacktestingEngine.

    Returns a dict with keys:
        - pnl_corr_mean
        - pnl_corr_std
        - pnl_corr_min
        - pnl_corr_max
        - pnl_corr_median
        - valid_individuals
        - total_pairs
    """
    # Generate PnL curves (single process, full population, no sampling)
    pnl_list: List[pd.Series] = []
    for ind in population:
        try:
            pnl = engine.get_pnl_curve(ind)
            if len(pnl) == 0 or pnl.std() == 0:
                continue
            pnl_list.append(pnl)
        except Exception:
            continue

    if len(pnl_list) < 2:
        return {
            "pnl_corr_mean": 0.0,
            "pnl_corr_std": 0.0,
            "pnl_corr_min": 0.0,
            "pnl_corr_max": 0.0,
            "pnl_corr_median": 0.0,
            "valid_individuals": len(pnl_list),
            "total_pairs": 0,
        }

    # Align curves by index and build matrix
    aligned = pd.concat(pnl_list, axis=1, join="inner")
    if aligned.shape[1] < 2:
        return {
            "pnl_corr_mean": 0.0,
            "pnl_corr_std": 0.0,
            "pnl_corr_min": 0.0,
            "pnl_corr_max": 0.0,
            "pnl_corr_median": 0.0,
            "valid_individuals": aligned.shape[1],
            "total_pairs": 0,
        }

    pnl_matrix = aligned.to_numpy()
    corr_matrix = np.corrcoef(pnl_matrix)
    if corr_matrix.shape[0] < 2:
        return {
            "pnl_corr_mean": 0.0,
            "pnl_corr_std": 0.0,
            "pnl_corr_min": 0.0,
            "pnl_corr_max": 0.0,
            "pnl_corr_median": 0.0,
            "valid_individuals": aligned.shape[1],
            "total_pairs": 0,
        }

    upper = np.triu_indices_from(corr_matrix, k=1)
    correlations = corr_matrix[upper]
    correlations = correlations[np.isfinite(correlations)]

    if correlations.size == 0:
        return {
            "pnl_corr_mean": 0.0,
            "pnl_corr_std": 0.0,
            "pnl_corr_min": 0.0,
            "pnl_corr_max": 0.0,
            "pnl_corr_median": 0.0,
            "valid_individuals": aligned.shape[1],
            "total_pairs": 0,
        }

    return {
        "pnl_corr_mean": float(np.mean(correlations)),
        "pnl_corr_std": float(np.std(correlations)),
        "pnl_corr_min": float(np.min(correlations)),
        "pnl_corr_max": float(np.max(correlations)),
        "pnl_corr_median": float(np.median(correlations)),
        "valid_individuals": int(aligned.shape[1]),
        "total_pairs": int(correlations.size),
    }


def _compute_ted_diversity(
    population: List[Any],
    n_workers: int,
) -> Dict[str, float]:
    """Compute tree-edit-distance-based diversity.

    Uses ParallelSimilarityMatrix with TreeEditDistance to compute
    pairwise distances, then normalizes each distance as

        normalized_ted = ted / max(len(tree_i), len(tree_j))

    and returns the mean of normalized TED as tree diversity.
    """
    if len(population) < 2:
        return {
            "ted_mean": 0.0,
            "ted_std": 0.0,
            "ted_min": 0.0,
            "ted_max": 0.0,
            "ted_mean_normalized": 0.0,
            "total_pairs": 0,
        }

    sim = ParallelSimilarityMatrix(population=population, n_workers=n_workers)
    sim.compute(show_progress=False)
    dist = sim.distance_matrix

    # Tree sizes
    sizes = np.array([len(ind) for ind in population], dtype=float)

    dists: List[float] = []
    norm_dists: List[float] = []
    n = dist.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            dists.append(d)
            max_len = max(sizes[i], sizes[j])
            if max_len > 0:
                norm_dists.append(d / max_len)

    if not dists or not norm_dists:
        return {
            "ted_mean": 0.0,
            "ted_std": 0.0,
            "ted_min": 0.0,
            "ted_max": 0.0,
            "ted_mean_normalized": 0.0,
            "total_pairs": 0,
        }

    dists_arr = np.array(dists, dtype=float)
    norm_arr = np.array(norm_dists, dtype=float)

    return {
        "ted_mean": float(dists_arr.mean()),
        "ted_std": float(dists_arr.std()),
        "ted_min": float(dists_arr.min()),
        "ted_max": float(dists_arr.max()),
        "ted_mean_normalized": float(norm_arr.mean()),
        "total_pairs": int(dists_arr.size),
    }


def _analyze_generations(
    records_dir: Path,
    generations: List[int],
    n_workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full analysis over selected generations.

    Returns:
        perf_df: in/out-of-sample performance trends
        diversity_full_df: diversity trends for full population
        diversity_top3_df: diversity trends for top-3 per generation
    """
    # Load config and data
    with open(records_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    data_cfg = config["data"]
    # tickers_dir is relative to project root; records_dir is also under
    # project root, so we can safely resolve via parent.
    tickers_dir = Path(records_dir).parent / data_cfg["tickers_dir"]

    if not tickers_dir.exists():
        raise FileNotFoundError(f"Tickers directory does not exist: {tickers_dir}")

    # Discover all tickers by scanning CSV files (same logic as main_evolution)
    csv_files = [f for f in os.listdir(tickers_dir) if f.endswith(".csv")]
    tickers = [f.replace(".csv", "") for f in csv_files]
    if not tickers:
        raise ValueError(f"No CSV files found in {tickers_dir}")

    # Load raw data and split train/test
    raw_data = load_and_process_data(str(tickers_dir), tickers)
    train_data, test_data = split_train_test_data(
        raw_data,
        train_data_start=data_cfg["train_data_start"],
        train_backtest_start=data_cfg["train_backtest_start"],
        train_backtest_end=data_cfg["train_backtest_end"],
        test_data_start=data_cfg["test_data_start"],
        test_backtest_start=data_cfg["test_backtest_start"],
        test_backtest_end=data_cfg["test_backtest_end"],
    )

    # Engines for train/test (portfolio-level)
    train_engine = _build_portfolio_engine_from_data(
        train_data,
        backtest_start=data_cfg["train_backtest_start"],
        backtest_end=data_cfg["train_backtest_end"],
    )
    test_engine = _build_portfolio_engine_from_data(
        test_data,
        backtest_start=data_cfg["test_backtest_start"],
        backtest_end=data_cfg["test_backtest_end"],
    )

    # Load populations
    pop_dir = records_dir / "populations"
    populations = _load_populations(pop_dir)

    perf_rows = []
    div_full_rows = []
    div_top_rows = []

    for gen in generations:
        if gen not in populations:
            continue
        population = populations[gen]

        # In-sample (train) fitness
        train_fitness_vals: List[float] = []
        for ind in population:
            if hasattr(ind, "fitness") and getattr(ind.fitness, "values", None):
                try:
                    train_fitness_vals.append(float(ind.fitness.values[0]))
                except Exception:
                    pass

        pop_train_mean = float(np.mean(train_fitness_vals)) if train_fitness_vals else 0.0

        # Top-3 by train fitness
        top3 = _select_top_k_by_train_fitness(population, k=3)
        top3_train_vals: List[float] = []
        for ind in top3:
            if hasattr(ind, "fitness") and getattr(ind.fitness, "values", None):
                try:
                    top3_train_vals.append(float(ind.fitness.values[0]))
                except Exception:
                    pass
        top3_train_mean = float(np.mean(top3_train_vals)) if top3_train_vals else 0.0

        # Out-of-sample performance (test)
        pop_oos_scores = _compute_oos_performance(population, test_engine, n_workers=1)
        pop_oos_mean = float(np.mean(pop_oos_scores)) if pop_oos_scores else 0.0

        top3_oos_scores = _compute_oos_performance(top3, test_engine, n_workers=1)
        top3_oos_mean = float(np.mean(top3_oos_scores)) if top3_oos_scores else 0.0

        perf_rows.append(
            {
                "generation": gen,
                "population_train_mean": pop_train_mean,
                "population_test_mean": pop_oos_mean,
                "top3_train_mean": top3_train_mean,
                "top3_test_mean": top3_oos_mean,
            }
        )

        # Diversity for full population
        pnl_full = _compute_pnl_correlation_diversity(population, train_engine, n_workers=n_workers)
        ted_full = _compute_ted_diversity(population, n_workers=n_workers)
        row_full = {"generation": gen}
        row_full.update(pnl_full)
        row_full.update(ted_full)
        # Add derived PnL diversity score
        row_full["pnl_diversity"] = 1.0 - row_full["pnl_corr_mean"]
        div_full_rows.append(row_full)

        # Diversity for top-3
        pnl_top = _compute_pnl_correlation_diversity(top3, train_engine, n_workers=n_workers)
        ted_top = _compute_ted_diversity(top3, n_workers=n_workers)
        row_top = {"generation": gen}
        row_top.update(pnl_top)
        row_top.update(ted_top)
        row_top["pnl_diversity"] = 1.0 - row_top["pnl_corr_mean"]
        div_top_rows.append(row_top)

    perf_df = pd.DataFrame(perf_rows).sort_values("generation")
    diversity_full_df = pd.DataFrame(div_full_rows).sort_values("generation")
    diversity_top3_df = pd.DataFrame(div_top_rows).sort_values("generation")

    return perf_df, diversity_full_df, diversity_top3_df


def _plot_performance_trends(perf_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot in-sample / out-of-sample performance trends."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gens = perf_df["generation"].values

    # Population vs Top3, train vs test
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Population
    axes[0].plot(gens, perf_df["population_train_mean"], label="Population Train", marker="o")
    axes[0].plot(gens, perf_df["population_test_mean"], label="Population Test", marker="s")
    axes[0].set_ylabel("Excess Return")
    axes[0].set_title("Population In-Sample vs Out-of-Sample Performance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Top3
    axes[1].plot(gens, perf_df["top3_train_mean"], label="Top3 Train", marker="o")
    axes[1].plot(gens, perf_df["top3_test_mean"], label="Top3 Test", marker="s")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Excess Return")
    axes[1].set_title("Top3 In-Sample vs Out-of-Sample Performance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "performance_in_out.png", dpi=200)
    plt.close(fig)


def _plot_diversity_trends(div_full: pd.DataFrame, div_top: pd.DataFrame, output_dir: Path) -> None:
    """Plot diversity trends (TED and PnL corr) for full population and top3."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full population
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    gens = div_full["generation"].values

    ax1.plot(gens, div_full["pnl_diversity"], label="PnL Diversity (1 - mean corr)", marker="o")
    ax1.plot(gens, div_full["ted_mean_normalized"], label="Tree Diversity (mean normalized TED)", marker="s")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity")
    ax1.set_title("Population Diversity Trends (PnL & TED)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(output_dir / "diversity_population.png", dpi=200)
    plt.close(fig1)

    # Top3
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    gens_top = div_top["generation"].values

    ax2.plot(gens_top, div_top["pnl_diversity"], label="PnL Diversity (1 - mean corr)", marker="o")
    ax2.plot(gens_top, div_top["ted_mean_normalized"], label="Tree Diversity (mean normalized TED)", marker="s")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Diversity")
    ax2.set_title("Top3 Diversity Trends (PnL & TED)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(output_dir / "diversity_top3.png", dpi=200)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment performance and diversity trends")
    parser.add_argument("--records-dir", required=True, help="Path to experiment records directory")
    parser.add_argument(
        "--generations",
        default="all",
        help="Generations to analyze: 'all', '0-10', '0,5,10', etc.",
    )
    parser.add_argument("--n-workers", type=int, default=6, help="Number of worker processes for heavy calculations")

    args = parser.parse_args()

    records_dir = Path(args.records_dir)
    if not records_dir.exists():
        raise SystemExit(f"Records directory does not exist: {records_dir}")

    # Determine max generation from populations dir
    pop_dir = records_dir / "populations"
    if not pop_dir.exists():
        raise SystemExit(f"Populations directory not found: {pop_dir}")

    gen_nums: List[int] = []
    for pkl_path in pop_dir.glob("generation_*.pkl"):
        gen_str = pkl_path.stem.split("_")[1]
        gen_nums.append(int(gen_str))
    if not gen_nums:
        raise SystemExit(f"No generation_*.pkl files found in {pop_dir}")

    max_gen = max(gen_nums)
    generations = _parse_generations_arg(args.generations, max_gen)
    if not generations:
        raise SystemExit("No valid generations selected")

    print(f"Analyzing generations: {generations}")
    perf_df, div_full_df, div_top_df = _analyze_generations(records_dir, generations, n_workers=args.n_workers)

    eval_dir = records_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    perf_df.to_csv(eval_dir / "performance_trends.csv", index=False)
    div_full_df.to_csv(eval_dir / "diversity_population.csv", index=False)
    div_top_df.to_csv(eval_dir / "diversity_top3.csv", index=False)

    # Plot
    _plot_performance_trends(perf_df, eval_dir)
    _plot_diversity_trends(div_full_df, div_top_df, eval_dir)

    print(f"Evaluation completed. Results saved under: {eval_dir}")


if __name__ == "__main__":
    main()
