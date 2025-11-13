"""
Main Evolutionary Engine

This module orchestrates the entire genetic programming process. It sets up the
DEAP toolbox with all the necessary operators and runs the evolutionary algorithm,
including a custom selection mechanism as specified in the project requirements.
"""
import random
import operator
import numpy as np
import pandas as pd
import dill
import os
import fcntl
import time
from typing import Dict, Union, Optional
from tqdm import trange, tqdm
from deap import base, creator, tools, gp

from gp_quant.backtesting.engine import BacktestingEngine, PortfolioBacktestingEngine
from gp_quant.gp.operators import pset


def ranked_selection(individuals, k, max_rank_fitness=1.8, min_rank_fitness=0.2):
    """
    Custom selection operator implementing Ranked Selection + SUS.

    As described in the PRD, this function first ranks individuals based on their
    raw fitness, then assigns a new fitness value based on rank. Finally, it uses
    Stochastic Universal Sampling (SUS) on the new rank-based fitness values.

    Args:
        individuals: A list of individuals to select from.
        k: The number of individuals to select.
        max_rank_fitness: The fitness value assigned to the best individual (Max in PRD).
        min_rank_fitness: The fitness value assigned to the worst individual (Min in PRD).

    Returns:
        A list of selected individuals.
    """
    # Sort individuals by their original fitness (descending)
    sorted_individuals = sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)
    pop_size = len(sorted_individuals)

    # Assign a temporary rank-based fitness for selection without overwriting the real fitness
    sum_rank_fitness = 0
    for i, ind in enumerate(sorted_individuals):
        rank = i + 1
        # Assign a temporary attribute for the rank-based fitness
        ind.rank_fitness = max_rank_fitness - ((max_rank_fitness - min_rank_fitness) * (rank - 1) / (pop_size - 1))
        sum_rank_fitness += ind.rank_fitness

    # Use Stochastic Universal Sampling on the temporary rank_fitness
    # DEAP's SUS function requires the fitness values to be in the .fitness.values attribute.
    # So, we'll temporarily store them there, perform selection, and then restore them if needed,
    # although since we're just selecting, modifying the values of the sorted_individuals list is fine.
    original_fitnesses = [ind.fitness.values for ind in sorted_individuals]
    for ind in sorted_individuals:
        ind.fitness.values = (ind.rank_fitness,)
    
    chosen = tools.selStochasticUniversalSampling(sorted_individuals, k)

    # Restore original fitness values to the main population to ensure logging is correct
    for ind, fit in zip(sorted_individuals, original_fitnesses):
        ind.fitness.values = fit

    return chosen

def save_population(population, generation, individual_records_dir, max_retries=3):
    """
    Save the current population to a pickle file using dill with global file locking.
    
    Uses blocking file lock to ensure all processes wait their turn to write,
    guaranteeing that all generations are saved without timeout issues.
    
    Args:
        population: The population to save
        generation: The current generation number
        individual_records_dir: The base directory for saving populations
        max_retries: Maximum number of retry attempts for write failures
    """
    if individual_records_dir is None:
        return
    
    # Create generation-specific directory
    gen_dir = os.path.join(individual_records_dir, f"generation_{generation:03d}")
    os.makedirs(gen_dir, exist_ok=True)
    
    pickle_file = os.path.join(gen_dir, "population.pkl")
    
    # Use a global lock file in the parent directory to serialize writes
    # This prevents I/O contention when multiple processes try to write simultaneously
    global_lock_file = os.path.join(os.path.dirname(individual_records_dir), ".population_save.lock")
    
    for attempt in range(max_retries):
        lock_fd = None
        try:
            # Create and acquire exclusive lock on global lock file
            lock_fd = os.open(global_lock_file, os.O_CREAT | os.O_WRONLY, 0o644)
            
            # Use BLOCKING lock - wait indefinitely until lock is available
            # This ensures all processes will eventually write, no timeouts
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            # Lock acquired, now save the population
            with open(pickle_file, 'wb') as f:
                dill.dump(population, f)
                f.flush()  # Force write to buffer
                os.fsync(f.fileno())  # Force write to disk
            
            # Verify the file was written successfully
            file_size = os.path.getsize(pickle_file) if os.path.exists(pickle_file) else 0
            if file_size > 0:
                # Success! Release lock and return
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                lock_fd = None
                return
            else:
                # File is empty, retry
                print(f"Warning: Gen {generation} - File empty (attempt {attempt+1}/{max_retries})")
                
        except Exception as e:
            # Only retry on actual write errors, not lock errors
            print(f"Warning: Failed to save gen {generation} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(0.2 * (attempt + 1))  # Exponential backoff
            else:
                import traceback
                traceback.print_exc()
                
        finally:
            # Always release lock and close file descriptor
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                except:
                    pass
    
    # If we get here, all retries failed
    print(f"ERROR: Failed to save population for generation {generation} after {max_retries} attempts")

def run_evolution(data, population_size=500, n_generations=50, 
                  reproduction_prob=0.35, crossover_prob=0.60, mutation_prob=0.05,
                  individual_records_dir: Optional[str] = None,
                  generation_callback=None,
                  fitness_metric='excess_return',
                  tournament_size=3,
                  hof_size=1):
    """
    Configures and runs the main evolutionary algorithm.
    
    Implements the paper's "pick-one" mechanism where each offspring is created by
    exactly one of three operations: reproduction, crossover, or mutation.

    Args:
        data: The historical stock data. Can be either:
              - A single Pandas DataFrame (for single ticker evolution)
              - A Dict[str, DataFrame] (for portfolio evolution)
        population_size: The number of individuals in the population.
        n_generations: The number of generations to run.
        reproduction_prob: Probability of reproduction (copying) operation (default: 0.35).
        crossover_prob: Probability of crossover operation (default: 0.60).
        mutation_prob: Probability of mutation operation (default: 0.05).
        individual_records_dir: Optional directory path to save population snapshots.
                               If provided, each generation's population will be saved as a pickle file.
        generation_callback: Optional callback function called after each generation.
                           Signature: callback(gen, pop, hof, logbook, record) -> Optional[custom_selector]
        fitness_metric: Fitness metric to use ('excess_return' or 'sharpe_ratio')
        tournament_size: Tournament size for selection
        hof_size: Size of hall of fame

    Returns:
        A tuple containing the final population, the logbook, and the hall of fame.
    """
    # Validate probabilities sum to 1.0
    prob_sum = reproduction_prob + crossover_prob + mutation_prob
    assert abs(prob_sum - 1.0) < 1e-6, \
        f"Reproduction, crossover, and mutation probabilities must sum to 1.0 (got {prob_sum})"
    # --- Setup DEAP Toolbox ---
    toolbox = base.Toolbox()
    
    # Determine if we're doing single ticker or portfolio evolution
    if isinstance(data, dict):
        # Portfolio mode: multiple tickers
        # Check if data contains the new structure with 'data' and 'backtest_start'
        first_ticker = list(data.keys())[0]
        if isinstance(data[first_ticker], dict) and 'data' in data[first_ticker]:
            # New structure: extract data and backtest config
            data_dict = {ticker: data[ticker]['data'] for ticker in data.keys()}
            backtest_config = {
                ticker: {
                    'backtest_start': data[ticker]['backtest_start'],
                    'backtest_end': data[ticker]['backtest_end']
                }
                for ticker in data.keys()
            }
            backtester = PortfolioBacktestingEngine(data_dict, backtest_config=backtest_config)
        else:
            # Old structure: direct DataFrame dict (backward compatibility)
            backtester = PortfolioBacktestingEngine(data)
        print(f"Running PORTFOLIO evolution with {len(data)} tickers")
    else:
        # Single ticker mode (backward compatibility)
        backtester = BacktestingEngine(data)
        print(f"Running SINGLE TICKER evolution")

    # Attribute generator
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Operator registration
    toolbox.register("evaluate", lambda ind: backtester.evaluate(ind, fitness_metric=fitness_metric))
    # Use Ranked Selection + SUS as per paper requirements
    toolbox.register("select", ranked_selection)
    # Use leaf-biased crossover: 90% internal nodes, 10% terminals (Koza's standard)
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Note: We don't use staticLimit decorator here because it only rejects operations
    # Instead, we implement retry logic in the evolution loop to ensure compliant offspring

    # --- Run Evolution ---
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hof_size)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # The DEAP `eaSimple` algorithm is a good starting point, but it doesn't handle
    # the separation of raw fitness and selection fitness. We use `eaMuPlusLambda`
    # or a custom loop for more control. For simplicity here, we'll use a custom loop.
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    # Evaluate the initial population with a progress bar
    print("Evaluating initial population...")
    fitnesses = list(tqdm(toolbox.map(toolbox.evaluate, pop), total=len(pop), desc="Initial Evaluation"))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    print(logbook.stream)
    
    # Save initial population (generation 0)
    save_population(pop, 0, individual_records_dir)

    # Statistics for retry failures
    retry_stats = {
        'crossover_failures': 0,  # Number of times crossover retry failed
        'mutation_failures': 0,   # Number of times mutation retry failed
        'total_crossovers': 0,    # Total crossover operations
        'total_mutations': 0      # Total mutation operations
    }
    
    # Use trange for a progress bar
    for gen in (pbar := trange(1, n_generations + 1, desc="Generation")):
        # Create offspring using the paper's "pick-one" mechanism
        # Each offspring is created by exactly ONE of: reproduction, crossover, or mutation
        offspring = []
        
        while len(offspring) < population_size:
            operation = random.random()
            
            if operation < reproduction_prob:
                # Reproduction: Copy an individual directly
                parent = toolbox.select(pop, 1)[0]
                child = toolbox.clone(parent)
                offspring.append(child)
                
            elif operation < reproduction_prob + crossover_prob:
                # Crossover: Select two parents and create two children
                # Retry until we get compliant offspring (depth <= 17)
                retry_stats['total_crossovers'] += 1
                max_retries = 10
                for attempt in range(max_retries):
                    parent1, parent2 = toolbox.select(pop, 2)
                    child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)
                    toolbox.mate(child1, child2)
                    
                    # Check if children are compliant
                    if child1.height <= 17 and child2.height <= 17:
                        del child1.fitness.values
                        del child2.fitness.values
                        offspring.append(child1)
                        # Only add second child if there's room
                        if len(offspring) < population_size:
                            offspring.append(child2)
                        break
                else:
                    # If all retries failed, generate new random individuals
                    retry_stats['crossover_failures'] += 1
                    child1 = toolbox.individual()
                    offspring.append(child1)
                    if len(offspring) < population_size:
                        child2 = toolbox.individual()
                        offspring.append(child2)
                    
            else:
                # Mutation: Select one parent and mutate it
                # Retry until we get compliant offspring (depth <= 17)
                retry_stats['total_mutations'] += 1
                max_retries = 10
                for attempt in range(max_retries):
                    parent = toolbox.select(pop, 1)[0]
                    mutant = toolbox.clone(parent)
                    toolbox.mutate(mutant)
                    
                    # Check if mutant is compliant
                    if mutant.height <= 17:
                        del mutant.fitness.values
                        offspring.append(mutant)
                        break
                else:
                    # If all retries failed, generate a new random individual
                    retry_stats['mutation_failures'] += 1
                    mutant = toolbox.individual()
                    offspring.append(mutant)
        
        # Ensure offspring size is exactly population_size (crossover may overshoot by 1)
        offspring = offspring[:population_size]

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            # Final safeguard before assigning fitness
            # This is the ultimate gatekeeper against any unexpected crashes inside evaluate()
            if not np.isfinite(fit[0]) or fit[0] > 1e12:
                ind.fitness.values = (-100000.0,) # Penalty fitness
            else:
                ind.fitness.values = fit


        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Update the Hall of fame with the best individual
        hof.update(pop)

        # Append the current generation statistics to the logbook
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        # Update progress bar with stats
        pbar.set_description(f"Gen {gen} | Avg: {record['avg']:.2f} | Best: {record['max']:.2f}")
        
        # Save population for this generation
        save_population(pop, gen, individual_records_dir)
        
        # Call generation callback if provided (after evaluation and stats)
        # Callback can return True to stop evolution, or a custom selector for next generation
        if generation_callback:
            callback_result = generation_callback(gen, pop, hof, logbook, record)
            if callback_result is True:
                # Early stopping triggered - break out of evolution loop
                print(f"Evolution stopped early at generation {gen}")
                break
            elif callback_result and callable(callback_result):
                # Replace the default selector with custom one
                toolbox.unregister("select")
                toolbox.register("select", callback_result)
    
    # Save retry statistics if individual_records_dir is provided
    if individual_records_dir:
        import json
        retry_stats_file = os.path.join(individual_records_dir, 'retry_statistics.json')
        with open(retry_stats_file, 'w') as f:
            json.dump(retry_stats, f, indent=2)
        
        # Also print summary
        print(f"\n{'='*80}")
        print("Retry Statistics Summary")
        print(f"{'='*80}")
        print(f"Total Crossovers: {retry_stats['total_crossovers']}")
        print(f"Crossover Failures (10 retries): {retry_stats['crossover_failures']} ({retry_stats['crossover_failures']/max(retry_stats['total_crossovers'],1)*100:.2f}%)")
        print(f"Total Mutations: {retry_stats['total_mutations']}")
        print(f"Mutation Failures (10 retries): {retry_stats['mutation_failures']} ({retry_stats['mutation_failures']/max(retry_stats['total_mutations'],1)*100:.2f}%)")
        print(f"{'='*80}\n")
    
    return pop, logbook, hof
