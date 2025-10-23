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

def run_evolution(data, population_size=500, n_generations=50, crossover_prob=0.6, mutation_prob=0.05, 
                  individual_records_dir: Optional[str] = None):
    """
    Configures and runs the main evolutionary algorithm.

    Args:
        data: The historical stock data. Can be either:
              - A single Pandas DataFrame (for single ticker evolution)
              - A Dict[str, DataFrame] (for portfolio evolution)
        population_size: The number of individuals in the population.
        n_generations: The number of generations to run.
        crossover_prob: The probability of crossover.
        mutation_prob: The probability of mutation.
        individual_records_dir: Optional directory path to save population snapshots.
                               If provided, each generation's population will be saved as a pickle file.

    Returns:
        A tuple containing the final population, the logbook, and the hall of fame.
    """
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
    toolbox.register("evaluate", backtester.evaluate)
    toolbox.register("select", ranked_selection)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Decorators for size limit
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # --- Run Evolution ---
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    
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

    # Use trange for a progress bar
    for gen in (pbar := trange(1, n_generations + 1, desc="Generation")):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

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
    
    return pop, logbook, hof
