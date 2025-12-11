import os
import sys
import pickle
import glob
import re
from pathlib import Path

def load_latest_population(exp_dir):
    """Robustly load the latest population."""
    exp_path = Path(exp_dir)
    pop_dir = exp_path / 'populations'
    
    if not pop_dir.exists():
        raise FileNotFoundError(f"Populations directory not found: {pop_dir}")
        
    pop_files = glob.glob(str(pop_dir / 'generation_*.pkl'))
    if not pop_files:
        raise FileNotFoundError(f"No generation pickle files found in {pop_dir}")
        
    # Sort by generation number
    def get_gen_num(filepath):
        match = re.search(r'generation_(\d+)\.pkl', filepath)
        return int(match.group(1)) if match else -1
        
    pop_files.sort(key=get_gen_num, reverse=True)
    
    for pop_file in pop_files:
        try:
            with open(pop_file, 'rb') as f:
                population = pickle.load(f)
            if population:
                gen_num = get_gen_num(pop_file)
                print(f"Loaded generation {gen_num} from {pop_file}")
                return population, gen_num
        except Exception as e:
            print(f"Failed to load {pop_file}: {e}")
            
    raise RuntimeError("No valid population found")

def main():
    exp_path = "experiment_result/pnl_niche_elitist_records_20251130_2029"
    
    print(f"🔍 Analyzing experiment: {exp_path}")
    
    try:
        population, gen = load_latest_population(exp_path)
        
        # Sort by fitness descending
        # Handle cases where fitness might not be set or valid
        valid_pop = [ind for ind in population if hasattr(ind.fitness, 'values') and ind.fitness.values]
        valid_pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
        
        print(f"\n🏆 Top 50 Fitness Scores (Generation {gen}):")
        print("-" * 40)
        print(f"{'Rank':<6} | {'Fitness':<15} | {'ID':<10}")
        print("-" * 40)
        
        for i, ind in enumerate(valid_pop[:50]):
            fitness = ind.fitness.values[0]
            ind_id = str(ind.id)[:8] if hasattr(ind, 'id') else "N/A"
            print(f"{i+1:<6} | {fitness:<15.6f} | {ind_id:<10}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
