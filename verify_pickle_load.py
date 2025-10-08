"""Verify that saved populations can be loaded correctly"""
import dill
import os
from deap import creator, base, gp

# Setup DEAP creator (required before loading)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Try to load a saved population
pickle_file = "experiments_results/ABX_TO/individual_records/generation_000/population.pkl"

if not os.path.exists(pickle_file):
    print(f"❌ File not found: {pickle_file}")
    exit(1)

print(f"Loading population from: {pickle_file}")
print(f"File size: {os.path.getsize(pickle_file)} bytes")

try:
    with open(pickle_file, 'rb') as f:
        population = dill.load(f)
    
    print(f"\n✅ Successfully loaded population!")
    print(f"Population size: {len(population)}")
    print(f"\nFirst 3 individuals:")
    for i, ind in enumerate(population[:3]):
        print(f"\n  Individual {i}:")
        print(f"    Fitness: {ind.fitness.values if ind.fitness.valid else 'Not evaluated'}")
        print(f"    Structure: {str(ind)[:100]}...")  # First 100 chars
        print(f"    Height: {ind.height}")
        print(f"    Length: {len(ind)}")
    
    print(f"\n✅ Verification successful! Populations can be loaded and analyzed.")
    
except Exception as e:
    print(f"\n❌ Failed to load population: {e}")
    import traceback
    traceback.print_exc()
