import dill
from deap import creator, base, gp
from gp_quant.gp.operators import pset

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

with open('portfolio_experiment_results/portfolio_exp_sharpe_20251023_125111/generations/generation_001.pkl', 'rb') as f:
    data = dill.load(f)

print('Type:', type(data))
if isinstance(data, dict):
    print('Keys:', list(data.keys()))
    if 'cluster_labels' in data:
        print('Has cluster_labels!')
        print('Cluster labels length:', len(data['cluster_labels']))
else:
    print('Length:', len(data))
    print('First item type:', type(data[0]))
