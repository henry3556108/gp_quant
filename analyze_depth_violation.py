"""
深入分析深度違規的原因
測試 DEAP 的 staticLimit 行為
"""
from deap import creator, base, gp, tools
import operator
import random
from gp_quant.gp.operators import pset

# 初始化 DEAP
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# 創建 toolbox
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 註冊操作
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 添加深度限制
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

print("="*100)
print("測試 DEAP staticLimit 的行為")
print("="*100)

# 測試 1: 突變行為
print("\n【測試 1: 突變深度限制】")
print("-"*100)

# 創建一個深度接近 17 的個體
pop = toolbox.population(n=100)
# 找一個深度較大的個體
deep_individuals = sorted(pop, key=lambda x: x.height, reverse=True)[:10]

print(f"找到 10 個較深的個體，深度範圍: {[ind.height for ind in deep_individuals]}")

# 測試突變
print("\n測試突變 100 次:")
mutation_results = []
for i in range(100):
    # 選一個深度較大的個體
    ind = toolbox.clone(deep_individuals[i % 10])
    original_height = ind.height
    
    # 執行突變
    mutant, = toolbox.mutate(ind)
    new_height = mutant.height
    
    mutation_results.append({
        'original': original_height,
        'new': new_height,
        'changed': original_height != new_height,
        'increased': new_height > original_height,
        'violated': new_height > 17
    })

import pandas as pd
df_mut = pd.DataFrame(mutation_results)

print(f"突變成功次數: {df_mut['changed'].sum()}")
print(f"深度增加次數: {df_mut['increased'].sum()}")
print(f"違反深度 17 限制: {df_mut['violated'].sum()}")
print(f"最大深度: {df_mut['new'].max()}")

# 測試 2: 交叉行為
print("\n【測試 2: 交叉深度限制】")
print("-"*100)

# 創建兩個深度接近 17 的個體
crossover_results = []
for i in range(100):
    ind1 = toolbox.clone(deep_individuals[i % 10])
    ind2 = toolbox.clone(deep_individuals[(i+1) % 10])
    
    original_h1 = ind1.height
    original_h2 = ind2.height
    
    # 執行交叉
    child1, child2 = toolbox.mate(ind1, ind2)
    
    new_h1 = child1.height
    new_h2 = child2.height
    
    crossover_results.append({
        'parent1_h': original_h1,
        'parent2_h': original_h2,
        'child1_h': new_h1,
        'child2_h': new_h2,
        'child1_violated': new_h1 > 17,
        'child2_violated': new_h2 > 17,
        'any_violated': new_h1 > 17 or new_h2 > 17
    })

df_cross = pd.DataFrame(crossover_results)

print(f"交叉成功次數: {len(df_cross)}")
print(f"Child1 違反深度 17: {df_cross['child1_violated'].sum()}")
print(f"Child2 違反深度 17: {df_cross['child2_violated'].sum()}")
print(f"任一子代違反: {df_cross['any_violated'].sum()}")
print(f"Child1 最大深度: {df_cross['child1_h'].max()}")
print(f"Child2 最大深度: {df_cross['child2_h'].max()}")

# 測試 3: staticLimit 的實際機制
print("\n【測試 3: staticLimit 機制分析】")
print("-"*100)

# 創建一個深度為 17 的個體（手動構建）
print("\n嘗試手動創建深度 > 17 的個體並測試...")

# 生成一個大族群，找深度最大的
large_pop = toolbox.population(n=1000)
max_depth_ind = max(large_pop, key=lambda x: x.height)
print(f"初始族群中最大深度: {max_depth_ind.height}")

# 測試 staticLimit 是否真的阻止了操作
print("\n測試 staticLimit 的阻止機制:")

# 創建一個深度為 17 的個體
deep_ind = toolbox.clone(max_depth_ind)
while deep_ind.height < 15:
    # 不斷突變直到深度接近 17
    temp, = toolbox.mutate(toolbox.clone(deep_ind))
    if temp.height > deep_ind.height:
        deep_ind = temp

print(f"創建了一個深度為 {deep_ind.height} 的個體")

# 測試突變是否會被阻止
print("\n對深度 {} 的個體執行 50 次突變:".format(deep_ind.height))
blocked_count = 0
success_count = 0
for i in range(50):
    test_ind = toolbox.clone(deep_ind)
    original_str = str(test_ind)
    
    mutant, = toolbox.mutate(test_ind)
    new_str = str(mutant)
    
    if original_str == new_str:
        blocked_count += 1
    else:
        success_count += 1
        if mutant.height > 17:
            print(f"  警告: 突變後深度 {mutant.height} > 17!")

print(f"被阻止的突變: {blocked_count}")
print(f"成功的突變: {success_count}")

# 測試 4: 檢查 staticLimit 的實際實作
print("\n【測試 4: staticLimit 源碼分析】")
print("-"*100)

import inspect
print("staticLimit 函數簽名:")
print(inspect.signature(gp.staticLimit))

print("\nstaticLimit 文檔:")
print(gp.staticLimit.__doc__)

# 測試 staticLimit 的返回值
print("\n測試 staticLimit 裝飾器的行為:")

# 創建未裝飾的操作
def test_mutate(individual):
    return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)

# 裝飾它
decorated_mutate = gp.staticLimit(key=operator.attrgetter("height"), max_value=17)(test_mutate)

# 測試
test_ind = toolbox.clone(deep_ind)
print(f"原始深度: {test_ind.height}")

result = decorated_mutate(test_ind)
print(f"裝飾後突變結果類型: {type(result)}")
print(f"結果: {result}")

if result:
    print(f"突變後深度: {result[0].height if isinstance(result, tuple) else result.height}")
else:
    print("突變被阻止（返回原個體）")

print("\n" + "="*100)
print("分析完成")
print("="*100)
