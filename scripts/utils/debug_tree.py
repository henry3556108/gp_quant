
import numpy as np
import pandas as pd
from deap import gp
from gp_quant.evolution.components.gp.operators import pset, NumVector

# Define the tree string from the log
tree_str = "gt(min(abs(vol(ARG0, 85)), id_int(id_int(id_int(82)))), sub(ROC(mul(ARG1, ARG0), id_int(187)), sub(min(ARG0, 40), add(0.5809647461822185, ARG0))))"

print(f"Tree String: {tree_str}")

# Create dummy data
n = 100
price_vec = np.random.rand(n).astype(float).view(NumVector)
volume_vec = np.random.rand(n).astype(float).view(NumVector)

print(f"Price Vec Type: {type(price_vec)}")
print(f"Volume Vec Type: {type(volume_vec)}")

# Compile
try:
    func = gp.compile(tree_str, pset)
    print("Compilation successful.")
except Exception as e:
    print(f"Compilation failed: {e}")
    exit(1)

# Execute
try:
    print("Executing function...")
    result = func(price_vec, volume_vec)
    print(f"Execution successful. Result type: {type(result)}")
    print(result)
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
