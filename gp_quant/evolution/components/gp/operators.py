"""
GP Operators and Primitive Set Configuration

This module configures the DEAP `PrimitiveSetTyped` which defines the grammar
of the trading rules. It brings together the custom primitives and standard
operators, enforcing a strict type system to ensure that only valid,
executable trees are generated.
"""
import random
import operator
import numpy as np
from deap import gp

from gp_quant.gp import primitives as prim

# --- Type System Definition ---
# For DEAP's typed system, we create dummy classes to represent our types.
# This allows the type checker to distinguish between numerical and boolean vectors.
class NumVector(np.ndarray): pass
class BoolVector(np.ndarray): pass

# --- Primitive Set Initialization ---
# The GP tree's inputs (terminals ARG0, ARG1) are numerical vectors.
# The final output of the entire tree MUST be a boolean vector.
pset = gp.PrimitiveSetTyped("MAIN", [NumVector, NumVector], BoolVector)

# --- Registering Primitives ---

# Boolean operators: These operate on and return boolean vectors.
pset.addPrimitive(np.logical_and, [BoolVector, BoolVector], BoolVector, name="logical_and")
pset.addPrimitive(np.logical_or, [BoolVector, BoolVector], BoolVector, name="logical_or")
pset.addPrimitive(prim.logical_not, [BoolVector], BoolVector, name="logical_not")

# Relational operators: These are the bridge. They take numerical vectors and return a boolean vector.
# 使用 NumPy 的向量化比較運算，而不是 Python 的 operator
pset.addPrimitive(np.less, [NumVector, NumVector], BoolVector, name="lt")
pset.addPrimitive(np.greater, [NumVector, NumVector], BoolVector, name="gt")

# Arithmetic operators: These operate on and return numerical vectors.
# Arithmetic operators: These operate on and return numerical vectors.
pset.addPrimitive(prim.add, [NumVector, NumVector], NumVector, name="add")
pset.addPrimitive(prim.sub, [NumVector, NumVector], NumVector, name="sub")
pset.addPrimitive(prim.mul, [NumVector, NumVector], NumVector, name="mul")
pset.addPrimitive(prim.protected_div, [NumVector, NumVector], NumVector, name="div")
pset.addPrimitive(prim.norm, [NumVector, NumVector], NumVector, name="norm")
pset.addPrimitive(prim.abs_val, [NumVector], NumVector, name="abs")
pset.addPrimitive(prim.log, [NumVector], NumVector, name="log")
pset.addPrimitive(prim.sign, [NumVector], NumVector, name="sign")

# Financial primitives: These take a numerical vector and an integer, returning a numerical vector.
pset.addPrimitive(prim.moving_average, [NumVector, int], NumVector, name="avg")
pset.addPrimitive(prim.moving_max, [NumVector, int], NumVector, name="max")
pset.addPrimitive(prim.moving_min, [NumVector, int], NumVector, name="min")
pset.addPrimitive(prim.lag, [NumVector, int], NumVector, name="lag")
pset.addPrimitive(prim.volatility, [NumVector, int], NumVector, name="vol")
pset.addPrimitive(prim.volatility, [NumVector, int], NumVector, name="std") # Alias for volatility
pset.addPrimitive(prim.rate_of_change, [NumVector, int], NumVector, name="ROC")
pset.addPrimitive(prim.relative_strength_index, [NumVector, int], NumVector, name="RSI")
pset.addPrimitive(prim.ts_rank, [NumVector, int], NumVector, name="ts_rank")
pset.addPrimitive(prim.correlation, [NumVector, NumVector, int], NumVector, name="corr")
pset.addPrimitive(prim.decay_linear, [NumVector, int], NumVector, name="decay_linear")

# Add a harmless identity primitive for integers to satisfy the generator
pset.addPrimitive(prim.identity_int, [int], int, name="id_int")

# Special 'if-then-else' primitive
# Note: DEAP's gp.if_then_else is not typed. We can create a typed version if needed,
# but for now, we can rely on boolean logic to achieve similar results.

# --- Registering Terminals ---

# Add boolean constant terminals. This is crucial to solve the IndexError when
# the generator needs to create a tree of height 0 with a BoolVector return type.
pset.addTerminal(True, BoolVector, name="V_TRUE")
pset.addTerminal(False, BoolVector, name="V_FALSE")

# Terminals are now implicitly ARG0 and ARG1 as defined in the PrimitiveSetTyped constructor.

# Ephemeral constants for generating random values at runtime

# 使用 functools.partial 替代 lambda 函數以支持 pickle 序列化
from functools import partial

def rand_float():
    """生成隨機浮點數"""
    return random.uniform(-1, 1)

def rand_int_n():
    """生成隨機整數（用於回望期間）"""
    return random.randint(5, 200)

# Ephemeral constants for generating random float values
# Register as NumVector to enable implicit broadcasting with vector operators
pset.addEphemeralConstant("rand_float", rand_float, NumVector)

# Random integer constants for lookback periods (e.g., 5 to 200 days)
pset.addEphemeralConstant("rand_int_n", rand_int_n, int)

print("Primitive set configured successfully.")
