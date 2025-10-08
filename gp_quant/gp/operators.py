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
pset.addPrimitive(operator.lt, [NumVector, NumVector], BoolVector, name="lt")
pset.addPrimitive(operator.gt, [NumVector, NumVector], BoolVector, name="gt")

# Arithmetic operators: These operate on and return numerical vectors.
pset.addPrimitive(prim.add, [NumVector, NumVector], NumVector, name="add")
pset.addPrimitive(prim.sub, [NumVector, NumVector], NumVector, name="sub")
pset.addPrimitive(prim.mul, [NumVector, NumVector], NumVector, name="mul")
pset.addPrimitive(prim.protected_div, [NumVector, NumVector], NumVector, name="div")

# Financial primitives: These take a numerical vector and an integer, returning a numerical vector.
pset.addPrimitive(prim.moving_average, [NumVector, int], NumVector, name="avg")
pset.addPrimitive(prim.moving_max, [NumVector, int], NumVector, name="max")
pset.addPrimitive(prim.moving_min, [NumVector, int], NumVector, name="min")
pset.addPrimitive(prim.lag, [NumVector, int], NumVector, name="lag")
pset.addPrimitive(prim.volatility, [NumVector, int], NumVector, name="vol")
pset.addPrimitive(prim.rate_of_change, [NumVector, int], NumVector, name="ROC")
pset.addPrimitive(prim.relative_strength_index, [NumVector, int], NumVector, name="RSI")

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

# Ephemeral constants for generating random float values
pset.addEphemeralConstant("rand_float", lambda: random.uniform(-1, 1), float)

# Random integer constants for lookback periods (e.g., 5 to 200 days)
pset.addEphemeralConstant("rand_int_n", lambda: random.randint(5, 200), int)

# # Add some fixed common lookback periods as terminals
# pset.addTerminal(10, int)
# pset.addTerminal(20, int)
# pset.addTerminal(50, int)
# pset.addTerminal(100, int)

print("Primitive set configured successfully.")
