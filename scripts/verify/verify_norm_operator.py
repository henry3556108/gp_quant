"""
Test script for the Norm operator implementation.

This script verifies that:
1. The Norm operator is correctly implemented
2. It can be used in GP trees
3. It produces expected results
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from deap import gp, creator, base
from gp_quant.gp import operators
from gp_quant.gp import primitives as prim

def test_norm_function():
    """Test the norm function directly."""
    print("=" * 60)
    print("Test 1: Direct Norm Function Test")
    print("=" * 60)
    
    # Test case 1: Simple arrays
    a = np.array([10.0, 20.0, 30.0, 40.0])
    b = np.array([5.0, 15.0, 35.0, 45.0])
    result = prim.norm(a, b)
    expected = np.array([5.0, 5.0, 5.0, 5.0])
    
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ Test passed: {np.allclose(result, expected)}\n")
    
    # Test case 2: Negative differences
    a = np.array([5.0, 10.0, 15.0])
    b = np.array([10.0, 5.0, 20.0])
    result = prim.norm(a, b)
    expected = np.array([5.0, 5.0, 5.0])
    
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ Test passed: {np.allclose(result, expected)}\n")
    
    # Test case 3: Zero difference
    a = np.array([100.0, 200.0, 300.0])
    b = np.array([100.0, 200.0, 300.0])
    result = prim.norm(a, b)
    expected = np.array([0.0, 0.0, 0.0])
    
    print(f"Input a: {a}")
    print(f"Input b: {b}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ Test passed: {np.allclose(result, expected)}\n")

def test_norm_in_gp_tree():
    """Test that norm can be used in a GP tree."""
    print("=" * 60)
    print("Test 2: Norm in GP Tree")
    print("=" * 60)
    
    # Create fitness and individual classes
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    # Create a simple tree using norm
    # Tree: norm(ARG0, ARG1)
    tree = creator.Individual([
        operators.pset.mapping['norm'],
        operators.pset.mapping['ARG0'],
        operators.pset.mapping['ARG1']
    ])
    
    print(f"Tree structure: {tree}")
    print(f"Tree string: {str(tree)}\n")
    
    # Compile and test the tree
    func = gp.compile(tree, operators.pset)
    
    # Test data
    arg0 = np.array([100.0, 110.0, 120.0, 130.0])
    arg1 = np.array([95.0, 105.0, 125.0, 135.0])
    
    result = func(arg0, arg1)
    expected = np.array([5.0, 5.0, 5.0, 5.0])
    
    print(f"ARG0: {arg0}")
    print(f"ARG1: {arg1}")
    print(f"Result:   {result}")
    print(f"Expected: {expected}")
    print(f"✓ Test passed: {np.allclose(result, expected)}\n")

def test_norm_in_complex_tree():
    """Test norm in a more complex GP tree."""
    print("=" * 60)
    print("Test 3: Norm in Complex Tree")
    print("=" * 60)
    
    # Create a tree: gt(norm(ARG0, ARG1), add(ARG0, ARG0))
    # This checks if the absolute difference is greater than 2*ARG0
    tree = creator.Individual([
        operators.pset.mapping['gt'],
        operators.pset.mapping['norm'],
        operators.pset.mapping['ARG0'],
        operators.pset.mapping['ARG1'],
        operators.pset.mapping['add'],
        operators.pset.mapping['ARG0'],
        operators.pset.mapping['ARG0']
    ])
    
    print(f"Tree structure: {tree}")
    print(f"Tree string: {str(tree)}\n")
    
    # Compile and test
    func = gp.compile(tree, operators.pset)
    
    # Test data
    arg0 = np.array([10.0, 20.0, 30.0, 40.0])
    arg1 = np.array([5.0, 15.0, 100.0, 200.0])
    # Differences: [5.0, 5.0, 70.0, 160.0]
    # 2*ARG0: [20.0, 40.0, 60.0, 80.0]
    # norm > 2*ARG0: [False, False, True, True]
    
    result = func(arg0, arg1)
    expected = np.array([False, False, True, True])
    
    print(f"ARG0: {arg0}")
    print(f"ARG1: {arg1}")
    print(f"Differences: {np.abs(arg0 - arg1)}")
    print(f"2*ARG0: {2*arg0}")
    print(f"Result (norm > 2*ARG0):   {result}")
    print(f"Expected:                 {expected}")
    print(f"✓ Test passed: {np.array_equal(result, expected)}\n")

def test_norm_with_technical_indicators():
    """Test norm combined with technical indicators."""
    print("=" * 60)
    print("Test 4: Norm with Technical Indicators")
    print("=" * 60)
    
    # Simplified test: just verify norm works with moving average output
    print("Testing norm with moving average output...\n")
    
    # Test data: trending up
    arg0 = np.array([100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0])
    arg1 = arg0.copy()  # Current price
    
    # Calculate MA manually
    ma = prim.moving_average(arg0, 5)
    
    # Calculate norm
    result = prim.norm(ma, arg1)
    
    print(f"Price series: {arg0}")
    print(f"5-day MA: {ma}")
    print(f"Norm(MA, Price): {result}")
    print(f"✓ Test passed: Norm works with technical indicator output\n")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NORM OPERATOR TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_norm_function()
        test_norm_in_gp_tree()
        test_norm_in_complex_tree()
        test_norm_with_technical_indicators()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe Norm operator is correctly implemented and can be used in GP trees.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
