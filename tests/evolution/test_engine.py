"""
Integration test for the evolutionary engine.
"""
import unittest
import pandas as pd
from deap import creator, base, gp, tools

# Add project root to path to allow imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from gp_quant.evolution.engine import run_evolution
from gp_quant.gp.operators import pset

class TestEvolutionEngine(unittest.TestCase):
    """Integration test suite for the main evolutionary engine."""

    def setUp(self):
        """Set up mock data and DEAP creator for a test run."""
        # Create a small, predictable dataset
        data = {
            'Open':  [100, 102, 104, 103, 100, 98, 101, 103, 105, 107],
            'Close': [101, 103, 102, 101, 99, 97, 100, 102, 104, 106],
            'Volume':[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=10))
        self.mock_data = pd.DataFrame(data, index=dates)

        # DEAP setup
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    def test_run_evolution_completes(self):
        """
        Test that a small-scale evolution runs to completion without errors.
        This is an integration test, not a performance test.
        """
        try:
            pop, log, hof = run_evolution(
                data=self.mock_data,
                population_size=10,  # Small population
                n_generations=2,     # Few generations
                crossover_prob=0.5,
                mutation_prob=0.2
            )
            
            # Check that the outputs are of the correct type
            self.assertIsInstance(pop, list)
            self.assertIsInstance(log, tools.Logbook)
            self.assertIsInstance(hof, tools.HallOfFame)

            # Check that the Hall of Fame contains an individual
            self.assertEqual(len(hof), 1)
            self.assertIsInstance(hof[0], creator.Individual)
            self.assertTrue(hof[0].fitness.valid)

        except Exception as e:
            self.fail(f"Evolutionary run failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
