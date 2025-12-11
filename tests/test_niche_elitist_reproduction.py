import unittest
from unittest.mock import MagicMock, Mock
from gp_quant.evolution.components.strategies.reproduction import NicheElitistReproduction
from gp_quant.evolution.components.individual import EvolutionIndividual
from deap import creator, base

class TestNicheElitistReproduction(unittest.TestCase):
    def setUp(self):
        # Initialize DEAP creator
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", EvolutionIndividual, fitness=creator.FitnessMax)
            
        self.strategy = NicheElitistReproduction()
        self.engine = Mock()
        self.engine.current_generation = 0  # Set to integer
        self.strategy.set_engine(self.engine)
        
        # Mock Selection Strategy
        self.selection_strategy = Mock()
        self.engine.strategies = {'selection': self.selection_strategy}
        
    def create_individual(self, fitness_value, id_val):
        # Pass dummy content list for PrimitiveTree
        ind = EvolutionIndividual([])
        ind.fitness = Mock()
        ind.fitness.values = (fitness_value,)
        ind.id = id_val
        return ind
        
    def test_round_robin_selection(self):
        """Test that individuals are selected in a round-robin fashion from clusters."""
        # Setup clusters
        # Cluster 1: [10.0, 9.0]
        # Cluster 2: [8.0, 7.0]
        # Cluster 3: [6.0]
        c1 = [self.create_individual(10.0, 'c1_1'), self.create_individual(9.0, 'c1_2')]
        c2 = [self.create_individual(8.0, 'c2_1'), self.create_individual(7.0, 'c2_2')]
        c3 = [self.create_individual(6.0, 'c3_1')]
        
        clusters = [c1, c2, c3]
        
        # Mock _get_or_compute_niching
        self.selection_strategy._get_or_compute_niching = Mock(return_value=(clusters, []))
        
        population = c1 + c2 + c3
        data = {'generation': 1}
        
        # Request 4 individuals
        # Expected order: 
        # 1. c1_1 (Cluster 1 best)
        # 2. c2_1 (Cluster 2 best)
        # 3. c3_1 (Cluster 3 best)
        # 4. c1_2 (Cluster 1 second best)
        selected = self.strategy.reproduce(population, 4, data)
        
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0].id, 'c1_1')
        self.assertEqual(selected[1].id, 'c2_1')
        self.assertEqual(selected[2].id, 'c3_1')
        self.assertEqual(selected[3].id, 'c1_2')
        
        # Verify operation tag
        for ind in selected:
            self.assertEqual(ind.operation, 'reproduction_niche_elite')
            
    def test_fallback_when_no_clusters(self):
        """Test fallback to standard elitist reproduction when no clusters are returned."""
        # Mock _get_or_compute_niching to return None
        self.selection_strategy._get_or_compute_niching = Mock(return_value=(None, None))
        
        population = [
            self.create_individual(1.0, 'ind1'),
            self.create_individual(2.0, 'ind2'),
            self.create_individual(3.0, 'ind3')
        ]
        
        # Request 2 individuals -> Should pick top 2 by fitness (3.0, 2.0)
        selected = self.strategy.reproduce(population, 2, {})
        
        self.assertEqual(len(selected), 2)
        # Sort by fitness to verify (ElitistReproduction sorts them)
        selected.sort(key=lambda x: x.fitness.values[0], reverse=True)
        self.assertEqual(selected[0].id, 'ind3')
        self.assertEqual(selected[1].id, 'ind2')
        
    def test_insufficient_individuals(self):
        """Test behavior when k > total population."""
        c1 = [self.create_individual(10.0, 'c1_1')]
        clusters = [c1]
        self.selection_strategy._get_or_compute_niching = Mock(return_value=(clusters, []))
        
        population = c1
        
        # Request 2 individuals, but only 1 exists
        # Should return the 1 available, then fill with random (which is the same one here)
        selected = self.strategy.reproduce(population, 2, {})
        
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].id, 'c1_1')
        # The second one is a clone of the same individual (randomly picked)
        self.assertEqual(selected[1].id, 'c1_1')

if __name__ == '__main__':
    unittest.main()
