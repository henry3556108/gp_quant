import unittest
from unittest.mock import MagicMock
from gp_quant.evolution.components.strategies.reproduction import StandardReproduction, ElitistReproduction

class MockIndividual:
    def __init__(self, fitness_value):
        self.fitness = MagicMock()
        self.fitness.values = (fitness_value,)
        self.id = fitness_value  # Use fitness as ID for simplicity
        self.operation = None

class TestReproductionStrategy(unittest.TestCase):
    
    def setUp(self):
        self.population = [MockIndividual(i) for i in range(10)]  # 0 to 9
        self.data = {}
        self.engine = MagicMock()
        self.engine.current_generation = 0
        
    def test_elitist_reproduction(self):
        strategy = ElitistReproduction(elite_ratio=1.0)
        strategy.set_engine(self.engine)
        
        # Select top 3
        k = 3
        reproduced = strategy.reproduce(self.population, k, self.data)
        
        self.assertEqual(len(reproduced), 3)
        # Should be 9, 8, 7
        fitnesses = [ind.fitness.values[0] for ind in reproduced]
        self.assertEqual(fitnesses, [9, 8, 7])
        self.assertEqual(reproduced[0].operation, 'reproduction_elite')
        
    def test_elitist_reproduction_partial(self):
        strategy = ElitistReproduction(elite_ratio=0.5)
        strategy.set_engine(self.engine)
        
        # Mock selection strategy for the non-elite part
        selection_mock = MagicMock()
        # Return individuals with fitness 0 and 1 for the non-elite part
        selection_mock.select_individuals.return_value = [self.population[0], self.population[1]]
        self.engine.strategies = {'selection': selection_mock}
        
        # Select 4 (2 elites, 2 standard)
        k = 4
        reproduced = strategy.reproduce(self.population, k, self.data)
        
        self.assertEqual(len(reproduced), 4)
        # Top 2 elites: 9, 8
        self.assertEqual(reproduced[0].fitness.values[0], 9)
        self.assertEqual(reproduced[1].fitness.values[0], 8)
        # Next 2 from selection mock: 0, 1
        self.assertEqual(reproduced[2].fitness.values[0], 0)
        self.assertEqual(reproduced[3].fitness.values[0], 1)
        
    def test_standard_reproduction(self):
        strategy = StandardReproduction()
        strategy.set_engine(self.engine)
        
        # Mock selection strategy
        selection_mock = MagicMock()
        selection_mock.select_individuals.return_value = [self.population[5], self.population[6]]
        self.engine.strategies = {'selection': selection_mock}
        
        k = 2
        reproduced = strategy.reproduce(self.population, k, self.data)
        
        self.assertEqual(len(reproduced), 2)
        self.assertEqual(reproduced[0].fitness.values[0], 5)
        self.assertEqual(reproduced[1].fitness.values[0], 6)

if __name__ == '__main__':
    unittest.main()
