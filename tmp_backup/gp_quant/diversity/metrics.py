"""
Diversity Metrics

Provides various diversity metrics for analyzing genetic programming populations.
"""

import numpy as np
from typing import List, Dict, Any
from collections import Counter


class DiversityMetrics:
    """Calculate various diversity metrics for a population of GP individuals."""
    
    @staticmethod
    def structural_diversity(population: List[Any]) -> Dict[str, float]:
        """
        Calculate structural diversity based on tree properties.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            Dictionary containing:
                - height_std: Standard deviation of tree heights
                - height_mean: Mean tree height
                - length_std: Standard deviation of tree lengths
                - length_mean: Mean tree length
                - complexity_mean: Average tree complexity (height * length)
        """
        heights = [ind.height for ind in population]
        lengths = [len(ind) for ind in population]
        complexities = [h * l for h, l in zip(heights, lengths)]
        
        return {
            'height_std': float(np.std(heights)),
            'height_mean': float(np.mean(heights)),
            'length_std': float(np.std(lengths)),
            'length_mean': float(np.mean(lengths)),
            'complexity_mean': float(np.mean(complexities)),
            'complexity_std': float(np.std(complexities))
        }
    
    @staticmethod
    def genotypic_diversity(population: List[Any]) -> Dict[str, float]:
        """
        Calculate genotypic diversity based on unique tree structures.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            Dictionary containing:
                - unique_ratio: Ratio of unique individuals to total population
                - unique_count: Number of unique individuals
                - duplicate_ratio: Ratio of duplicates
        """
        # Convert trees to strings for comparison
        tree_strings = [str(ind) for ind in population]
        unique_count = len(set(tree_strings))
        total_count = len(population)
        
        return {
            'unique_ratio': unique_count / total_count if total_count > 0 else 0.0,
            'unique_count': unique_count,
            'duplicate_ratio': (total_count - unique_count) / total_count if total_count > 0 else 0.0,
            'total_count': total_count
        }
    
    @staticmethod
    def fitness_diversity(population: List[Any]) -> Dict[str, float]:
        """
        Calculate fitness diversity.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            Dictionary containing:
                - fitness_std: Standard deviation of fitness values
                - fitness_mean: Mean fitness
                - fitness_cv: Coefficient of variation (std/mean)
                - fitness_range: Range (max - min)
                - fitness_min: Minimum fitness
                - fitness_max: Maximum fitness
        """
        # Extract fitness values (handle invalid fitness)
        fitness_values = [ind.fitness.values[0] for ind in population if ind.fitness.valid]
        
        if not fitness_values:
            return {
                'fitness_std': 0.0,
                'fitness_mean': 0.0,
                'fitness_cv': 0.0,
                'fitness_range': 0.0,
                'fitness_min': 0.0,
                'fitness_max': 0.0
            }
        
        fitness_mean = float(np.mean(fitness_values))
        fitness_std = float(np.std(fitness_values))
        
        return {
            'fitness_std': fitness_std,
            'fitness_mean': fitness_mean,
            'fitness_cv': fitness_std / abs(fitness_mean) if fitness_mean != 0 else 0.0,
            'fitness_range': float(np.max(fitness_values) - np.min(fitness_values)),
            'fitness_min': float(np.min(fitness_values)),
            'fitness_max': float(np.max(fitness_values))
        }
    
    @staticmethod
    def phenotypic_diversity(population: List[Any]) -> Dict[str, Any]:
        """
        Calculate phenotypic diversity based on used primitives and terminals.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            Dictionary containing:
                - unique_primitives: Number of unique primitives used
                - unique_terminals: Number of unique terminals used
                - primitive_usage: Distribution of primitive usage
                - terminal_usage: Distribution of terminal usage
        """
        all_primitives = []
        all_terminals = []
        
        for ind in population:
            # Extract primitives and terminals from the tree
            for node in ind:
                node_str = str(node)
                # Simple heuristic: if it starts with uppercase or is a function, it's a primitive
                if hasattr(node, 'name'):
                    if node.arity > 0:  # It's a function/primitive
                        all_primitives.append(node.name)
                    else:  # It's a terminal
                        all_terminals.append(node_str)
        
        primitive_counter = Counter(all_primitives)
        terminal_counter = Counter(all_terminals)
        
        return {
            'unique_primitives': len(primitive_counter),
            'unique_terminals': len(terminal_counter),
            'primitive_usage': dict(primitive_counter),
            'terminal_usage': dict(terminal_counter),
            'total_primitives_used': sum(primitive_counter.values()),
            'total_terminals_used': sum(terminal_counter.values())
        }
    
    @classmethod
    def calculate_all(cls, population: List[Any]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all diversity metrics for a population.
        
        Args:
            population: List of DEAP individuals
            
        Returns:
            Dictionary containing all metrics organized by category
        """
        return {
            'structural': cls.structural_diversity(population),
            'genotypic': cls.genotypic_diversity(population),
            'fitness': cls.fitness_diversity(population),
            'phenotypic': cls.phenotypic_diversity(population)
        }
