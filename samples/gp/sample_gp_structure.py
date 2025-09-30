"""
Sample script to demonstrate the Genetic Programming structure and visualization.

This script performs the following actions:
1. Imports the configured Primitive Set (`pset`).
2. Uses DEAP to generate a random, valid trading rule tree.
3. Prints the string representation of the generated rule.
4. Visualizes the tree and saves it to a file ('samples/outputs/gp_tree.png').

Note: This script requires `networkx` and `pygraphviz` to be installed for tree
visualization. `pygraphviz` also requires the Graphviz system library.
  - `pip install networkx pygraphviz`
  - On macOS, you might need: `brew install graphviz`
"""
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from deap import gp, creator, base, tools
from gp_quant.gp.operators import pset

def main():
    """Main function to run the demonstration."""
    print("--- Running GP Structure and Visualization Sample ---")

    # DEAP creator setup (needed for tree generation)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # Generate a single random individual (trading rule tree)
    individual = toolbox.individual()

    print("\nGenerated Trading Rule (string representation):")
    print(str(individual))

    # --- Visualize the Tree ---
    nodes, edges, labels = gp.graph(individual)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(g, pos, node_size=900, node_color='lightblue')
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=10)
    plt.title("Generated GP Tree", fontsize=16)
    
    # Save the plot
    output_dir = os.path.join(project_root, 'samples', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'gp_tree.png')
    
    try:
        plt.savefig(save_path)
        print(f"\nGP tree visualization saved to: {save_path}")
    except Exception as e:
        print(f"\nCould not save visualization. Error: {e}")
        print("Please ensure you have `pygraphviz` and Graphviz system library installed.")

    print("\n--- Sample script finished. ---")

if __name__ == "__main__":
    main()
