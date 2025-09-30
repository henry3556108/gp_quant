#!/bin/bash

# This script automates the setup and testing of the GP Quant project.
# It ensures that all commands are run in a clean, isolated environment.

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the name of the conda environment
ENV_NAME="gp_quant"

# --- Step 1: Create Conda Environment ---
echo "--- Creating Conda environment: $ENV_NAME ---"

# Check if the environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    conda create --name $ENV_NAME python=3.10 -y
fi

echo "Conda environment setup complete."

# --- Step 2: Install Dependencies ---
echo "\n--- Installing dependencies from requirements.txt into $ENV_NAME ---"

# Use 'conda run' to execute commands within the specified environment
conda run -n $ENV_NAME pip install -r requirements.txt

# Special step for pygraphviz on macOS if needed
if [[ "$(uname)" == "Darwin" ]]; then
    echo "\nChecking for Graphviz system library (required for pygraphviz)..."
    if ! brew list graphviz &>/dev/null; then
        echo "Warning: Graphviz not found. Please run 'brew install graphviz' for visualization features."
    fi
fi

echo "Dependency installation complete."

# --- Step 3: Run Tests ---
echo "\n--- Running test suite in $ENV_NAME environment ---"

conda run -n $ENV_NAME python -m unittest discover tests

echo "\n--- Setup and test script finished successfully! ---"
