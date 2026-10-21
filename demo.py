"""
Turnkey Master Demonstration Script for UrbanSim.
Executes the full pipeline:
1. Synthesizes Mumbai urban morphology graph and diurnal weather.
2. Trains the Physics-Informed Graph Neural Network (PI-GNN) surrogate.
3. Evaluates and benchmarks against MLP, Non-Physics GAT, and Empirical TEB.
4. Executes differentiable heat mitigation optimization and exports Pareto curves.
5. Launches the interactive web digital twin dashboard.
"""

import os
import sys
import time

def run_master_pipeline():
    print("=" * 80)
    print("  URBANSIM: AI-DRIVEN MICROCLIMATE DIGITAL TWIN & HEAT MITIGATION")
    print("  Physics-Informed Graph Neural Networks for Climate Change Adaptation")
    print("=" * 80)

    # Step 1: Model Training
    print("\n>>> STEP 1: TRAINING PHYSICS-INFORMED GNN SURROGATE MODEL...")
    from train import train_urbansim
    train_urbansim(epochs=80, lr=0.005)

    # Step 2: Benchmarking against Baselines
    print("\n>>> STEP 2: BENCHMARKING AGAINST BASELINE MODELS...")
    from evaluate import run_benchmark
    run_benchmark()

    # Step 3: Mitigation Optimization & Heatmaps
    print("\n>>> STEP 3: RUNNING MITIGATION OPTIMIZATION & GEOSPATIAL HEATMAP GENERATION...")
    from run_optimization import run_optimization_pipeline
    run_optimization_pipeline()

    # Step 4: Web Dashboard
    print("\n>>> STEP 4: LAUNCHING INTERACTIVE WEB DASHBOARD...")
    print("  Server starting on http://127.0.0.1:5000")
    print("  Press Ctrl+C to terminate.")
    from urbansim.visualizer.server import run_server
    run_server(port=5000)


if __name__ == "__main__":
    run_master_pipeline()
