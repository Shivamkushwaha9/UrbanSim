"""
Benchmarking script comparing UrbanSim (PI-GNN) against:
1. Standard MLP (Tabular baseline)
2. Non-Physics GAT (Pure MSE loss)
3. Empirical TEB (Diagnostic Urban Canopy Model)
Produces quantitative metrics (MAE, RMSE, R2, Energy Balance Residual, Latency)
and generates publication-quality plots.
"""

import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from urbansim.data.mumbai_generator import (
    create_mumbai_urban_graph,
    generate_mumbai_weather_scenario,
    simulate_ground_truth_lst
)
from urbansim.models.pignn import UrbanSimPIGNN
from urbansim.models.baselines import StandardMLP, NonPhysicsGAT, EmpiricalTEB
from urbansim.physics.thermodynamics import solve_surface_energy_balance_residual, celsius_to_kelvin


def compute_metrics(pred_c, target_c, graph, weather_dict):
    """Computes MAE, RMSE, R2, and physical SEB residual."""
    mae = torch.mean(torch.abs(pred_c - target_c)).item()
    rmse = torch.sqrt(torch.mean((pred_c - target_c) ** 2)).item()
    ss_tot = torch.sum((target_c - target_c.mean()) ** 2).item()
    ss_res = torch.sum((target_c - pred_c) ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # Evaluate physical Surface Energy Balance Residual
    pred_k = celsius_to_kelvin(pred_c.squeeze(-1))
    air_k = celsius_to_kelvin(weather_dict["air_temp_c"])
    dew_k = celsius_to_kelvin(weather_dict["dew_temp_c"])

    residuals, _ = solve_surface_energy_balance_residual(
        surface_temp_k=pred_k,
        air_temp_k=air_k,
        dew_point_k=dew_k,
        solar_irradiance=weather_dict["solar_irradiance"],
        albedo=graph.node_features[:, 3],
        emissivity=graph.node_features[:, 4],
        sky_view_factor=graph.node_features[:, 7],
        wind_speed=weather_dict["wind_speed"],
        relative_humidity=weather_dict["relative_humidity"],
        vegetation_fraction=graph.node_features[:, 5],
        thermal_conductivity=graph.node_features[:, 6],
        traffic_density=graph.node_features[:, 8]
    )

    mean_res = torch.mean(torch.abs(residuals)).item()
    return mae, rmse, r2, mean_res


def run_benchmark():
    print("=" * 70)
    print("  URBANSIM: ARCHITECTURE BENCHMARK & COMPARATIVE EVALUATION")
    print("=" * 70)

    device = "cpu"
    graph = create_mumbai_urban_graph(grid_size=20, seed=101)
    graph.build_tensors(device=device)

    # Test under severe peak heatwave conditions (Hour 14:00)
    weather = generate_mumbai_weather_scenario(hour=14.0)
    target_lst = simulate_ground_truth_lst(graph, weather, noise_std=0.2).to(device)

    # 1. Load or instantiate models
    pignn = UrbanSimPIGNN(hidden_dim=64, num_layers=3).to(device)
    if torch.cuda.is_available() and False:
        pass
    try:
        chk = torch.load("urbansim_checkpoint.pt", map_location=device)
        pignn.load_state_dict(chk["model_state_dict"])
        print("  Loaded trained UrbanSim PI-GNN weights.")
    except Exception:
        print("  Notice: Pretrained checkpoint not found, evaluating initialized weights.")

    mlp = StandardMLP().to(device)
    gat = NonPhysicsGAT().to(device)
    teb = EmpiricalTEB()

    # Quick calibration of baseline models on 30 epochs
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=0.01)
    opt_gat = torch.optim.Adam(gat.parameters(), lr=0.01)
    for _ in range(40):
        opt_mlp.zero_grad()
        p_mlp = mlp(graph.node_features, None, None, weather)
        l_mlp = nn.functional.mse_loss(p_mlp, target_lst)
        l_mlp.backward()
        opt_mlp.step()

        opt_gat.zero_grad()
        p_gat = gat(graph.node_features, graph.edge_index, None, weather)
        l_gat = nn.functional.mse_loss(p_gat, target_lst)
        l_gat.backward()
        opt_gat.step()

    # Benchmark timings & inference
    pignn.eval()
    mlp.eval()
    gat.eval()

    models = {
        "Empirical TEB": lambda: teb.predict(graph, weather),
        "Standard MLP": lambda: mlp(graph.node_features, None, None, weather),
        "Non-Physics GAT": lambda: gat(graph.node_features, graph.edge_index, None, weather),
        "UrbanSim (PI-GNN)": lambda: pignn(graph.node_features, graph.edge_index, graph.edge_attr, weather)
    }

    results = {}
    print("\nEvaluating models...")

    for name, fn in models.items():
        # Timing
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                preds = fn()
        elapsed_ms = (time.time() - t0) * 100.0  # ms per run

        mae, rmse, r2, seb_res = compute_metrics(preds, target_lst, graph, weather)
        results[name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "SEB_Residual_W_m2": round(seb_res, 2),
            "Latency_ms": round(elapsed_ms, 2)
        }
        print(f"  {name:20s} | MAE: {mae:.3f} C | RMSE: {rmse:.3f} C | R2: {r2:.4f} | "
              f"SEB Residual: {seb_res:.1f} W/m2 | Latency: {elapsed_ms:.2f} ms")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate Publication Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=150)
    names = list(results.keys())

    maes = [results[n]["MAE"] for n in names]
    residuals = [results[n]["SEB_Residual_W_m2"] for n in names]
    r2s = [results[n]["R2"] for n in names]

    colors = ["#4A90E2", "#F5A623", "#9013FE", "#2ECC71"]

    axes[0].bar(names, maes, color=colors)
    axes[0].set_title("LST Prediction MAE (deg C) [Lower is Better]")
    axes[0].set_ylabel("MAE (deg C)")
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    axes[1].bar(names, residuals, color=colors)
    axes[1].set_title("Surface Energy Balance Residual (W/m2) [Lower is Better]")
    axes[1].set_ylabel("Mean |R_SEB| (W/m2)")
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    axes[2].bar(names, r2s, color=colors)
    axes[2].set_title("R2 Score [Higher is Better]")
    axes[2].set_ylabel("R2")
    axes[2].set_ylim([0.0, 1.05])
    axes[2].tick_params(axis='x', rotation=20)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("benchmark_comparison.png")
    print("\nSaved benchmark plot to benchmark_comparison.png")


if __name__ == "__main__":
    run_benchmark()
