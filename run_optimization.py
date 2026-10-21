"""
Runs urban heat mitigation optimization scenarios on the Mumbai digital twin:
1. Multi-budget Pareto Frontier generation
2. Hotspot-targeted Cool & Green Infrastructure placement
3. Exports publication heatmaps (Baseline vs Optimized vs Temperature Delta)
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from urbansim.data.mumbai_generator import (
    create_mumbai_urban_graph,
    generate_mumbai_weather_scenario
)
from urbansim.models.pignn import UrbanSimPIGNN
from urbansim.optimization.mitigation_engine import UrbanMitigationOptimizer


def run_optimization_pipeline():
    print("=" * 70)
    print("  URBANSIM: HEAT MITIGATION OPTIMIZATION PIPELINE")
    print("=" * 70)

    device = "cpu"
    graph = create_mumbai_urban_graph(grid_size=20, seed=42)
    graph.build_tensors(device=device)

    # Extreme summer heat scenario (Hour 14:00)
    weather = generate_mumbai_weather_scenario(hour=14.0)

    # Load model
    model = UrbanSimPIGNN(hidden_dim=64, num_layers=3).to(device)
    try:
        chk = torch.load("urbansim_checkpoint.pt", map_location=device)
        model.load_state_dict(chk["model_state_dict"])
        print("  Loaded trained UrbanSim model.")
    except Exception:
        print("  Notice: Trained checkpoint not found, running with initialized weights.")

    optimizer = UrbanMitigationOptimizer(
        model=model,
        graph=graph,
        weather_dict=weather,
        device=device
    )

    # 1. Run Differentiable Optimization under $1.5M budget
    budget = 1_500_000.0
    print(f"\n[1/3] Solving Differentiable Allocation under ${budget:,.0f} Budget...")
    opt_res = optimizer.optimize_differentiable_budget(budget_usd=budget, num_steps=120)

    print(f"  Baseline Mean LST:     {opt_res['baseline_mean_lst']:.2f} deg C")
    print(f"  Optimized Mean LST:    {opt_res['optimized_mean_lst']:.2f} deg C")
    print(f"  Average Heat Reduction:{opt_res['mean_cooling_c']:.2f} deg C")
    print(f"  Peak Hotspot Cooling:  {opt_res['max_cooling_c']:.2f} deg C")
    print(f"  Allocated Capital:     ${opt_res['final_cost_usd']:,.2f}")

    # 2. Generate Pareto Frontier
    print("\n[2/3] Generating Pareto Trade-off Frontier (Cooling vs Budget)...")
    budgets = [250_000.0, 500_000.0, 1_000_000.0, 1_750_000.0, 3_000_000.0]
    frontier = optimizer.generate_pareto_frontier(budgets)

    for pt in frontier:
        print(f"  Budget: ${pt['budget_usd']:,.0f} | Actual: ${pt['actual_cost_usd']:,.0f} | "
              f"Mean Cooling: -{pt['mean_cooling_c']:.2f} C | Peak: -{pt['max_cooling_c']:.2f} C")

    # 3. Save Summary
    summary = {
        "budget_usd": budget,
        "results": {
            "baseline_mean_lst": opt_res["baseline_mean_lst"],
            "optimized_mean_lst": opt_res["optimized_mean_lst"],
            "mean_cooling_c": opt_res["mean_cooling_c"],
            "max_cooling_c": opt_res["max_cooling_c"],
            "actual_cost_usd": opt_res["final_cost_usd"]
        },
        "pareto_frontier": frontier
    }

    with open("optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4. Generate Visual Heatmaps (20x20 spatial grids)
    print("\n[3/3] Rendering Geospatial Microclimate Heatmaps...")
    grid_size = 20
    base_grid = opt_res["baseline_lst"].reshape(grid_size, grid_size)
    opt_grid = opt_res["optimized_lst"].reshape(grid_size, grid_size)
    delta_grid = opt_res["delta_t"].reshape(grid_size, grid_size)
    cool_alloc_grid = opt_res["best_cool_allocation"].reshape(grid_size, grid_size)
    green_alloc_grid = opt_res["best_green_allocation"].reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # Heatmap 1: Baseline LST
    im0 = axes[0].imshow(base_grid, cmap="inferno", origin="lower", extent=[0, 1000, 0, 1000])
    axes[0].set_title("Baseline Surface Temperature (deg C)\n[Pre-Mitigation Peak Heat]")
    axes[0].set_xlabel("X (meters)")
    axes[0].set_ylabel("Y (meters)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Heatmap 2: Optimized LST
    im1 = axes[1].imshow(opt_grid, cmap="inferno", origin="lower", extent=[0, 1000, 0, 1000],
                        vmin=base_grid.min(), vmax=base_grid.max())
    axes[1].set_title("Optimized Surface Temperature (deg C)\n[With Cool & Green Infrastructure]")
    axes[1].set_xlabel("X (meters)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Heatmap 3: Cooling Differential (Delta T)
    im2 = axes[2].imshow(delta_grid, cmap="YlGnBu", origin="lower", extent=[0, 1000, 0, 1000])
    axes[2].set_title("Net Cooling Effect (Delta T deg C)\n[Hotspot Mitigation Magnitude]")
    axes[2].set_xlabel("X (meters)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("mumbai_mitigation_heatmaps.png")
    print("  Saved geospatial heatmaps to mumbai_mitigation_heatmaps.png")

    # Pareto plot
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=150)
    p_budgets = [pt["actual_cost_usd"] / 1e6 for pt in frontier]
    p_cooling = [pt["mean_cooling_c"] for pt in frontier]
    p_peak = [pt["max_cooling_c"] for pt in frontier]

    ax2.plot(p_budgets, p_cooling, marker="o", color="#2ECC71", linewidth=2.5, label="Mean Cooling (deg C)")
    ax2.plot(p_budgets, p_peak, marker="s", color="#3498DB", linewidth=2.0, linestyle="--", label="Peak Hotspot Cooling (deg C)")
    ax2.set_xlabel("Capital Expenditure (Million USD)")
    ax2.set_ylabel("Temperature Reduction (deg C)")
    ax2.set_title("Pareto Efficiency Frontier: Cooling Return vs Municipal Investment")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("pareto_frontier.png")
    print("  Saved Pareto frontier curve to pareto_frontier.png")


if __name__ == "__main__":
    run_optimization_pipeline()
