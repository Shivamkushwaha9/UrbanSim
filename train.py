"""
End-to-end training script for UrbanSim Physics-Informed Graph Neural Network (PI-GNN).
Trains on Mumbai microclimate data, enforcing Surface Energy Balance conservation.
"""

import os
import json
import time
import torch
import torch.optim as optim
import numpy as np

from urbansim.data.mumbai_generator import (
    create_mumbai_urban_graph,
    generate_mumbai_weather_scenario,
    simulate_ground_truth_lst
)
from urbansim.models.pignn import UrbanSimPIGNN
from urbansim.physics.loss import PhysicsInformedLoss


def train_urbansim(
    epochs: int = 150,
    lr: float = 0.005,
    lambda_phys: float = 0.35,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "urbansim_checkpoint.pt"
):
    print("=" * 70)
    print("  URBANSIM: PHYSICS-INFORMED GNN TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device: {device.upper()} | PyTorch: {torch.__version__}")

    # 1. Generate Mumbai Urban Graph
    print("\n[1/5] Synthesizing Mumbai Urban Morphology Graph (BKC, Dharavi, Kurla)...")
    graph = create_mumbai_urban_graph(grid_size=20, seed=42)
    graph.build_tensors(device=device)
    print(f"  Nodes (Parcels/Buildings): {graph.num_nodes}")
    print(f"  Edges (Canyon Interactions): {graph.edge_index.size(1)}")

    # 2. Generate Multi-Hour Diurnal Scenarios
    print("\n[2/5] Simulating Diurnal Meteorological Scenarios & Ground Truth LST...")
    hours = [10.0, 12.0, 14.0, 16.0]
    scenarios = []

    for h in hours:
        weather = generate_mumbai_weather_scenario(hour=h)
        # Move weather tensors to device
        weather_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in weather.items()}
        target_lst = simulate_ground_truth_lst(graph, weather).to(device)
        scenarios.append({
            "hour": h,
            "weather": weather_dev,
            "target_lst": target_lst
        })
        print(f"  Hour {h:04.1f}: Air Temp = {weather['air_temp_c'].item():.1f} C, "
              f"Solar = {weather['solar_irradiance'].item():.1f} W/m2, "
              f"Mean LST = {target_lst.mean().item():.1f} C")

    # 3. Model & Loss Setup
    print("\n[3/5] Initializing PI-GNN Architecture and Physics-Informed Loss...")
    model = UrbanSimPIGNN(
        node_in_dim=9,
        edge_in_dim=4,
        weather_dim=5,
        hidden_dim=64,
        num_layers=3
    ).to(device)

    criterion = PhysicsInformedLoss(
        lambda_phys=lambda_phys,
        lambda_smooth=0.05,
        lambda_cons=0.02
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    print("\n[4/5] Training PI-GNN Surrogate...")
    start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_data_loss = 0.0
        total_phys_loss = 0.0
        total_residual_w_m2 = 0.0

        for sc in scenarios:
            optimizer.zero_grad()
            weather = sc["weather"]
            target_lst = sc["target_lst"]

            pred_lst = model(
                graph.node_features,
                graph.edge_index,
                graph.edge_attr,
                weather
            )

            loss_dict = criterion(
                pred_lst_c=pred_lst,
                target_lst_c=target_lst,
                node_features=graph.node_features,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                weather_dict=weather
            )

            loss_dict["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss_dict["loss_total"].item()
            total_data_loss += loss_dict["loss_data"].item()
            total_phys_loss += loss_dict["loss_physics"].item()
            total_residual_w_m2 += loss_dict["mean_seb_residual_w_m2"].item()

        scheduler.step()

        num_sc = len(scenarios)
        avg_loss = total_loss / num_sc
        avg_data = total_data_loss / num_sc
        avg_phys = total_phys_loss / num_sc
        avg_res = total_residual_w_m2 / num_sc

        history.append({
            "epoch": epoch,
            "loss_total": avg_loss,
            "loss_data": avg_data,
            "loss_phys": avg_phys,
            "mean_residual_w_m2": avg_res
        })

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:03d}/{epochs:03d}] | Loss: {avg_loss:.4f} "
                  f"(Data: {avg_data:.4f}, Phys: {avg_phys:.4f}) | "
                  f"SEB Residual: {avg_res:.2f} W/m2")

    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.2f} seconds.")

    # 5. Validation Metrics on Peak Heat Hour (14:00)
    print("\n[5/5] Computing Final Evaluation Metrics on Peak Heat Hour (14:00)...")
    model.eval()
    peak_sc = scenarios[2]  # Hour 14:00
    with torch.no_grad():
        pred_lst = model(
            graph.node_features,
            graph.edge_index,
            graph.edge_attr,
            peak_sc["weather"]
        )

    mae = torch.mean(torch.abs(pred_lst - peak_sc["target_lst"])).item()
    rmse = torch.sqrt(torch.mean((pred_lst - peak_sc["target_lst"]) ** 2)).item()
    ss_tot = torch.sum((peak_sc["target_lst"] - peak_sc["target_lst"].mean()) ** 2).item()
    ss_res = torch.sum((peak_sc["target_lst"] - pred_lst) ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    print(f"  -> LST MAE:  {mae:.3f} deg C")
    print(f"  -> LST RMSE: {rmse:.3f} deg C")
    print(f"  -> R2 Score: {r2:.4f}")

    # Save Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "hidden_dim": 64,
        "num_layers": 3,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "history": history
    }
    torch.save(checkpoint, save_path)
    print(f"  Saved model checkpoint to {save_path}")

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, graph, peak_sc["weather"]


if __name__ == "__main__":
    train_urbansim()
