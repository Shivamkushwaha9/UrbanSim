"""
Flask backend API server for UrbanSim interactive microclimate digital twin.
Provides real-time inference, scenario simulation, and differentiable optimization endpoints.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, render_template
import torch
import numpy as np

from urbansim.data.mumbai_generator import (
    create_mumbai_urban_graph,
    generate_mumbai_weather_scenario
)
from urbansim.models.pignn import UrbanSimPIGNN
from urbansim.optimization.mitigation_engine import UrbanMitigationOptimizer
from urbansim.physics.thermodynamics import solve_surface_energy_balance_residual, celsius_to_kelvin

app = Flask(__name__, template_folder="templates")

# Initialize shared simulation state
DEVICE = "cpu"
GRAPH = create_mumbai_urban_graph(grid_size=20, seed=42)
GRAPH.build_tensors(device=DEVICE)

MODEL = UrbanSimPIGNN(hidden_dim=64, num_layers=3).to(DEVICE)
checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "urbansim_checkpoint.pt")
if os.path.exists(checkpoint_path):
    try:
        chk = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(chk["model_state_dict"])
        print("[UrbanSim Server] Loaded trained model checkpoint.")
    except Exception as e:
        print(f"[UrbanSim Server] Failed to load checkpoint: {e}")
MODEL.eval()

CURRENT_WEATHER = generate_mumbai_weather_scenario(hour=14.0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/research-paper")
def view_paper():
    paper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "research_paper.md")
    content = "Research paper not found."
    if os.path.exists(paper_path):
        with open(paper_path, "r", encoding="utf-8") as f:
            content = f.read()
    # Simple clean HTML reader
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>UrbanSim Research Paper</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown.min.css">
    <style>
        body {{ background: #0b0f19; color: #f1f5f9; padding: 2rem; display: flex; justify-content: center; }}
        .markdown-body {{ background: #0f172a !important; color: #f1f5f9 !important; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 3rem; max-width: 900px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
        .markdown-body a {{ color: #00f2fe; }}
        .markdown-body h1, .markdown-body h2, .markdown-body h3 {{ color: #00f2fe; border-bottom-color: rgba(255,255,255,0.1); }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <article class="markdown-body" id="content"></article>
    <script>
        const raw = {repr(content)};
        document.getElementById('content').innerHTML = marked.parse(raw);
    </script>
</body>
</html>"""
    return html


@app.route("/api/state", methods=["GET"])
def get_state():
    """Returns static urban morphology and current microclimate state."""
    with torch.no_grad():
        pred_lst = MODEL(
            GRAPH.node_features,
            GRAPH.edge_index,
            GRAPH.edge_attr,
            CURRENT_WEATHER
        ).squeeze(-1).numpy().tolist()

    nodes_data = []
    for i, n in enumerate(GRAPH.nodes):
        nodes_data.append({
            "id": n["id"],
            "x": n["x"],
            "y": n["y"],
            "zone": n["zone"],
            "height": round(n["height"], 1),
            "albedo": round(n["albedo"], 3),
            "veg_fraction": round(n["vegetation_fraction"], 3),
            "sky_view_factor": round(n["sky_view_factor"], 3),
            "traffic": round(n["traffic_density"], 2),
            "temperature_c": round(pred_lst[i], 2)
        })

    weather_data = {
        "hour": CURRENT_WEATHER["hour"],
        "air_temp_c": round(CURRENT_WEATHER["air_temp_c"].item(), 1),
        "dew_temp_c": round(CURRENT_WEATHER["dew_temp_c"].item(), 1),
        "solar_irradiance": round(CURRENT_WEATHER["solar_irradiance"].item(), 1),
        "wind_speed": round(CURRENT_WEATHER["wind_speed"].item(), 1),
        "wind_direction_deg": round(CURRENT_WEATHER["wind_direction_deg"].item(), 1),
        "relative_humidity": round(CURRENT_WEATHER["relative_humidity"].item(), 1)
    }

    mean_lst = float(np.mean(pred_lst))
    max_lst = float(np.max(pred_lst))

    return jsonify({
        "num_nodes": GRAPH.num_nodes,
        "nodes": nodes_data,
        "weather": weather_data,
        "stats": {
            "mean_lst_c": round(mean_lst, 2),
            "max_lst_c": round(max_lst, 2),
            "hotspot_count": int(np.sum(np.array(pred_lst) > 42.0))
        }
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Re-runs fast surrogate inference under modified meteorological conditions."""
    data = request.json or {}
    hour = float(data.get("hour", 14.0))
    air_temp_c = float(data.get("air_temp_c", 36.5))
    solar_in = float(data.get("solar_irradiance", 920.0))
    wind_u = float(data.get("wind_speed", 3.2))
    rh = float(data.get("relative_humidity", 65.0))
    dew_temp_c = air_temp_c - ((100.0 - rh) / 5.0)

    custom_weather = {
        "hour": hour,
        "air_temp_c": torch.tensor(air_temp_c, dtype=torch.float32),
        "dew_temp_c": torch.tensor(dew_temp_c, dtype=torch.float32),
        "solar_irradiance": torch.tensor(solar_in, dtype=torch.float32),
        "wind_speed": torch.tensor(wind_u, dtype=torch.float32),
        "wind_direction_deg": torch.tensor(245.0, dtype=torch.float32),
        "relative_humidity": torch.tensor(rh, dtype=torch.float32)
    }

    with torch.no_grad():
        pred_lst = MODEL(
            GRAPH.node_features,
            GRAPH.edge_index,
            GRAPH.edge_attr,
            custom_weather
        ).squeeze(-1).numpy()

    # Calculate physical SEB residual for verification
    pred_k = celsius_to_kelvin(torch.tensor(pred_lst))
    residuals, _ = solve_surface_energy_balance_residual(
        surface_temp_k=pred_k,
        air_temp_k=celsius_to_kelvin(custom_weather["air_temp_c"]),
        dew_point_k=celsius_to_kelvin(custom_weather["dew_temp_c"]),
        solar_irradiance=custom_weather["solar_irradiance"],
        albedo=GRAPH.node_features[:, 3],
        emissivity=GRAPH.node_features[:, 4],
        sky_view_factor=GRAPH.node_features[:, 7],
        wind_speed=custom_weather["wind_speed"],
        relative_humidity=custom_weather["relative_humidity"],
        vegetation_fraction=GRAPH.node_features[:, 5],
        thermal_conductivity=GRAPH.node_features[:, 6],
        traffic_density=GRAPH.node_features[:, 8]
    )

    mean_res = float(torch.mean(torch.abs(residuals)).item())

    return jsonify({
        "temperatures_c": [round(float(t), 2) for t in pred_lst],
        "stats": {
            "mean_lst_c": round(float(np.mean(pred_lst)), 2),
            "max_lst_c": round(float(np.max(pred_lst)), 2),
            "min_lst_c": round(float(np.min(pred_lst)), 2),
            "mean_energy_residual_w_m2": round(mean_res, 2)
        }
    })


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """Runs differentiable heat mitigation optimization."""
    data = request.json or {}
    budget = float(data.get("budget_usd", 1_500_000.0))
    target_zones = data.get("target_zones", None)

    optimizer = UrbanMitigationOptimizer(
        model=MODEL,
        graph=GRAPH,
        weather_dict=CURRENT_WEATHER,
        device=DEVICE
    )

    res = optimizer.optimize_differentiable_budget(
        budget_usd=budget,
        num_steps=100,
        target_zones=target_zones
    )

    return jsonify({
        "baseline_mean_lst": round(res["baseline_mean_lst"], 2),
        "optimized_mean_lst": round(res["optimized_mean_lst"], 2),
        "mean_cooling_c": round(res["mean_cooling_c"], 2),
        "max_cooling_c": round(res["max_cooling_c"], 2),
        "actual_cost_usd": round(res["final_cost_usd"], 2),
        "budget_usd": budget,
        "baseline_lst": [round(float(t), 2) for t in res["baseline_lst"]],
        "optimized_lst": [round(float(t), 2) for t in res["optimized_lst"]],
        "delta_t": [round(float(d), 2) for d in res["delta_t"]],
        "cool_allocation": [round(float(c), 3) for c in res["best_cool_allocation"]],
        "green_allocation": [round(float(g), 3) for g in res["best_green_allocation"]]
    })


def run_server(port=5000):
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    run_server()
