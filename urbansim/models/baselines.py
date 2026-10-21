"""
Baseline benchmark models for Urban Heat Island prediction:
1. StandardMLP: Feedforward neural network ignoring graph topology and physics.
2. NonPhysicsGAT: Graph Attention Network trained without physical loss constraints.
3. EmpiricalTEB: Simplified diagnostic Urban Canopy Model (Town Energy Balance).
"""

import math
import torch
import torch.nn as nn
from urbansim.physics.thermodynamics import solve_surface_energy_balance_residual, kelvin_to_celsius, celsius_to_kelvin


class StandardMLP(nn.Module):
    """Feedforward Multi-Layer Perceptron operating solely on isolated node features."""

    def __init__(self, in_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_features: torch.Tensor, edge_index, edge_attr, weather_dict: dict):
        num_nodes = node_features.size(0)
        device = node_features.device
        air_t = weather_dict["air_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        dew_t = weather_dict["dew_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        solar = weather_dict["solar_irradiance"].to(device).view(1, 1).expand(num_nodes, 1) / 1000.0
        wind = weather_dict["wind_speed"].to(device).view(1, 1).expand(num_nodes, 1)
        rh = weather_dict["relative_humidity"].to(device).view(1, 1).expand(num_nodes, 1) / 100.0

        x = torch.cat([node_features, air_t, dew_t, solar, wind, rh], dim=-1)
        return self.net(x) + air_t


class NonPhysicsGAT(nn.Module):
    """Standard Graph Attention Network trained solely with empirical MSE (no physics loss)."""

    def __init__(self, in_dim: int = 14, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Linear(in_dim, hidden_dim)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.w = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_attr, weather_dict: dict):
        num_nodes = node_features.size(0)
        device = node_features.device
        air_t = weather_dict["air_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        dew_t = weather_dict["dew_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        solar = weather_dict["solar_irradiance"].to(device).view(1, 1).expand(num_nodes, 1) / 1000.0
        wind = weather_dict["wind_speed"].to(device).view(1, 1).expand(num_nodes, 1)
        rh = weather_dict["relative_humidity"].to(device).view(1, 1).expand(num_nodes, 1) / 100.0

        x = torch.cat([node_features, air_t, dew_t, solar, wind, rh], dim=-1)
        h = self.encoder(x)

        # 1-layer standard GAT message passing
        src, dst = edge_index[0], edge_index[1]
        h_src, h_dst = h[src], h[dst]
        a_input = torch.cat([h_src, h_dst], dim=-1)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))

        # Softmax over incoming edges
        alpha = torch.sigmoid(e).unsqueeze(-1)
        wh_src = self.w(h_src)
        weighted_msgs = alpha * wh_src

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, weighted_msgs)

        out = self.head(h + agg) + air_t
        return out


class EmpiricalTEB:
    """Diagnostic empirical urban canopy approximation based on localized energy balance."""

    def predict(self, graph, weather_dict: dict) -> torch.Tensor:
        num_nodes = graph.num_nodes
        preds = []
        air_c = weather_dict["air_temp_c"].item()
        solar = weather_dict["solar_irradiance"].item()

        for i in range(num_nodes):
            n = graph.nodes[i]
            # Simple diagnostic equation: delta_T = (1 - albedo)*Solar / (h_c + 4*eps*sigma*T^3)
            solar_absorbed = (1.0 - n["albedo"]) * solar * n["sky_view_factor"]
            cooling_veg = 5.0 * n["vegetation_fraction"]
            h_c = 15.0 + 3.0 * weather_dict["wind_speed"].item()
            delta_t = (solar_absorbed / h_c) - cooling_veg + (n["traffic_density"] * 1.5)
            preds.append(air_c + max(0.0, delta_t))

        return torch.tensor(preds, dtype=torch.float32).unsqueeze(-1)
