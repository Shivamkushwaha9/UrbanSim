"""
Physics-Informed Graph Neural Network (PI-GNN) for Urban Thermodynamics.
Integrates spatial graph message passing with thermodynamic attention,
advection vector alignment, and radiative view factor conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalMessagePassingLayer(nn.Module):
    """
    Message passing layer that aggregates thermodynamic fluxes from neighboring
    canyons, modulated by radiative view factors and wind advection vectors.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message transformation
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Attention projection
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)

        # Physics modulation scale parameters
        self.gamma_rad = nn.Parameter(torch.tensor(0.5))
        self.gamma_wind = nn.Parameter(torch.tensor(0.5))

        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        h: [N, hidden_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        src, dst = edge_index[0], edge_index[1]

        # Construct raw messages along edges
        h_src = h[src]
        h_dst = h[dst]
        msg_inputs = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_inputs)  # [E, hidden_dim]

        # Physics-modulated attention scores
        q = self.q_proj(h_dst)
        k = self.k_proj(h_src)
        dot_score = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)

        # edge_attr[:, 1] is view factor F_ij, edge_attr[:, 2] is wind cosine alignment
        phys_bias = (
            self.gamma_rad * edge_attr[:, 1]
            + self.gamma_wind * torch.clamp(edge_attr[:, 2], min=0.0)
        )
        attention_weights = torch.sigmoid(dot_score + phys_bias).unsqueeze(-1)

        weighted_messages = messages * attention_weights

        # Aggregate incoming messages onto destination nodes
        num_nodes = h.size(0)
        aggregated = torch.zeros((num_nodes, self.hidden_dim), device=h.device, dtype=h.dtype)
        aggregated.index_add_(0, dst, weighted_messages)

        # Update node states with residual connection
        combined = torch.cat([h, aggregated], dim=-1)
        h_next = self.norm(h + self.update_mlp(combined))
        return h_next


class UrbanSimPIGNN(nn.Module):
    """
    Complete Physics-Informed Graph Neural Network surrogate model.
    Predicts hyper-local Land Surface Temperature (LST) and sensible heat flux
    from building morphology, material properties, and ambient meteorology.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 4,
        weather_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.weather_dim = weather_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim + weather_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, edge_in_dim)
        )

        # Message Passing Layers
        self.mp_layers = nn.ModuleList([
            PhysicalMessagePassingLayer(hidden_dim, edge_in_dim)
            for _ in range(num_layers)
        ])

        # Dual Readout Heads
        # Head 1: Land Surface Temperature (LST in Celsius)
        self.lst_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        nn.init.constant_(self.lst_head[-1].bias, 15.0)

        # Head 2: Turbulent Sensible Heat Flux (H in W/m^2)
        self.flux_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        weather_dict: dict
    ) -> torch.Tensor:
        """
        Forward pass predicting LST (°C).
        """
        device = next(self.parameters()).device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        num_nodes = node_features.size(0)

        # Broadcast ambient weather conditions across all nodes on the same device
        air_t = weather_dict["air_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        dew_t = weather_dict["dew_temp_c"].to(device).view(1, 1).expand(num_nodes, 1)
        solar = (weather_dict["solar_irradiance"].to(device).view(1, 1).expand(num_nodes, 1)) / 1000.0  # Scale kW
        wind = weather_dict["wind_speed"].to(device).view(1, 1).expand(num_nodes, 1)
        rh = (weather_dict["relative_humidity"].to(device).view(1, 1).expand(num_nodes, 1)) / 100.0

        weather_tensor = torch.cat([air_t, dew_t, solar, wind, rh], dim=-1)

        # Encode initial node representations
        combined_node_input = torch.cat([node_features, weather_tensor], dim=-1)
        h = self.node_encoder(combined_node_input)

        # Encode edge features
        encoded_edge_attr = self.edge_encoder(edge_attr)

        # Pass through thermodynamic message passing layers
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, encoded_edge_attr)

        # Predict surface temperature
        pred_lst_c = self.lst_head(h)

        # Add air temperature residual prior (ensures physical baseline scale)
        pred_lst_c = pred_lst_c + air_t

        return pred_lst_c
