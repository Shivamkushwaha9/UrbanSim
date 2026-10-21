"""
Physics-informed loss functions for the Urban Heat Island Graph Neural Network.
Enforces conservation of energy via the Surface Energy Balance (SEB),
turbulent heat transfer constraints, and radiative view factor equilibrium.
"""

import torch
import torch.nn as nn
from urbansim.physics.thermodynamics import (
    solve_surface_energy_balance_residual,
    celsius_to_kelvin
)


class PhysicsInformedLoss(nn.Module):
    """
    Combines empirical observation loss with physical conservation constraints.
    L_total = L_data + lambda_phys * L_physics + lambda_smooth * L_smoothness
    """

    def __init__(
        self,
        lambda_phys: float = 0.35,
        lambda_smooth: float = 0.05,
        lambda_cons: float = 0.02,
        residual_norm_scale: float = 100.0
    ):
        super().__init__()
        self.lambda_phys = lambda_phys
        self.lambda_smooth = lambda_smooth
        self.lambda_cons = lambda_cons
        self.residual_norm_scale = residual_norm_scale
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        pred_lst_c: torch.Tensor,
        target_lst_c: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        weather_dict: dict,
        mask: torch.Tensor = None
    ) -> dict:
        """
        Args:
            pred_lst_c: Predicted surface temperatures in Celsius [N, 1]
            target_lst_c: Ground truth satellite/observed LST in Celsius [N, 1]
            node_features: Feature tensor containing [elev, height, area, albedo, eps, veg_frac, cond, svf, traffic, ...]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge attributes [E, d_e] (distance, view factor, wind alignment, etc.)
            weather_dict: Ambient weather forcings (air_temp_c, dew_temp_c, solar_irradiance, wind_speed, rh)
            mask: Optional observation mask for semi-supervised training [N]
        """
        # 1. Supervised Data Fidelity Loss
        if mask is not None:
            l_data = self.mse_loss(pred_lst_c[mask], target_lst_c[mask])
        else:
            l_data = self.mse_loss(pred_lst_c, target_lst_c)

        # 2. Thermodynamic Surface Energy Balance Residual Loss
        device = pred_lst_c.device
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        pred_lst_k = celsius_to_kelvin(pred_lst_c.squeeze(-1))
        air_temp_k = celsius_to_kelvin(weather_dict["air_temp_c"].to(device))
        dew_temp_k = celsius_to_kelvin(weather_dict["dew_temp_c"].to(device))
        solar_in = weather_dict["solar_irradiance"].to(device)
        wind_speed = weather_dict["wind_speed"].to(device)
        rh = weather_dict["relative_humidity"].to(device)

        # Extract node physical properties (assumes standard feature order)
        albedo = node_features[:, 3]
        emissivity = node_features[:, 4]
        veg_fraction = node_features[:, 5]
        thermal_cond = node_features[:, 6]
        svf = node_features[:, 7]
        traffic = node_features[:, 8]

        # Compute thermodynamic SEB residual across all nodes
        residuals, flux_dict = solve_surface_energy_balance_residual(
            surface_temp_k=pred_lst_k,
            air_temp_k=air_temp_k,
            dew_point_k=dew_temp_k,
            solar_irradiance=solar_in,
            albedo=albedo,
            emissivity=emissivity,
            sky_view_factor=svf,
            wind_speed=wind_speed,
            relative_humidity=rh,
            vegetation_fraction=veg_fraction,
            thermal_conductivity=thermal_cond,
            traffic_density=traffic
        )

        # Normalized SEB residual loss: ||R_SEB / norm||^2
        scaled_residual = residuals / self.residual_norm_scale
        l_physics = torch.mean(scaled_residual ** 2)

        # 3. Spatial Smoothness / Canyon Advection Graph Regularization
        # Heat should diffuse smoothly along connected canyon corridors
        src, dst = edge_index[0], edge_index[1]
        temp_diff = pred_lst_c[src] - pred_lst_c[dst]
        # Weight by canyon proximity (edge_attr[:, 0] is normalized distance)
        dist_weight = torch.exp(-edge_attr[:, 0])
        l_smooth = torch.mean((temp_diff ** 2) * dist_weight.unsqueeze(-1))

        # 4. View Factor Geometric Closure Constraint (Sum of SVF and inter-node view factors <= 1)
        # edge_attr[:, 1] is the view factor F_ij
        view_factor_sum = torch.zeros_like(svf)
        view_factor_sum.index_add_(0, src, edge_attr[:, 1])
        l_cons = torch.mean(torch.clamp(svf + view_factor_sum - 1.05, min=0.0) ** 2)

        # Total multi-task loss
        l_total = (
            l_data
            + (self.lambda_phys * l_physics)
            + (self.lambda_smooth * l_smooth)
            + (self.lambda_cons * l_cons)
        )

        return {
            "loss_total": l_total,
            "loss_data": l_data,
            "loss_physics": l_physics,
            "loss_smooth": l_smooth,
            "loss_cons": l_cons,
            "mean_seb_residual_w_m2": torch.mean(torch.abs(residuals)),
            "flux_dict": flux_dict
        }
