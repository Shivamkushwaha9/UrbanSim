"""
Simulation-based Urban Heat Mitigation Optimization Engine.
Uses the differentiable UrbanSim PI-GNN surrogate to solve for optimal
spatial placement of green roofs, cool pavements, and vegetative buffers.
"""

from typing import Dict, List, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
from urbansim.data.urban_morphology import UrbanMorphologyGraph


class UrbanMitigationOptimizer:
    """
    Solves for the optimal allocation of heat mitigation strategies:
    1. Cool Roofs / Reflective Pavements (Albedo increase: Delta alpha in [0, 0.5])
    2. Green Roofs / Urban Canopies (Vegetation increase: Delta veg in [0, 0.7])
    """

    def __init__(
        self,
        model: nn.Module,
        graph: UrbanMorphologyGraph,
        weather_dict: dict,
        cost_cool_per_m2: float = 12.0,   # USD per m^2 for reflective coating
        cost_green_per_m2: float = 38.0,  # USD per m^2 for extensive green roof
        device: str = "cpu"
    ):
        self.model = model
        self.graph = graph
        self.weather_dict = weather_dict
        self.cost_cool = cost_cool_per_m2
        self.cost_green = cost_green_per_m2
        self.device = device
        self.model.eval()

    def evaluate_scenario(
        self,
        action_cool: torch.Tensor,
        action_green: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Runs the differentiable surrogate with modified albedo and vegetation fractions.
        action_cool: [N] float in [0, 0.5]
        action_green: [N] float in [0, 0.7]
        Returns:
            predicted_lst: [N, 1]
            total_cost: float
        """
        node_feats = self.graph.node_features.clone()

        # Update albedo (column 3) and vegetation fraction (column 5)
        node_feats[:, 3] = torch.clamp(node_feats[:, 3] + action_cool, 0.1, 0.75)
        node_feats[:, 5] = torch.clamp(node_feats[:, 5] + action_green, 0.0, 0.85)

        with torch.no_grad():
            pred_lst = self.model(
                node_feats,
                self.graph.edge_index,
                self.graph.edge_attr,
                self.weather_dict
            )

        # Calculate budget cost
        areas = self.graph.node_features[:, 2] * 2500.0
        cost = torch.sum(action_cool * self.cost_cool * areas + action_green * self.cost_green * areas).item()
        return pred_lst, cost

    def optimize_differentiable_budget(
        self,
        budget_usd: float = 1_500_000.0,
        num_steps: int = 120,
        lr: float = 0.015,
        target_zones: List[str] = None
    ) -> Dict:
        """
        Uses Projected Gradient Descent through the differentiable neural surrogate
        to minimize peak LST under a strict budget constraint.
        """
        num_nodes = self.graph.num_nodes
        device = self.device

        # Zone mask (e.g. only prioritize Dharavi or BKC if specified)
        mask = torch.ones(num_nodes, device=device)
        if target_zones is not None:
            for i, n in enumerate(self.graph.nodes):
                if n["zone"] not in target_zones:
                    mask[i] = 0.0

        # Unconstrained optimization parameters (logits)
        raw_cool = nn.Parameter(torch.zeros(num_nodes, device=device, requires_grad=True))
        raw_green = nn.Parameter(torch.zeros(num_nodes, device=device, requires_grad=True))

        optimizer = torch.optim.Adam([raw_cool, raw_green], lr=lr)
        node_feats_base = self.graph.node_features.clone().to(device)
        areas = node_feats_base[:, 2] * 2500.0

        best_loss = float("inf")
        best_cool = None
        best_green = None

        for step in range(num_steps):
            optimizer.zero_grad()

            # Bound actions via sigmoid scaling
            act_cool = torch.sigmoid(raw_cool) * 0.45 * mask
            act_green = torch.sigmoid(raw_green) * 0.65 * mask

            raw_cost = torch.sum(act_cool * self.cost_cool * areas + act_green * self.cost_green * areas)
            if raw_cost > budget_usd:
                scale = budget_usd / (raw_cost + 1e-6)
                act_cool = act_cool * scale
                act_green = act_green * scale

            total_cost = torch.sum(act_cool * self.cost_cool * areas + act_green * self.cost_green * areas)

            # Modified node features
            node_feats_mod = node_feats_base.clone()
            node_feats_mod[:, 3] = torch.clamp(node_feats_mod[:, 3] + act_cool, 0.1, 0.75)
            node_feats_mod[:, 5] = torch.clamp(node_feats_mod[:, 5] + act_green, 0.0, 0.85)

            # Forward pass through surrogate
            pred_lst = self.model(
                node_feats_mod,
                self.graph.edge_index.to(device),
                self.graph.edge_attr.to(device),
                self.weather_dict
            )

            # Objective: minimize mean temperature + peak hotspot penalty
            mean_temp = torch.mean(pred_lst)
            max_temp = torch.max(pred_lst)
            loss = mean_temp + 0.5 * max_temp

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss and total_cost.item() <= (budget_usd * 1.01):
                best_loss = loss.item()
                best_cool = act_cool.detach().clone()
                best_green = act_green.detach().clone()

        if best_cool is None:
            best_cool = torch.sigmoid(raw_cool).detach() * 0.45 * mask
            best_green = torch.sigmoid(raw_green).detach() * 0.65 * mask
            raw_c = torch.sum(best_cool * self.cost_cool * areas + best_green * self.cost_green * areas)
            if raw_c > budget_usd:
                scale = budget_usd / (raw_c + 1e-6)
                best_cool = best_cool * scale
                best_green = best_green * scale

        # Final evaluation
        final_lst, final_cost = self.evaluate_scenario(best_cool, best_green)

        # Baseline evaluation
        zero_act = torch.zeros(num_nodes, device=device)
        baseline_lst, _ = self.evaluate_scenario(zero_act, zero_act)

        delta_t = baseline_lst - final_lst

        return {
            "best_cool_allocation": best_cool.cpu().numpy(),
            "best_green_allocation": best_green.cpu().numpy(),
            "baseline_mean_lst": baseline_lst.mean().item(),
            "optimized_mean_lst": final_lst.mean().item(),
            "mean_cooling_c": delta_t.mean().item(),
            "max_cooling_c": delta_t.max().item(),
            "final_cost_usd": final_cost,
            "budget_usd": budget_usd,
            "baseline_lst": baseline_lst.squeeze(-1).cpu().numpy(),
            "optimized_lst": final_lst.squeeze(-1).cpu().numpy(),
            "delta_t": delta_t.squeeze(-1).cpu().numpy()
        }

    def generate_pareto_frontier(
        self,
        budget_range: List[float] = None
    ) -> List[Dict]:
        """
        Runs optimization across an array of budgets to generate the complete Pareto Trade-Off Curve.
        """
        if budget_range is None:
            budget_range = [250_000.0, 500_000.0, 1_000_000.0, 2_000_000.0, 3_500_000.0]

        frontier = []
        for b in budget_range:
            res = self.optimize_differentiable_budget(budget_usd=b, num_steps=80)
            frontier.append({
                "budget_usd": b,
                "actual_cost_usd": res["final_cost_usd"],
                "mean_cooling_c": res["mean_cooling_c"],
                "max_cooling_c": res["max_cooling_c"],
                "optimized_mean_lst": res["optimized_mean_lst"]
            })

        return frontier
