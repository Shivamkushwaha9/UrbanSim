"""
Urban morphology and geospatial topology representations.
Computes geometric descriptors: Sky View Factor (SVF), canyon aspect ratio (H/W),
and inter-building radiative view factors.
"""

import math
import numpy as np
import torch


def compute_geometric_view_factor(dist, height_src, height_dst, canyon_width):
    """
    Computes approximate radiative view factor between two opposing canyon facades / parcels.
    F_ij = 0.5 * (1.0 - (dist / sqrt(dist^2 + canyon_width^2)))
    """
    effective_dist = max(dist, 5.0)
    denom = math.sqrt(effective_dist ** 2 + canyon_width ** 2)
    f_ij = 0.5 * (1.0 - (effective_dist / denom))
    return max(0.01, min(0.45, f_ij))


def estimate_sky_view_factor(height, canyon_width, is_roof=False):
    """
    Estimates Sky View Factor (SVF) for an urban canyon parcel or building roof.
    For roofs: SVF is typically near 0.85 - 0.98.
    For narrow street canyons: SVF = cos(arctan(2 * H / W)).
    """
    if is_roof:
        return np.random.uniform(0.85, 0.98)
    
    aspect_ratio = (2.0 * max(1.0, height)) / max(4.0, canyon_width)
    theta = math.atan(aspect_ratio)
    svf = math.cos(theta)
    return max(0.12, min(0.95, svf))


class UrbanMorphologyGraph:
    """
    Represents an urban sector as an attributed spatial graph G = (V, E).
    """

    def __init__(self, num_nodes: int, name: str = "UrbanDistrict"):
        self.num_nodes = num_nodes
        self.name = name
        self.nodes = []
        self.edges = []
        self.edge_index = None
        self.edge_attr = None
        self.node_features = None
        self.coordinates = None

    def build_tensors(self, device="cpu"):
        """Converts graph attributes to PyTorch tensors."""
        # Node features: [elev, height, area, albedo, eps, veg_frac, cond, svf, traffic]
        feat_list = []
        coords_list = []
        for n in self.nodes:
            feat_list.append([
                n["elevation"] / 20.0,
                n["height"] / 100.0,
                n["area"] / 2500.0,
                n["albedo"],
                n["emissivity"],
                n["vegetation_fraction"],
                n["thermal_conductivity"],
                n["sky_view_factor"],
                n["traffic_density"],
            ])
            coords_list.append([n["x"], n["y"]])

        self.node_features = torch.tensor(feat_list, dtype=torch.float32, device=device)
        self.coordinates = torch.tensor(coords_list, dtype=torch.float32, device=device)

        # Edges
        if len(self.edges) > 0:
            src = [e["src"] for e in self.edges]
            dst = [e["dst"] for e in self.edges]
            self.edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)

            # Edge attr: [norm_dist, view_factor, wind_align, canyon_hw]
            edge_attrs = []
            for e in self.edges:
                edge_attrs.append([
                    e["dist"] / 100.0,
                    e["view_factor"],
                    e["wind_align"],
                    e["aspect_ratio"]
                ])
            self.edge_attr = torch.tensor(edge_attrs, dtype=torch.float32, device=device)
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            self.edge_attr = torch.empty((0, 4), dtype=torch.float32, device=device)

        return self
