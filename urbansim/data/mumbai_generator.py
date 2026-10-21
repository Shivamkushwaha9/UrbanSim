"""
High-fidelity urban generator for Mumbai megacity case study.
Models diverse urban zones: Bandra-Kurla Complex (commercial), Dharavi (informal high-density),
Kurla Transit Corridor, and Mithi River vegetative buffer.
Calibrated with tropical coastal meteorological forcings.
"""

import math
import numpy as np
import torch
from urbansim.data.urban_morphology import (
    UrbanMorphologyGraph,
    estimate_sky_view_factor,
    compute_geometric_view_factor
)
from urbansim.physics.thermodynamics import solve_surface_energy_balance_residual, celsius_to_kelvin, kelvin_to_celsius


def generate_mumbai_weather_scenario(hour: float = 14.0) -> dict:
    """
    Returns realistic meteorological forcing for Mumbai pre-monsoon summer heatwave.
    Peak solar irradiance occurs around 13:00 - 14:00.
    """
    # Diurnal solar cycle
    solar_zenith = math.sin(math.pi * max(0.0, min(1.0, (hour - 6.0) / 12.0)))
    solar_in = max(0.0, 960.0 * (solar_zenith ** 1.2))

    # Air temperature: base 29 C at dawn, peaking at 37.5 C at 14:30
    temp_diurnal = math.sin(math.pi * max(0.0, min(1.0, (hour - 8.0) / 14.0)))
    air_temp_c = 29.5 + 8.0 * max(0.0, temp_diurnal)

    # Dew point and Relative Humidity (coastal maritime humidity)
    rh = 76.0 - 22.0 * max(0.0, temp_diurnal)  # 54% to 76%
    # Approx dew point
    dew_temp_c = air_temp_c - ((100.0 - rh) / 5.0)

    # Sea breeze kicks in during afternoon from WSW (245 degrees)
    wind_speed = 2.2 + 2.5 * max(0.0, temp_diurnal)
    wind_direction_deg = 245.0

    return {
        "hour": hour,
        "air_temp_c": torch.tensor(air_temp_c, dtype=torch.float32),
        "dew_temp_c": torch.tensor(dew_temp_c, dtype=torch.float32),
        "solar_irradiance": torch.tensor(solar_in, dtype=torch.float32),
        "wind_speed": torch.tensor(wind_speed, dtype=torch.float32),
        "wind_direction_deg": torch.tensor(wind_direction_deg, dtype=torch.float32),
        "relative_humidity": torch.tensor(rh, dtype=torch.float32)
    }


def create_mumbai_urban_graph(
    grid_size: int = 20,
    seed: int = 42,
    canyon_radius: float = 45.0
) -> UrbanMorphologyGraph:
    """
    Creates an attributed graph representing a 1 km x 1 km sector of Mumbai
    divided into 4 distinct microclimatic zones.
    """
    np.random.seed(seed)
    num_nodes = grid_size * grid_size
    graph = UrbanMorphologyGraph(num_nodes=num_nodes, name="Mumbai_BKC_Dharavi_District")

    step = 1000.0 / grid_size  # 50 meters per node parcel

    # Zone definitions across the grid
    # Quadrant 1 (Top-Left): BKC Commercial Core (tall buildings, AC heat, moderate albedo)
    # Quadrant 2 (Top-Right): Kurla Highway (high traffic, asphalt, low albedo)
    # Quadrant 3 (Bottom-Left): Dharavi Informal (dense low-rise, tin roofs, low veg, low inertia)
    # Quadrant 4 (Bottom-Right): Mithi River / Mangrove Buffer (high veg, water presence)

    node_id = 0
    for r in range(grid_size):
        for c in range(grid_size):
            x = c * step + step / 2.0
            y = r * step + step / 2.0

            # Determine Zone
            is_top = r < (grid_size // 2)
            is_left = c < (grid_size // 2)

            if is_top and is_left:
                zone = "BKC_Commercial"
                height = np.random.uniform(35.0, 85.0)
                albedo = np.random.uniform(0.20, 0.32)
                emissivity = 0.90
                veg_fraction = np.random.uniform(0.05, 0.18)
                conductivity = 1.6  # Concrete / reinforced steel
                traffic = np.random.uniform(0.3, 0.6)
                is_roof = True
                canyon_w = 28.0
            elif is_top and not is_left:
                zone = "Kurla_Highway"
                height = np.random.uniform(8.0, 22.0)
                albedo = np.random.uniform(0.12, 0.18)  # Dark asphalt / aged tar
                emissivity = 0.94
                veg_fraction = np.random.uniform(0.02, 0.08)
                conductivity = 1.3
                traffic = np.random.uniform(0.7, 1.0)  # Heavy traffic
                is_roof = False
                canyon_w = 40.0
            elif not is_top and is_left:
                zone = "Dharavi_Informal"
                height = np.random.uniform(4.0, 10.0)  # Dense 1-2 story structures
                albedo = np.random.uniform(0.18, 0.25)  # Corrugated metal / weathered asbestos
                emissivity = 0.88
                veg_fraction = np.random.uniform(0.01, 0.04)  # Negligible vegetation
                conductivity = 2.2  # Thin metal / brick (heats rapidly)
                traffic = np.random.uniform(0.2, 0.4)
                is_roof = True
                canyon_w = 6.0  # Ultra-narrow alleys (H/W > 2.5)
            else:
                zone = "Mithi_Buffer"
                height = np.random.uniform(1.0, 6.0)
                albedo = np.random.uniform(0.15, 0.22)
                emissivity = 0.96
                veg_fraction = np.random.uniform(0.55, 0.88)  # Dense green mangroves
                conductivity = 0.7  # Wet organic soil
                traffic = np.random.uniform(0.0, 0.1)
                is_roof = False
                canyon_w = 50.0

            svf = estimate_sky_view_factor(height, canyon_w, is_roof=is_roof)

            node_data = {
                "id": node_id,
                "x": x,
                "y": y,
                "zone": zone,
                "elevation": np.random.uniform(4.0, 12.0),
                "height": height,
                "area": step * step,
                "albedo": albedo,
                "emissivity": emissivity,
                "vegetation_fraction": veg_fraction,
                "thermal_conductivity": conductivity,
                "sky_view_factor": svf,
                "traffic_density": traffic,
                "canyon_width": canyon_w
            }
            graph.nodes.append(node_data)
            node_id += 1

    # Connect nodes within spatial canyon radius
    wind_rad = math.radians(245.0)  # WSW wind vector
    wind_vec = np.array([math.cos(wind_rad), math.sin(wind_rad)])

    for i in range(num_nodes):
        pos_i = np.array([graph.nodes[i]["x"], graph.nodes[i]["y"]])
        for j in range(num_nodes):
            if i == j:
                continue
            pos_j = np.array([graph.nodes[j]["x"], graph.nodes[j]["y"]])
            dist = np.linalg.norm(pos_i - pos_j)

            # Spatial adjacency cutoff (e.g. 75 meters)
            if dist <= 75.0:
                dir_vec = (pos_j - pos_i) / max(dist, 1e-4)
                # Cosine alignment with wind direction (advection influence)
                wind_align = float(np.dot(dir_vec, wind_vec))

                avg_height = 0.5 * (graph.nodes[i]["height"] + graph.nodes[j]["height"])
                canyon_w = 0.5 * (graph.nodes[i]["canyon_width"] + graph.nodes[j]["canyon_width"])
                vf = compute_geometric_view_factor(dist, graph.nodes[i]["height"], graph.nodes[j]["height"], canyon_w)

                edge_data = {
                    "src": i,
                    "dst": j,
                    "dist": dist,
                    "view_factor": vf,
                    "wind_align": wind_align,
                    "aspect_ratio": avg_height / max(canyon_w, 2.0)
                }
                graph.edges.append(edge_data)

    return graph


def simulate_ground_truth_lst(
    graph: UrbanMorphologyGraph,
    weather: dict,
    noise_std: float = 0.25
) -> torch.Tensor:
    """
    Computes physical equilibrium Land Surface Temperature (LST)
    using numerical thermodynamic root finding, plus realistic satellite sensor noise.
    """
    if graph.node_features is None:
        graph.build_tensors()
    num_nodes = graph.num_nodes
    target_temps_c = np.zeros(num_nodes, dtype=np.float32)

    air_temp_c = weather["air_temp_c"].item()
    dew_temp_c = weather["dew_temp_c"].item()
    solar_in = weather["solar_irradiance"].item()
    wind_u = weather["wind_speed"].item()
    rh = weather["relative_humidity"].item()

    air_k = celsius_to_kelvin(air_temp_c)
    dew_k = celsius_to_kelvin(dew_temp_c)

    for i in range(num_nodes):
        node = graph.nodes[i]
        albedo = node["albedo"]
        emissivity = node["emissivity"]
        svf = node["sky_view_factor"]
        veg = node["vegetation_fraction"]
        cond = node["thermal_conductivity"]
        traffic = node["traffic_density"]
        is_comm = (node["zone"] == "BKC_Commercial")

        # Newton-Raphson / bisection solver for equilibrium temperature
        # where R_SEB(T) = 0
        t_low = air_k - 5.0
        t_high = air_k + 35.0

        for _ in range(25):
            t_mid = 0.5 * (t_low + t_high)
            res, _ = solve_surface_energy_balance_residual(
                surface_temp_k=t_mid,
                air_temp_k=air_k,
                dew_point_k=dew_k,
                solar_irradiance=solar_in,
                albedo=albedo,
                emissivity=emissivity,
                sky_view_factor=svf,
                wind_speed=wind_u,
                relative_humidity=rh,
                vegetation_fraction=veg,
                thermal_conductivity=cond,
                traffic_density=traffic,
                is_commercial=is_comm
            )
            if res > 0:
                # Excess energy -> temperature must rise to radiate/convect more
                t_low = t_mid
            else:
                t_high = t_mid

        t_eq_c = kelvin_to_celsius(0.5 * (t_low + t_high))
        # Add micro-scale spatial turbulence and satellite observation noise
        t_noisy = t_eq_c + np.random.normal(0.0, noise_std)
        target_temps_c[i] = t_noisy

    return torch.tensor(target_temps_c, dtype=torch.float32).unsqueeze(-1)
