"""
Thermodynamic governing equations for microclimate and Urban Heat Island (UHI) simulation.
Implements the full Surface Energy Balance (SEB), Stefan-Boltzmann radiative exchange,
turbulent sensible and latent heat fluxes, substrate conduction, and anthropogenic fluxes.
"""

import math
from typing import Dict, Tuple, Optional
import torch

# Fundamental Physical Constants
STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 * K^4)
AIR_DENSITY = 1.184                 # kg/m^3 at 25 deg C, 1 atm
AIR_SPECIFIC_HEAT = 1005.0          # J/(kg * K)
PSYCHROMETRIC_CONST = 0.066         # kPa / K
VON_KARMAN = 0.40                   # dimensionless


def celsius_to_kelvin(temp_c):
    """Convert Celsius to Kelvin."""
    return temp_c + 273.15


def kelvin_to_celsius(temp_k):
    """Convert Kelvin to Celsius."""
    return temp_k - 273.15


def compute_sky_temperature(air_temp_k, dew_point_k):
    """
    Berdahl & Martin formulation for effective clear sky emissivity and temperature.
    eps_sky = 0.711 + 0.56 * (T_dew_c / 100) + 0.73 * (T_dew_c / 100)^2
    T_sky = T_air * eps_sky^(0.25)
    """
    dew_c = kelvin_to_celsius(dew_point_k)
    dew_c_scaled = dew_c / 100.0
    eps_sky = 0.711 + 0.56 * dew_c_scaled + 0.73 * (dew_c_scaled ** 2)
    # Clamp emissivity to physically plausible bounds
    if isinstance(eps_sky, torch.Tensor):
        eps_sky = torch.clamp(eps_sky, 0.65, 0.98)
        return air_temp_k * (eps_sky ** 0.25)
    else:
        eps_sky = max(0.65, min(0.98, eps_sky))
        return air_temp_k * (eps_sky ** 0.25)


def compute_net_shortwave_radiation(
    albedo,
    solar_irradiance,
    sky_view_factor,
    diffuse_fraction=0.3,
    shadow_mask=1.0
):
    """
    Calculates net absorbed shortwave radiation (W/m^2).
    S_net = (1 - alpha) * [SVF * S_diff + shadow_mask * S_dir]
    """
    s_diff = solar_irradiance * diffuse_fraction
    s_dir = solar_irradiance * (1.0 - diffuse_fraction)
    incident = sky_view_factor * s_diff + shadow_mask * s_dir
    return (1.0 - albedo) * incident


def compute_net_longwave_radiation(
    surface_temp_k,
    air_temp_k,
    sky_temp_k,
    canyon_temp_k,
    emissivity,
    sky_view_factor
):
    """
    Calculates net longwave radiative exchange with sky and surrounding street canyon surfaces.
    L_net = eps * [ SVF * sigma * T_sky^4 + (1 - SVF) * sigma * T_canyon^4 - sigma * T_s^4 ]
    """
    sigma = STEFAN_BOLTZMANN
    flux_in_sky = sky_view_factor * sigma * (sky_temp_k ** 4)
    flux_in_canyon = (1.0 - sky_view_factor) * sigma * (canyon_temp_k ** 4)
    flux_out = sigma * (surface_temp_k ** 4)
    return emissivity * (flux_in_sky + flux_in_canyon - flux_out)


def compute_sensible_heat_flux(
    surface_temp_k,
    air_temp_k,
    wind_speed,
    roughness_length=0.05
):
    """
    Calculates turbulent sensible heat transfer to the canopy air layer (W/m^2).
    H = h_c * (T_s - T_air)
    h_c = 5.62 + 3.9 * wind_speed * (roughness / 0.05)^0.2
    """
    if isinstance(wind_speed, torch.Tensor):
        u_eff = torch.clamp(wind_speed, min=0.1)
        hc = 5.62 + 3.9 * (u_eff ** 0.8) * ((roughness_length / 0.05) ** 0.2)
    else:
        u_eff = max(0.1, wind_speed)
        hc = 5.62 + 3.9 * (u_eff ** 0.8) * ((roughness_length / 0.05) ** 0.2)
    return hc * (surface_temp_k - air_temp_k)


def compute_latent_heat_flux(
    net_radiation,
    ground_heat_flux,
    air_temp_k,
    relative_humidity,
    vegetation_fraction,
    soil_moisture=0.5
):
    """
    Calculates latent heat flux from evapotranspiration (W/m^2) using simplified Penman-Monteith.
    For impermeable concrete/asphalt (veg_fraction = 0), latent flux is near 0.
    """
    t_c = kelvin_to_celsius(air_temp_k)
    # Saturated vapor pressure (kPa) via Tetens equation
    if isinstance(t_c, torch.Tensor):
        e_sat = 0.61078 * torch.exp((17.27 * t_c) / (t_c + 237.3))
        # Slope of saturation vapor pressure curve Delta (kPa/K)
        delta = (4098.0 * e_sat) / ((t_c + 237.3) ** 2)
        gamma = PSYCHROMETRIC_CONST
        rh_frac = torch.clamp(relative_humidity / 100.0, 0.05, 0.99)
        vpd = e_sat * (1.0 - rh_frac)
        evap_efficiency = torch.clamp(vegetation_fraction * (0.4 + 0.6 * soil_moisture), 0.0, 1.0)
        available_energy = torch.clamp(net_radiation - ground_heat_flux, min=0.0)
        le = evap_efficiency * ((delta / (delta + gamma)) * available_energy + 15.0 * vpd)
        return torch.clamp(le, min=0.0)
    else:
        e_sat = 0.61078 * math.exp((17.27 * t_c) / (t_c + 237.3))
        delta = (4098.0 * e_sat) / ((t_c + 237.3) ** 2)
        gamma = PSYCHROMETRIC_CONST
        rh_frac = max(0.05, min(0.99, relative_humidity / 100.0))
        vpd = e_sat * (1.0 - rh_frac)
        evap_efficiency = max(0.0, min(1.0, vegetation_fraction * (0.4 + 0.6 * soil_moisture)))
        available_energy = max(0.0, net_radiation - ground_heat_flux)
        le = evap_efficiency * ((delta / (delta + gamma)) * available_energy + 15.0 * vpd)
        return max(0.0, le)


def compute_ground_heat_flux(
    surface_temp_k,
    core_temp_k,
    thermal_conductivity=1.2,
    layer_thickness=0.25
):
    """
    Calculates conductive heat storage into the building envelope or ground substrate (W/m^2).
    G = (k / d) * (T_s - T_core)
    """
    conductance = thermal_conductivity / max(0.05, layer_thickness)
    return conductance * (surface_temp_k - core_temp_k)


def compute_anthropogenic_heat_flux(
    traffic_density,
    air_temp_k,
    is_commercial=False
):
    """
    Calculates anthropogenic heat release from traffic and building HVAC (W/m^2).
    """
    t_c = kelvin_to_celsius(air_temp_k)
    q_traffic = traffic_density * 35.0  # Normalized peak traffic flux (up to ~35 W/m^2)
    # AC heat rejection if cooling active above 24 deg C
    if isinstance(t_c, torch.Tensor):
        cooling_demand = torch.clamp(t_c - 24.0, min=0.0)
        multiplier = 4.5 if is_commercial else 2.0
        q_hvac = multiplier * cooling_demand
        return q_traffic + q_hvac
    else:
        cooling_demand = max(0.0, t_c - 24.0)
        multiplier = 4.5 if is_commercial else 2.0
        q_hvac = multiplier * cooling_demand
        return q_traffic + q_hvac


def solve_surface_energy_balance_residual(
    surface_temp_k,
    air_temp_k,
    dew_point_k,
    solar_irradiance,
    albedo,
    emissivity,
    sky_view_factor,
    wind_speed,
    relative_humidity,
    vegetation_fraction,
    thermal_conductivity=1.2,
    layer_thickness=0.25,
    traffic_density=0.5,
    is_commercial=False,
    canyon_temp_k=None,
    shadow_mask=1.0
):
    """
    Computes the complete Surface Energy Balance (SEB) residual (W/m^2):
    R_SEB = R_n - H - LE - G + Q_F
    Under thermodynamic equilibrium, R_SEB == 0.
    """
    sky_temp_k = compute_sky_temperature(air_temp_k, dew_point_k)
    if canyon_temp_k is None:
        canyon_temp_k = air_temp_k

    s_net = compute_net_shortwave_radiation(
        albedo=albedo,
        solar_irradiance=solar_irradiance,
        sky_view_factor=sky_view_factor,
        shadow_mask=shadow_mask
    )

    l_net = compute_net_longwave_radiation(
        surface_temp_k=surface_temp_k,
        air_temp_k=air_temp_k,
        sky_temp_k=sky_temp_k,
        canyon_temp_k=canyon_temp_k,
        emissivity=emissivity,
        sky_view_factor=sky_view_factor
    )

    r_net = s_net + l_net

    h_flux = compute_sensible_heat_flux(
        surface_temp_k=surface_temp_k,
        air_temp_k=air_temp_k,
        wind_speed=wind_speed
    )

    core_temp_k = air_temp_k - 1.5  # Approximate internal thermal core baseline
    g_flux = compute_ground_heat_flux(
        surface_temp_k=surface_temp_k,
        core_temp_k=core_temp_k,
        thermal_conductivity=thermal_conductivity,
        layer_thickness=layer_thickness
    )

    le_flux = compute_latent_heat_flux(
        net_radiation=r_net,
        ground_heat_flux=g_flux,
        air_temp_k=air_temp_k,
        relative_humidity=relative_humidity,
        vegetation_fraction=vegetation_fraction
    )

    q_f = compute_anthropogenic_heat_flux(
        traffic_density=traffic_density,
        air_temp_k=air_temp_k,
        is_commercial=is_commercial
    )

    residual = r_net - h_flux - le_flux - g_flux + q_f
    return residual, {
        "s_net": s_net,
        "l_net": l_net,
        "r_net": r_net,
        "h_flux": h_flux,
        "le_flux": le_flux,
        "g_flux": g_flux,
        "q_f": q_f
    }
