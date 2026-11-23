"""Antenna TX/RX world positions and separable cosine-power antenna gain model."""

from __future__ import annotations

import math
from typing import Sequence

from radarsim.config import RadarProfile, compute_derived_metrics
from radarsim.scene import RadarCoordinateFrame

Vec3 = tuple[float, float, float]


def tx_rx_world_positions(
    radar_profile: RadarProfile,
    radar_frame: RadarCoordinateFrame,
) -> tuple[tuple[Vec3, ...], tuple[Vec3, ...]]:
    """Compute world-frame 3D positions [m] of every TX and RX element.

    Aperture mapping convention:
    Configured planar array coordinates ``[u, v, w]`` in ``radar_config.yaml``
    span azimuth (``u`` -> radar local ``+X``), elevation (``v`` -> radar local
    ``+Z``), and out-of-plane offset (``w`` -> radar local ``+Y`` boresight).
    Coordinates in ``wavelengths`` are scaled by ``wavelength_m = c / f_c``.
    """
    arr = radar_profile.antenna_array
    scale = (
        compute_derived_metrics(radar_profile).wavelength_m
        if arr.coordinate_units == "wavelengths"
        else 1.0
    )

    def _to_world(pos_cfg: Sequence[float]) -> Vec3:
        local_m = (
            float(pos_cfg[0]) * scale,
            float(pos_cfg[2]) * scale,
            float(pos_cfg[1]) * scale,
        )
        return radar_frame.local_to_world_point(local_m)

    tx_world = tuple(_to_world(p) for p in arr.tx_positions_lambda)
    rx_world = tuple(_to_world(p) for p in arr.rx_positions_lambda)
    return tx_world, rx_world


def _cosine_exponent_for_fov(fov_3db_deg: float, configured_exp: float) -> float:
    """Resolve cosine power exponent matching configured 3 dB beamwidth."""
    half_rad = 0.5 * math.radians(fov_3db_deg)
    c_half = math.cos(half_rad)
    if c_half <= 1e-6 or c_half >= 1.0 - 1e-9:
        return max(0.1, float(configured_exp))
    # For fov_3db_deg = 120 deg, cos(60 deg) = 0.5 -> log(0.5)/log(0.5) = 1.0
    # Matches configured cosine_power_exponent=1.0 identically at 120 deg FoV
    return float(math.log(0.5) / math.log(c_half))


def antenna_gain_linear(
    radar_profile: RadarProfile,
    radar_frame: RadarCoordinateFrame,
    world_target_point_m: Sequence[float],
) -> float:
    """Evaluate approximate directional antenna power gain (linear, dimensionless) toward a world point.

    Uses ``rf.antenna_gain_dbi_peak`` and ``rf.antenna_pattern`` (separable
    cosine-power approximation with -30 dB back-hemisphere floor).
    """
    peak_linear = 10.0 ** (radar_profile.rf.antenna_gain_dbi_peak / 10.0)
    pattern = radar_profile.rf.antenna_pattern
    lx, ly, lz = radar_frame.world_to_local_point(world_target_point_m)
    if ly <= 1e-9:
        # Behind antenna plane: -30 dB back-lobe floor
        return peak_linear * 1e-3

    horiz = math.hypot(lx, ly)
    az = math.atan2(lx, ly)
    el = math.atan2(lz, horiz)
    if abs(az) >= 0.5 * math.pi or abs(el) >= 0.5 * math.pi:
        return peak_linear * 1e-3

    p_az = _cosine_exponent_for_fov(pattern.azimuth_fov_3db_deg, pattern.cosine_power_exponent)
    p_el = _cosine_exponent_for_fov(pattern.elevation_fov_3db_deg, pattern.cosine_power_exponent)
    rel_gain = (max(0.0, math.cos(az)) ** p_az) * (max(0.0, math.cos(el)) ** p_el)
    return peak_linear * max(1e-3, rel_gain)
