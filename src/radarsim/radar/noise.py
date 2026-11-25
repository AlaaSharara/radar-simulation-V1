"""Receiver thermal noise (k*T*B*F) and antenna-to-ADC-full-scale power conversion.

Reference plane:
    Input-referred thermal noise at the RX antenna port:
        P_n_ant = k_B * T * B_eq * F   [W]
    where F = 10^(NF_dB / 10).

    Receiver chain gain G_rx = 10^(gain_dB / 10) maps RX antenna power to ADC
    input power across impedance Z_0:
        P_adc = G_rx * P_ant   [W]

    Normalized complex IQ full scale (amplitude = 1.0) corresponds to peak
    voltage V_peak_FS = adc_full_scale_vpp / 2 across Z_0, with average power:
        P_FS = V_peak_FS^2 / (2 * Z_0) = adc_full_scale_vpp^2 / (8 * Z_0)   [W]

    Therefore, the normalized complex noise variance per ADC sample is:
        E[|w|^2] = (P_n_ant * G_rx) / P_FS
    with i.i.d. real/imag components each having variance E[|w|^2] / 2.
"""

from __future__ import annotations

import math

from radarsim.config import RadarProfile
from radarsim.radar.types import ThermalNoiseBudget

# Exact 2019 SI Boltzmann constant [J/K]
BOLTZMANN_K_J_PER_K: float = 1.380649e-23

NOISE_REFERENCE_PLANE: str = "rx_antenna_port_referred_to_adc_normalized_full_scale"


def compute_thermal_noise_budget(radar_profile: RadarProfile) -> ThermalNoiseBudget:
    """Compute exact receiver thermal noise power and normalized ADC scaling from ``RadarProfile``."""
    rf = radar_profile.rf
    rx = rf.receiver

    t_k = rx.noise_temperature_k
    b_eq_hz = rx.noise_equivalent_bandwidth_hz
    nf_db = rf.receiver_noise_figure_db
    nf_linear = 10.0 ** (nf_db / 10.0)

    # P_n = k * T * B * F [W] at RX antenna port
    p_noise_ant_w = BOLTZMANN_K_J_PER_K * t_k * b_eq_hz * nf_linear

    rx_gain_db = rx.receiver_gain_db
    rx_gain_linear = 10.0 ** (rx_gain_db / 10.0)

    tx_power_dbm = rf.tx_power_dbm_per_active_tx
    tx_power_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)

    vpp = rx.adc_full_scale_vpp
    z0 = rx.adc_input_impedance_ohm
    p_fs_w = (vpp * vpp) / (8.0 * z0)

    # Scale converting channel transfer amplitude h = sqrt(P_r / P_t) -> normalized ADC IQ amplitude
    # |A_adc|^2 = P_tx * |h|^2 * G_rx / P_FS
    antenna_to_adc_scale = math.sqrt((tx_power_w * rx_gain_linear) / p_fs_w)

    # Complex thermal noise variance E[|w|^2] in normalized full-scale units
    noise_var_complex = (p_noise_ant_w * rx_gain_linear) / p_fs_w
    noise_std_per_comp = math.sqrt(0.5 * noise_var_complex)

    return ThermalNoiseBudget(
        reference_plane=NOISE_REFERENCE_PLANE,
        boltzmann_k_j_per_k=BOLTZMANN_K_J_PER_K,
        noise_temperature_k=t_k,
        noise_equivalent_bandwidth_hz=b_eq_hz,
        receiver_noise_figure_db=nf_db,
        noise_factor_linear=nf_linear,
        input_referred_noise_power_w=p_noise_ant_w,
        receiver_gain_db=rx_gain_db,
        receiver_gain_linear=rx_gain_linear,
        tx_power_dbm=tx_power_dbm,
        tx_power_w=tx_power_w,
        adc_full_scale_vpp=vpp,
        adc_input_impedance_ohm=z0,
        adc_full_scale_power_w=p_fs_w,
        antenna_to_adc_scale=antenna_to_adc_scale,
        noise_variance_adc_complex=noise_var_complex,
        noise_std_adc_per_component=noise_std_per_comp,
    )
