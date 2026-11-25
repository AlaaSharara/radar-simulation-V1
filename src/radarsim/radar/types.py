"""Data contracts for the FMCW/TDM-MIMO front-end following ``INTERFACES.md`` §6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThermalNoiseBudget:
    """Explicit receiver thermal-noise budget and antenna-to-ADC scaling."""

    reference_plane: str
    boltzmann_k_j_per_k: float
    noise_temperature_k: float
    noise_equivalent_bandwidth_hz: float
    receiver_noise_figure_db: float
    noise_factor_linear: float
    input_referred_noise_power_w: float
    receiver_gain_db: float
    receiver_gain_linear: float
    tx_power_dbm: float
    tx_power_w: float
    adc_full_scale_vpp: float
    adc_input_impedance_ohm: float
    adc_full_scale_power_w: float
    antenna_to_adc_scale: float
    noise_variance_adc_complex: float
    noise_std_adc_per_component: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_plane": self.reference_plane,
            "boltzmann_k_j_per_k": self.boltzmann_k_j_per_k,
            "noise_temperature_k": self.noise_temperature_k,
            "noise_equivalent_bandwidth_hz": self.noise_equivalent_bandwidth_hz,
            "receiver_noise_figure_db": self.receiver_noise_figure_db,
            "noise_factor_linear": self.noise_factor_linear,
            "input_referred_noise_power_w": self.input_referred_noise_power_w,
            "receiver_gain_db": self.receiver_gain_db,
            "receiver_gain_linear": self.receiver_gain_linear,
            "tx_power_dbm": self.tx_power_dbm,
            "tx_power_w": self.tx_power_w,
            "adc_full_scale_vpp": self.adc_full_scale_vpp,
            "adc_input_impedance_ohm": self.adc_input_impedance_ohm,
            "adc_full_scale_power_w": self.adc_full_scale_power_w,
            "antenna_to_adc_scale": self.antenna_to_adc_scale,
            "noise_variance_adc_complex": self.noise_variance_adc_complex,
            "noise_std_adc_per_component": self.noise_std_adc_per_component,
        }


@dataclass(frozen=True)
class ADCCube:
    """Complex dechirped FMCW ADC cube adhering to ``INTERFACES.md`` §6.

    Canonical shape of ``complex_samples``:
        ``[loop_index, tx_index, rx_index, adc_sample]``
    with dtype ``complex64`` or ``complex128`` (documented via ``dtype``).
    """

    frame_id: int
    sim_time_s: float
    config_hash: str
    complex_samples: np.ndarray
    chirp_time_s: np.ndarray  # shape [loops, tx], actual transmit timestamps [s]
    tx_schedule: tuple[int, ...]
    rx_ids: tuple[int, ...]
    sample_rate_hz: float
    slope_hz_per_s: float
    fc_start_hz: float
    carrier_frequency_hz: float
    chirp_period_s: float
    same_tx_interval_s: float
    adc_start_time_s: float
    dtype: str
    sample_units: str
    adc_full_scale: float
    adc_bits_effective: int
    noise_reference_plane: str
    noise_equivalent_bandwidth_hz: float
    enabled_impairments: tuple[str, ...]
    processing_mode: str = "full_sim"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        s = self.complex_samples.shape
        return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))

    @property
    def num_loops(self) -> int:
        return int(self.complex_samples.shape[0])

    @property
    def num_tx(self) -> int:
        return int(self.complex_samples.shape[1])

    @property
    def num_rx(self) -> int:
        return int(self.complex_samples.shape[2])

    @property
    def num_adc_samples(self) -> int:
        return int(self.complex_samples.shape[3])

    def as_chronological(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(chirps[M, rx, adc], tx_id_per_chirp[M], timestamp_s[M])`` in chronological TX order."""
        n_loops, _, n_rx, n_adc = self.shape
        slots = self.tx_schedule
        n_slots = len(slots)
        total = n_loops * n_slots
        chron = np.empty((total, n_rx, n_adc), dtype=self.complex_samples.dtype)
        tx_ids = np.empty(total, dtype=np.int32)
        times = np.empty(total, dtype=np.float64)
        idx = 0
        for loop_idx in range(n_loops):
            for tx_id in slots:
                chron[idx] = self.complex_samples[loop_idx, tx_id]
                tx_ids[idx] = tx_id
                times[idx] = self.chirp_time_s[loop_idx, tx_id]
                idx += 1
        return chron, tx_ids, times

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sim_time_s": self.sim_time_s,
            "config_hash": self.config_hash,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sample_units": self.sample_units,
            "adc_full_scale": self.adc_full_scale,
            "adc_bits_effective": self.adc_bits_effective,
            "tx_schedule": list(self.tx_schedule),
            "rx_ids": list(self.rx_ids),
            "sample_rate_hz": self.sample_rate_hz,
            "slope_hz_per_s": self.slope_hz_per_s,
            "fc_start_hz": self.fc_start_hz,
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "chirp_period_s": self.chirp_period_s,
            "same_tx_interval_s": self.same_tx_interval_s,
            "adc_start_time_s": self.adc_start_time_s,
            "noise_reference_plane": self.noise_reference_plane,
            "noise_equivalent_bandwidth_hz": self.noise_equivalent_bandwidth_hz,
            "enabled_impairments": list(self.enabled_impairments),
            "processing_mode": self.processing_mode,
            "diagnostics": dict(self.diagnostics),
        }
