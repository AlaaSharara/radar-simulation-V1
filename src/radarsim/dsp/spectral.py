"""Range FFT, dual-branch clutter suppression, same-TX Doppler FFT, and TDM phase compensation.

Conventions & normalization (documented explicitly per ``PLAN.md`` rule 11):
- **Amplitude-preserving window normalization:** ``w_norm = w / sum(w)`` so a
  unit-amplitude complex tone has magnitude ``1.0`` at its FFT peak.
- **Zero-padding:** Increases FFT bin density (interpolation) only; it does NOT
  improve physical range, Doppler, or angular resolution.
- **Dual slow-time branches:**
  1. ``stationary_range_cube`` — untouched complex slow-time data (preserves
     static targets and stationary breathing/phase signatures).
  2. ``moving_range_cube`` — slow-time mean-subtracted clutter-suppressed branch
     (attenuates static walls, furniture, and TX leakage for moving-target detection).
- **TDM-MIMO Doppler phase compensation:** Corrects motion-induced inter-TX phase
  offsets using actual transmit timestamps ``cube.chirp_time_s[0, tx] - cube.chirp_time_s[0, 0]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from radarsim.config import RadarProfile
from radarsim.radar.types import ADCCube


@dataclass(frozen=True)
class WindowStats:
    name: str
    length: int
    coherent_gain: float
    incoherent_noise_gain: float
    enbw_bins: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "length": self.length,
            "coherent_gain": self.coherent_gain,
            "incoherent_noise_gain": self.incoherent_noise_gain,
            "enbw_bins": self.enbw_bins,
        }


def make_window(name: str, length: int) -> tuple[np.ndarray, WindowStats]:
    """Return amplitude-normalized window ``w / sum(w)`` and exact coherent/ENBW gains."""
    if length < 1:
        raise ValueError(f"Window length must be >= 1, got {length}")
    if name == "hann":
        raw = np.hanning(length).astype(np.float64)
        if length == 1:
            raw = np.ones(1, dtype=np.float64)
    elif name == "hamming":
        raw = np.hamming(length).astype(np.float64)
    elif name == "blackman":
        raw = np.blackman(length).astype(np.float64)
    elif name in ("rectangular", "none"):
        raw = np.ones(length, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window name: {name!r}")

    w_sum = float(np.sum(raw))
    w_sq_sum = float(np.sum(raw * raw))
    coherent_gain = w_sum / float(length)
    enbw_bins = float(length) * w_sq_sum / max(w_sum * w_sum, 1e-30)
    w_norm = raw / max(w_sum, 1e-30)
    incoherent_noise_gain = float(np.sum(w_norm * w_norm))  # equals enbw_bins / length

    return w_norm, WindowStats(
        name=name,
        length=length,
        coherent_gain=coherent_gain,
        incoherent_noise_gain=incoherent_noise_gain,
        enbw_bins=enbw_bins,
    )


@dataclass(frozen=True)
class SpectralCubeResult:
    """Outputs of Range FFT, dual slow-time branches, Doppler FFT, and TDM compensation."""

    range_axis_m: np.ndarray                   # shape (N_r_valid,)
    velocity_axis_mps: np.ndarray              # shape (N_d,)
    range_cube_stationary: np.ndarray          # untouched [loops, tx, rx, N_r_valid]
    range_cube_moving: np.ndarray              # clutter-suppressed [loops, tx, rx, N_r_valid]
    rd_cube_moving: np.ndarray                 # TDM-compensated [N_d, tx, rx, N_r_valid]
    rd_cube_stationary: np.ndarray             # untouched [N_d, tx, rx, N_r_valid]
    rd_power_moving: np.ndarray                # non-coherent avg power [N_d, N_r_valid]
    rd_power_stationary: np.ndarray            # non-coherent avg power [N_d, N_r_valid]
    same_tx_interval_s: float
    tx_time_offsets_s: np.ndarray              # shape (tx,)
    range_window_stats: WindowStats
    doppler_window_stats: WindowStats
    range_fft_size: int
    doppler_fft_size: int


def compute_range_doppler_cubes(
    adc_cube: ADCCube,
    radar_profile: RadarProfile,
    *,
    apply_tdm_compensation: bool = True,
) -> SpectralCubeResult:
    """Run Range FFT, preserve untouched stationary branch, suppress clutter, and run same-TX Doppler FFT."""
    rf = radar_profile.rf
    wf = radar_profile.waveform
    samples = adc_cube.complex_samples.astype(np.complex128, copy=False)
    n_loops, n_tx, n_rx, n_adc = samples.shape

    # 1. Range window & FFT along fast-time axis -1
    n_r_fft = max(wf.range_fft_size, n_adc)
    w_r, r_stats = make_window(wf.window.range, n_adc)
    r_fft_full = np.fft.fft(samples * w_r[None, None, None, :], n=n_r_fft, axis=-1)

    # Keep positive IF bins up to configured max_if_frequency_hz (bounded by Nyquist)
    c = rf.speed_of_light_m_per_s
    fs = adc_cube.sample_rate_hz
    slope = adc_cube.slope_hz_per_s
    freq_bins = np.arange(n_r_fft // 2, dtype=np.float64) * (fs / n_r_fft)
    valid_mask = freq_bins <= rf.receiver.max_if_frequency_hz + 1e-6
    n_r_valid = max(1, int(np.count_nonzero(valid_mask)))
    range_axis_m = (c * freq_bins[:n_r_valid]) / (2.0 * slope)

    # Untouched stationary complex slow-time branch (preserved for stationary/breathing analysis)
    range_cube_stationary = r_fft_full[:, :, :, :n_r_valid]

    # Moving-target clutter-suppressed branch (slow-time DC/mean subtraction per channel & range bin)
    if n_loops > 1:
        range_cube_moving = range_cube_stationary - np.mean(
            range_cube_stationary, axis=0, keepdims=True
        )
    else:
        range_cube_moving = range_cube_stationary.copy()

    # 2. Same-TX Doppler FFT using actual timestamps from adc_cube.chirp_time_s
    if n_loops > 1:
        dt_same = float(np.median(np.diff(adc_cube.chirp_time_s, axis=0)))
    else:
        dt_same = float(adc_cube.same_tx_interval_s)
    tx_offsets_s = (adc_cube.chirp_time_s[0, :] - adc_cube.chirp_time_s[0, 0]).astype(np.float64)

    n_d_fft = max(wf.doppler_fft_size, n_loops)
    w_d, d_stats = make_window(wf.window.doppler, n_loops)

    rd_stationary_raw = np.fft.fftshift(
        np.fft.fft(range_cube_stationary * w_d[:, None, None, None], n=n_d_fft, axis=0),
        axes=0,
    )
    rd_moving_raw = np.fft.fftshift(
        np.fft.fft(range_cube_moving * w_d[:, None, None, None], n=n_d_fft, axis=0),
        axes=0,
    )

    # Physical Doppler velocity axis [m/s] (positive = receding, consistent with T4)
    wavelength_m = c / adc_cube.carrier_frequency_hz
    doppler_freqs_hz = (np.arange(n_d_fft, dtype=np.float64) - (n_d_fft // 2)) / (
        n_d_fft * dt_same
    )
    velocity_axis_mps = 0.5 * wavelength_m * doppler_freqs_hz

    # 3. TDM-MIMO Doppler phase compensation across TX channels:
    # A target at Doppler f_D accumulates +2*pi * f_D * dt_tx[tx] on TX channel `tx`.
    # Multiply by exp(-j * 2*pi * f_D * dt_tx[tx]) to restore coherent virtual array phase.
    if apply_tdm_compensation and n_tx > 1:
        comp = np.exp(-1j * 2.0 * math.pi * doppler_freqs_hz[:, None] * tx_offsets_s[None, :])
        rd_moving = rd_moving_raw * comp[:, :, None, None]
        rd_stationary = rd_stationary_raw * comp[:, :, None, None]
    else:
        rd_moving = rd_moving_raw
        rd_stationary = rd_stationary_raw

    # Non-coherent average power across virtual channels [N_d, N_r_valid]
    rd_power_moving = np.mean(np.abs(rd_moving) ** 2, axis=(1, 2))
    rd_power_stationary = np.mean(np.abs(rd_stationary) ** 2, axis=(1, 2))

    return SpectralCubeResult(
        range_axis_m=range_axis_m,
        velocity_axis_mps=velocity_axis_mps,
        range_cube_stationary=range_cube_stationary,
        range_cube_moving=range_cube_moving,
        rd_cube_moving=rd_moving,
        rd_cube_stationary=rd_stationary,
        rd_power_moving=rd_power_moving,
        rd_power_stationary=rd_power_stationary,
        same_tx_interval_s=dt_same,
        tx_time_offsets_s=tx_offsets_s,
        range_window_stats=r_stats,
        doppler_window_stats=d_stats,
        range_fft_size=n_r_fft,
        doppler_fft_size=n_d_fft,
    )
