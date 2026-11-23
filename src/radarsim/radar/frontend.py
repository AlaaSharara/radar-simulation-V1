"""Vectorized linear FMCW / TDM-MIMO front-end converting ``PathSet`` into ``ADCCube``.

Signal conventions (explicitly documented per ``PLAN.md`` rule 11):
- **Sampling representation:** ``complex_iq_normalized_to_full_scale`` (full
  scale = ``1.0`` on both I and Q axes; ``complex64`` or ``complex128``).
- **Dechirp mixer convention:** ``s_beat = s_tx * conj(s_rx)``.
  - Positive range delay ``tau = L / c > 0`` produces positive fast-time beat
    frequency ``f_beat = +S * tau > 0`` (peaks in positive range FFT bins).
  - Positive receding radial velocity ``v_r > 0`` increases delay across chirps
    ``tau(t_slow) = tau_0 + (2 * v_r / c) * dt_chirp``, producing positive
    slow-time beat Doppler ``f_D_beat = +2 * v_r / lambda_c > 0``.
- **IF filter:** Configured Butterworth low-pass magnitude transfer function
  evaluated at each path's beat frequency ``f_beat``.
- **Thermal noise:** ``P_n = k * T * B_eq * F`` at the RX antenna port,
  referred to normalized ADC full-scale units using ``receiver_gain_db``,
  ``adc_input_impedance_ohm``, and ``adc_full_scale_vpp``.
- **Vendor neutrality:** Reads all parameters from ``RadarProfile``; contains
  zero hardcoded hardware-specific constants.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from radarsim.config import RadarProfile, compute_derived_metrics
from radarsim.radar.noise import compute_thermal_noise_budget
from radarsim.radar.types import ADCCube, ThermalNoiseBudget
from radarsim.rf.types import PathSet, RFPath
from radarsim.rng import make_generator


def _compute_tdm_chirp_offsets(
    radar_profile: RadarProfile, num_loops: int
) -> tuple[np.ndarray, float]:
    """Compute relative chirp start times ``[loops, tx]`` [s] and ``same_tx_interval_s``."""
    rf = radar_profile.rf
    wf = radar_profile.waveform
    n_tx = rf.tx_count
    chirp_period = wf.chirp_period_s

    rel_t = np.zeros((num_loops, n_tx), dtype=np.float64)
    if rf.tdm_mode:
        schedule = rf.tx_schedule
        n_slots = len(schedule)
        slot_for_tx = {tx_id: slot for slot, tx_id in enumerate(schedule)}
        loops = np.arange(num_loops, dtype=np.float64)
        for tx_id in range(n_tx):
            slot = slot_for_tx[tx_id]
            rel_t[:, tx_id] = (loops * n_slots + slot) * chirp_period
        same_tx_interval = n_slots * chirp_period
    else:
        loops = np.arange(num_loops, dtype=np.float64)
        for tx_id in range(n_tx):
            rel_t[:, tx_id] = loops * chirp_period
        same_tx_interval = chirp_period

    return rel_t, same_tx_interval


class FMCWFrontEnd:
    """Vectorized FMCW/TDM-MIMO signal synthesizer with 10 independently toggleable impairments."""

    def __init__(
        self,
        radar_profile: RadarProfile,
        *,
        config_hash: str = "unhashed",
        global_seed: int = 20261123,
        dtype: str = "complex64",
    ) -> None:
        if dtype not in ("complex64", "complex128"):
            raise ValueError(f"dtype must be 'complex64' or 'complex128', got {dtype!r}")
        self.radar_profile = radar_profile
        self.config_hash = str(config_hash)
        self.global_seed = int(global_seed)
        self.dtype_name = dtype
        self.np_dtype = np.complex64 if dtype == "complex64" else np.complex128

        self.derived = compute_derived_metrics(radar_profile)
        self.noise_budget: ThermalNoiseBudget = compute_thermal_noise_budget(radar_profile)

        # Static per-board/profile calibration errors drawn deterministically from global_seed
        self._channel_mismatch_matrix = self._draw_static_channel_mismatch()
        self._iq_imbalance_params = self._draw_static_iq_imbalance()
        self._tx_leakage_phases = self._draw_static_leakage_phases()

    def _draw_static_channel_mismatch(self) -> np.ndarray:
        """Draw static per-(tx, rx) complex gain/phase mismatch from configured RMS priors."""
        n_tx = self.radar_profile.rf.tx_count
        n_rx = self.radar_profile.rf.rx_count
        arr = self.radar_profile.antenna_array
        rng = make_generator(self.global_seed, "impairments", subkey="tx_rx_channel_mismatch")
        gain_db = rng.normal(0.0, arr.channel_gain_mismatch_db_rms, size=(n_tx, n_rx))
        phase_rad = np.deg2rad(
            rng.normal(0.0, arr.channel_phase_mismatch_deg_rms, size=(n_tx, n_rx))
        )
        return (10.0 ** (gain_db / 20.0)) * np.exp(1j * phase_rad)

    def _draw_static_iq_imbalance(self) -> tuple[np.ndarray, np.ndarray]:
        """Draw static per-RX IQ amplitude mismatch (linear epsilon) and phase skew [rad]."""
        n_rx = self.radar_profile.rf.rx_count
        imp = self.radar_profile.impairments
        rng = make_generator(self.global_seed, "impairments", subkey="iq_imbalance")
        amp_db = rng.normal(0.0, imp.iq_imbalance_amplitude_mismatch_db_sigma, size=n_rx)
        # Convert dB imbalance to fractional amplitude mismatch epsilon
        gain_ratio = 10.0 ** (amp_db / 20.0)
        epsilon = (gain_ratio - 1.0) / (gain_ratio + 1.0)
        phase_skew = np.deg2rad(
            rng.normal(0.0, imp.iq_imbalance_phase_mismatch_deg_sigma, size=n_rx)
        )
        return epsilon, phase_skew

    def _draw_static_leakage_phases(self) -> np.ndarray:
        n_tx = self.radar_profile.rf.tx_count
        n_rx = self.radar_profile.rf.rx_count
        rng = make_generator(self.global_seed, "impairments", subkey="tx_leakage")
        return rng.uniform(-math.pi, math.pi, size=(n_tx, n_rx))

    def _butterworth_if_gain(self, beat_freq_hz: np.ndarray) -> np.ndarray:
        rx = self.radar_profile.rf.receiver
        order = rx.if_filter_order
        f_cut = rx.max_if_frequency_hz
        ratio = np.abs(beat_freq_hz) / max(f_cut, 1.0)
        return 1.0 / np.sqrt(1.0 + np.power(ratio, 2 * order))

    def _generate_phase_noise(
        self,
        rng: np.random.Generator,
        n_loops: int,
        n_tx: int,
        n_adc: int,
        fs_hz: float,
    ) -> np.ndarray:
        """Generate Lorentzian/AR(1) + white LO phase noise approximation ``[loops, tx, adc]`` [rad]."""
        pn_cfg = self.radar_profile.rf.phase_noise
        scale = self.radar_profile.impairments.phase_noise_strength_scale
        if scale <= 0.0:
            return np.zeros((n_loops, n_tx, n_adc), dtype=np.float64)

        # Calibrate white floor from L(1 MHz) [dBc/Hz] integrated over sample rate
        l_1mhz_lin = 10.0 ** (pn_cfg.reference_dbc_per_hz_at_1mhz / 10.0)
        # Two-sided integrated white phase std
        sigma_white = scale * math.sqrt(max(1e-18, 2.0 * l_1mhz_lin * fs_hz))

        # Lorentzian AR(1) colored component parameterized by linewidth_hz
        alpha = math.exp(-2.0 * math.pi * pn_cfg.linewidth_hz / max(fs_hz, 1.0))
        # Innovation std chosen so steady-state AR(1) std matches scaled linewidth phase floor
        sigma_lor_ss = scale * math.sqrt(max(1e-18, 2.0 * l_1mhz_lin * (1.0e6 ** 2) / max(pn_cfg.linewidth_hz, 1.0))) * 1e-2
        sigma_innov = sigma_lor_ss * math.sqrt(max(0.0, 1.0 - alpha * alpha))

        white = rng.normal(0.0, sigma_white, size=(n_loops, n_tx, n_adc))
        innov = rng.normal(0.0, sigma_innov, size=(n_loops, n_tx, n_adc))
        colored = np.empty_like(innov)
        colored[:, :, 0] = rng.normal(0.0, sigma_lor_ss, size=(n_loops, n_tx))
        for n in range(1, n_adc):
            colored[:, :, n] = alpha * colored[:, :, n - 1] + innov[:, :, n]
        return white + colored

    def synthesize(
        self,
        path_set: PathSet,
        *,
        processing_mode: str = "full_sim",
        num_loops_override: int | None = None,
        frame_seed_offset: int | None = None,
    ) -> ADCCube:
        """Synthesize a complex dechirped ``ADCCube`` from ``path_set``."""
        if processing_mode not in ("full_sim", "preview", "analysis_export"):
            raise ValueError(
                f"processing_mode must be 'full_sim', 'preview', or 'analysis_export', got {processing_mode!r}"
            )

        rf = self.radar_profile.rf
        wf = self.radar_profile.waveform
        imp = self.radar_profile.impairments

        n_loops = wf.chirps_per_tx_per_frame if num_loops_override is None else int(num_loops_override)
        if n_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {n_loops}")
        n_tx = rf.tx_count
        n_rx = rf.rx_count
        n_adc = wf.adc_samples_per_chirp
        c = rf.speed_of_light_m_per_s
        fc_start = wf.start_frequency_hz
        fc = rf.carrier_frequency_hz
        fs = wf.adc_sample_rate_hz
        slope_nom = wf.slope_hz_per_s

        subkey = path_set.frame_id if frame_seed_offset is None else frame_seed_offset
        rng_imp = make_generator(self.global_seed, "impairments", subkey=f"frame_{subkey}")
        rng_noise = make_generator(self.global_seed, "noise", subkey=f"frame_{subkey}")

        # 1. Chronological TDM chirp timestamps
        chirp_rel_t, same_tx_interval = _compute_tdm_chirp_offsets(self.radar_profile, n_loops)
        chirp_time_s = float(path_set.sim_time_s) + chirp_rel_t
        t_fast = wf.adc_start_time_s + np.arange(n_adc, dtype=np.float64) / fs

        # 2. Per-chirp waveform impairments (slope error, start frequency error, timing jitter)
        if imp.chirp_slope_error_enabled and imp.chirp_slope_error_sigma_ppm > 0.0:
            slope_frac = rng_imp.normal(0.0, imp.chirp_slope_error_sigma_ppm * 1e-6, size=(n_loops, n_tx))
            slope_eff = slope_nom * (1.0 + slope_frac)
        else:
            slope_eff = np.full((n_loops, n_tx), slope_nom, dtype=np.float64)

        if imp.chirp_start_frequency_error_enabled and imp.chirp_start_frequency_error_sigma_hz > 0.0:
            df0 = rng_imp.normal(0.0, imp.chirp_start_frequency_error_sigma_hz, size=(n_loops, n_tx))
        else:
            df0 = np.zeros((n_loops, n_tx), dtype=np.float64)

        if imp.timing_jitter_enabled and imp.timing_jitter_sigma_s > 0.0:
            dt_trig = rng_imp.normal(0.0, imp.timing_jitter_sigma_s, size=(n_loops, n_tx))
            dt_adc = rng_imp.normal(0.0, imp.timing_jitter_sigma_s, size=(n_loops, n_tx, n_adc))
        else:
            dt_trig = None
            dt_adc = None

        # 3. Group RF paths by (tx_id, rx_id) and sum coherently in complex128
        grouped: dict[tuple[int, int], list[RFPath]] = {}
        for p in path_set.paths:
            if 0 <= p.tx_id < n_tx and 0 <= p.rx_id < n_rx:
                grouped.setdefault((p.tx_id, p.rx_id), []).append(p)

        cube = np.zeros((n_loops, n_tx, n_rx, n_adc), dtype=np.complex128)
        scale_ant_to_adc = self.noise_budget.antenna_to_adc_scale
        two_pi = 2.0 * math.pi
        chunk_size = 16
        use_separable_preview = processing_mode == "preview"
        t_fast_mid = wf.adc_start_time_s + 0.5 * (n_adc - 1) / fs
        fc_mid = fc_start + slope_nom * t_fast_mid
        t_fast_centered = t_fast - t_fast_mid

        for (tx_id, rx_id), plist in grouped.items():
            rel_t_tx = chirp_rel_t[:, tx_id]      # (L,)
            slope_tx = slope_eff[:, tx_id]        # (L,)
            df0_tx = df0[:, tx_id]                # (L,)
            dt_trig_tx = dt_trig[:, tx_id] if dt_trig is not None else None
            t_fast_mat = (
                t_fast[None, :] + dt_adc[:, tx_id, :]
                if dt_adc is not None
                else t_fast[None, :]
            )  # (1, N) or (L, N)

            # Fast BLAS separable matmul path: exact for preview mode (center-referenced chirp)
            # or for purely stationary paths when per-chirp slope/timing errors are disabled.
            if use_separable_preview or (
                (not imp.chirp_slope_error_enabled)
                and (not imp.timing_jitter_enabled)
                and all(p.radial_velocity_mps == 0.0 for p in plist)
            ):
                tau0 = np.array([p.delay_s for p in plist], dtype=np.float64)
                vr = np.array([p.radial_velocity_mps for p in plist], dtype=np.float64)
                amp_h = np.array([p.amplitude_linear for p in plist], dtype=np.float64)
                refl_phase = np.array(
                    [math.atan2(p.reflection_coeff_imag, p.reflection_coeff_real) for p in plist],
                    dtype=np.float64,
                )
                h_if = self._butterworth_if_gain(slope_nom * tau0)
                amp_adc = amp_h * scale_ant_to_adc * h_if  # (P,)

                tau_pl = tau0[:, None] + (2.0 * vr[:, None] / c) * rel_t_tx[None, :]  # (P, L)
                fc_mid_tx = (fc_start + df0_tx) + slope_tx * t_fast_mid
                phi_slow = (
                    two_pi * fc_mid_tx[None, :] * tau_pl
                    - math.pi * slope_tx[None, :] * (tau_pl * tau_pl)
                    - refl_phase[:, None]
                )
                if dt_trig_tx is not None:
                    phi_slow = phi_slow + two_pi * slope_tx[None, :] * tau_pl * dt_trig_tx[None, :]
                slow_mat = (amp_adc[:, None] * np.exp(1j * phi_slow)).T  # (L, P)
                fast_mat = np.exp(1j * (two_pi * slope_nom * tau0[:, None] * t_fast_centered[None, :]))  # (P, N)
                if use_separable_preview:
                    cube[:, tx_id, rx_id, :] = (
                        slow_mat.astype(np.complex64) @ fast_mat.astype(np.complex64)
                    )
                else:
                    cube[:, tx_id, rx_id, :] = slow_mat @ fast_mat
                continue

            acc = np.zeros((n_loops, n_adc), dtype=np.complex128)
            for start in range(0, len(plist), chunk_size):
                batch = plist[start : start + chunk_size]
                tau0 = np.array([p.delay_s for p in batch], dtype=np.float64)
                vr = np.array([p.radial_velocity_mps for p in batch], dtype=np.float64)
                amp_h = np.array([p.amplitude_linear for p in batch], dtype=np.float64)
                # Reflection/transmission complex phase (subtracted under s_tx * conj(s_rx))
                refl_phase = np.array(
                    [math.atan2(p.reflection_coeff_imag, p.reflection_coeff_real) for p in batch],
                    dtype=np.float64,
                )

                # IF Butterworth low-pass filter response at nominal beat frequency S * tau0
                h_if = self._butterworth_if_gain(slope_nom * tau0)
                amp_adc = amp_h * scale_ant_to_adc * h_if  # (P,)

                # Time-varying round-trip delay across chirps: tau(p, l) = tau0 + (2*v_r/c) * t_rel
                tau_pl = tau0[:, None] + (2.0 * vr[:, None] / c) * rel_t_tx[None, :]  # (P, L)

                # Exact linear FMCW dechirp phase: Phi_tx(t) - Phi_tx(t - tau)
                # Slow-time carrier + residual video phase:
                # 2*pi * (fc_start + df0) * tau(p, l) - pi * S_eff * tau(p, l)^2 - theta_refl
                phi_slow = (
                    two_pi * (fc_start + df0_tx[None, :]) * tau_pl
                    - math.pi * slope_tx[None, :] * (tau_pl * tau_pl)
                    - refl_phase[:, None]
                )
                if dt_trig_tx is not None:
                    # Trigger timing jitter shifts instantaneous chirp frequency by S * dt_trig
                    phi_slow = phi_slow + two_pi * slope_tx[None, :] * tau_pl * dt_trig_tx[None, :]

                # Fast-time beat frequency term: 2*pi * S_eff * tau(p, l) * t_fast
                f_beat_pl = slope_tx[None, :] * tau_pl  # (P, L)
                if dt_trig_tx is not None:
                    f_beat_pl = f_beat_pl + slope_tx[None, :] * dt_trig_tx[None, :]

                # Evaluate vectorized phase (P, L, N)
                phase = phi_slow[:, :, None] + two_pi * f_beat_pl[:, :, None] * t_fast_mat[None, :, :]
                sig = amp_adc[:, None, None] * np.exp(1j * phase)
                acc += sig.sum(axis=0)

            cube[:, tx_id, rx_id, :] = acc

        # 4. LO Phase Noise (applied coherently to signal before additive receiver noise)
        if imp.phase_noise_enabled and rf.phase_noise.enabled:
            phi_pn = self._generate_phase_noise(rng_imp, n_loops, n_tx, n_adc, fs)
            cube *= np.exp(1j * phi_pn[:, :, None, :])

        # 5. Direct TX-to-RX Leakage (near-DC self-interference tone)
        if imp.tx_leakage_enabled:
            leak_amp = 10.0 ** (imp.tx_leakage_level_db_relative_to_full_scale / 20.0)
            tau_leak = 0.15e-9  # ~4.5 cm on-board/radome coupling delay
            leak_fast = np.exp(1j * two_pi * slope_nom * tau_leak * t_fast)
            leak_tone = leak_amp * np.exp(1j * self._tx_leakage_phases)[None, :, :, None] * leak_fast[None, None, None, :]
            cube += leak_tone

        # 6. Static TX/RX Channel Gain & Phase Mismatch
        if imp.tx_rx_channel_mismatch_enabled:
            cube *= self._channel_mismatch_matrix[None, :, :, None]

        # 7. Receiver Thermal Noise (k * T * B_eq * F referred to ADC normalized IQ)
        if imp.thermal_noise_enabled:
            std_comp = self.noise_budget.noise_std_adc_per_component
            noise_re = rng_noise.normal(0.0, std_comp, size=cube.shape)
            noise_im = rng_noise.normal(0.0, std_comp, size=cube.shape)
            cube += noise_re + 1j * noise_im

        # 8. Receiver IQ Imbalance (per-RX gain & quadrature phase skew)
        if imp.iq_imbalance_enabled:
            eps_rx, phi_rx = self._iq_imbalance_params
            i_in = cube.real
            q_in = cube.imag
            e = eps_rx[None, None, :, None]
            cp = np.cos(phi_rx)[None, None, :, None]
            sp = np.sin(phi_rx)[None, None, :, None]
            i_out = (1.0 + 0.5 * e) * i_in
            q_out = (1.0 - 0.5 * e) * (q_in * cp + i_in * sp)
            cube = i_out + 1j * q_out

        # 9. ADC Clipping (symmetric [-1.0, +1.0] full-scale saturation on I and Q)
        clipped_count = 0
        if imp.adc_clipping_enabled and rf.receiver.adc_clip_enabled:
            clipped_mask = (np.abs(cube.real) > 1.0) | (np.abs(cube.imag) > 1.0)
            clipped_count = int(np.count_nonzero(clipped_mask))
            cube = np.clip(cube.real, -1.0, 1.0) + 1j * np.clip(cube.imag, -1.0, 1.0)

        # 10. ADC Uniform Mid-Tread Quantization (adc_bits over [-1.0, +1.0])
        adc_bits = rf.receiver.adc_bits
        if imp.adc_quantization_enabled and rf.receiver.adc_quantization_enabled:
            q_step = 2.0 / float(2**adc_bits)
            q_re = np.round(cube.real / q_step) * q_step
            q_im = np.round(cube.imag / q_step) * q_step
            cube = q_re + 1j * q_im

        out_samples = cube.astype(self.np_dtype, copy=False)

        diagnostics: dict[str, Any] = {
            "num_input_paths": len(path_set.paths),
            "clipped_sample_count": clipped_count,
            "peak_abs_sample": float(np.max(np.abs(out_samples))) if out_samples.size else 0.0,
            "rms_abs_sample": float(np.sqrt(np.mean(np.abs(out_samples) ** 2))) if out_samples.size else 0.0,
            "noise_budget": self.noise_budget.to_dict(),
            "conventions": {
                "dechirp_mixer": "s_tx * conj(s_rx)",
                "range_beat_sign": "positive frequency f_beat = +S * tau for positive range",
                "doppler_beat_sign": "positive slow-time frequency +2 * v_r / lambda_c for receding v_r > 0",
                "adc_full_scale": "1.0 normalized (real & imag independently in [-1, +1])",
                "dtype": self.dtype_name,
            },
        }

        return ADCCube(
            frame_id=path_set.frame_id,
            sim_time_s=path_set.sim_time_s,
            config_hash=self.config_hash,
            complex_samples=out_samples,
            chirp_time_s=chirp_time_s,
            tx_schedule=rf.tx_schedule,
            rx_ids=tuple(range(n_rx)),
            sample_rate_hz=fs,
            slope_hz_per_s=slope_nom,
            fc_start_hz=wf.start_frequency_hz,
            carrier_frequency_hz=fc,
            chirp_period_s=wf.chirp_period_s,
            same_tx_interval_s=same_tx_interval,
            adc_start_time_s=wf.adc_start_time_s,
            dtype=self.dtype_name,
            sample_units=rf.receiver.sample_representation,
            adc_full_scale=1.0,
            adc_bits_effective=adc_bits,
            noise_reference_plane=self.noise_budget.reference_plane,
            noise_equivalent_bandwidth_hz=self.noise_budget.noise_equivalent_bandwidth_hz,
            enabled_impairments=imp.enabled_names(),
            processing_mode=processing_mode,
            diagnostics=diagnostics,
        )
