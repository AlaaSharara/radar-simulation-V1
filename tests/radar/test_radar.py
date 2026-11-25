"""Tests for T4: FMCW/TDM-MIMO front-end, analytical range/Doppler, kTBF noise, and 10 impairment ablations."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from radarsim.config import RadarProfile, load_radar_profile, parse_radar_profile
from radarsim.radar import (
    ADCCube,
    BOLTZMANN_K_J_PER_K,
    FMCWFrontEnd,
    NOISE_REFERENCE_PLANE,
    compute_thermal_noise_budget,
)
from radarsim.rf import PathSet, PointTarget, RFPath, RFPropagationConfig, RFPropagationEngine
from radarsim.scene import load_scene

ROOT = Path(__file__).resolve().parents[2]

ALL_IMPAIRMENTS: tuple[str, ...] = (
    "phase_noise",
    "chirp_slope_error",
    "chirp_start_frequency_error",
    "tx_rx_channel_mismatch",
    "iq_imbalance",
    "adc_quantization",
    "adc_clipping",
    "tx_leakage",
    "timing_jitter",
    "thermal_noise",
)


def _profile_with_only(base: RadarProfile, enabled: set[str]) -> RadarProfile:
    prof = base
    for name in ALL_IMPAIRMENTS:
        prof = prof.with_impairment_toggle(name, name in enabled)
    return prof


@pytest.fixture(scope="module")
def default_scene():
    return load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")


@pytest.fixture(scope="module")
def default_radar():
    return load_radar_profile(ROOT / "radar_config.yaml")


def _single_target_pathset(
    scene,
    radar: RadarProfile,
    *,
    range_m: float = 3.00,
    radial_vel_mps: float = 0.0,
    rcs_m2: float = 0.1,
) -> PathSet:
    cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_second_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
    )
    engine = RFPropagationEngine(scene, radar, config=cfg, global_seed=101)
    tx0 = engine.tx_world[0]
    bx, by, bz = scene.radar_frame.boresight_world
    pos = (tx0[0] + range_m * bx, tx0[1] + range_m * by, tx0[2] + range_m * bz)
    vel = (radial_vel_mps * bx, radial_vel_mps * by, radial_vel_mps * bz)
    target = PointTarget(target_id="pt_test", position_m=pos, velocity_mps=vel, rcs_m2=rcs_m2)
    return engine.propagate(frame_id=0, sim_time_s=0.0, point_targets=[target], include_scene_surfaces=False)


# ---------------------------------------------------------------------------
# 1. Analytical stationary target at 3.00 m -> range beat & bin
# ---------------------------------------------------------------------------


def test_stationary_target_at_3m_appears_at_analytical_range_beat_and_bin(
    default_scene, default_radar
) -> None:
    clean_profile = _profile_with_only(default_radar, set())
    path_set = _single_target_pathset(default_scene, clean_profile, range_m=3.00, radial_vel_mps=0.0)
    fe = FMCWFrontEnd(clean_profile, config_hash="test_hash", global_seed=20261123, dtype="complex128")
    cube = fe.synthesize(path_set, processing_mode="full_sim")

    # Canonical shape & INTERFACES.md §6 metadata
    assert cube.shape == (128, 3, 4, 192)
    assert cube.dtype == "complex128"
    assert cube.tx_schedule == (0, 1, 2)
    assert cube.rx_ids == (0, 1, 2, 3)
    assert cube.noise_reference_plane == NOISE_REFERENCE_PLANE
    assert cube.enabled_impairments == ()
    assert json.dumps(cube.metadata_dict())

    # Independent analytical calculation (no helper calls)
    c = 299792458.0
    slope = 50.0e12
    fs = 5.0e6
    n_fft_cfg = 256
    expected_f_beat = slope * (2.0 * 3.00 / c)
    expected_bin_256 = int(round(expected_f_beat / fs * n_fft_cfg))

    chirp_00 = cube.complex_samples[0, 0, 0, :]
    win = np.hanning(len(chirp_00))
    spec_256 = np.abs(np.fft.fft(chirp_00 * win, n=n_fft_cfg))
    peak_bin_256 = int(np.argmax(spec_256))
    assert peak_bin_256 == expected_bin_256

    # High-resolution zero-padded beat frequency check
    n_zp = 16384
    spec_zp = np.abs(np.fft.fft(chirp_00 * win, n=n_zp))
    est_f_beat = int(np.argmax(spec_zp)) * (fs / n_zp)
    est_range_m = c * est_f_beat / (2.0 * slope)
    assert abs(est_f_beat - expected_f_beat) < 500.0  # < 0.05% error
    assert abs(est_range_m - 3.00) < 0.002  # < 2 mm

    # Stationary target peaks at zero Doppler bin
    r_fft = np.fft.fft(cube.complex_samples[:, 0, 0, :] * win[None, :], n=n_fft_cfg, axis=-1)
    d_fft = np.fft.fftshift(np.fft.fft(r_fft[:, peak_bin_256], n=128))
    assert int(np.argmax(np.abs(d_fft))) == 64  # DC / 0 m/s


# ---------------------------------------------------------------------------
# 2. Moving +1.00 m/s receding & -1.00 m/s approaching target -> Doppler sign/bin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radial_vel_mps", [1.00, -1.00])
def test_moving_target_appears_at_analytical_doppler_sign_and_bin(
    default_scene, default_radar, radial_vel_mps: float
) -> None:
    clean_profile = _profile_with_only(default_radar, set())
    path_set = _single_target_pathset(
        default_scene, clean_profile, range_m=3.00, radial_vel_mps=radial_vel_mps
    )
    fe = FMCWFrontEnd(clean_profile, global_seed=20261123, dtype="complex128")
    cube = fe.synthesize(path_set)

    # Independent analytical Doppler bin calculation
    c = 299792458.0
    fc = 61.0e9
    lam = c / fc
    t_same = 3 * 60.0e-6
    n_dop = 128
    v_res = lam / (2.0 * n_dop * t_same)
    expected_signed_bin = int(round(radial_vel_mps / v_res))  # +9 for +1.0 m/s, -9 for -1.0 m/s
    expected_shifted_idx = (n_dop // 2) + expected_signed_bin

    win_r = np.hanning(192)
    win_d = np.hanning(n_dop)
    r_fft = np.fft.fft(cube.complex_samples[:, 0, 0, :] * win_r[None, :], n=256, axis=-1)
    peak_r = int(np.argmax(np.abs(r_fft[0])))
    d_spec = np.abs(np.fft.fftshift(np.fft.fft(r_fft[:, peak_r] * win_d, n=n_dop)))
    peak_d = int(np.argmax(d_spec))

    assert peak_d == expected_shifted_idx
    if radial_vel_mps > 0:
        assert peak_d > n_dop // 2
    else:
        assert peak_d < n_dop // 2


# ---------------------------------------------------------------------------
# 3. Chronological TDM timestamps & same-TX interval
# ---------------------------------------------------------------------------


def test_tdm_timestamps_same_tx_interval_and_chronological_order(
    default_scene, default_radar
) -> None:
    clean_profile = _profile_with_only(default_radar, set())
    path_set = _single_target_pathset(default_scene, clean_profile, range_m=3.00, radial_vel_mps=1.00)
    fe = FMCWFrontEnd(clean_profile, global_seed=77)
    cube = fe.synthesize(path_set)

    assert cube.same_tx_interval_s == pytest.approx(180.0e-6, abs=1e-15)
    assert cube.chirp_time_s.shape == (128, 3)

    # Exact 180 us same-TX interval and 60 us inter-TX spacing
    diffs_same_tx = np.diff(cube.chirp_time_s, axis=0)
    np.testing.assert_allclose(diffs_same_tx, 180.0e-6, atol=1e-15)
    np.testing.assert_allclose(cube.chirp_time_s[:, 1] - cube.chirp_time_s[:, 0], 60.0e-6, atol=1e-15)
    np.testing.assert_allclose(cube.chirp_time_s[:, 2] - cube.chirp_time_s[:, 1], 60.0e-6, atol=1e-15)

    # Chronological unpacking preserves exact 60 us spacing and [0, 1, 2] TX schedule
    chron, tx_ids, times = cube.as_chronological()
    assert chron.shape == (384, 4, 192)
    assert list(tx_ids[:6]) == [0, 1, 2, 0, 1, 2]
    np.testing.assert_allclose(np.diff(times), 60.0e-6, atol=1e-15)


# ---------------------------------------------------------------------------
# 4. Statistical thermal noise test (k * T * B * F)
# ---------------------------------------------------------------------------


def test_thermal_noise_matches_ktbf_budget_statistically(default_radar) -> None:
    noise_only_profile = _profile_with_only(default_radar, {"thermal_noise"})
    empty_paths = PathSet(frame_id=0, sim_time_s=0.0, paths=[], diagnostics={})
    fe = FMCWFrontEnd(noise_only_profile, global_seed=20261123, dtype="complex128")
    cube = fe.synthesize(empty_paths)

    # Independent analytical k*T*B*F -> normalized ADC full-scale variance
    k_b = 1.380649e-23
    t_k = 290.0
    b_eq = 2.3e6
    f_lin = 10.0 ** (12.0 / 10.0)
    p_n_ant = k_b * t_k * b_eq * f_lin
    g_rx = 10.0 ** (30.0 / 10.0)
    p_fs = (1.0 ** 2) / (8.0 * 50.0)
    expected_var_complex = (p_n_ant * g_rx) / p_fs

    budget = compute_thermal_noise_budget(noise_only_profile)
    assert budget.noise_variance_adc_complex == pytest.approx(expected_var_complex, rel=1e-12)

    samples = cube.complex_samples.reshape(-1)
    n_samples = samples.size
    emp_var_complex = float(np.mean(np.abs(samples) ** 2))
    emp_var_re = float(np.var(samples.real))
    emp_var_im = float(np.var(samples.imag))
    emp_mean_abs = float(abs(np.mean(samples)))
    corr_re_im = float(np.corrcoef(samples.real, samples.imag)[0, 1])

    # Statistical tolerances for N = 294,912 complex samples (1-sigma rel std of var ~ 0.002)
    assert abs(emp_var_complex - expected_var_complex) / expected_var_complex < 0.02
    assert abs(emp_var_re - 0.5 * expected_var_complex) / (0.5 * expected_var_complex) < 0.025
    assert abs(emp_var_im - 0.5 * expected_var_complex) / (0.5 * expected_var_complex) < 0.025
    assert emp_mean_abs < 5.0 * math.sqrt(expected_var_complex / n_samples)
    assert abs(corr_re_im) < 0.02


# ---------------------------------------------------------------------------
# 5. Independent ablation test for EVERY ONE of the 10 impairments
# ---------------------------------------------------------------------------


def test_ablation_all_ten_impairments_toggle_independently(default_scene, default_radar) -> None:
    clean_prof = _profile_with_only(default_radar, set())
    ps_target = _single_target_pathset(default_scene, clean_prof, range_m=3.00, radial_vel_mps=0.50, rcs_m2=0.5)
    ps_empty = PathSet(frame_id=0, sim_time_s=0.0, paths=[], diagnostics={})

    cube_clean = FMCWFrontEnd(clean_prof, global_seed=55, dtype="complex128").synthesize(ps_target)

    # 1. thermal_noise
    c_no_noise = FMCWFrontEnd(clean_prof, global_seed=55, dtype="complex128").synthesize(ps_empty)
    c_noise = FMCWFrontEnd(_profile_with_only(default_radar, {"thermal_noise"}), global_seed=55, dtype="complex128").synthesize(ps_empty)
    assert float(np.max(np.abs(c_no_noise.complex_samples))) == 0.0
    assert float(np.mean(np.abs(c_noise.complex_samples) ** 2)) > 0.0

    # 2. tx_leakage (near-DC strong tone matching -35 dBFS ~ 0.01778 FS)
    c_leak = FMCWFrontEnd(_profile_with_only(default_radar, {"tx_leakage"}), global_seed=55, dtype="complex128").synthesize(ps_empty)
    expected_leak_amp = 10.0 ** (-35.0 / 20.0)
    assert float(np.mean(np.abs(c_leak.complex_samples))) == pytest.approx(expected_leak_amp, rel=1e-3)

    # 3. tx_rx_channel_mismatch (perturbs inter-channel amplitude/phase ratios)
    c_ch = FMCWFrontEnd(_profile_with_only(default_radar, {"tx_rx_channel_mismatch"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    ratios_clean = np.abs(cube_clean.complex_samples[0, 0, :, 0]) / np.abs(cube_clean.complex_samples[0, 0, 0, 0])
    ratios_ch = np.abs(c_ch.complex_samples[0, 0, :, 0]) / np.abs(c_ch.complex_samples[0, 0, 0, 0])
    assert float(np.max(np.abs(ratios_ch - ratios_clean))) > 0.01

    # 4. iq_imbalance (creates measurable negative-frequency mirror image at -f_beat)
    c_iq = FMCWFrontEnd(_profile_with_only(default_radar, {"iq_imbalance"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    win = np.hanning(192)
    spec_clean = np.abs(np.fft.fft(cube_clean.complex_samples[0, 0, 0] * win, n=256))
    spec_iq = np.abs(np.fft.fft(c_iq.complex_samples[0, 0, 0] * win, n=256))
    pos_bin = int(np.argmax(spec_clean[:128]))
    neg_bin = (256 - pos_bin) % 256
    assert spec_iq[neg_bin] > 10.0 * spec_clean[neg_bin]

    # 5. adc_clipping (strong saturating path clips to <= 1.0 when ON, > 1.0 when OFF)
    strong_path = RFPath(
        path_id="strong", tx_id=0, rx_id=0, target_id="close", surface_id=None, path_type="los",
        vertices_m=((0, 0, 0), (0, 1, 0), (0, 0, 0)), tx_leg_length_m=1.0, rx_leg_length_m=1.0,
        total_path_length_m=2.0, delay_s=2.0 / 299792458.0, radial_velocity_mps=0.0, doppler_hz=0.0,
        phase_rad=0.0, path_loss_db=10.0, reflection_coeff_real=1.0, reflection_coeff_imag=0.0,
        rcs_m2=10.0, amplitude_linear=0.05, amplitude_phase_rad=0.0,
    )
    ps_strong = PathSet(frame_id=0, sim_time_s=0.0, paths=[strong_path])
    c_noclip = FMCWFrontEnd(clean_prof, global_seed=55, dtype="complex128").synthesize(ps_strong)
    c_clip = FMCWFrontEnd(_profile_with_only(default_radar, {"adc_clipping"}), global_seed=55, dtype="complex128").synthesize(ps_strong)
    assert float(np.max(np.abs(c_noclip.complex_samples.real))) > 1.0
    assert c_noclip.diagnostics["clipped_sample_count"] == 0
    assert float(np.max(np.abs(c_clip.complex_samples.real))) <= 1.0 + 1e-12
    assert c_clip.diagnostics["clipped_sample_count"] > 0

    # 6. adc_quantization (snaps all samples to exact 12-bit quantization grid 2 / 4096)
    c_quant = FMCWFrontEnd(_profile_with_only(default_radar, {"adc_quantization"}), global_seed=55, dtype="complex128").synthesize(ps_strong)
    step = 2.0 / (2 ** 12)
    residual_re = np.abs(c_quant.complex_samples.real / step - np.round(c_quant.complex_samples.real / step))
    assert float(np.max(residual_re)) < 1e-9
    assert not np.allclose(c_quant.complex_samples, c_noclip.complex_samples)

    # 7. phase_noise (perturbs signal phase & raises spectral skirt)
    c_pn = FMCWFrontEnd(_profile_with_only(default_radar, {"phase_noise"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    phase_diff_pn = np.angle(c_pn.complex_samples[:, 0, 0, :] * np.conj(cube_clean.complex_samples[:, 0, 0, :]))
    assert float(np.std(phase_diff_pn)) > 1e-4

    # 8. chirp_slope_error (perturbs beat phase across chirps)
    c_slope = FMCWFrontEnd(_profile_with_only(default_radar, {"chirp_slope_error"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    diff_slope = float(np.max(np.abs(c_slope.complex_samples - cube_clean.complex_samples)))
    assert diff_slope > 0.0

    # 9. chirp_start_frequency_error (perturbs chirp-to-chirp carrier phase 2*pi*df0*tau)
    c_f0 = FMCWFrontEnd(_profile_with_only(default_radar, {"chirp_start_frequency_error"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    phase_diff_f0 = np.angle(c_f0.complex_samples[:, 0, 0, 0] * np.conj(cube_clean.complex_samples[:, 0, 0, 0]))
    assert float(np.std(phase_diff_f0)) > 1e-3

    # 10. timing_jitter (perturbs beat phase/samples via trigger & ADC jitter)
    c_jit = FMCWFrontEnd(_profile_with_only(default_radar, {"timing_jitter"}), global_seed=55, dtype="complex128").synthesize(ps_target)
    diff_jit = float(np.max(np.abs(c_jit.complex_samples - cube_clean.complex_samples)))
    assert diff_jit > 0.0


# ---------------------------------------------------------------------------
# 6. Determinism & vendor-neutral profile check
# ---------------------------------------------------------------------------


def test_full_impairments_determinism_and_vendor_neutral_profile(default_scene, default_radar) -> None:
    ps = _single_target_pathset(default_scene, default_radar, range_m=3.00, radial_vel_mps=0.75)
    c1 = FMCWFrontEnd(default_radar, global_seed=20261123).synthesize(ps)
    c2 = FMCWFrontEnd(default_radar, global_seed=20261123).synthesize(ps)
    np.testing.assert_array_equal(c1.complex_samples, c2.complex_samples)
    np.testing.assert_array_equal(c1.chirp_time_s, c2.chirp_time_s)

    # Preview mode explicitly tags processing_mode="preview"
    c_prev = FMCWFrontEnd(default_radar, global_seed=20261123).synthesize(
        ps, processing_mode="preview", num_loops_override=32
    )
    assert c_prev.shape == (32, 3, 4, 192)
    assert c_prev.processing_mode == "preview"

    # Substituted 77 GHz 2-TX / 4-RX profile synthesizes with zero hardcoded TI assumptions
    raw = yaml.safe_load((ROOT / "radar_config.yaml").read_text(encoding="utf-8"))
    raw.pop("expected_derived", None)
    raw["profile"]["id"] = "generic_77ghz_2x4"
    raw["rf"]["carrier_frequency_hz"] = 77.0e9
    raw["rf"]["operating_band_hz"] = [76.0e9, 81.0e9]
    raw["rf"]["tx_count"] = 2
    raw["rf"]["tx_schedule"] = [0, 1]
    raw["waveform"]["start_frequency_hz"] = 76.5e9
    raw["waveform"]["chirps_per_tx_per_frame"] = 64
    raw["waveform"]["doppler_fft_size"] = 64
    raw["antenna_array"]["tx_positions_lambda"] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    alt_prof, _ = parse_radar_profile(raw, source="alt.yaml")
    ps_alt = _single_target_pathset(default_scene, alt_prof, range_m=3.00)
    c_alt = FMCWFrontEnd(alt_prof, global_seed=99).synthesize(ps_alt)
    assert c_alt.shape == (64, 2, 4, 192)
    assert c_alt.same_tx_interval_s == pytest.approx(120.0e-6, abs=1e-15)
