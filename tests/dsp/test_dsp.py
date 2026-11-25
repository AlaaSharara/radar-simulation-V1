"""Tests for T5: Range/Doppler FFT, TDM compensation, OS-CFAR, 4x3 angle, tracking, fall baseline, and eval."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from radarsim.config import load_motion_config, load_radar_profile
from radarsim.dsp import (
    DSPConfig,
    DSPProcessor,
    DetectionCluster,
    OSCFARConfig,
    RadarTracker,
    RuleBasedFallDetector,
    compute_range_doppler_cubes,
    make_window,
)
from radarsim.eval import evaluate_fall_baseline, evaluate_tracker
from radarsim.motion import generate_motion_sequence
from radarsim.radar import FMCWFrontEnd
from radarsim.rf import PointTarget, RFPropagationConfig, RFPropagationEngine
from radarsim.scene import load_scene

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def default_scene():
    return load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")


@pytest.fixture(scope="module")
def default_radar():
    return load_radar_profile(ROOT / "radar_config.yaml")


@pytest.fixture(scope="module")
def default_motion_cfg():
    return load_motion_config(ROOT / "motion_config.yaml")


def _clean_radar_with_noise(radar):
    prof = radar
    for name in (
        "phase_noise",
        "chirp_slope_error",
        "chirp_start_frequency_error",
        "tx_rx_channel_mismatch",
        "iq_imbalance",
        "adc_quantization",
        "adc_clipping",
        "tx_leakage",
        "timing_jitter",
    ):
        prof = prof.with_impairment_toggle(name, False)
    return prof.with_impairment_toggle("thermal_noise", True)


# ---------------------------------------------------------------------------
# 1. Window normalization, coherent gain, and 4x3 aperture limitations
# ---------------------------------------------------------------------------


def test_window_coherent_gain_and_unit_tone_amplitude() -> None:
    w_hann, stats_h = make_window("hann", 192)
    assert stats_h.coherent_gain == pytest.approx(0.5, abs=0.01)
    assert stats_h.enbw_bins == pytest.approx(1.5, abs=0.02)

    w_rect, stats_r = make_window("none", 192)
    assert stats_r.coherent_gain == pytest.approx(1.0, abs=1e-12)
    assert stats_r.enbw_bins == pytest.approx(1.0, abs=1e-12)

    # Unit-amplitude tone at exact FFT bin k=16 has magnitude 1.0 after normalized window + FFT
    n = np.arange(192, dtype=np.float64)
    tone = np.exp(1j * 2.0 * math.pi * 16.0 * n / 192.0)
    peak = float(np.max(np.abs(np.fft.fft(tone * w_hann, n=192))))
    assert peak == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Synthetic single-target analytical range, Doppler, 2D angle, and TDM compensation
# ---------------------------------------------------------------------------


def test_single_moving_target_range_doppler_angle_and_tdm_compensation(
    default_scene, default_radar
) -> None:
    radar = _clean_radar_with_noise(default_radar)
    rf_cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_second_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
    )
    rf_eng = RFPropagationEngine(default_scene, radar, config=rf_cfg, global_seed=10)
    fe = FMCWFrontEnd(radar, config_hash="dsp_test", global_seed=10, dtype="complex128")

    # Target at exact spherical coordinates: range=3.00 m, az=+20 deg, el=-10 deg, receding +1.50 m/s
    r_true = 3.00
    az_true = math.radians(20.0)
    el_true = math.radians(-10.0)
    vr_true = 1.50
    pos_world = default_scene.radar_frame.spherical_to_world_point(r_true, az_true, el_true)
    # Radial unit vector from radar to target
    r_vec = np.asarray(pos_world) - np.asarray(default_scene.radar_mount.position_m)
    u_rad = r_vec / float(np.linalg.norm(r_vec))
    vel_world = tuple(float(vr_true * v) for v in u_rad)

    pt = PointTarget(target_id="moving_pt", position_m=pos_world, velocity_mps=vel_world, rcs_m2=0.2)
    ps = rf_eng.propagate(0, 0.0, point_targets=[pt], include_scene_surfaces=False)
    cube = fe.synthesize(ps, processing_mode="full_sim")

    dsp = DSPProcessor(radar, default_scene.radar_frame)
    frame = dsp.process(cube)

    assert len(frame.detections) >= 1
    det = frame.detections[0]
    assert det.range_m == pytest.approx(r_true, abs=0.05)
    assert det.radial_velocity_mps == pytest.approx(vr_true, abs=0.12)
    assert math.degrees(det.azimuth_rad) == pytest.approx(20.0, abs=4.0)
    assert math.degrees(det.elevation_rad) == pytest.approx(-10.0, abs=6.0)
    assert det.snr_db > 15.0

    # Verify explicit 4x3 aperture limitations & zero-padding note in diagnostics
    lim = frame.diagnostics["aperture_limitations"]
    assert lim["virtual_channels"] == 12
    assert lim["azimuth_elements"] == 4
    assert lim["elevation_elements"] == 3
    assert 20.0 < lim["azimuth_beamwidth_3db_deg"] < 35.0
    assert 25.0 < lim["elevation_beamwidth_3db_deg"] < 45.0
    assert "does NOT improve" in lim["zero_padding_note"]

    # Verify TDM Doppler phase compensation reduces elevation bias at high radial speed (+3.5 m/s)
    vel_fast = tuple(float(3.50 * v) for v in u_rad)
    pt_fast = PointTarget(target_id="fast_pt", position_m=pos_world, velocity_mps=vel_fast, rcs_m2=0.5)
    ps_fast = rf_eng.propagate(1, 0.05, point_targets=[pt_fast], include_scene_surfaces=False)
    cube_fast = fe.synthesize(ps_fast, processing_mode="full_sim")

    dsp_comp = DSPProcessor(radar, default_scene.radar_frame, config=DSPConfig(apply_tdm_compensation=True))
    dsp_nocomp = DSPProcessor(radar, default_scene.radar_frame, config=DSPConfig(apply_tdm_compensation=False))
    f_comp = dsp_comp.process(cube_fast)
    f_nocomp = dsp_nocomp.process(cube_fast)
    err_comp = abs(math.degrees(f_comp.detections[0].elevation_rad) - (-10.0))
    err_nocomp = abs(math.degrees(f_nocomp.detections[0].elevation_rad) - (-10.0))
    assert err_comp < err_nocomp


# ---------------------------------------------------------------------------
# 3. Dual-branch clutter suppression vs untouched stationary phase branch
# ---------------------------------------------------------------------------


def test_clutter_suppression_preserves_untouched_stationary_breathing_branch(
    default_scene, default_radar
) -> None:
    radar = _clean_radar_with_noise(default_radar)
    rf_cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
    )
    rf_eng = RFPropagationEngine(default_scene, radar, config=rf_cfg, global_seed=22)
    fe = FMCWFrontEnd(radar, global_seed=22, dtype="complex128")

    # Strong static clutter at 2.0 m (v=0) + weaker moving target at 3.2 m (v=+1.0 m/s)
    p_static = default_scene.radar_frame.spherical_to_world_point(2.0, 0.0, 0.0)
    p_moving = default_scene.radar_frame.spherical_to_world_point(3.2, math.radians(10.0), 0.0)
    u_mov = (np.asarray(p_moving) - np.asarray(default_scene.radar_mount.position_m))
    u_mov /= float(np.linalg.norm(u_mov))

    ps = rf_eng.propagate(
        0,
        0.0,
        point_targets=[
            PointTarget("static_clutter", p_static, (0.0, 0.0, 0.0), rcs_m2=2.0),
            PointTarget("moving_target", p_moving, tuple(float(1.0 * v) for v in u_mov), rcs_m2=0.1),
        ],
        include_scene_surfaces=False,
    )
    cube = fe.synthesize(ps)
    spec = compute_range_doppler_cubes(cube, radar)

    # Static clutter bin at 2.0 m is suppressed > 35 dB in moving branch while preserved in stationary branch
    r_bin_static = int(np.argmin(np.abs(spec.range_axis_m - 2.0)))
    dc_bin = spec.rd_power_stationary.shape[0] // 2
    p_stat_dc = float(spec.rd_power_stationary[dc_bin, r_bin_static])
    p_mov_dc = float(spec.rd_power_moving[dc_bin, r_bin_static])
    assert p_stat_dc > 1e4 * max(p_mov_dc, 1e-30)

    # DSP clutter-suppressed branch detects the moving target at 3.2 m and ignores the static clutter
    dsp = DSPProcessor(radar, default_scene.radar_frame)
    frame0 = dsp.process(cube)
    assert any(abs(d.range_m - 3.2) < 0.12 for d in frame0.detections)
    assert not any(abs(d.range_m - 2.0) < 0.12 and abs(d.radial_velocity_mps) < 0.15 for d in frame0.detections)

    # Stationary breathing branch tracks a 1.0 mm range shift of a stationary target cleanly
    p_shift = default_scene.radar_frame.spherical_to_world_point(2.001, 0.0, 0.0)
    ps1 = rf_eng.propagate(
        1, 0.05, point_targets=[PointTarget("static_clutter", p_shift, (0.0, 0.0, 0.0), rcs_m2=2.0)], include_scene_surfaces=False
    )
    frame1 = dsp.process(fe.synthesize(ps1))
    br = frame1.features["breathing_phase_branch"]
    assert br["status"] == "optional_experimental_branch_not_validated_claim"
    assert abs(br["displacement_mm"] - 1.0) < 0.2


# ---------------------------------------------------------------------------
# 4. Transparent rule-based fall baseline vs hard negatives
# ---------------------------------------------------------------------------


def _run_synthetic_track_profile(z_series: list[float], dt_s: float = 0.05) -> list[bool]:
    """Feed synthetic radar-like cluster centroids through RadarTracker + RuleBasedFallDetector."""
    tracker = RadarTracker()
    detector = RuleBasedFallDetector()
    alerts: list[bool] = []
    for k, z in enumerate(z_series):
        t = k * dt_s
        vz = (z - z_series[k - 1]) / dt_s if k > 0 else 0.0
        cl = DetectionCluster(
            cluster_id=0,
            centroid_m=(1.5, 2.0, z),
            mean_radial_velocity_mps=vz * 0.3,
            max_abs_radial_velocity_mps=abs(vz) * 0.4,
            mean_snr_db=22.0,
            z_min_m=max(0.0, z - 0.15),
            z_max_m=z + 0.15,
            detection_indices=(0,),
        )
        tracker.update(t, [cl])
        alert, _ = detector.evaluate(t, tracker.internal_tracks())
        alerts.append(alert.active)
    return alerts


def test_rule_based_fall_baseline_separates_falls_from_hard_negatives() -> None:
    # 1. True fall: standing (1.05 m) -> rapid 0.5 s drop to floor (0.20 m) -> still 1.2 s
    fall_z = [1.05] * 20 + list(np.linspace(1.05, 0.20, 10)) + [0.20] * 25
    fall_alerts = _run_synthetic_track_profile(fall_z)
    assert any(fall_alerts), "Expected baseline rule to trigger on rapid floor fall + stillness"

    # 2. Hard negative: quick sit on chair/sofa (drops to 0.68 m, elevated above floor)
    sit_z = [1.05] * 20 + list(np.linspace(1.05, 0.68, 8)) + [0.68] * 25
    assert not any(_run_synthetic_track_profile(sit_z))

    # 3. Hard negative: bend to pick up object (brief dip to 0.56 m then returns to standing)
    bend_z = [1.05] * 20 + list(np.linspace(1.05, 0.56, 10)) + [0.56] * 6 + list(np.linspace(0.56, 1.05, 12)) + [1.05] * 15
    assert not any(_run_synthetic_track_profile(bend_z))

    # 4. Hard negative: intentional slow lying down on bed (slow 2.0 s transition to 0.72 m)
    lie_bed_z = [1.05] * 20 + list(np.linspace(1.05, 0.72, 40)) + [0.72] * 25
    assert not any(_run_synthetic_track_profile(lie_bed_z))

    # 5. Hard negative: controlled kneeling to floor & standing back up (slow descent, rises back up)
    kneel_z = [1.05] * 20 + list(np.linspace(1.05, 0.55, 22)) + [0.55] * 12 + list(np.linspace(0.55, 1.05, 20)) + [1.05] * 10
    assert not any(_run_synthetic_track_profile(kneel_z))


# ---------------------------------------------------------------------------
# 5. End-to-end radar-only pipeline (T2 -> T3 -> T4 -> T5) & separate eval metrics
# ---------------------------------------------------------------------------


def test_end_to_end_radar_only_dsp_responds_to_motion_without_gt_leakage(
    default_scene, default_radar, default_motion_cfg
) -> None:
    seq = generate_motion_sequence(
        default_scene,
        default_motion_cfg,
        seed=20261123,
        scenario_mode="normal_daily_routine",
        max_duration_scale=0.3,
    )
    # Sample 12 frames at 5 Hz during active walking inside the room
    walk_frames = [f for f in seq.frames if f.action_label == "walk" and f.root_position_m[1] > 0.6]
    selected = walk_frames[::20][:12]
    assert len(selected) >= 8

    rf_eng = RFPropagationEngine(
        default_scene,
        default_radar,
        config=RFPropagationConfig(preview_mode=True),
        global_seed=20261123,
    )
    fe = FMCWFrontEnd(default_radar, config_hash="e2e_t5_test", global_seed=20261123)
    dsp = DSPProcessor(default_radar, default_scene.radar_frame)

    dsp_frames = []
    for mf in selected:
        ps = rf_eng.propagate(mf.frame_id, mf.sim_time_s, motion_frame=mf)
        cube = fe.synthesize(ps, processing_mode="preview", num_loops_override=32)
        df = dsp.process(cube)
        dsp_frames.append(df)

    # Verify radar-only detections, occupancy provenance, JSON payloads, and motion response
    assert any(len(df.detections) > 0 for df in dsp_frames)
    assert all(df.occupancy.provenance == "radar_derived" for df in dsp_frames)
    assert json.dumps(dsp_frames[-1].to_websocket_radar_dict())
    assert json.dumps(dsp_frames[-1].to_websocket_analytics_dict())

    # Evaluate separately via radarsim.eval
    trk_metrics = evaluate_tracker(dsp_frames, selected)
    fall_metrics = evaluate_fall_baseline(dsp_frames, selected)

    assert trk_metrics.detection_rate > 0.5
    assert trk_metrics.ground_truth_leakage_detected is False
    assert fall_metrics.ground_truth_has_fall is False
    assert fall_metrics.outcome == "TN"


def test_range_azimuth_bscan_map_and_end_to_end_fall_detection(
    default_scene, default_radar, default_motion_cfg
) -> None:
    """Verify Range-Azimuth B-scan peak accuracy and end-to-end fall_demo seed 7001 TP without GT leakage."""
    radar = _clean_radar_with_noise(default_radar)
    rf_eng = RFPropagationEngine(
        default_scene,
        radar,
        config=RFPropagationConfig(
            enable_los=True,
            enable_first_order_reflections=False,
            enable_transmission=False,
            enable_diffuse_scattering=False,
        ),
        global_seed=11,
    )
    fe = FMCWFrontEnd(radar, global_seed=11, dtype="complex128")
    dsp = DSPProcessor(radar, default_scene.radar_frame)

    r_true = 3.00
    az_true = math.radians(20.0)
    pos = default_scene.radar_frame.spherical_to_world_point(r_true, az_true, 0.0)
    r_vec = np.asarray(pos) - np.asarray(default_scene.radar_mount.position_m)
    u_rad = r_vec / float(np.linalg.norm(r_vec))
    pt = PointTarget("az_pt", pos, tuple(float(1.2 * v) for v in u_rad), rcs_m2=0.25)
    cube = fe.synthesize(rf_eng.propagate(0, 0.0, point_targets=[pt], include_scene_surfaces=False))
    df = dsp.process(cube)

    assert df.range_angle.range_azimuth_db is not None
    ra_db = df.range_angle.range_azimuth_db
    r_ax = df.range_angle.range_m
    az_ax = df.range_angle.azimuth_rad
    assert r_ax is not None and ra_db.shape == (len(r_ax), len(az_ax))
    flat = int(np.argmax(ra_db))
    ir, iaz = divmod(flat, ra_db.shape[1])
    assert abs(float(r_ax[ir]) - r_true) < 0.15
    assert abs(math.degrees(float(az_ax[iaz])) - 20.0) < 4.5
    ws_analytics = df.to_websocket_analytics_dict()
    assert "range_azimuth_db" in ws_analytics["range_angle"]
    assert "range_m" in ws_analytics["range_angle"]

    # End-to-end fall_demo (seed=7001, fall_forward) over the pre-fall -> fall -> settling window
    seq = generate_motion_sequence(
        default_scene,
        default_motion_cfg,
        seed=7001,
        scenario_mode="fall_demo",
        fall_style="fall_forward",
        max_duration_scale=0.6,
    )
    sampled = seq.sample_at_rate(20.0)
    window_frames = sampled[220:]  # t = 11.0s .. 16.35s covering walk, fall, impact, stillness

    from radarsim.pipeline import SimulationPipeline

    pipe = SimulationPipeline(ROOT)
    res = pipe.run_scenario(
        seed=7001,
        scenario_mode="fall_demo",
        fall_style="fall_forward",
        mode="preview",
        start_frame=220,
        adc_window_frames=8,
    )
    assert res.total_frames == len(window_frames)
    assert res.fall_eval.ground_truth_has_fall is True
    assert res.fall_eval.outcome == "TP", f"Expected TP on fall_demo window, got {res.fall_eval.to_dict()}"
    assert res.tracker_eval.ground_truth_leakage_detected is False
