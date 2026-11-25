"""Tests for T3: LOS/occlusion, image reflections, Fresnel/TMM passivity, human RCS, and PathSet."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from radarsim.config import load_motion_config, load_radar_profile
from radarsim.motion import BodyCapsule, generate_motion_sequence
from radarsim.rf import (
    PointTarget,
    RFPropagationConfig,
    RFPropagationEngine,
    evaluate_surface_em,
    find_first_order_reflection,
    fresnel_single_interface,
    multilayer_reflection_transmission,
    rayleigh_roughness_factor,
)
from radarsim.scene import SurfaceLayer, load_scene

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


# ---------------------------------------------------------------------------
# 1. Analytical direct LOS delay 2R/c and Doppler sign
# ---------------------------------------------------------------------------


def test_direct_los_path_at_3m_matches_analytical_2r_over_c_and_doppler(
    default_scene, default_radar
) -> None:
    cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_second_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
    )
    engine = RFPropagationEngine(default_scene, default_radar, config=cfg, global_seed=42)

    # Place target exactly 3.00 m from TX0 / RX0 (which both sit at radar_position_m)
    r_target = 3.00
    tx0 = engine.tx_world[0]
    bx, by, bz = default_scene.radar_frame.boresight_world
    target_pos = (tx0[0] + r_target * bx, tx0[1] + r_target * by, tx0[2] + r_target * bz)
    # Receding velocity +1.00 m/s along boresight
    target_vel = (1.0 * bx, 1.0 * by, 1.0 * bz)

    target = PointTarget(
        target_id="pt_3m",
        position_m=target_pos,
        velocity_mps=target_vel,
        rcs_m2=0.1,
    )
    path_set = engine.propagate(
        frame_id=0,
        sim_time_s=0.0,
        point_targets=[target],
        include_scene_surfaces=False,
    )

    assert len(path_set.paths) == 12  # 3 TX x 4 RX
    p00 = path_set.paths_for_pair(0, 0)[0]

    # Independent analytical expectations
    c = 299792458.0
    lam = c / 61.0e9
    expected_delay_s = 2.0 * r_target / c
    expected_doppler_hz = -2.0 * 1.0 / lam

    assert p00.path_type == "los"
    assert p00.tx_leg_length_m == pytest.approx(3.00, abs=1e-12)
    assert p00.rx_leg_length_m == pytest.approx(3.00, abs=1e-12)
    assert p00.total_path_length_m == pytest.approx(6.00, abs=1e-12)
    assert p00.delay_s == pytest.approx(expected_delay_s, rel=0.0, abs=1e-15)
    assert p00.radial_velocity_mps == pytest.approx(1.00, abs=1e-12)
    assert p00.doppler_hz == pytest.approx(expected_doppler_hz, abs=1e-9)
    assert p00.amplitude_linear > 0.0
    assert abs(p00.complex_amplitude) == pytest.approx(p00.amplitude_linear, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. First-order image-method wall reflection vs independent analytical geometry
# ---------------------------------------------------------------------------


def test_wall_reflection_matches_independent_image_geometry(default_scene, default_radar) -> None:
    tx0 = default_scene.radar_mount.position_m  # (2.0, 0.08, 2.25)
    target = (2.0, 2.50, 1.50)

    # Locate x_min wall (x = 0, inward normal = (+1, 0, 0))
    wall_x_min = next(s for s in default_scene.surfaces_by_kind("wall") if s.metadata["wall"] == "x_min")
    refl = find_first_order_reflection(tx0, target, wall_x_min, default_scene.surfaces)
    assert refl is not None

    # Independent analytical image solution for plane x = 0:
    # Mirror of TX0 = (-2.0, 0.08, 2.25); line to (2.0, 2.50, 1.50) crosses x = 0 at u = 0.5
    q_expected = (0.0, 0.5 * (0.08 + 2.50), 0.5 * (2.25 + 1.50))
    leg1_expected = math.dist(tx0, q_expected)
    leg2_expected = math.dist(q_expected, target)
    total_expected = math.dist((-2.0, 0.08, 2.25), target)
    cos_inc_expected = abs((q_expected[0] - tx0[0]) / leg1_expected)
    inc_expected = math.acos(cos_inc_expected)

    assert refl.bounce_point_m == pytest.approx(q_expected, abs=1e-12)
    assert refl.leg1_length_m == pytest.approx(leg1_expected, abs=1e-12)
    assert refl.leg2_length_m == pytest.approx(leg2_expected, abs=1e-12)
    assert (refl.leg1_length_m + refl.leg2_length_m) == pytest.approx(total_expected, abs=1e-12)
    assert refl.incidence_angle_rad == pytest.approx(inc_expected, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Blocked LOS removal & finite-thickness slab transmission
# ---------------------------------------------------------------------------


def test_blocked_los_is_removed_and_transmitted_path_attenuated(
    default_scene, default_radar
) -> None:
    # Place target behind wardrobe_01 (wardrobe occupies x in [3.12, 3.82], y in [0.48, 1.58], z in [0, 2.10])
    # Ray from radar (2.0, 0.08, 2.25) to (3.88, 1.20, 0.90) crosses wardrobe_01
    blocked_target = PointTarget(
        target_id="behind_wardrobe",
        position_m=(3.88, 1.20, 0.90),
        rcs_m2=0.1,
    )

    # First check with transmission OFF -> LOS must be completely removed
    engine_no_tx = RFPropagationEngine(
        default_scene,
        default_radar,
        config=RFPropagationConfig(
            enable_los=True,
            enable_first_order_reflections=False,
            enable_transmission=False,
            enable_diffuse_scattering=False,
        ),
    )
    ps_blocked = engine_no_tx.propagate(0, 0.0, point_targets=[blocked_target], tx_ids=[0], rx_ids=[0])
    assert len(ps_blocked.paths_by_type("los")) == 0
    assert ps_blocked.diagnostics["occluded_los_count"] == 1

    # Now enable transmission -> transmitted path appears and is strictly weaker than free-space LOS
    engine_tx = RFPropagationEngine(
        default_scene,
        default_radar,
        config=RFPropagationConfig(
            enable_los=True,
            enable_first_order_reflections=False,
            enable_transmission=True,
            enable_diffuse_scattering=False,
        ),
    )
    ps_trans = engine_tx.propagate(0, 0.0, point_targets=[blocked_target], tx_ids=[0], rx_ids=[0])
    ps_free = engine_tx.propagate(
        0, 0.0, point_targets=[blocked_target], tx_ids=[0], rx_ids=[0], include_scene_surfaces=False
    )
    trans_paths = ps_trans.paths_by_type("transmitted")
    free_paths = ps_free.paths_by_type("los")
    assert len(trans_paths) == 1
    assert len(free_paths) == 1
    assert trans_paths[0].amplitude_linear < free_paths[0].amplitude_linear
    assert abs(trans_paths[0].reflection_coeff) < 1.0

    # Human capsule occlusion check: place target directly behind a torso capsule
    mid_cap = BodyCapsule(
        segment_id="blocker_torso",
        a_m=(2.0, 1.50, 1.40),
        b_m=(2.0, 1.50, 2.00),
        radius_m=0.18,
        material_id="human_clothing",
    )
    behind_human = PointTarget(target_id="behind_person", position_m=(2.0, 3.00, 1.40), rcs_m2=0.1)
    # Build minimal mock motion frame with that capsule
    from radarsim.motion import MotionFrame, JOINT_NAMES

    dummy_frame = MotionFrame(
        frame_id=0,
        sim_time_s=0.0,
        actor_id="person_01",
        action_label="pause",
        scenario_mode="normal_daily_routine",
        fall_event_id=None,
        root_position_m=(2.0, 1.5, 0.0),
        root_yaw_rad=0.0,
        joints_m={name: (2.0, 1.5, 1.0) for name in JOINT_NAMES},
        body_capsules=[mid_cap],
        contacts=[],
        avatar_asset_id="adult_male_default",
    )
    ps_human_occ = engine_no_tx.propagate(
        0, 0.0, motion_frame=dummy_frame, point_targets=[behind_human], tx_ids=[0], rx_ids=[0], include_scene_surfaces=False
    )
    assert not any(p.target_id == "behind_person" for p in ps_human_occ.paths_by_type("los"))


# ---------------------------------------------------------------------------
# 4. Complex Fresnel TE/TM & Transfer-Matrix energy conservation / passivity
# ---------------------------------------------------------------------------


def test_fresnel_brewster_halfwave_and_multilayer_passivity(default_scene) -> None:
    c = 299792458.0
    f_hz = 61.0e9
    lam = c / f_hz

    # Lossless dielectric eps_r = 4.0 (n = 2.0)
    eps4 = 4.0 + 0.0j
    assert abs(fresnel_single_interface(eps4, 0.0, polarization="TE")) == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert abs(fresnel_single_interface(eps4, 0.0, polarization="TM")) == pytest.approx(1.0 / 3.0, abs=1e-12)

    # Exact Brewster angle theta_B = atan(2) -> Gamma_TM = 0
    theta_b = math.atan(2.0)
    assert abs(fresnel_single_interface(eps4, theta_b, polarization="TM")) < 1e-14

    # Half-wave lossless slab -> zero reflection, unit transmission
    d_half = lam / (2.0 * math.sqrt(4.0))
    r_hw, t_hw = multilayer_reflection_transmission([eps4], [d_half], 0.0, f_hz, polarization="TE")
    assert abs(r_hw) < 1e-12
    assert abs(t_hw) == pytest.approx(1.0, abs=1e-12)

    # Energy conservation for arbitrary lossless 3-layer stack: |R|^2 + |T|^2 == 1
    for pol in ("TE", "TM"):
        for deg in (0.0, 20.0, 45.0, 70.0):
            r, t = multilayer_reflection_transmission(
                [2.94 + 0j, 1.0 + 0j, 2.94 + 0j],
                [0.0127, 0.0889, 0.0127],
                math.radians(deg),
                f_hz,
                polarization=pol,
            )
            assert abs(r) ** 2 + abs(t) ** 2 == pytest.approx(1.0, abs=1e-11)

    # Passive bounds on ALL configured room materials & default 3-layer cavity wall
    cavity_layers = (
        SurfaceLayer("plasterboard", 0.0127),
        SurfaceLayer("air", 0.0889),
        SurfaceLayer("plasterboard", 0.0127),
    )
    for pol in ("TE", "TM"):
        for deg in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0):
            resp = evaluate_surface_em(
                cavity_layers, default_scene.materials, math.radians(deg), f_hz, polarization=pol
            )
            assert 0.0 <= abs(resp.reflection_smooth) <= 1.0 + 1e-12
            assert 0.0 <= abs(resp.reflection_coherent) <= abs(resp.reflection_smooth) + 1e-12
            assert 0.0 <= abs(resp.transmission) <= 1.0 + 1e-12
            assert abs(resp.reflection_smooth) ** 2 + abs(resp.transmission) ** 2 < 1.0

    # Thick lossy concrete slab (15 cm): strictly passive and strongly attenuates transmission
    conc_resp = evaluate_surface_em(
        (SurfaceLayer("concrete", 0.15),), default_scene.materials, 0.0, f_hz, polarization="TE"
    )
    assert 0.0 < abs(conc_resp.reflection_smooth) < 1.0
    assert abs(conc_resp.transmission) < 0.05

    # PEC metal boundary: |R_smooth| == 1, T == 0
    metal_resp = evaluate_surface_em(
        (SurfaceLayer("metal", 0.001),), default_scene.materials, 0.3, f_hz, polarization="TE"
    )
    assert abs(metal_resp.reflection_smooth) == pytest.approx(1.0, abs=1e-12)
    assert abs(metal_resp.transmission) == 0.0

    # Rayleigh roughness factor in (0, 1]
    assert rayleigh_roughness_factor(0.0, 0.0, lam) == 1.0
    assert 0.0 < rayleigh_roughness_factor(0.0005, 0.0, lam) < 1.0


# ---------------------------------------------------------------------------
# 5. Full room scene with human MotionFrame: LOS, reflection, diffuse, 2nd-order & reproducibility
# ---------------------------------------------------------------------------


def test_full_scene_pathset_contains_los_reflection_diffuse_and_is_reproducible(
    default_scene, default_radar, default_motion_cfg
) -> None:
    seq = generate_motion_sequence(
        default_scene, default_motion_cfg, seed=20261123, scenario_mode="normal_daily_routine", max_duration_scale=0.3
    )
    # Pick a frame after the person has entered well inside the room
    inside_frame = next(f for f in seq.frames if f.root_position_m[1] > 1.2)

    cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=True,
        enable_second_order_reflections=True,
        max_second_order_paths=12,
        enable_transmission=True,
        enable_diffuse_scattering=True,
        preview_mode=True,
    )
    eng_a = RFPropagationEngine(default_scene, default_radar, config=cfg, global_seed=20261123)
    eng_b = RFPropagationEngine(default_scene, default_radar, config=cfg, global_seed=20261123)

    ps_a = eng_a.propagate(inside_frame.frame_id, inside_frame.sim_time_s, motion_frame=inside_frame)
    ps_b = eng_b.propagate(inside_frame.frame_id, inside_frame.sim_time_s, motion_frame=inside_frame)

    # All three core path families appear
    counts = ps_a.diagnostics["counts_by_type"]
    assert counts["los"] > 0
    assert counts["specular_reflection"] > 0
    assert counts["diffuse_scatter"] > 0
    assert 0 < ps_a.diagnostics["second_order_count"] <= 12
    assert ps_a.diagnostics["model_disclaimer"] == "geometric_optics_and_stochastic_diffuse_not_full_wave_em"

    # Second-order switch OFF produces zero second-order paths
    eng_no2 = RFPropagationEngine(
        default_scene,
        default_radar,
        config=RFPropagationConfig(enable_second_order_reflections=False, preview_mode=True),
        global_seed=20261123,
    )
    ps_no2 = eng_no2.propagate(inside_frame.frame_id, inside_frame.sim_time_s, motion_frame=inside_frame)
    assert ps_no2.diagnostics["second_order_count"] == 0

    # Passive reflection magnitude bound on every emitted path
    for path in ps_a.paths:
        assert abs(path.reflection_coeff) <= 1.0 + 1e-9
        assert path.delay_s > 0.0
        assert path.model_flags.get("full_wave_em") is False

    # Same seed reproduces identical PathSet
    assert ps_a.to_dict() == ps_b.to_dict()

    # Pure JSON serialization (no complex numbers)
    encoded = json.dumps(ps_a.to_dict(max_paths=50))
    assert "geometric_optics" in encoded

    # Different seed changes stochastic diffuse / scatterer realizations
    eng_c = RFPropagationEngine(default_scene, default_radar, config=cfg, global_seed=999999)
    ps_c = eng_c.propagate(inside_frame.frame_id, inside_frame.sim_time_s, motion_frame=inside_frame)
    assert ps_c.to_dict() != ps_a.to_dict()


def test_human_capsule_rcs_is_aspect_dependent(default_scene, default_radar) -> None:
    from radarsim.motion import JOINT_NAMES, MotionFrame

    cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
        enable_human_body_occlusion=False,
        preview_mode=True,
    )
    engine = RFPropagationEngine(default_scene, default_radar, config=cfg, global_seed=123)
    radar_p = default_scene.radar_mount.position_m
    bx, by, bz = default_scene.radar_frame.boresight_world
    center = (radar_p[0] + 2.5 * bx, radar_p[1] + 2.5 * by, radar_p[2] + 2.5 * bz)

    # Broadside capsule (vertical axis perpendicular to nearly-horizontal boresight) vs end-on capsule (along boresight)
    cap_broad = BodyCapsule("torso", (center[0], center[1], center[2] - 0.2), (center[0], center[1], center[2] + 0.2), 0.12, "human_clothing", -15.0)
    cap_endon = BodyCapsule("torso", (center[0] - 0.2 * bx, center[1] - 0.2 * by, center[2] - 0.2 * bz), (center[0] + 0.2 * bx, center[1] + 0.2 * by, center[2] + 0.2 * bz), 0.12, "human_clothing", -15.0)

    def _frame(cap: BodyCapsule) -> MotionFrame:
        return MotionFrame(
            frame_id=0, sim_time_s=0.0, actor_id="p", action_label="pause", scenario_mode="normal_daily_routine",
            fall_event_id=None, root_position_m=(center[0], center[1], 0.0), root_yaw_rad=0.0,
            joints_m={k: center for k in JOINT_NAMES}, body_capsules=[cap], contacts=[], avatar_asset_id="adult_male_default",
        )

    ps_broad = engine.propagate(0, 0.0, motion_frame=_frame(cap_broad), tx_ids=[0], rx_ids=[0], include_scene_surfaces=False)
    ps_endon = engine.propagate(0, 0.0, motion_frame=_frame(cap_endon), tx_ids=[0], rx_ids=[0], include_scene_surfaces=False)
    det_broad = next(p for p in ps_broad.paths if not p.model_flags["is_stochastic_scatterer"])
    det_endon = next(p for p in ps_endon.paths if not p.model_flags["is_stochastic_scatterer"])
    assert det_broad.rcs_m2 is not None and det_endon.rcs_m2 is not None
    assert det_broad.rcs_m2 > det_endon.rcs_m2


def test_tmm_tm_pec_backing_continuity_and_reciprocal_multipath(default_scene, default_radar) -> None:
    """Verify TM PEC limit continuity (d -> 0), SurfaceRayAccelerator parity, and reciprocal target multipath."""
    from radarsim.rf.raytrace import SurfaceRayAccelerator, find_segment_surface_hits

    f_hz = 61.0e9
    for pol in ("TE", "TM"):
        r0, t0 = multilayer_reflection_transmission([1.0 + 0.0j], [0.0], 0.35, f_hz, polarization=pol, layer_is_pec=[True])
        r_thin, t_thin = multilayer_reflection_transmission(
            [2.94 + 0.0j, 1.0 + 0.0j], [1e-9, 0.0], 0.35, f_hz, polarization=pol, layer_is_pec=[False, True]
        )
        assert abs(r0 - (-1.0 + 0.0j)) < 1e-12
        assert abs(r_thin - r0) < 1e-5
        assert abs(t0) == 0.0 and abs(t_thin) == 0.0

    accel = SurfaceRayAccelerator(default_scene.surfaces)
    segs = [
        ((2.0, 0.08, 2.25), (3.88, 1.20, 0.90)),
        ((2.0, 0.08, 2.25), (1.0, 2.50, 1.20)),
        ((0.5, 0.5, 0.5), (3.5, 4.5, 2.0)),
    ]
    for a, b in segs:
        h_ref = find_segment_surface_hits(a, b, default_scene.surfaces)
        h_acc = accel.find_segment_hits(a, b)
        assert [h.surface.surface_id for h in h_acc] == [h.surface.surface_id for h in h_ref]
        assert accel.is_segment_blocked(a, b) == bool(h_ref)

    eng = RFPropagationEngine(
        default_scene,
        default_radar,
        config=RFPropagationConfig(
            enable_los=True,
            enable_first_order_reflections=True,
            enable_target_multipath=True,
            enable_diffuse_scattering=False,
        ),
    )
    pt = PointTarget("person_proxy", (2.0, 2.4, 1.0), velocity_mps=(0.0, 0.5, 0.0), rcs_m2=0.25)
    ps = eng.propagate(0, 0.0, point_targets=[pt], tx_ids=[0], rx_ids=[0])
    legs = {
        p.model_flags.get("multipath_leg")
        for p in ps.paths
        if p.target_id == "person_proxy" and p.path_type == "specular_reflection"
    }
    assert "tx_surface_target_rx" in legs
    assert "tx_target_surface_rx" in legs
