"""Tests for T2: rigged human GLB avatar, 17-joint 100 Hz motion, A* navigation, falls, and 20+ seeds."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from radarsim.config import load_motion_config
from radarsim.motion import (
    AVATAR_ASSET_ID,
    AStarNavigator,
    FALL_PHASES,
    FALL_STYLES,
    JOINT_NAMES,
    build_procedural_human_glb,
    check_body_capsule_collisions,
    generate_motion_sequence,
    validate_glb_asset,
)
from radarsim.scene import load_scene

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def default_scene():
    return load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")


@pytest.fixture(scope="module")
def default_motion_cfg():
    return load_motion_config(ROOT / "motion_config.yaml")


# ---------------------------------------------------------------------------
# 1. Browser-ready full-body human avatar asset & rig mapping
# ---------------------------------------------------------------------------


def test_browser_ready_human_avatar_glb_and_rig_spec() -> None:
    avatar_dir = ROOT / "assets" / "avatars"
    glb_path = avatar_dir / "adult_male_default.glb"
    rig_path = avatar_dir / "adult_male_default_rig.json"
    attr_path = avatar_dir / "ATTRIBUTION.md"

    if not glb_path.exists():
        build_procedural_human_glb(glb_path, rig_path)

    assert glb_path.exists()
    assert rig_path.exists()
    assert attr_path.exists()

    report = validate_glb_asset(glb_path, rig_path)
    assert report["valid"] is True
    assert report["joint_count"] == 17
    assert report["material_count"] == 5
    assert report["vertex_count"] >= 2000
    assert report["triangle_count"] >= 4000
    assert 1.65 <= report["height_m"] <= 1.90

    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    assert rig["avatar_asset_id"] == AVATAR_ASSET_ID
    assert rig["draw_skeleton"] is False
    assert rig["hide_bones"] is True
    assert set(rig["joint_names"]) == set(JOINT_NAMES)
    mat = np.asarray(rig["coordinate_convention"]["world_to_gltf_matrix"], dtype=np.float64)
    assert np.linalg.det(mat) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Collision-aware A* navigation
# ---------------------------------------------------------------------------


def test_astar_navigation_plans_collision_free_path_around_furniture(default_scene) -> None:
    nav = AStarNavigator(default_scene, resolution_m=0.10, body_radius_m=0.15, obstacle_clearance_m=0.12)

    # Plan between two reachable points in the default room
    start = (0.9, 0.55)
    goal = (2.50, 1.15)
    path = nav.plan_path(start, goal)
    assert path is not None
    assert len(path) >= 2
    assert path[0] == start
    assert math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) < 0.15
    for i in range(len(path) - 1):
        assert nav.is_segment_clear(path[i], path[i + 1])

    # Verify A* genuinely detours around a blocking obstacle (table_01 sits on straight line)
    detour_start = (0.60, 2.50)
    detour_goal = (2.50, 1.20)
    detour_path = nav.plan_path(detour_start, detour_goal)
    assert detour_path is not None
    for i in range(len(detour_path) - 1):
        assert nav.is_segment_clear(detour_path[i], detour_path[i + 1])

    # Physically blocked pocket behind bed_01/table_01/chair_01/wardrobe_01 (gaps <= 0.18 m)
    # must be rejected rather than cutting through furniture
    assert nav.plan_path((0.9, 0.55), (2.05, 4.35)) is None


# ---------------------------------------------------------------------------
# 3. Entry through doorway, 100 Hz 17-joint contract, collisions & contacts
# ---------------------------------------------------------------------------


def test_normal_daily_routine_entry_collisions_contacts_and_hard_negatives(
    default_scene, default_motion_cfg
) -> None:
    seq = generate_motion_sequence(
        default_scene,
        default_motion_cfg,
        seed=20261123,
        scenario_mode="normal_daily_routine",
    )

    assert seq.sample_rate_hz == 100.0
    assert seq.avatar_asset_id == AVATAR_ASSET_ID
    assert seq.fall_events == []
    assert len(seq.frames) > 300

    # Starts outside doorway (y < 0), passes through front_door x-span, enters room (y >= 0.5)
    door = default_scene.get_opening("front_door")
    s0, s1 = door.span_along_wall_m
    assert seq.frames[0].root_position_m[1] < 0.0
    assert seq.frames[0].action_label == "enter"
    crossing = [f for f in seq.frames if abs(f.root_position_m[1]) < 0.08]
    assert len(crossing) > 0
    for cf in crossing:
        assert s0 <= cf.root_position_m[0] <= s1
    assert any(f.root_position_m[1] >= 0.5 for f in seq.frames)

    # Hard negatives are present and NEVER labeled as falls
    assert any(f.is_hard_negative for f in seq.frames)
    assert all(f.fall_event_id is None for f in seq.frames)

    # Verify strict 100 Hz timing, 17 joints, 14 capsules, smooth continuity, and full-body collisions
    event_allowed_furn: dict[int, str] = {}
    for ev in seq.events:
        if ev.target_object_id is not None:
            for fid in range(ev.start_frame, ev.end_frame + 1):
                event_allowed_furn[fid] = ev.target_object_id

    for idx, frame in enumerate(seq.frames):
        assert frame.frame_id == idx
        assert frame.sim_time_s == pytest.approx(idx * 0.01, abs=1e-6)
        assert set(frame.joints_m.keys()) == set(JOINT_NAMES)
        assert len(frame.body_capsules) == 14

        if idx > 0:
            # Verify no teleportation between consecutive 100 Hz frames
            prev_p = seq.frames[idx - 1].joints_m["pelvis"]
            cur_p = frame.joints_m["pelvis"]
            step_m = math.dist(prev_p, cur_p)
            assert step_m < 0.06, f"Teleport jump of {step_m:.3f} m at frame {idx} ({frame.action_label})"

        # Sample full-body capsule collision checks every 5 frames (20 Hz)
        if idx % 5 == 0:
            allowed = (event_allowed_furn[idx],) if idx in event_allowed_furn else ()
            violations = check_body_capsule_collisions(
                frame.body_capsules,
                default_scene,
                allowed_furniture_ids=allowed,
                allow_doorway_exterior=(frame.action_label == "enter"),
            )
            assert not violations, f"Frame {idx} ({frame.action_label}): {violations}"

    # Ground-truth JSON serialization check
    encoded = json.dumps(seq.frames[0].to_ground_truth_dict())
    assert AVATAR_ASSET_ID in encoded

    # Downsampling to 20 Hz radar rate works deterministically
    frames_20hz = seq.sample_at_rate(20.0)
    assert len(frames_20hz) >= 10
    assert frames_20hz[1].sim_time_s - frames_20hz[0].sim_time_s == pytest.approx(0.05, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. All 5 contact-aware fall styles & 5 phases (no pose teleportation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fall_style", FALL_STYLES)
def test_all_five_fall_styles_have_five_phases_contacts_and_no_teleportation(
    default_scene, default_motion_cfg, fall_style: str
) -> None:
    seq = generate_motion_sequence(
        default_scene,
        default_motion_cfg,
        seed=7001,
        scenario_mode="fall_demo",
        fall_style=fall_style,
        max_duration_scale=0.5,
    )

    assert len(seq.fall_events) == 1
    fall_ev = seq.fall_events[0]
    assert fall_ev["fall_style"] == fall_style
    fall_id = fall_ev["fall_event_id"]

    # All 5 phases present in strictly monotonic chronological order
    phase_ts = fall_ev["phase_timestamps_s"]
    assert list(phase_ts.keys()) == list(FALL_PHASES)
    ts_vals = [phase_ts[p] for p in FALL_PHASES]
    assert all(ts_vals[i] < ts_vals[i + 1] for i in range(len(ts_vals) - 1))

    # Extract fall frames and check continuous descent + impact contact
    fall_frames = [f for f in seq.frames if f.fall_event_id == fall_id]
    assert len(fall_frames) > 100
    assert fall_frames[0].sim_time_s == pytest.approx(fall_ev["start_time_s"], abs=1e-6)
    assert fall_frames[-1].sim_time_s == pytest.approx(fall_ev["end_time_s"], abs=1e-6)

    pelvis_z_start = fall_frames[0].joints_m["pelvis"][2]
    pelvis_z_min = min(f.joints_m["pelvis"][2] for f in fall_frames)
    assert pelvis_z_start > 0.80
    # Sideways fall rests on shoulder width (~0.23 m pelvis centerline); supine/prone ~0.12 m
    assert pelvis_z_min < 0.28

    # Continuous descent: no teleportation jump across fall frames
    for i in range(1, len(fall_frames)):
        dp = math.dist(fall_frames[i - 1].joints_m["pelvis"], fall_frames[i].joints_m["pelvis"])
        assert dp < 0.055, f"{fall_style}: pelvis jumped {dp:.3f} m in 10 ms"

    # Unintentional floor impact contact recorded at impact_contact phase
    impact_frames = [f for f in fall_frames if f.action_phase == "impact_contact"]
    assert len(impact_frames) >= 5
    assert any(
        any((not c.intentional) and c.surface_kind == "floor" and c.impact_speed_mps > 0.0 for c in f.contacts)
        for f in impact_frames
    )

    # Full-body capsules during fall stay inside walls and clear of furniture
    for f in fall_frames[::5]:
        violations = check_body_capsule_collisions(f.body_capsules, default_scene)
        assert not violations, f"{fall_style} frame {f.frame_id}: {violations}"


# ---------------------------------------------------------------------------
# 5. Determinism & 20+ seeds variation across random_mix
# ---------------------------------------------------------------------------


def test_fixed_seed_reproducibility_and_24_seed_diversity(default_scene, default_motion_cfg) -> None:
    # Exact reproducibility for fixed seed
    seq_a = generate_motion_sequence(default_scene, default_motion_cfg, seed=20261123, scenario_mode="random_mix", max_duration_scale=0.35)
    seq_b = generate_motion_sequence(default_scene, default_motion_cfg, seed=20261123, scenario_mode="random_mix", max_duration_scale=0.35)

    assert seq_a.seed_map == seq_b.seed_map
    assert seq_a.height_m == seq_b.height_m
    assert seq_a.body_scale == seq_b.body_scale
    assert len(seq_a.frames) == len(seq_b.frames)
    assert [e.to_dict() for e in seq_a.events] == [e.to_dict() for e in seq_b.events]
    for fa, fb in zip(seq_a.frames[::20], seq_b.frames[::20]):
        assert fa.joints_m == fb.joints_m
        assert fa.action_label == fb.action_label
        assert fa.fall_event_id == fb.fall_event_id

    # 24 seeds show varied action orders, paths, body dimensions, and both fall / no-fall runs
    action_orders: set[tuple[str, ...]] = set()
    final_xy_rounded: set[tuple[float, float]] = set()
    heights: set[float] = set()
    fall_run_count = 0
    no_fall_run_count = 0

    for seed in range(5000, 5024):
        seq = generate_motion_sequence(
            default_scene,
            default_motion_cfg,
            seed=seed,
            scenario_mode="random_mix",
            max_duration_scale=0.25,
        )
        action_orders.add(tuple(e.action_label for e in seq.events))
        end_pos = seq.frames[-1].root_position_m
        final_xy_rounded.add((round(end_pos[0], 2), round(end_pos[1], 2)))
        heights.add(round(seq.height_m, 4))
        if seq.fall_events:
            fall_run_count += 1
        else:
            no_fall_run_count += 1

    assert len(action_orders) >= 18
    assert len(final_xy_rounded) >= 18
    assert len(heights) >= 20
    assert fall_run_count > 0
    assert no_fall_run_count > 0
