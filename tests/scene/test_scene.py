"""Tests for T1: furnished room geometry, layered surfaces, transforms, materials, and 100 seeded layouts."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from radarsim.config import load_material_db, load_scene_config
from radarsim.scene import (
    EPSILON_0_F_PER_M,
    LayoutGenerationBounds,
    OrientedBox,
    RadarCoordinateFrame,
    boresight_from_yaw_tilt,
    build_scene,
    complex_permittivity,
    generate_layout_batch,
    generate_random_layout,
    load_scene,
    lookup_material_properties,
    radial_velocity_mps,
    validate_scene_geometry,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Default scene loading, layered walls, openings, and furniture
# ---------------------------------------------------------------------------


def test_default_scene_loads_all_surfaces_openings_and_furniture() -> None:
    scene = load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")

    assert scene.scene_id == "room_4x5_default"
    assert (scene.width_x_m, scene.depth_y_m, scene.height_z_m) == (4.0, 5.0, 2.7)
    assert abs(scene.room.wall_thickness_m - 0.1143) < 1e-12

    # All 5 expected furniture objects exist
    object_ids = {obj.object_id for obj in scene.objects}
    assert object_ids == {"bed_01", "sofa_01", "table_01", "chair_01", "wardrobe_01"}
    for obj in scene.objects:
        assert len(obj.surfaces) == 6

    # Component material mapping respected on furniture surfaces
    bed_faces = {s.metadata["face"]: s.material_id for s in scene.get_object("bed_01").surfaces}
    assert bed_faces["top"] == "fabric"
    assert bed_faces["front"] == "wood"

    # Door and window openings exist with exact construction layer thicknesses
    door = scene.get_opening("front_door")
    window = scene.get_opening("back_window")
    assert door.kind == "door" and door.wall == "y_min"
    assert len(door.layers) == 1 and abs(door.layers[0].thickness_m - 0.045) < 1e-12
    assert door.panel_surface is not None
    assert window.kind == "window" and window.wall == "y_max"
    assert len(window.layers) == 1 and abs(window.layers[0].thickness_m - 0.0095) < 1e-12
    assert window.panel_surface is not None

    # Floor and ceiling layers and inward normals
    floor = scene.surfaces_by_kind("floor")[0]
    ceiling = scene.surfaces_by_kind("ceiling")[0]
    assert floor.normal_m == (0.0, 0.0, 1.0)
    assert abs(floor.area_m2 - 20.0) < 1e-12
    assert abs(floor.total_thickness_m - 0.15) < 1e-12
    assert ceiling.normal_m == (0.0, 0.0, -1.0)
    assert abs(ceiling.area_m2 - 20.0) < 1e-12
    assert [l.material_id for l in ceiling.layers] == ["plasterboard", "air"]
    assert abs(ceiling.total_thickness_m - (0.0127 + 0.08)) < 1e-12

    # Layered walls: plasterboard / air cavity / plasterboard matching wall_thickness_m
    wall_surfaces = scene.surfaces_by_kind("wall")
    assert len(wall_surfaces) > 0
    for ws in wall_surfaces:
        assert [l.material_id for l in ws.layers] == ["plasterboard", "air", "plasterboard"]
        assert [l.thickness_m for l in ws.layers] == pytest.approx([0.0127, 0.0889, 0.0127])
        assert ws.total_thickness_m == pytest.approx(0.1143)

    # Exact area conservation on each wall (solid wall + opening aperture == full wall area)
    wall_areas = {"y_min": 0.0, "y_max": 0.0, "x_min": 0.0, "x_max": 0.0}
    expected_normals = {
        "y_min": (0.0, 1.0, 0.0),
        "y_max": (0.0, -1.0, 0.0),
        "x_min": (1.0, 0.0, 0.0),
        "x_max": (-1.0, 0.0, 0.0),
    }
    for ws in wall_surfaces:
        w = ws.metadata["wall"]
        wall_areas[w] += ws.area_m2
        assert ws.normal_m == pytest.approx(expected_normals[w])

    assert wall_areas["x_min"] == pytest.approx(5.0 * 2.7)
    assert wall_areas["x_max"] == pytest.approx(5.0 * 2.7)
    assert wall_areas["y_min"] + (0.9 * 2.1) == pytest.approx(4.0 * 2.7)
    assert wall_areas["y_max"] + (1.5 * 1.1) == pytest.approx(4.0 * 2.7)


# ---------------------------------------------------------------------------
# 2. Coordinate transforms (INTERFACES.md §1)
# ---------------------------------------------------------------------------


def test_radar_coordinate_frame_conventions_and_roundtrips() -> None:
    scene = load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")
    frame = scene.radar_frame

    # Boresight from yaw=0, tilt_down=15 deg matches scene.json independently
    bx, by, bz = boresight_from_yaw_tilt(0.0, 15.0)
    assert (bx, by, bz) == pytest.approx(
        (0.0, math.cos(math.radians(15.0)), -math.sin(math.radians(15.0))),
        abs=1e-12,
    )
    assert frame.boresight_world == pytest.approx((bx, by, bz), abs=1e-9)

    # Rotation matrix is right-handed orthonormal (R^T R = I, det = +1)
    R = frame.rotation_local_to_world
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    # Point 3.0 m straight along boresight -> range=3.0, az=0, el=0
    p_bore = tuple(
        frame.position_m[i] + 3.0 * frame.boresight_world[i] for i in range(3)
    )
    rng, az, el = frame.point_to_spherical(p_bore)
    assert rng == pytest.approx(3.0, abs=1e-12)
    assert az == pytest.approx(0.0, abs=1e-12)
    assert el == pytest.approx(0.0, abs=1e-12)

    # Point shifted toward local +X (right) has positive azimuth; local +Z (up) has positive elevation
    p_right_up = frame.local_to_world_point((1.0, 2.0, 0.5))
    rng2, az2, el2 = frame.point_to_spherical(p_right_up)
    assert az2 > 0.0 and az2 == pytest.approx(math.atan2(1.0, 2.0), abs=1e-12)
    assert el2 > 0.0 and el2 == pytest.approx(math.atan2(0.5, math.hypot(1.0, 2.0)), abs=1e-12)
    assert rng2 == pytest.approx(math.sqrt(1.0**2 + 2.0**2 + 0.5**2), abs=1e-12)

    # Exact spherical round-trip back to world coordinates
    reconstructed = frame.spherical_to_world_point(rng2, az2, el2)
    assert reconstructed == pytest.approx(p_right_up, abs=1e-12)

    # Radial velocity sign convention: positive receding, negative approaching, zero tangential
    v_receding = tuple(1.5 * c for c in frame.boresight_world)
    v_approaching = tuple(-1.5 * c for c in frame.boresight_world)
    v_tangential = frame.axis_x_world
    assert frame.radial_velocity(p_bore, v_receding) == pytest.approx(1.5, abs=1e-12)
    assert frame.radial_velocity(p_bore, v_approaching) == pytest.approx(-1.5, abs=1e-12)
    assert frame.radial_velocity(p_bore, v_tangential) == pytest.approx(0.0, abs=1e-12)
    assert radial_velocity_mps((0.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 2.0, 0.0)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. OrientedBox SAT collision & geometry validation
# ---------------------------------------------------------------------------


def test_oriented_box_sat_distinguishes_rotated_non_overlapping_boxes() -> None:
    # Two thin boxes rotated 45 deg placed parallel side-by-side: AABBs overlap, OBBs do NOT!
    box_a = OrientedBox(center_m=(0.0, 0.0, 0.5), half_extents_m=(1.0, 0.1, 0.5), yaw_rad=math.pi / 4.0)
    # Shift perpendicular to the 45-deg long axis by 0.5 m (> 2 * 0.1 = 0.2 m)
    shift = 0.5 / math.sqrt(2.0)
    box_b = OrientedBox(center_m=(-shift, shift, 0.5), half_extents_m=(1.0, 0.1, 0.5), yaw_rad=math.pi / 4.0)

    # Verify their AABBs overlap while SAT OBB check correctly reports separation ~0.3 m
    assert box_a.aabb().max_m[0] > box_b.aabb().min_m[0]
    assert not box_a.intersects_box(box_b)
    assert box_a.distance_2d(box_b) == pytest.approx(0.3, abs=1e-9)

    # Overlapping rotated boxes return True and distance 0.0
    box_c = OrientedBox(center_m=(0.0, 0.0, 0.5), half_extents_m=(1.0, 0.1, 0.5), yaw_rad=-math.pi / 4.0)
    assert box_a.intersects_box(box_c)
    assert box_a.distance_2d(box_c) == 0.0


def test_default_scene_passes_all_geometry_validation_checks() -> None:
    scene = load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")
    report = validate_scene_geometry(scene)

    assert report.valid, f"Default scene unexpectedly invalid: {report.reasons}"
    assert all(report.checks.values())
    assert scene.is_point_free_2d(0.9, 0.55, radius_m=0.25)
    # Entry path from outside through front_door is permitted; solid wall is blocked
    assert scene.is_point_free_2d(0.9, -0.3, radius_m=0.2, allow_doorway_exterior=True)
    assert not scene.is_point_free_2d(2.5, -0.3, radius_m=0.2, allow_doorway_exterior=True)


def test_geometry_validation_detects_overlap_out_of_bounds_and_blocked_doorway() -> None:
    base_cfg = load_scene_config(ROOT / "scene.json")
    mat_db = load_material_db(ROOT / "material_db.yaml")

    # 1. Force sofa_01 onto bed_01 -> OBJECT_OVERLAP
    bed = base_cfg.objects[0]
    bad_objects = list(base_cfg.objects)
    bad_objects[1] = replace(bad_objects[1], position_m=bed.position_m)
    overlap_scene = build_scene(replace(base_cfg, objects=tuple(bad_objects)), mat_db)
    rep_overlap = validate_scene_geometry(overlap_scene)
    assert not rep_overlap.valid
    assert not rep_overlap.checks["object_non_overlap"]
    assert any(issue.code == "OBJECT_OVERLAP" for issue in rep_overlap.issues)

    # 2. Push wardrobe_01 past the x_max wall -> OBJECT_OUT_OF_BOUNDS
    bad_wall = list(base_cfg.objects)
    bad_wall[4] = replace(bad_wall[4], position_m=(3.6, 0.48, 0.0))
    oob_scene = build_scene(replace(base_cfg, objects=tuple(bad_wall)), mat_db)
    rep_oob = validate_scene_geometry(oob_scene)
    assert not rep_oob.valid
    assert not rep_oob.checks["wall_boundaries"]
    assert any(issue.code == "OBJECT_OUT_OF_BOUNDS" for issue in rep_oob.issues)

    # 3. Place table_01 directly in front of front_door -> DOORWAY_BLOCKED
    bad_door = list(base_cfg.objects)
    bad_door[2] = replace(bad_door[2], position_m=(0.55, 0.20, 0.0))
    blocked_scene = build_scene(replace(base_cfg, objects=tuple(bad_door)), mat_db)
    rep_door = validate_scene_geometry(blocked_scene)
    assert not rep_door.valid
    assert not rep_door.checks["doorway_clearance"]
    assert any(issue.code == "DOORWAY_BLOCKED" for issue in rep_door.issues)

    # 4. Nearly closed door (open_fraction=0.1) -> DOOR_TOO_NARROW
    bad_openings = list(base_cfg.openings)
    bad_openings[0] = replace(bad_openings[0], open_fraction=0.1)
    narrow_scene = build_scene(replace(base_cfg, openings=tuple(bad_openings)), mat_db)
    rep_narrow = validate_scene_geometry(narrow_scene)
    assert not rep_narrow.valid
    assert any(issue.code == "DOOR_TOO_NARROW" for issue in rep_narrow.issues)


# ---------------------------------------------------------------------------
# 4. Material lookup & complex permittivity helper
# ---------------------------------------------------------------------------


def test_material_lookup_and_complex_permittivity_sign_convention() -> None:
    mat_db = load_material_db(ROOT / "material_db.yaml")

    # Direct dielectric entries follow epsilon_r = epsilon_real - j * epsilon_imag
    assert complex_permittivity(mat_db, "air", 61.0e9) == complex(1.0, 0.0)
    assert complex_permittivity(mat_db, "wood", 60.5e9) == complex(1.99, -0.1135)
    assert complex_permittivity(mat_db, "plasterboard", 60.5e9) == complex(2.94, -0.0628)
    assert complex_permittivity(mat_db, "glass", 60.5e9) == complex(6.27, -0.1703)
    assert complex_permittivity(mat_db, "fabric", 61.0e9) == complex(1.5, -0.05)

    # Aliases resolve cleanly
    drywall = lookup_material_properties(mat_db, "drywall", 60.5e9)
    assert drywall.resolved_id == "plasterboard"
    assert drywall.epsilon_r == complex(2.94, -0.0628)
    assert drywall.frequency_extrapolated is False

    # Extrapolation warning flag activates at 61.0 GHz vs 60.5 GHz table entry
    plaster_61 = lookup_material_properties(mat_db, "plasterboard", 61.0e9)
    assert plaster_61.frequency_extrapolated is True

    # Concrete uses power-law conductivity converted once to epsilon_imag (independent formula)
    f_hz = 61.0e9
    expected_sigma = 0.0462 * ((f_hz / 1.0e9) ** 0.7822)
    expected_eps_imag = expected_sigma / (2.0 * math.pi * f_hz * EPSILON_0_F_PER_M)
    concrete = lookup_material_properties(mat_db, "concrete", f_hz)
    assert concrete.conductivity_s_per_m == pytest.approx(expected_sigma, rel=1e-12)
    assert concrete.epsilon_real == pytest.approx(5.24)
    assert concrete.epsilon_imag == pytest.approx(expected_eps_imag, rel=1e-12)
    assert concrete.epsilon_r.real == pytest.approx(5.24)
    assert concrete.epsilon_r.imag == pytest.approx(-expected_eps_imag, rel=1e-12)

    # Metal is tagged as PEC
    metal = lookup_material_properties(mat_db, "metal", 61.0e9)
    assert metal.is_pec is True


# ---------------------------------------------------------------------------
# 5. Seeded random-layout generator (100 layouts accepted/rejected with reasons)
# ---------------------------------------------------------------------------


def test_100_seeded_randomized_layouts_single_shot_accept_or_reject_with_reasons() -> None:
    base_cfg = load_scene_config(ROOT / "scene.json")
    mat_db = load_material_db(ROOT / "material_db.yaml")
    seeds = list(range(1000, 1100))

    results = generate_layout_batch(base_cfg, mat_db, seeds, max_attempts=1)
    assert len(results) == 100

    # Determinism check on same seeds
    results_repeat = generate_layout_batch(base_cfg, mat_db, seeds[:10], max_attempts=1)
    for a, b in zip(results[:10], results_repeat):
        assert a.accepted == b.accepted
        assert a.rejection_reasons == b.rejection_reasons
        if a.scene is not None and b.scene is not None:
            assert a.scene.to_dict() == b.scene.to_dict()

    accepted_count = 0
    rejected_count = 0
    for res in results:
        if res.accepted:
            accepted_count += 1
            assert res.scene is not None
            assert res.rejection_reasons == ()
            # Independent verification of every accepted layout
            verify = validate_scene_geometry(
                res.scene,
                min_object_clearance_m=LayoutGenerationBounds().min_object_clearance_m,
                wall_margin_m=LayoutGenerationBounds().wall_margin_m,
            )
            assert verify.valid, f"Seed {res.seed} accepted invalid scene: {verify.reasons}"
            assert res.scene.width_x_m <= 8.0 and res.scene.depth_y_m <= 6.0
        else:
            rejected_count += 1
            assert res.scene is None
            assert len(res.rejection_reasons) > 0
            assert not res.validation.valid

    # Single-shot random sampling must produce both valid accepted scenes and explicit rejections
    assert accepted_count > 0
    assert rejected_count > 0


def test_100_seeded_randomized_layouts_multi_attempt_and_impossible_room_rejection() -> None:
    base_cfg = load_scene_config(ROOT / "scene.json")
    mat_db = load_material_db(ROOT / "material_db.yaml")

    # Multi-attempt search across 100 seeds up to 8x6 m produces valid layouts cleanly
    results = generate_layout_batch(base_cfg, mat_db, range(2000, 2100), max_attempts=40)
    assert len(results) == 100
    accepted = [r for r in results if r.accepted]
    rejected = [r for r in results if not r.accepted]
    assert len(accepted) >= 85
    for r in accepted:
        assert r.scene is not None
        assert validate_scene_geometry(r.scene).valid
    for r in rejected:
        assert r.scene is None
        assert len(r.rejection_reasons) > 0

    # Impossible tiny room must always be rejected with explicit reasons (never silently clipped)
    tiny_bounds = LayoutGenerationBounds(
        width_x_range_m=(2.0, 2.2),
        depth_y_range_m=(2.0, 2.2),
        height_z_range_m=(2.5, 2.6),
    )
    impossible = generate_random_layout(
        base_cfg, mat_db, seed=42, bounds=tiny_bounds, max_attempts=5
    )
    assert not impossible.accepted
    assert impossible.scene is None
    assert len(impossible.rejection_reasons) > 0


# ---------------------------------------------------------------------------
# 6. Normalized scene export (motion, RF, frontend) & preview summary
# ---------------------------------------------------------------------------


def test_normalized_scene_export_occupancy_grid_and_summary() -> None:
    scene = load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")

    # JSON-serializable payload for frontend / RF / motion
    payload = scene.to_dict(frequency_hz=61.0e9)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["scene_id"] == "room_4x5_default"
    assert decoded["room"]["width_x_m"] == 4.0
    assert len(decoded["objects"]) == 5
    assert len(decoded["openings"]) == 2
    assert len(decoded["surfaces"]) == len(scene.surfaces)

    # Occupancy grid derived directly from normalized scene
    grid = scene.build_occupancy_grid_2d(resolution_m=0.10, clearance_m=0.12)
    assert grid.shape == (40, 50)
    assert grid.blocked.any()
    assert (~grid.blocked).any()

    # Geometry summary and ASCII floorplan contain expected entities
    summary = scene.geometry_summary_markdown(frequency_hz=61.0e9)
    assert "room_4x5_default" in summary
    assert "bed_01" in summary and "wardrobe_01" in summary
    assert "front_door" in summary and "back_window" in summary
    assert "concrete" in summary and "plasterboard" in summary
