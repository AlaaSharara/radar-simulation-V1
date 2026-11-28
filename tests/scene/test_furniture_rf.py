"""Tests for the realistic furniture RF model (hollow carcass, open-underneath, foam upholstery)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from radarsim.rf import PointTarget, RFPropagationConfig, RFPropagationEngine, evaluate_surface_em
from radarsim.rf.raytrace import SurfaceRayAccelerator
from radarsim.config import load_radar_profile, parse_scene_config, load_material_db
from radarsim.scene import build_scene, load_scene

ROOT = Path(__file__).resolve().parents[2]
F_HZ = 61.0e9


@pytest.fixture(scope="module")
def scene():
    return load_scene(ROOT / "scene.json", ROOT / "material_db.yaml")


@pytest.fixture(scope="module")
def accel(scene):
    return SurfaceRayAccelerator(scene.surfaces)


def _hits(accel, a, b):
    return accel.find_segment_hits(tuple(a), tuple(b))


def _local_bounds_ok(obj, tol=1e-6) -> bool:
    c = np.asarray(obj.box.center_m)
    ax = obj.box.axes_2d()
    hx, hy, hz = obj.box.half_extents_m
    for sf in obj.surfaces:
        for v in sf.vertices_m:
            d = np.asarray(v) - c
            lx = d[0] * ax[0, 0] + d[1] * ax[0, 1]
            ly = d[0] * ax[1, 0] + d[1] * ax[1, 1]
            if abs(lx) > hx + tol or abs(ly) > hy + tol or d[2] < -hz - tol or d[2] > hz + tol:
                return False
    return True


def test_every_panel_lies_inside_its_objects_bounding_box(scene) -> None:
    for obj in scene.objects:
        assert _local_bounds_ok(obj), obj.object_id


def test_no_panel_is_coplanar_with_the_floor(scene) -> None:
    for obj in scene.objects:
        for sf in obj.surfaces:
            zs = [v[2] for v in sf.vertices_m]
            assert not (max(zs) < 1e-3), f"{sf.surface_id} duplicates the floor"


def test_table_and_chair_are_open_underneath(scene, accel) -> None:
    t = scene.get_object("table_01")
    cx, cy, _ = t.box.center_m
    assert _hits(accel, (cx, cy, 0.05), (cx, cy, 0.40)) == []
    # ... but the tabletop is solid
    assert len(_hits(accel, (cx, cy, 0.05), (cx, cy, 0.80))) == 1

    ch = scene.get_object("chair_01")
    cx, cy, _ = ch.box.center_m
    assert _hits(accel, (cx, cy, 0.05), (cx, cy, 0.40)) == []
    assert len(_hits(accel, (cx, cy, 0.05), (cx, cy, 0.70))) == 1  # seat


def test_wardrobe_is_a_hollow_carcass(scene, accel) -> None:
    w = scene.get_object("wardrobe_01")
    cx, cy, _ = w.box.center_m
    # inside the cabinet: nothing between two interior points
    assert _hits(accel, (cx, cy, 0.4), (cx, cy, 1.8)) == []
    # from outside to the interior along +x crosses exactly one side panel
    assert len(_hits(accel, (cx - 1.5, cy, 1.0), (cx, cy, 1.0))) == 1


def test_upholstery_is_not_an_antireflection_artifact(scene) -> None:
    """The legacy 10 mm fabric slab gave |R| ~ -56 dB (anti-reflection at 61 GHz)."""
    for obj in scene.objects:
        for sf in obj.surfaces:
            for deg in (0.0, 45.0):
                em = evaluate_surface_em(sf.layers, scene.materials, math.radians(deg), F_HZ)
                r_db = 20.0 * math.log10(max(abs(em.reflection_smooth), 1e-12))
                assert r_db > -40.0, (sf.surface_id, deg, r_db)
                assert r_db < -1.0 or "metal" in sf.material_id, (sf.surface_id, deg, r_db)


def test_board_furniture_reflectivity_is_in_published_range(scene) -> None:
    """Chipboard/wood at 60 GHz: normal-incidence |R| roughly -9..-20 dB for 18-50 mm panels."""
    for sid in ("wardrobe_01_carcass_front", "table_01_top_top", "bed_01_rail_front"):
        sf = next(s for s in scene.surfaces if s.surface_id == sid)
        em = evaluate_surface_em(sf.layers, scene.materials, 0.0, F_HZ)
        r_db = 20.0 * math.log10(abs(em.reflection_smooth))
        assert -22.0 < r_db < -8.0, (sid, r_db)


def test_unknown_kind_falls_back_to_legacy_solid_box(scene) -> None:
    import json

    raw = json.loads((ROOT / "scene.json").read_text(encoding="utf-8"))
    for o in raw["objects"]:
        if o["id"] == "chair_01":
            o["kind"] = "shelf_unit"
    cfg = parse_scene_config(raw)
    sc = build_scene(cfg, load_material_db(ROOT / "material_db.yaml"))
    assert len(sc.get_object("chair_01").surfaces) == 6


def test_diffuse_budget_includes_furniture_panels(scene) -> None:
    prof = load_radar_profile(ROOT / "radar_config.yaml")
    eng = RFPropagationEngine(scene, prof, config=RFPropagationConfig(), global_seed=3)
    kinds = {s.kind for s, _, _ in eng._diffuse_surface_points}
    assert "furniture" in kinds
    n_f = sum(1 for s, _, _ in eng._diffuse_surface_points if s.kind == "furniture")
    assert n_f <= RFPropagationConfig().max_diffuse_furniture_surfaces * RFPropagationConfig().diffuse_points_per_surface


def test_person_seated_on_sofa_is_visible_in_preview_mode(scene) -> None:
    """With the legacy solid box the sofa swallowed a seated person's hips (no LOS, no transmission in preview)."""
    prof = load_radar_profile(ROOT / "radar_config.yaml")
    cfg = RFPropagationConfig(
        enable_los=True,
        enable_first_order_reflections=False,
        enable_transmission=False,
        enable_diffuse_scattering=False,
        enable_human_body_occlusion=False,
        preview_mode=True,
    )
    eng = RFPropagationEngine(scene, prof, config=cfg, global_seed=1)
    tgt = PointTarget(target_id="hip_on_sofa", position_m=(3.0, 4.0, 0.55), rcs_m2=0.1)
    ps = eng.propagate(0, 0.0, point_targets=[tgt], tx_ids=[0], rx_ids=[0])
    assert len(ps.paths_by_type("los")) == 1
