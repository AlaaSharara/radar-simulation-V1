"""Tests for T0: package structure, strict config validation, derived metrics, RNG, and viz manifest."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from radarsim.config import (
    ConfigValidationError,
    DERIVED_METRIC_KEYS,
    RadarProfile,
    compare_derived_metrics,
    compute_derived_metrics,
    load_material_db,
    load_motion_config,
    load_project_config,
    load_radar_config_with_expected,
    load_radar_profile,
    load_scene_config,
    load_sim_profile,
    parse_material_db,
    parse_motion_config,
    parse_radar_profile,
    parse_scene_config,
    parse_sim_profile,
    validate_derived_metrics,
)
from radarsim.rng import DEFAULT_STREAM_NAMES, SeedManager, derive_stream_seed, make_generator

ROOT = Path(__file__).resolve().parents[1]


def _load_raw_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def _load_raw_json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Package layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "submodule",
    ["scene", "motion", "rf", "radar", "dsp", "viz", "eval"],
)
def test_radarsim_submodules_exist_and_import(submodule: str) -> None:
    module = importlib.import_module(f"radarsim.{submodule}")
    assert module is not None


# ---------------------------------------------------------------------------
# 2. Loading valid default configs & vendor-neutral RadarProfile
# ---------------------------------------------------------------------------


def test_load_all_default_configs_individually_and_together() -> None:
    project = load_project_config(ROOT)

    assert isinstance(project.radar, RadarProfile)
    assert project.radar.profile_id == "iwr6843isk_ods_class_room_v1"
    assert project.radar.virtual_channel_count == 12
    assert project.radar.adc_cube_shape == (128, 3, 4, 192)
    assert len(project.radar.virtual_positions_lambda()) == 12
    assert len(project.radar.virtual_positions_m()) == 12
    assert len(project.config_hash) == 64

    # Individual loaders succeed
    assert load_radar_profile(ROOT / "radar_config.yaml") == project.radar
    assert load_scene_config(ROOT / "scene.json") == project.scene
    assert load_material_db(ROOT / "material_db.yaml") == project.materials
    assert load_motion_config(ROOT / "motion_config.yaml") == project.motion
    assert load_sim_profile(ROOT / "sim_profile.yaml") == project.sim_profile

    # Material alias resolution
    assert project.materials.resolve("drywall").material_id == "plasterboard"
    assert project.materials.resolve("human_clothing").material_id == "fabric"
    assert project.materials.resolve("concrete").conductivity_model is not None


def test_vendor_neutral_radar_profile_substitution() -> None:
    """Another vendor's 77 GHz 2-TX/4-RX profile substitutes cleanly without TI assumptions."""
    raw = _load_raw_yaml("radar_config.yaml")
    raw.pop("expected_derived", None)
    raw["profile"] = {
        "id": "generic_77ghz_2x4_v1",
        "name": "Generic 77 GHz 2-TX / 4-RX FMCW radar",
        "status": "synthetic_alternative_profile",
    }
    raw["rf"]["carrier_frequency_hz"] = 77.0e9
    raw["rf"]["operating_band_hz"] = [76.0e9, 81.0e9]
    raw["rf"]["tx_count"] = 2
    raw["rf"]["rx_count"] = 4
    raw["rf"]["tx_schedule"] = [0, 1]
    raw["rf"]["tdm_mode"] = True
    raw["waveform"]["start_frequency_hz"] = 76.5e9
    raw["waveform"]["slope_hz_per_s"] = 60.0e12
    raw["waveform"]["ramp_time_s"] = 50.0e-6
    raw["waveform"]["idle_time_s"] = 10.0e-6
    raw["waveform"]["chirp_period_s"] = 60.0e-6
    raw["waveform"]["nominal_sweep_bandwidth_hz"] = 60.0e12 * 50.0e-6
    raw["waveform"]["adc_sample_rate_hz"] = 10.0e6
    raw["waveform"]["adc_samples_per_chirp"] = 256
    raw["waveform"]["range_fft_size"] = 256
    raw["waveform"]["chirps_per_tx_per_frame"] = 64
    raw["waveform"]["doppler_fft_size"] = 64
    raw["rf"]["receiver"]["max_if_frequency_hz"] = 4.5e6
    raw["antenna_array"]["tx_positions_lambda"] = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    profile, expected = parse_radar_profile(raw, source="alt_vendor.yaml")
    assert expected is None
    assert profile.profile_id == "generic_77ghz_2x4_v1"
    assert profile.virtual_channel_count == 8
    assert profile.adc_cube_shape == (64, 2, 4, 256)

    # Verify derived metrics independently for the substituted profile
    c = 299792458.0
    bw_eff = 60.0e12 * (256 / 10.0e6)
    lam = c / 77.0e9
    t_same = 60.0e-6 * 2
    derived = compute_derived_metrics(profile)
    assert abs(derived.effective_bandwidth_hz - bw_eff) < 1e-6
    assert abs(derived.wavelength_m - lam) < 1e-15
    assert abs(derived.range_resolution_m - c / (2.0 * bw_eff)) < 1e-12
    assert abs(derived.max_unambiguous_range_m - c * 4.5e6 / (2.0 * 60.0e12)) < 1e-12
    assert abs(derived.same_tx_chirp_interval_s - t_same) < 1e-15
    assert abs(derived.max_unambiguous_radial_velocity_mps - lam / (4.0 * t_same)) < 1e-12
    assert abs(derived.velocity_resolution_mps - lam / (2.0 * 64 * t_same)) < 1e-12


def test_radar_profile_impairment_toggle_is_immutable() -> None:
    profile = load_radar_profile(ROOT / "radar_config.yaml")
    assert profile.impairments.phase_noise_enabled is True
    toggled = profile.with_impairment_toggle("phase_noise", False)
    assert toggled.impairments.phase_noise_enabled is False
    assert profile.impairments.phase_noise_enabled is True
    assert "phase_noise" not in toggled.impairments.enabled_names()

    with pytest.raises(ConfigValidationError, match="Unknown impairment"):
        profile.with_impairment_toggle("nonexistent_impairment", True)


# ---------------------------------------------------------------------------
# 3. Independent derived metrics & expected_derived comparison
# ---------------------------------------------------------------------------


def test_derived_metrics_do_not_use_expected_derived_as_input() -> None:
    raw = _load_raw_yaml("radar_config.yaml")
    # Deliberately corrupt expected_derived in the raw dict
    corrupted = copy.deepcopy(raw)
    corrupted["expected_derived"]["effective_bandwidth_hz"] = 12345.0
    corrupted["expected_derived"]["wavelength_m"] = 999.0

    # Parse without validating expected_derived -> compute_derived_metrics still gives true values
    profile, bad_expected = parse_radar_profile(
        corrupted, validate_expected=False, require_expected_derived=True
    )
    assert bad_expected is not None
    actual = compute_derived_metrics(profile)

    # Independent analytical calculation
    c = 299792458.0
    expected_bw = 50.0e12 * (192 / 5.0e6)
    expected_lam = c / 61.0e9
    assert abs(actual.effective_bandwidth_hz - expected_bw) < 1e-6
    assert abs(actual.wavelength_m - expected_lam) < 1e-15

    # Tolerance validation MUST catch the corrupted expected values
    with pytest.raises(ConfigValidationError, match="effective_bandwidth_hz"):
        validate_derived_metrics(actual, bad_expected)


def test_default_expected_derived_all_pass_tolerances() -> None:
    profile, expected = load_radar_config_with_expected(
        ROOT / "radar_config.yaml",
        require_expected_derived=True,
        validate_expected=False,
    )
    assert expected is not None
    actual = compute_derived_metrics(profile)
    comparisons = compare_derived_metrics(actual, expected)
    assert [c.name for c in comparisons] == list(DERIVED_METRIC_KEYS)
    assert all(c.passed for c in comparisons)


# ---------------------------------------------------------------------------
# 4. Strict validation error reporting across all 5 config files
# ---------------------------------------------------------------------------


def test_invalid_radar_config_fails_clearly() -> None:
    # Missing required field
    raw = _load_raw_yaml("radar_config.yaml")
    del raw["waveform"]["slope_hz_per_s"]
    with pytest.raises(ConfigValidationError, match="waveform.slope_hz_per_s"):
        parse_radar_profile(raw)

    # Nyquist violation: max_if >= fs / 2
    raw = _load_raw_yaml("radar_config.yaml")
    raw["rf"]["receiver"]["max_if_frequency_hz"] = 2.6e6
    with pytest.raises(ConfigValidationError, match="Nyquist"):
        parse_radar_profile(raw, validate_expected=False)

    # Boolean rejected where numeric expected
    raw = _load_raw_yaml("radar_config.yaml")
    raw["rf"]["tx_count"] = True
    with pytest.raises(ConfigValidationError, match="rf.tx_count"):
        parse_radar_profile(raw, validate_expected=False)

    # ADC window exceeds ramp_time_s
    raw = _load_raw_yaml("radar_config.yaml")
    raw["waveform"]["adc_samples_per_chirp"] = 250
    with pytest.raises(ConfigValidationError, match="exceeding ramp_time_s"):
        parse_radar_profile(raw, validate_expected=False)

    # Invalid window name
    raw = _load_raw_yaml("radar_config.yaml")
    raw["waveform"]["window"]["range"] = "mystery_window"
    with pytest.raises(ConfigValidationError, match="waveform.window.range"):
        parse_radar_profile(raw, validate_expected=False)

    # Mismatched rx_positions_lambda length
    raw = _load_raw_yaml("radar_config.yaml")
    raw["antenna_array"]["rx_positions_lambda"].pop()
    with pytest.raises(ConfigValidationError, match="antenna_array.rx_positions_lambda"):
        parse_radar_profile(raw, validate_expected=False)


def test_invalid_scene_config_fails_clearly() -> None:
    # Wrong units
    raw = _load_raw_json("scene.json")
    raw["units"] = "cm"
    with pytest.raises(ConfigValidationError, match="units"):
        parse_scene_config(raw)

    # Non-unit boresight vector
    raw = _load_raw_json("scene.json")
    raw["radar"]["boresight_unit_vector_world"] = [0.0, 2.0, 0.0]
    with pytest.raises(ConfigValidationError, match="boresight_unit_vector_world"):
        parse_scene_config(raw)

    # Radar outside room bounds
    raw = _load_raw_json("scene.json")
    raw["radar"]["position_m"] = [9.0, 0.1, 2.0]
    with pytest.raises(ConfigValidationError, match="radar.position_m"):
        parse_scene_config(raw)

    # Duplicate object ID
    raw = _load_raw_json("scene.json")
    raw["objects"].append(copy.deepcopy(raw["objects"][0]))
    with pytest.raises(ConfigValidationError, match="duplicate object id"):
        parse_scene_config(raw)


def test_invalid_material_db_fails_clearly() -> None:
    # Unphysical epsilon_real < 1.0
    raw = _load_raw_yaml("material_db.yaml")
    raw["materials"]["wood"]["epsilon_real"] = 0.5
    with pytest.raises(ConfigValidationError, match="materials.wood.epsilon_real"):
        parse_material_db(raw)

    # Double-counted loss: both epsilon_imag > 0 and conductivity_model
    raw = _load_raw_yaml("material_db.yaml")
    raw["materials"]["concrete"]["epsilon_imag"] = 0.25
    with pytest.raises(ConfigValidationError, match="double-counting"):
        parse_material_db(raw)

    # Dangling alias target
    raw = _load_raw_yaml("material_db.yaml")
    raw["materials"]["broken_alias"] = {"alias_of": "unobtanium"}
    with pytest.raises(ConfigValidationError, match="unobtanium"):
        parse_material_db(raw)

    # Cyclic alias
    raw = _load_raw_yaml("material_db.yaml")
    raw["materials"]["loop_a"] = {"alias_of": "loop_b"}
    raw["materials"]["loop_b"] = {"alias_of": "loop_a"}
    with pytest.raises(ConfigValidationError, match="cyclic"):
        parse_material_db(raw)


def test_invalid_motion_config_fails_clearly() -> None:
    # Skeleton must never be drawn
    raw = _load_raw_yaml("motion_config.yaml")
    raw["actor"]["draw_skeleton"] = True
    with pytest.raises(ConfigValidationError, match="actor.draw_skeleton"):
        parse_motion_config(raw)

    # Visible avatar is mandatory
    raw = _load_raw_yaml("motion_config.yaml")
    raw["actor"]["visible_avatar_required"] = False
    with pytest.raises(ConfigValidationError, match="actor.visible_avatar_required"):
        parse_motion_config(raw)

    # Joint count contract
    raw = _load_raw_yaml("motion_config.yaml")
    raw["actor"]["internal_joint_count"] = 15
    with pytest.raises(ConfigValidationError, match="internal_joint_count"):
        parse_motion_config(raw)

    # Default mode not in modes
    raw = _load_raw_yaml("motion_config.yaml")
    raw["scenario"]["default_mode"] = "unknown_mode"
    with pytest.raises(ConfigValidationError, match="scenario.default_mode"):
        parse_motion_config(raw)

    # Inverted range
    raw = _load_raw_yaml("motion_config.yaml")
    raw["motion"]["walking_speed_mps_range"] = [2.0, 0.5]
    with pytest.raises(ConfigValidationError, match="walking_speed_mps_range"):
        parse_motion_config(raw)


def test_invalid_sim_profile_fails_clearly() -> None:
    # show_skeleton must be false
    raw = _load_raw_yaml("sim_profile.yaml")
    raw["frontend"]["show_skeleton"] = True
    with pytest.raises(ConfigValidationError, match="frontend.show_skeleton"):
        parse_sim_profile(raw)

    # Invalid websocket path
    raw = _load_raw_yaml("sim_profile.yaml")
    raw["backend"]["websocket_path"] = "ws/sim"
    with pytest.raises(ConfigValidationError, match="backend.websocket_path"):
        parse_sim_profile(raw)

    # Four panel layout requires 4 panels
    raw = _load_raw_yaml("sim_profile.yaml")
    raw["panels"].pop()
    with pytest.raises(ConfigValidationError, match="exactly 4 panels"):
        parse_sim_profile(raw)


def test_cross_file_validation_rejects_unknown_scene_material(tmp_path: Path) -> None:
    for fname in (
        "radar_config.yaml",
        "scene.json",
        "material_db.yaml",
        "motion_config.yaml",
        "sim_profile.yaml",
    ):
        shutil.copy2(ROOT / fname, tmp_path / fname)

    scene_data = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    scene_data["objects"][0]["material_id"] = "mythril"
    (tmp_path / "scene.json").write_text(json.dumps(scene_data), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="mythril"):
        load_project_config(tmp_path)


# ---------------------------------------------------------------------------
# 5. Stable independent RNG streams
# ---------------------------------------------------------------------------


def test_rng_streams_are_stable_reproducible_and_independent() -> None:
    motion = load_motion_config(ROOT / "motion_config.yaml")
    sm1 = SeedManager.from_motion_config(motion)
    sm2 = SeedManager.from_motion_config(motion)

    assert sm1.seed_map() == sm2.seed_map()
    assert set(sm1.seed_map().keys()) == set(motion.seeding.independent_streams)
    assert len(set(sm1.seed_map().values())) == len(motion.seeding.independent_streams)

    # Independent check of stable SHA-256 derivation (no Python hash() randomness)
    expected_digest = hashlib.sha256(
        f"radarsim.rng.v1\0{motion.scenario.random_seed}\0path".encode("utf-8")
    ).digest()
    expected_seed = int.from_bytes(expected_digest[:8], byteorder="big", signed=False)
    assert derive_stream_seed(motion.scenario.random_seed, "path") == expected_seed

    # Same seed reproduces exact draws
    draws_a = sm1.rng("path").normal(size=16)
    draws_b = sm2.rng("path").normal(size=16)
    np.testing.assert_array_equal(draws_a, draws_b)

    # Drawing heavily from another stream does NOT affect 'speed'
    _ = sm1.rng("noise").random(size=5000)
    speed_1 = sm1.rng("speed").uniform(0.45, 1.55, size=10)
    speed_2 = sm2.rng("speed").uniform(0.45, 1.55, size=10)
    np.testing.assert_array_equal(speed_1, speed_2)

    # Reset reproduces the initial sequence
    sm1.reset("path")
    np.testing.assert_array_equal(sm1.rng("path").normal(size=16), draws_a)

    # Different global seeds produce different streams
    sm3 = SeedManager(global_seed=motion.scenario.random_seed + 1, stream_names=DEFAULT_STREAM_NAMES)
    assert sm3.seed_for("path") != sm1.seed_for("path")
    assert not np.array_equal(sm3.rng("path").normal(size=16), draws_a)

    # Subkey spawning is reproducible and distinct across subkeys
    frame0_a = sm1.spawn_rng("rf_diffuse", subkey=0).random(8)
    frame0_b = make_generator(motion.scenario.random_seed, "rf_diffuse", subkey=0).random(8)
    frame1 = sm1.spawn_rng("rf_diffuse", subkey=1).random(8)
    np.testing.assert_array_equal(frame0_a, frame0_b)
    assert not np.array_equal(frame0_a, frame1)


def test_rng_validation_rejects_bad_inputs() -> None:
    with pytest.raises(TypeError):
        derive_stream_seed(True, "path")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        derive_stream_seed(42, "   ")
    sm = SeedManager(42, stream_names=["path", "noise"], strict_streams=True)
    with pytest.raises(KeyError):
        sm.rng("unregistered")


# ---------------------------------------------------------------------------
# 6. Pre-created viz/web Vite/React package manifest
# ---------------------------------------------------------------------------


def test_viz_web_manifest_is_valid_and_matches_sim_profile() -> None:
    web_dir = ROOT / "viz" / "web"
    pkg_path = web_dir / "package.json"
    vite_cfg_path = web_dir / "vite.config.js"
    index_html_path = web_dir / "index.html"

    assert pkg_path.exists()
    assert vite_cfg_path.exists()
    assert index_html_path.exists()

    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    assert pkg["type"] == "module"
    for script_name in ("dev", "build", "preview"):
        assert script_name in pkg["scripts"]
    for dep in ("react", "react-dom", "three", "plotly.js-dist-min"):
        assert dep in pkg["dependencies"]
    for dev_dep in ("vite", "@vitejs/plugin-react"):
        assert dev_dep in pkg["devDependencies"]

    sim_profile = load_sim_profile(ROOT / "sim_profile.yaml")
    vite_cfg_text = vite_cfg_path.read_text(encoding="utf-8")
    assert sim_profile.frontend.host in vite_cfg_text
    assert str(sim_profile.frontend.port) in vite_cfg_text
    assert "/src/main.jsx" in index_html_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. CLI script scripts/validate_config.py from repository root
# ---------------------------------------------------------------------------


def test_validate_config_cli_runs_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/validate_config.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "All 5 configuration files are valid" in result.stdout


def test_validate_config_cli_fails_on_tolerance_mismatch(tmp_path: Path) -> None:
    for fname in (
        "radar_config.yaml",
        "scene.json",
        "material_db.yaml",
        "motion_config.yaml",
        "sim_profile.yaml",
    ):
        shutil.copy2(ROOT / fname, tmp_path / fname)

    radar_data = yaml.safe_load((tmp_path / "radar_config.yaml").read_text(encoding="utf-8"))
    radar_data["expected_derived"]["range_resolution_m"] = 0.50
    (tmp_path / "radar_config.yaml").write_text(yaml.safe_dump(radar_data), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/validate_config.py", "--root", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "range_resolution_m" in result.stderr
