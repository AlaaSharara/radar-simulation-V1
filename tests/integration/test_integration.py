"""Synchronous end-to-end integration tests for scene -> motion -> RF -> ADC -> DSP -> UI session."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from radarsim.pipeline import SimulationPipeline, export_run_npz
from radarsim.viz import ControlMessageError, SimulationSession

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_fixed_seed_reproducibility() -> None:
    """Identical seed + scenario parameters produce bit-identical ADC/DSP results in full_sim mode."""
    pipeline = SimulationPipeline(ROOT)

    res_a = pipeline.run_scenario(
        seed=7001,
        scenario_mode="fall_demo",
        fall_style="fall_forward",
        mode="full_sim",
        start_frame=40,
        max_frames=4,
        adc_window_frames=4,
    )
    res_b = pipeline.run_scenario(
        seed=7001,
        scenario_mode="fall_demo",
        fall_style="fall_forward",
        mode="full_sim",
        start_frame=40,
        max_frames=4,
        adc_window_frames=4,
    )

    assert res_a.config_hash == res_b.config_hash
    assert len(res_a.config_hash) == 64
    assert res_a.seed_map == res_b.seed_map
    assert res_a.adc_cube_shape == (128, 3, 4, 192)
    assert res_a.adc_window_samples is not None and res_b.adc_window_samples is not None
    assert np.array_equal(res_a.adc_window_samples, res_b.adc_window_samples)

    for df_a, df_b in zip(res_a.dsp_frames, res_b.dsp_frames):
        assert df_a.frame_id == df_b.frame_id
        assert df_a.processing_mode == "full_sim"
        assert np.array_equal(df_a.range_doppler.values_db, df_b.range_doppler.values_db)
        assert [d.to_dict() for d in df_a.detections] == [d.to_dict() for d in df_b.detections]


def test_three_fixed_scenarios_and_no_ground_truth_leakage() -> None:
    """Run normal_daily_routine, fall_demo (fall_forward), and random_mix synchronously."""
    pipeline = SimulationPipeline(ROOT)

    # 1. Normal daily routine (no falls)
    res_norm = pipeline.run_scenario(
        seed=20261123,
        scenario_mode="normal_daily_routine",
        mode="preview",
        max_frames=40,
        adc_window_frames=8,
    )
    assert res_norm.fall_eval.ground_truth_has_fall is False
    assert res_norm.tracker_eval.ground_truth_leakage_detected is False
    assert len(res_norm.dsp_frames) == 40
    assert all(df.processing_mode == "preview" for df in res_norm.dsp_frames)

    # 2. Fall demo (full preview duration so fall impact & post-fall state are covered)
    res_fall = pipeline.run_scenario(
        seed=7001,
        scenario_mode="fall_demo",
        fall_style="fall_forward",
        mode="preview",
        max_duration_scale=0.35,
        adc_window_frames=8,
    )
    assert res_fall.fall_eval.ground_truth_has_fall is True
    assert res_fall.tracker_eval.ground_truth_leakage_detected is False
    assert any(mf.fall_style == "fall_forward" for mf in res_fall.motion_frames)
    assert res_fall.tracker_eval.frames_with_detections > 0

    # 3. Random mix
    res_mix = pipeline.run_scenario(
        seed=20261125,
        scenario_mode="random_mix",
        mode="preview",
        max_frames=40,
        adc_window_frames=8,
    )
    assert res_mix.tracker_eval.ground_truth_leakage_detected is False
    assert len(res_mix.seed_map) > 0


def test_session_synchronization_controls_and_rejection() -> None:
    """Synchronously verify SimulationSession payloads, controls, and invalid command rejection."""
    session = SimulationSession(
        root_dir=ROOT,
        mode="preview",
        seed=20261123,
        scenario_mode="random_mix",
    )

    health = session.health_payload()
    assert health["status"] == "ok"
    assert len(health["config_hash"]) == 64

    cfg = session.config_summary_payload()
    assert cfg["scene"]["scene_id"] == "room_4x5_default"
    assert cfg["avatar_rig"]["avatar_asset_id"] == "adult_male_default"
    assert cfg["avatar_rig"]["draw_skeleton"] is False
    assert cfg["sim_profile"]["show_skeleton"] is False
    assert cfg["sim_profile"]["show_ground_truth_overlay_on_point_cloud_default"] is False

    modes = session.scenario_modes_payload()
    assert set(modes["modes"].keys()) == {"normal_daily_routine", "fall_demo", "random_mix"}

    # Initial synchronized frame
    f0 = session.compute_frame(advance=False)
    assert f0["type"] == "simulation_frame"
    assert f0["ground_truth"]["avatar_asset_id"] == "adult_male_default"
    assert f0["radar"]["provenance"] == "radar_derived"
    assert f0["provenance"]["config_hash"] == health["config_hash"]

    # Pause & Step advance monotonically
    ack_pause = session.apply_control({"type": "control", "action": "pause"})
    assert ack_pause["control_state"]["playing"] is False

    f1 = session.compute_frame(advance=True)
    assert f1["frame_id"] > f0["frame_id"]
    assert f1["sim_time_s"] > f0["sim_time_s"]

    # Regenerate scenario with explicit seed and fall_style
    ack_reg = session.apply_control(
        {
            "type": "control",
            "action": "regenerate_scenario",
            "scenario_mode": "fall_demo",
            "value": 7001,
            "fall_style": "fall_forward",
        }
    )
    assert ack_reg["control_state"]["scenario_mode"] == "fall_demo"
    assert ack_reg["control_state"]["seed"] == 7001
    assert ack_reg["control_state"]["fall_style"] == "fall_forward"

    # Toggle impairment and mode cleanly
    ack_imp = session.apply_control(
        {"type": "control", "action": "set_impairment", "name": "phase_noise", "enabled": False}
    )
    assert ack_imp["control_state"]["impairments"]["phase_noise"] is False

    # Reject unknown / malformed control commands strictly
    with pytest.raises(ControlMessageError):
        session.apply_control({"type": "control", "action": "invalid_action_xyz"})
    with pytest.raises(ControlMessageError):
        session.apply_control({"type": "control", "action": "pause", "extra_key": 123})


def test_cli_analysis_export_npz_roundtrip(tmp_path: Path) -> None:
    """Execute python -m radarsim.main --mode analysis_export and verify NPZ/JSON export."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    cmd = [
        sys.executable,
        "-m",
        "radarsim.main",
        "--root-dir",
        str(ROOT),
        "--mode",
        "analysis_export",
        "--scenario-mode",
        "random_mix",
        "--seed",
        "20261125",
        "--start-frame",
        "20",
        "--max-frames",
        "4",
        "--adc-window-frames",
        "2",
        "--output-dir",
        str(tmp_path),
        "--export-prefix",
        "itest",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"CLI failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    summary = json.loads(proc.stdout)
    assert summary["mode"] == "analysis_export"
    assert summary["total_frames"] == 4
    assert summary["adc_cube_shape"] == [128, 3, 4, 192]
    assert summary["eval_provenance"]["physical_measurements_performed"] is False

    npz_path = Path(summary["npz_path"])
    meta_path = Path(summary["metadata_path"])
    assert npz_path.exists()
    assert meta_path.exists()

    with np.load(npz_path, allow_pickle=False) as data:
        assert tuple(data["adc_window_complex"].shape) == (2, 128, 3, 4, 192)
        assert data["adc_window_complex"].dtype == np.complex64
        assert tuple(data["frame_ids"].shape) == (4,)
        assert tuple(data["sim_times_s"].shape) == (4,)
        assert tuple(data["dsp_range_doppler_db"].shape[:1]) == (4,)
        assert len(data["action_label_vocab"]) >= 1
        meta_embedded = json.loads(str(data["metadata_json"]))
        assert meta_embedded["config_hash"] == summary["config_hash"]
        assert meta_embedded["seed_map"] == summary["seed_map"]
