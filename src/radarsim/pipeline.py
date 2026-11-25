"""End-to-end synchronous simulation pipeline & reproducible NPZ exporter (T7).

Connects Scene (T1) -> Motion (T2) -> RF Propagation (T3) -> FMCW/TDM-MIMO ADC (T4)
-> Radar-Only DSP (T5) -> Separate Evaluation Metrics & Session Integration (T6)
without duplicating physics logic or feeding ground-truth XYZ coordinates into DSP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np

from radarsim.config import ProjectConfig, RadarProfile, load_project_config
from radarsim.dsp import DSPConfig, DSPFrame, DSPProcessor
from radarsim.eval import (
    FallBaselineEvalMetrics,
    TrackerEvalMetrics,
    evaluate_fall_baseline,
    evaluate_tracker,
)
from radarsim.motion import MotionFrame, MotionSequence, generate_motion_sequence
from radarsim.radar import ADCCube, FMCWFrontEnd
from radarsim.rf import PathSet, RFPropagationConfig, RFPropagationEngine
from radarsim.scene import Scene, build_scene
from radarsim.viz.protocol import VALID_MODES
from radarsim.viz.session import SimulationSession, _PREVIEW_CAPSULE_IDS


def _collect_software_versions() -> dict[str, str]:
    packages = ("numpy", "PyYAML", "matplotlib", "fastapi", "uvicorn", "pytest")
    versions: dict[str, str] = {"python": platform.python_version()}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def _select_adc_window_indices(
    motion_frames: Sequence[MotionFrame],
    window_frames: int,
) -> tuple[int, int, str]:
    """Select a deterministic contiguous frame window ``[start, end)`` for raw ADC export."""
    n = len(motion_frames)
    if n == 0 or window_frames <= 0:
        return 0, 0, "empty_window"
    w = min(n, int(window_frames))
    if w == n:
        return 0, n, (
            f"All {n} simulated frames exported as contiguous complex64 ADC cubes "
            f"(no temporal decimation needed)."
        )

    # Prefer centering on fall impact if present; otherwise center on peak active motion.
    impact_idx: int | None = None
    for idx, mf in enumerate(motion_frames):
        if mf.fall_event_id is not None and mf.action_phase == "impact_contact":
            impact_idx = idx
            break
    if impact_idx is None:
        for idx, mf in enumerate(motion_frames):
            if mf.fall_event_id is not None:
                impact_idx = idx
                break

    if impact_idx is not None:
        anchor = impact_idx
        reason_tag = f"centered on fall impact frame index {anchor} (t={motion_frames[anchor].sim_time_s:.2f}s)"
    else:
        # Pick midpoint of frames where actor is active (walking, sitting, bending, etc.)
        active_indices = [
            idx for idx, mf in enumerate(motion_frames) if mf.action_label != "entry_wait"
        ]
        anchor = active_indices[len(active_indices) // 2] if active_indices else (n // 2)
        reason_tag = f"centered on active motion frame index {anchor} (t={motion_frames[anchor].sim_time_s:.2f}s)"

    start = max(0, anchor - (w // 2))
    end = start + w
    if end > n:
        end = n
        start = max(0, end - w)

    return start, end, (
        f"Contiguous {w}-frame window [{start}, {end}) selected ({reason_tag}); "
        f"retaining only {w}/{n} full ADC cubes bounds archive size cleanly while still "
        f"saving per-frame timestamps, labels, RF summaries, tracks, and Range-Doppler maps "
        f"for all {n} frames."
    )


@dataclass(frozen=True)
class ScenarioRunResult:
    """Complete reproducible record for one headless ``SimulationPipeline`` execution."""

    config_hash: str
    seed: int
    scenario_mode: str
    fall_style: str | None
    mode: str
    frame_rate_hz: float
    total_frames: int
    start_frame_index: int
    total_sequence_frames: int
    duration_s: float
    wall_runtime_s: float
    achieved_fps: float
    seed_map: dict[str, int]
    fall_events: list[dict[str, Any]]
    action_segments: list[dict[str, Any]]
    derived_metrics: dict[str, float]
    motion_frames: list[MotionFrame]
    path_sets: list[PathSet]
    dsp_frames: list[DSPFrame]
    adc_window_samples: np.ndarray | None
    adc_window_frame_ids: list[int]
    adc_window_sim_times_s: list[float]
    adc_window_chirp_time_s: np.ndarray | None
    adc_window_rationale: str
    adc_cube_shape: tuple[int, int, int, int]
    tracker_eval: TrackerEvalMetrics
    fall_eval: FallBaselineEvalMetrics
    labels_per_frame: list[dict[str, Any]]
    eval_provenance: dict[str, Any]

    def summary_dict(self) -> dict[str, Any]:
        """Compact JSON-serializable summary of run configuration, metrics, and timing."""
        adc_bytes = (
            int(self.adc_window_samples.nbytes) if self.adc_window_samples is not None else 0
        )
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "seed_map": dict(self.seed_map),
            "scenario_mode": self.scenario_mode,
            "fall_style": self.fall_style,
            "mode": self.mode,
            "frame_rate_hz": self.frame_rate_hz,
            "total_frames": self.total_frames,
            "start_frame_index": self.start_frame_index,
            "total_sequence_frames": self.total_sequence_frames,
            "duration_s": round(self.duration_s, 6),
            "wall_runtime_s": round(self.wall_runtime_s, 4),
            "achieved_fps": round(self.achieved_fps, 3),
            "adc_cube_shape": list(self.adc_cube_shape),
            "adc_window": {
                "frame_count": len(self.adc_window_frame_ids),
                "frame_ids": list(self.adc_window_frame_ids),
                "sim_times_s": [round(float(t), 6) for t in self.adc_window_sim_times_s],
                "raw_bytes": adc_bytes,
                "rationale": self.adc_window_rationale,
            },
            "derived_metrics": dict(self.derived_metrics),
            "action_segments": list(self.action_segments),
            "fall_events": list(self.fall_events),
            "tracker_eval": self.tracker_eval.to_dict(),
            "fall_eval": self.fall_eval.to_dict(),
            "eval_provenance": dict(self.eval_provenance),
        }


class SimulationPipeline:
    """Authoritative orchestrator for headless runs, NPZ export, and WebSocket server creation."""

    def __init__(
        self,
        root_dir: str | Path = ".",
        *,
        radar_path: str | Path | None = None,
        scene_path: str | Path | None = None,
        material_db_path: str | Path | None = None,
        motion_path: str | Path | None = None,
        sim_profile_path: str | Path | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self._radar_path = Path(radar_path) if radar_path is not None else None
        self._scene_path = Path(scene_path) if scene_path is not None else None
        self._material_db_path = Path(material_db_path) if material_db_path is not None else None
        self._motion_path = Path(motion_path) if motion_path is not None else None
        self._sim_profile_path = Path(sim_profile_path) if sim_profile_path is not None else None

        self.project: ProjectConfig = load_project_config(
            self.root_dir,
            radar_path=self._radar_path,
            scene_path=self._scene_path,
            material_db_path=self._material_db_path,
            motion_path=self._motion_path,
            sim_profile_path=self._sim_profile_path,
        )
        self.scene: Scene = build_scene(self.project.scene, self.project.materials)

    def create_session(
        self,
        *,
        mode: str = "preview",
        seed: int | None = None,
        scenario_mode: str | None = None,
        fall_style: str | None = None,
    ) -> SimulationSession:
        """Instantiate a live ``SimulationSession`` bound to the validated config paths."""
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        session = SimulationSession(
            root_dir=self.root_dir,
            radar_path=self._radar_path,
            scene_path=self._scene_path,
            mode=mode,
            seed=seed,
            scenario_mode=scenario_mode,
        )
        if fall_style is not None:
            session.apply_control(
                {
                    "type": "control",
                    "action": "regenerate_scenario",
                    "scenario_mode": session.scenario_mode,
                    "value": session.seed,
                    "fall_style": fall_style,
                }
            )
        return session

    def run_scenario(
        self,
        *,
        seed: int | None = None,
        scenario_mode: str | None = None,
        fall_style: str | None = None,
        mode: str = "full_sim",
        max_frames: int | None = None,
        start_frame: int = 0,
        adc_window_frames: int = 16,
        max_duration_scale: float | None = None,
        impairment_overrides: Mapping[str, bool] | None = None,
    ) -> ScenarioRunResult:
        """Execute ``scene -> motion -> RF -> radar/ADC -> DSP -> eval`` synchronously."""
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(VALID_MODES)}")

        resolved_seed = (
            self.project.motion.scenario.random_seed if seed is None else int(seed)
        )
        resolved_scenario_mode = (
            self.project.motion.scenario.default_mode
            if scenario_mode is None
            else str(scenario_mode)
        )
        eff_scale = (
            (0.6 if mode == "preview" else 1.0)
            if max_duration_scale is None
            else float(max_duration_scale)
        )

        # 1. Deterministic 100 Hz motion -> sampled radar frames
        motion_seq: MotionSequence = generate_motion_sequence(
            self.scene,
            self.project.motion,
            seed=resolved_seed,
            scenario_mode=resolved_scenario_mode,
            fall_style=fall_style,
            max_duration_scale=eff_scale,
        )
        fps = float(self.project.sim_profile.backend.frame_rate_hz)
        sampled_all = motion_seq.sample_at_rate(fps)
        total_seq_frames = len(sampled_all)

        start_idx = max(0, int(start_frame))
        if start_idx >= total_seq_frames:
            raise ValueError(
                f"start_frame={start_idx} exceeds available sequence length {total_seq_frames}"
            )
        sampled = sampled_all[start_idx:]
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError(f"max_frames must be >= 1, got {max_frames}")
            sampled = sampled[: int(max_frames)]

        # 2. Configure RF, ADC front-end, and radar-only DSP
        base_profile: RadarProfile = self.project.radar
        if impairment_overrides:
            for imp_name, enabled in impairment_overrides.items():
                base_profile = base_profile.with_impairment_toggle(imp_name, bool(enabled))

        is_preview = mode == "preview"
        if is_preview:
            active_profile = dc_replace(
                base_profile,
                human_scattering=dc_replace(
                    base_profile.human_scattering,
                    deterministic_micro_scatterers_per_body_segment=1,
                    random_scatterers_per_body_segment=0,
                ),
                waveform=dc_replace(
                    base_profile.waveform,
                    doppler_fft_size=64,
                ),
            )
            static_eng = RFPropagationEngine(
                self.scene,
                active_profile,
                config=RFPropagationConfig(
                    enable_los=False,
                    enable_first_order_reflections=True,
                    enable_target_multipath=False,
                    enable_second_order_reflections=False,
                    enable_transmission=False,
                    enable_diffuse_scattering=True,
                    diffuse_points_per_surface=1,
                    max_diffuse_surfaces=4,
                    enable_human_body_occlusion=False,
                    preview_mode=True,
                ),
                global_seed=resolved_seed,
            )
            all_static = static_eng.propagate(0, 0.0).paths
            by_pair: dict[tuple[int, int], list[Any]] = {}
            for p in all_static:
                by_pair.setdefault((p.tx_id, p.rx_id), []).append(p)
            static_preview_paths: list[Any] = []
            for plist in by_pair.values():
                plist.sort(key=lambda x: x.amplitude_linear, reverse=True)
                static_preview_paths.extend(plist[:4])
            rf_engine = RFPropagationEngine(
                self.scene,
                active_profile,
                config=RFPropagationConfig(
                    enable_los=True,
                    enable_first_order_reflections=True,
                    enable_static_reflections=False,
                    enable_target_multipath=True,
                    enable_second_order_reflections=False,
                    enable_transmission=True,
                    max_transmission_slabs=1,
                    enable_diffuse_scattering=False,
                    enable_human_body_occlusion=False,
                    preview_mode=True,
                ),
                global_seed=resolved_seed,
            )
        else:
            active_profile = base_profile
            static_preview_paths = []
            rf_engine = RFPropagationEngine(
                self.scene,
                active_profile,
                config=RFPropagationConfig(
                    enable_los=True,
                    enable_first_order_reflections=True,
                    enable_target_multipath=True,
                    enable_second_order_reflections=False,
                    enable_transmission=True,
                    enable_diffuse_scattering=True,
                    diffuse_points_per_surface=2,
                    max_diffuse_surfaces=10,
                    enable_human_body_occlusion=True,
                    preview_mode=False,
                ),
                global_seed=resolved_seed,
            )

        frontend = FMCWFrontEnd(
            active_profile,
            config_hash=self.project.config_hash,
            global_seed=resolved_seed,
            dtype="complex64",
        )
        dsp = DSPProcessor(
            active_profile,
            self.scene.radar_frame,
            config=DSPConfig(),
        )

        # 3. Determine bounded contiguous ADC window to keep for archive export
        win_start, win_end, base_rationale = _select_adc_window_indices(
            sampled, adc_window_frames
        )

        # 4. Run frame loop sequentially (strictly passing only ADCCube to DSP)
        path_sets: list[PathSet] = []
        dsp_frames: list[DSPFrame] = []
        labels_per_frame: list[dict[str, Any]] = []
        adc_window_cubes: list[np.ndarray] = []
        adc_window_fids: list[int] = []
        adc_window_ts: list[float] = []
        adc_window_chirp_t: list[np.ndarray] = []

        last_cube_shape: tuple[int, int, int, int] = (
            32 if is_preview else active_profile.adc_cube_shape[0],
            active_profile.rf.tx_count,
            active_profile.rf.rx_count,
            active_profile.waveform.adc_samples_per_chirp,
        )

        t0 = time.perf_counter()
        for local_idx, mf in enumerate(sampled):
            fid = int(start_idx + local_idx)
            sim_t = float(mf.sim_time_s)

            if is_preview:
                preview_caps = [
                    c for c in mf.body_capsules if c.segment_id in _PREVIEW_CAPSULE_IDS
                ] or mf.body_capsules[:6]
                mf_preview = dc_replace(mf, body_capsules=preview_caps)
                dyn_ps = rf_engine.propagate(fid, sim_t, motion_frame=mf_preview)
                merged_paths = list(dyn_ps.paths) + static_preview_paths
                counts = {
                    "los": sum(1 for p in merged_paths if p.path_type == "los"),
                    "specular_reflection": sum(
                        1 for p in merged_paths if p.path_type == "specular_reflection"
                    ),
                    "diffuse_scatter": sum(
                        1 for p in merged_paths if p.path_type == "diffuse_scatter"
                    ),
                    "transmitted": sum(1 for p in merged_paths if p.path_type == "transmitted"),
                }
                diag = dict(dyn_ps.diagnostics)
                diag["total_paths"] = len(merged_paths)
                diag["counts_by_type"] = counts
                diag["preview_static_room_cached"] = True
                ps = PathSet(frame_id=fid, sim_time_s=sim_t, paths=merged_paths, diagnostics=diag)
                adc_cube: ADCCube = frontend.synthesize(
                    ps, processing_mode="preview", num_loops_override=32
                )
            else:
                ps = rf_engine.propagate(fid, sim_t, motion_frame=mf)
                # Label processing_mode accurately as full_sim during analysis_export as well
                adc_cube = frontend.synthesize(ps, processing_mode="full_sim")

            last_cube_shape = adc_cube.shape
            dsp_frame = dsp.process(adc_cube)

            path_sets.append(ps)
            dsp_frames.append(dsp_frame)
            labels_per_frame.append(
                {
                    "frame_id": fid,
                    "sim_time_s": round(sim_t, 6),
                    "action_label": mf.action_label,
                    "action_phase": mf.action_phase,
                    "fall_event_id": mf.fall_event_id,
                    "fall_style": mf.fall_style,
                    "is_hard_negative": mf.is_hard_negative,
                }
            )

            if win_start <= local_idx < win_end:
                adc_window_cubes.append(adc_cube.complex_samples.astype(np.complex64, copy=False))
                adc_window_fids.append(fid)
                adc_window_ts.append(sim_t)
                adc_window_chirp_t.append(adc_cube.chirp_time_s)

        wall_runtime_s = time.perf_counter() - t0
        achieved_fps = (len(sampled) / wall_runtime_s) if wall_runtime_s > 0.0 else 0.0

        # 5. Strictly separate ground-truth evaluation (after DSP)
        tracker_eval = evaluate_tracker(dsp_frames, sampled)
        fall_eval = evaluate_fall_baseline(dsp_frames, sampled)

        adc_win_arr = (
            np.stack(adc_window_cubes, axis=0) if adc_window_cubes else None
        )
        adc_win_chirp_arr = (
            np.stack(adc_window_chirp_t, axis=0) if adc_window_chirp_t else None
        )
        bytes_per_cube = int(math.prod(last_cube_shape) * 8)  # complex64 = 8 B
        full_est_mb = (bytes_per_cube * len(sampled)) / (1024.0 * 1024.0)
        adc_rationale = (
            f"{base_rationale} Each ADC cube shape={last_cube_shape} complex64 uses "
            f"{bytes_per_cube / (1024.0 * 1024.0):.3f} MiB "
            f"(full {len(sampled)}-frame ADC archive would be {full_est_mb:.1f} MiB uncompressed)."
        )

        resolved_duration_s = (
            float(sampled[-1].sim_time_s - sampled[0].sim_time_s) if len(sampled) > 1 else 0.0
        )

        return ScenarioRunResult(
            config_hash=self.project.config_hash,
            seed=resolved_seed,
            scenario_mode=resolved_scenario_mode,
            fall_style=fall_style,
            mode=mode,
            frame_rate_hz=fps,
            total_frames=len(sampled),
            start_frame_index=start_idx,
            total_sequence_frames=total_seq_frames,
            duration_s=resolved_duration_s,
            wall_runtime_s=wall_runtime_s,
            achieved_fps=achieved_fps,
            seed_map=dict(motion_seq.seed_map),
            fall_events=[dict(fe) for fe in motion_seq.fall_events],
            action_segments=[seg.to_dict() for seg in motion_seq.events],
            derived_metrics=self.project.derived.as_dict(),
            motion_frames=sampled,
            path_sets=path_sets,
            dsp_frames=dsp_frames,
            adc_window_samples=adc_win_arr,
            adc_window_frame_ids=adc_window_fids,
            adc_window_sim_times_s=adc_window_ts,
            adc_window_chirp_time_s=adc_win_chirp_arr,
            adc_window_rationale=adc_rationale,
            adc_cube_shape=last_cube_shape,
            tracker_eval=tracker_eval,
            fall_eval=fall_eval,
            labels_per_frame=labels_per_frame,
            eval_provenance={
                "evaluation_scope": "simulated_synthetic_benchmark_not_real_hardware",
                "physical_measurements_performed": False,
                "ground_truth_leakage_detected": tracker_eval.ground_truth_leakage_detected,
                "dsp_input_contract": "ADCCube_and_RadarProfile_and_RadarCoordinateFrame_only",
            },
        )


def export_run_npz(
    result: ScenarioRunResult,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
) -> tuple[Path, Path]:
    """Export one ``ScenarioRunResult`` to compressed ``.npz`` + sidecar ``_metadata.json``."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_parts = []
    if prefix:
        stem_parts.append(str(prefix))
    stem_parts.extend([result.scenario_mode, f"seed{result.seed}", result.mode])
    if result.fall_style:
        stem_parts.append(result.fall_style)
    stem = "_".join(stem_parts)

    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}_metadata.json"

    n = result.total_frames
    vocab: list[str] = []
    vocab_map: dict[str, int] = {}
    action_idx = np.zeros(n, dtype=np.int32)
    is_hn = np.zeros(n, dtype=bool)
    gt_fall = np.zeros(n, dtype=bool)
    gt_root = np.zeros((n, 3), dtype=np.float32)
    gt_spine = np.zeros((n, 3), dtype=np.float32)
    gt_head = np.zeros((n, 3), dtype=np.float32)

    rf_total = np.zeros(n, dtype=np.int32)
    rf_los = np.zeros(n, dtype=np.int32)
    rf_spec = np.zeros(n, dtype=np.int32)
    rf_diff = np.zeros(n, dtype=np.int32)
    rf_trans = np.zeros(n, dtype=np.int32)

    dsp_det_cnt = np.zeros(n, dtype=np.int32)
    dsp_trk_cnt = np.zeros(n, dtype=np.int32)
    dsp_vox_cnt = np.zeros(n, dtype=np.int32)
    dsp_trk_xyz = np.full((n, 3), np.nan, dtype=np.float32)
    dsp_trk_vel = np.full((n, 3), np.nan, dtype=np.float32)
    dsp_est_h = np.full(n, np.nan, dtype=np.float32)
    dsp_est_vz = np.full(n, np.nan, dtype=np.float32)
    dsp_fall = np.zeros(n, dtype=bool)

    rd_list: list[np.ndarray] = []
    frame_ids = np.zeros(n, dtype=np.int32)
    sim_times = np.zeros(n, dtype=np.float64)

    for i, (mf, ps, df, lbl) in enumerate(
        zip(result.motion_frames, result.path_sets, result.dsp_frames, result.labels_per_frame)
    ):
        frame_ids[i] = int(lbl["frame_id"])
        sim_times[i] = float(mf.sim_time_s)

        al = mf.action_label
        if al not in vocab_map:
            vocab_map[al] = len(vocab)
            vocab.append(al)
        action_idx[i] = vocab_map[al]
        is_hn[i] = bool(mf.is_hard_negative)
        gt_fall[i] = mf.fall_event_id is not None

        gt_root[i] = mf.root_position_m
        gt_spine[i] = mf.joints_m.get("spine", mf.root_position_m)
        gt_head[i] = mf.joints_m.get("head", mf.root_position_m)

        cbt = ps.diagnostics.get("counts_by_type", {})
        rf_total[i] = int(ps.diagnostics.get("total_paths", len(ps.paths)))
        rf_los[i] = int(cbt.get("los", 0))
        rf_spec[i] = int(cbt.get("specular_reflection", 0))
        rf_diff[i] = int(cbt.get("diffuse_scatter", 0))
        rf_trans[i] = int(cbt.get("transmitted", 0))

        dsp_det_cnt[i] = len(df.detections)
        dsp_trk_cnt[i] = len(df.tracks)
        dsp_vox_cnt[i] = len(df.occupancy.voxels)
        if df.tracks:
            dsp_trk_xyz[i] = df.tracks[0].position_m
            dsp_trk_vel[i] = df.tracks[0].velocity_mps
        h_val = df.features.get("estimated_height_m")
        vz_val = df.features.get("vertical_velocity_mps")
        if h_val is not None:
            dsp_est_h[i] = float(h_val)
        if vz_val is not None:
            dsp_est_vz[i] = float(vz_val)
        dsp_fall[i] = bool(df.fall_alert.active)
        rd_list.append(df.range_doppler.values_db.astype(np.float32, copy=False))

    rd_stack = np.stack(rd_list, axis=0) if rd_list else np.empty((0, 0, 0), dtype=np.float32)
    rd_range_m = (
        result.dsp_frames[0].range_doppler.range_m.astype(np.float32)
        if result.dsp_frames
        else np.empty(0, dtype=np.float32)
    )
    rd_velocity_mps = (
        result.dsp_frames[0].range_doppler.velocity_mps.astype(np.float32)
        if result.dsp_frames
        else np.empty(0, dtype=np.float32)
    )

    metadata = {
        **result.summary_dict(),
        "software_versions": _collect_software_versions(),
        "labels_per_frame": result.labels_per_frame,
        "npz_file": npz_path.name,
    }
    meta_json_str = json.dumps(metadata, indent=2, sort_keys=False)

    arrays: dict[str, Any] = {
        "frame_ids": frame_ids,
        "sim_times_s": sim_times,
        "action_label_indices": action_idx,
        "action_label_vocab": np.array(vocab, dtype=str),
        "is_hard_negative": is_hn,
        "gt_fall_active": gt_fall,
        "gt_root_position_m": gt_root,
        "gt_spine_position_m": gt_spine,
        "gt_head_position_m": gt_head,
        "rf_total_paths": rf_total,
        "rf_counts_los": rf_los,
        "rf_counts_specular": rf_spec,
        "rf_counts_diffuse": rf_diff,
        "rf_counts_transmitted": rf_trans,
        "dsp_detection_count": dsp_det_cnt,
        "dsp_track_count": dsp_trk_cnt,
        "dsp_voxel_count": dsp_vox_cnt,
        "dsp_primary_track_xyz_m": dsp_trk_xyz,
        "dsp_primary_track_vel_mps": dsp_trk_vel,
        "dsp_est_height_m": dsp_est_h,
        "dsp_est_vz_mps": dsp_est_vz,
        "dsp_fall_alert_active": dsp_fall,
        "dsp_range_doppler_db": rd_stack,
        "rd_range_m": rd_range_m,
        "rd_velocity_mps": rd_velocity_mps,
        "adc_window_frame_ids": np.array(result.adc_window_frame_ids, dtype=np.int32),
        "adc_window_sim_times_s": np.array(result.adc_window_sim_times_s, dtype=np.float64),
        "metadata_json": np.array(meta_json_str, dtype=str),
    }
    if result.adc_window_samples is not None:
        arrays["adc_window_complex"] = result.adc_window_samples
    if result.adc_window_chirp_time_s is not None:
        arrays["adc_window_chirp_time_s"] = result.adc_window_chirp_time_s

    np.savez_compressed(npz_path, **arrays)
    meta_path.write_text(meta_json_str, encoding="utf-8")
    return npz_path, meta_path
