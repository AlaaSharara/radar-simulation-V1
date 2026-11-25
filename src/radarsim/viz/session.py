"""Authoritative backend simulation session connecting Scene -> Motion -> RF -> Radar -> DSP.

Owns simulation time, frame IDs, RNG seeds, impairment toggles, and playback
state so the browser frontend never invents actor motion or radar detections.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import random
from pathlib import Path
import threading
from typing import Any

from dataclasses import replace as dc_replace
from radarsim.config import ProjectConfig, RadarProfile, load_project_config
from radarsim.dsp import DSPConfig, DSPFrame, DSPProcessor
from radarsim.motion import FALL_STYLES, MotionFrame, MotionSequence, generate_motion_sequence
from radarsim.radar import ADCCube, FMCWFrontEnd
from radarsim.rf import PathSet, RFPropagationConfig, RFPropagationEngine
from radarsim.scene import Scene, build_scene
from radarsim.viz.protocol import (
    ControlCommand,
    build_simulation_frame_message,
    parse_control_message,
)

# Representative body capsules used in interactive 'preview' mode (PLAN.md §9:
# reduced scatterer count for interactive UI, explicitly labeled 'preview').
_PREVIEW_CAPSULE_IDS: frozenset[str] = frozenset(
    {"head", "upper_torso", "lower_torso", "left_thigh", "right_thigh"}
)


class SimulationSession:
    """Thread-safe stateful backend session producing synchronized four-panel frames."""

    def __init__(
        self,
        root_dir: str | Path = ".",
        *,
        radar_path: str | Path | None = None,
        scene_path: str | Path | None = None,
        mode: str = "preview",
        seed: int | None = None,
        scenario_mode: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.project: ProjectConfig = load_project_config(
            self.root_dir,
            radar_path=radar_path,
            scene_path=scene_path,
        )
        self.scene: Scene = build_scene(self.project.scene, self.project.materials)
        self.radar_profile: RadarProfile = self.project.radar

        self.mode: str = mode
        # No explicit seed -> a fresh random clip every run, and a NEW clip after each loop wrap.
        # An explicit seed (CLI --seed or the Seed box) keeps that clip looping for reproducibility.
        self.auto_new_clip: bool = seed is None
        self.seed: int = (
            random.SystemRandom().randrange(1, 2**31 - 1) if seed is None else int(seed)
        )
        self.scenario_mode: str = (
            self.project.motion.scenario.default_mode
            if scenario_mode is None
            else str(scenario_mode)
        )
        self.fall_style: str | None = None

        self.playing: bool = True
        self.speed: float = 1.0
        self.frame_rate_hz: float = float(self.project.sim_profile.backend.frame_rate_hz)
        self.max_cached_frames: int = int(self.project.sim_profile.backend.max_cached_frames)

        # Monotonic backend frame ID & authoritative simulation clock
        self._frame_id: int = 0
        self._sim_time_s: float = 0.0
        self._seq_cursor: int = 0
        self._speed_accum: float = 0.0
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

        # Load avatar rig metadata for frontend GLTF adapter
        self._avatar_rig = self._load_avatar_rig()

        # Initialize motion sequence and real RF -> Radar -> DSP pipeline stages
        self._motion_seq: MotionSequence
        self._sampled_motion: list[MotionFrame] = []
        self._rf_engine: RFPropagationEngine
        self._frontend: FMCWFrontEnd
        self._dsp: DSPProcessor
        self._rebuild_pipeline(regenerate_motion=True)

    def _load_avatar_rig(self) -> dict[str, Any]:
        rig_path = self.root_dir / "assets" / "avatars" / "adult_male_default_rig.json"
        if rig_path.exists():
            return json.loads(rig_path.read_text(encoding="utf-8"))
        return {}

    def _make_rf_config(self) -> RFPropagationConfig:
        if self.mode == "preview":
            return RFPropagationConfig(
                enable_los=True,
                enable_first_order_reflections=True,
                enable_target_multipath=False,
                enable_second_order_reflections=False,
                enable_transmission=False,
                enable_diffuse_scattering=True,
                diffuse_points_per_surface=1,
                max_diffuse_surfaces=6,
                enable_human_body_occlusion=False,
                preview_mode=True,
            )
        return RFPropagationConfig(
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
        )

    def _rebuild_pipeline(self, *, regenerate_motion: bool) -> None:
        self._cache.clear()
        if regenerate_motion:
            self._motion_seq = generate_motion_sequence(
                self.scene,
                self.project.motion,
                seed=self.seed,
                scenario_mode=self.scenario_mode,
                fall_style=self.fall_style,
                max_duration_scale=0.6 if self.mode == "preview" else 1.0,
            )
            self._sampled_motion = self._motion_seq.sample_at_rate(self.frame_rate_hz)
            self._seq_cursor = 0

        # Build preview-tuned radar profile view (fewer micro-scatterers per segment and 64-pt
        # Doppler FFT interpolation for 32 preview loops; explicitly labeled 'preview').
        preview_hs = dc_replace(
            self.radar_profile.human_scattering,
            deterministic_micro_scatterers_per_body_segment=1,
            random_scatterers_per_body_segment=0,
        )
        preview_wf = dc_replace(
            self.radar_profile.waveform,
            doppler_fft_size=64,
        )
        self._preview_profile = dc_replace(
            self.radar_profile,
            human_scattering=preview_hs,
            waveform=preview_wf,
        )

        self._rf_engine = RFPropagationEngine(
            self.scene,
            self.radar_profile,
            config=self._make_rf_config(),
            global_seed=self.seed,
        )
        # In preview mode, precompute static room reflections/diffuse paths once and only
        # ray-trace moving human scatterers per frame (real geometric optics, no fake paths).
        self._static_preview_paths = []
        self._rf_preview_dynamic: RFPropagationEngine | None = None
        if self.mode == "preview":
            static_eng = RFPropagationEngine(
                self.scene,
                self._preview_profile,
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
                global_seed=self.seed,
            )
            # Keep strongest static specular & diffuse paths per (tx, rx) pair for crisp interactive preview
            all_static = static_eng.propagate(0, 0.0).paths
            by_pair: dict[tuple[int, int], list[Any]] = {}
            for p in all_static:
                by_pair.setdefault((p.tx_id, p.rx_id), []).append(p)
            kept: list[Any] = []
            for plist in by_pair.values():
                plist.sort(key=lambda x: x.amplitude_linear, reverse=True)
                kept.extend(plist[:4])
            self._static_preview_paths = kept
            self._rf_preview_dynamic = RFPropagationEngine(
                self.scene,
                self._preview_profile,
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
                global_seed=self.seed,
            )

        active_profile = self._preview_profile if self.mode == "preview" else self.radar_profile
        self._frontend = FMCWFrontEnd(
            active_profile,
            config_hash=self.project.config_hash,
            global_seed=self.seed,
            dtype="complex64",
        )
        self._dsp = DSPProcessor(
            active_profile,
            self.scene.radar_frame,
            config=DSPConfig(),
        )

    def control_state(self) -> dict[str, Any]:
        return {
            "playing": self.playing,
            "speed": self.speed,
            "seed": self.seed,
            "scenario_mode": self.scenario_mode,
            "fall_style": self.fall_style,
            "mode": self.mode,
            "frame_rate_hz": self.frame_rate_hz,
            "clip_time_s": round(self._seq_cursor / self.frame_rate_hz, 2),
            "clip_duration_s": round(len(self._sampled_motion) / self.frame_rate_hz, 2),
            "impairments": self.radar_profile.impairments.as_toggle_dict(),
        }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "config_hash": self.project.config_hash,
                "scene_id": self.scene.scene_id,
                "profile_id": self.radar_profile.profile_id,
                "mode": self.mode,
                "frame_id": self._frame_id,
                "sim_time_s": round(self._sim_time_s, 6),
                "playing": self.playing,
            }

    def config_summary_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "config_hash": self.project.config_hash,
                "radar_profile": {
                    "id": self.radar_profile.profile.id,
                    "name": self.radar_profile.profile.name,
                    "status": self.radar_profile.profile.status,
                    "hardware_reference": self.radar_profile.profile.hardware_reference,
                    "tx_count": self.radar_profile.rf.tx_count,
                    "rx_count": self.radar_profile.rf.rx_count,
                    "virtual_channels": self.radar_profile.virtual_channel_count,
                    "carrier_frequency_hz": self.radar_profile.rf.carrier_frequency_hz,
                    "adc_cube_shape": list(self.radar_profile.adc_cube_shape),
                },
                "derived_metrics": self.project.derived.as_dict(),
                "scene": self.scene.to_dict(frequency_hz=self.radar_profile.rf.carrier_frequency_hz),
                "avatar_rig": self._avatar_rig,
                "sim_profile": {
                    "frame_rate_hz": self.frame_rate_hz,
                    "speed_values": list(self.project.sim_profile.controls.speed_values),
                    "show_skeleton": self.project.sim_profile.frontend.show_skeleton,
                    "show_ground_truth_overlay_on_point_cloud_default": (
                        self.project.sim_profile.frontend.show_ground_truth_overlay_on_point_cloud_default
                    ),
                },
                "control_state": self.control_state(),
            }

    def scenario_modes_payload(self) -> dict[str, Any]:
        modes = {
            name: {
                "fall_probability": cfg.fall_probability,
                "no_fall_probability": cfg.no_fall_probability,
                "exactly_one_fall": cfg.exactly_one_fall,
            }
            for name, cfg in self.project.motion.scenario.modes.items()
        }
        return {
            "default_mode": self.project.motion.scenario.default_mode,
            "modes": modes,
            "fall_styles": list(FALL_STYLES),
        }

    def apply_control(self, raw_message: Any) -> dict[str, Any]:
        """Validate and apply a frontend control message; returns updated control_state."""
        allowed_modes = tuple(self.project.motion.scenario.modes.keys())
        cmd: ControlCommand = parse_control_message(
            raw_message, allowed_scenario_modes=allowed_modes
        )
        with self._lock:
            if cmd.action == "play":
                self.playing = True
            elif cmd.action == "pause":
                self.playing = False
            elif cmd.action == "step":
                self.playing = False
            elif cmd.action == "set_speed":
                assert cmd.value is not None
                self.speed = float(cmd.value)
            elif cmd.action == "set_seed":
                assert cmd.value is not None
                self.seed = int(cmd.value)
                self.auto_new_clip = False
                self._rebuild_pipeline(regenerate_motion=True)
            elif cmd.action == "regenerate_scenario":
                assert cmd.scenario_mode is not None
                self.scenario_mode = cmd.scenario_mode
                if cmd.value is not None:
                    self.seed = int(cmd.value)
                self.fall_style = cmd.fall_style
                self._rebuild_pipeline(regenerate_motion=True)
            elif cmd.action == "set_impairment":
                assert cmd.name is not None and cmd.enabled is not None
                self.radar_profile = self.radar_profile.with_impairment_toggle(
                    cmd.name, cmd.enabled
                )
                self._rebuild_pipeline(regenerate_motion=False)
            elif cmd.action == "set_mode":
                assert cmd.mode is not None
                self.mode = cmd.mode
                self._rebuild_pipeline(regenerate_motion=False)

            return {
                "type": "control_ack",
                "action": cmd.action,
                "control_state": self.control_state(),
            }

    def compute_frame(self, *, advance: bool = True) -> dict[str, Any]:
        """Run Scene -> Motion -> RF -> Radar -> DSP for the current/next frame."""
        with self._lock:
            if not self._sampled_motion:
                raise RuntimeError("Motion sequence has no sampled frames")

            if advance:
                self._frame_id += 1
                # Speed > 1: the machine cannot compute faster than ~20 radar frames/s, so play the
                # motion faster by skipping motion frames (preview only). Speed <= 1 is handled by
                # the server slowing down its frame rate.
                self._speed_accum += max(1.0, float(self.speed))
                steps = max(1, int(self._speed_accum))
                self._speed_accum -= steps
                n_motion = len(self._sampled_motion)
                wrapped = self._seq_cursor + steps >= n_motion
                self._seq_cursor = (self._seq_cursor + steps) % n_motion
                if wrapped:
                    if self.auto_new_clip:
                        # Loop wrap -> brand-new random 10-20 s clip (also resets tracker/detector)
                        self.seed = random.SystemRandom().randrange(1, 2**31 - 1)
                        self._rebuild_pipeline(regenerate_motion=True)
                    else:
                        # Reset stateful tracker & fall detector cleanly on loop wrap
                        self._dsp.reset()

            mf = self._sampled_motion[self._seq_cursor]
            self._sim_time_s = round(self._frame_id / self.frame_rate_hz, 6)

            imp_sig = tuple(sorted(self.radar_profile.impairments.as_toggle_dict().items()))
            cache_key = (self._seq_cursor, self.seed, self.scenario_mode, self.mode, imp_sig)

            if self.mode == "preview" and cache_key in self._cache:
                cached = self._cache[cache_key]
                self._cache.move_to_end(cache_key)
                # Keep frame_id and sim_time_s strictly monotonic even on looped preview cache hit
                msg = dict(cached)
                msg["frame_id"] = self._frame_id
                msg["sim_time_s"] = self._sim_time_s
                msg["control_state"] = self.control_state()
                return msg

            # 1. Real RF propagation (T3)
            if self.mode == "preview" and self._rf_preview_dynamic is not None:
                preview_caps = [
                    c for c in mf.body_capsules if c.segment_id in _PREVIEW_CAPSULE_IDS
                ] or mf.body_capsules[:6]
                mf_preview = dc_replace(mf, body_capsules=preview_caps)
                dyn_ps = self._rf_preview_dynamic.propagate(
                    self._frame_id,
                    self._sim_time_s,
                    motion_frame=mf_preview,
                )
                merged_paths = list(dyn_ps.paths) + list(self._static_preview_paths)
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
                path_set = PathSet(
                    frame_id=self._frame_id,
                    sim_time_s=self._sim_time_s,
                    paths=merged_paths,
                    diagnostics=diag,
                )
            else:
                path_set = self._rf_engine.propagate(
                    self._frame_id,
                    self._sim_time_s,
                    motion_frame=mf,
                )

            # 2. Real FMCW/TDM-MIMO ADC synthesis (T4)
            if self.mode == "preview":
                adc_cube: ADCCube = self._synthesize_preview_fast(path_set)
            else:
                adc_cube = self._frontend.synthesize(path_set, processing_mode="full_sim")

            # 3. Real radar-only DSP chain (T5) — receives ONLY ADCCube!
            dsp_frame: DSPFrame = self._dsp.process(adc_cube)

            msg = build_simulation_frame_message(
                frame_id=self._frame_id,
                sim_time_s=self._sim_time_s,
                seed=self.seed,
                scene_id=self.scene.scene_id,
                motion_frame=mf,
                path_set=path_set,
                dsp_frame=dsp_frame,
                config_hash=self.project.config_hash,
                seed_map=self._motion_seq.seed_map,
                mode=self.mode,
                control_state=self.control_state(),
            )

            if self.mode == "preview" and self.max_cached_frames > 0:
                self._cache[cache_key] = msg
                while len(self._cache) > self.max_cached_frames:
                    self._cache.popitem(last=False)

            return msg

    def _synthesize_preview_fast(self, path_set: PathSet) -> ADCCube:
        """Synthesize a real 32-loop preview ADCCube across all 12 virtual channels.

        Never fabricates detections; computes real complex FMCW beat samples,
        real thermal noise, and real enabled impairments, explicitly labeled
        ``processing_mode='preview'``.
        """
        return self._frontend.synthesize(
            path_set,
            processing_mode="preview",
            num_loops_override=32,
        )
