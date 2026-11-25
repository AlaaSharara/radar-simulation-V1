"""WebSocket frame serialization and strict control-message validation (``INTERFACES.md`` §8–9)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from radarsim.dsp.types import DSPFrame
from radarsim.motion.types import MotionFrame
from radarsim.rf.types import PathSet

KNOWN_IMPAIRMENTS: frozenset[str] = frozenset(
    {
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
    }
)

VALID_MODES: frozenset[str] = frozenset({"preview", "full_sim", "analysis_export"})


class ControlMessageError(ValueError):
    """Raised when a frontend-to-backend control payload violates ``INTERFACES.md`` §9."""


@dataclass(frozen=True)
class ControlCommand:
    action: str
    value: Any = None
    scenario_mode: str | None = None
    name: str | None = None
    enabled: bool | None = None
    mode: str | None = None
    fall_style: str | None = None


def parse_control_message(
    raw: Any,
    *,
    allowed_scenario_modes: Sequence[str] = ("normal_daily_routine", "fall_demo", "random_mix"),
) -> ControlCommand:
    """Strictly validate a control JSON dict per ``INTERFACES.md`` §9 and reject unknown keys."""
    if not isinstance(raw, Mapping):
        raise ControlMessageError("Control message must be a JSON object")
    if raw.get("type") != "control":
        raise ControlMessageError(f"Expected type='control', got {raw.get('type')!r}")

    action = raw.get("action")
    if not isinstance(action, str):
        raise ControlMessageError("Control message must include string 'action'")

    keys = set(raw.keys())

    if action in ("play", "pause", "step"):
        extra = keys - {"type", "action"}
        if extra:
            raise ControlMessageError(f"Unexpected keys for action {action!r}: {sorted(extra)}")
        return ControlCommand(action=action)

    if action == "set_speed":
        extra = keys - {"type", "action", "value"}
        if extra or "value" not in raw:
            raise ControlMessageError(
                f"set_speed requires only 'value'; unexpected keys: {sorted(extra)}"
            )
        val = raw["value"]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
            raise ControlMessageError(f"set_speed value must be a finite number, got {val!r}")
        speed = float(val)
        if not (0.1 <= speed <= 4.0):
            raise ControlMessageError(f"set_speed value must be in [0.1, 4.0], got {speed}")
        return ControlCommand(action="set_speed", value=speed)

    if action == "set_seed":
        extra = keys - {"type", "action", "value"}
        if extra or "value" not in raw:
            raise ControlMessageError(
                f"set_seed requires only 'value'; unexpected keys: {sorted(extra)}"
            )
        val = raw["value"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ControlMessageError(f"set_seed value must be an integer, got {val!r}")
        return ControlCommand(action="set_seed", value=int(val))

    if action == "regenerate_scenario":
        extra = keys - {"type", "action", "scenario_mode", "value", "fall_style"}
        if extra:
            raise ControlMessageError(f"Unexpected keys for regenerate_scenario: {sorted(extra)}")
        scen_mode = raw.get("scenario_mode")
        if not isinstance(scen_mode, str) or scen_mode not in allowed_scenario_modes:
            raise ControlMessageError(
                f"Invalid scenario_mode {scen_mode!r}; expected one of {list(allowed_scenario_modes)}"
            )
        seed_val = raw.get("value")
        if seed_val is not None:
            if isinstance(seed_val, bool) or not isinstance(seed_val, int):
                raise ControlMessageError(f"Optional seed value must be an integer, got {seed_val!r}")
        fall_style = raw.get("fall_style")
        if fall_style is not None and not isinstance(fall_style, str):
            raise ControlMessageError("fall_style must be a string or null")
        return ControlCommand(
            action="regenerate_scenario",
            scenario_mode=scen_mode,
            value=seed_val,
            fall_style=fall_style,
        )

    if action == "set_impairment":
        extra = keys - {"type", "action", "name", "enabled"}
        if extra or "name" not in raw or "enabled" not in raw:
            raise ControlMessageError(
                f"set_impairment requires 'name' and 'enabled'; unexpected keys: {sorted(extra)}"
            )
        name = raw["name"]
        enabled = raw["enabled"]
        if not isinstance(name, str) or name not in KNOWN_IMPAIRMENTS:
            raise ControlMessageError(
                f"Unknown impairment {name!r}; expected one of {sorted(KNOWN_IMPAIRMENTS)}"
            )
        if not isinstance(enabled, bool):
            raise ControlMessageError(f"set_impairment 'enabled' must be boolean, got {enabled!r}")
        return ControlCommand(action="set_impairment", name=name, enabled=enabled)

    if action == "set_mode":
        extra = keys - {"type", "action", "mode"}
        if extra or "mode" not in raw:
            raise ControlMessageError(f"set_mode requires 'mode'; unexpected keys: {sorted(extra)}")
        mode = raw["mode"]
        if not isinstance(mode, str) or mode not in VALID_MODES:
            raise ControlMessageError(f"Invalid mode {mode!r}; expected one of {sorted(VALID_MODES)}")
        return ControlCommand(action="set_mode", mode=mode)

    raise ControlMessageError(f"Unknown control action: {action!r}")


def select_display_rf_paths(path_set: PathSet, *, max_paths: int = 72) -> list[dict[str, Any]]:
    """Select representative TX0/RX0 (or strongest) RF paths per type for crisp Panel 2 rendering."""
    # Prefer canonical TX0/RX0 paths so 12 near-identical virtual channels do not overlap visually,
    # falling back to strongest paths if TX0/RX0 is empty.
    tx0_rx0 = [p for p in path_set.paths if p.tx_id == 0 and p.rx_id == 0]
    pool = tx0_rx0 if tx0_rx0 else list(path_set.paths)

    by_type: dict[str, list[Any]] = {
        "los": [],
        "specular_reflection": [],
        "diffuse_scatter": [],
        "transmitted": [],
    }
    for p in pool:
        by_type.setdefault(p.path_type, []).append(p)
    for plist in by_type.values():
        plist.sort(key=lambda p: p.amplitude_linear, reverse=True)

    quotas = {
        "los": max(1, int(max_paths * 0.40)),
        "specular_reflection": max(1, int(max_paths * 0.30)),
        "diffuse_scatter": max(1, int(max_paths * 0.20)),
        "transmitted": max(1, int(max_paths * 0.10)),
    }
    selected: list[dict[str, Any]] = []
    for ptype, q in quotas.items():
        for p in by_type.get(ptype, [])[:q]:
            selected.append(
                {
                    "path_id": p.path_id,
                    "path_type": p.path_type,
                    "vertices_m": [[round(c, 4) for c in v] for v in p.vertices_m],
                    "total_path_length_m": round(p.total_path_length_m, 4),
                    "delay_s": p.delay_s,
                    "path_loss_db": round(p.path_loss_db, 2),
                    "amplitude_linear": p.amplitude_linear,
                    "radial_velocity_mps": round(p.radial_velocity_mps, 4),
                    "doppler_hz": round(p.doppler_hz, 2),
                    "reflection_coeff_mag": round(abs(p.reflection_coeff), 4),
                    "target_id": p.target_id,
                    "surface_id": p.surface_id,
                }
            )
    return selected[:max_paths]


def _summarize_link_budget(path_set: PathSet) -> dict[str, Any]:
    tx0_rx0 = [p for p in path_set.paths if p.tx_id == 0 and p.rx_id == 0]
    pool = tx0_rx0 if tx0_rx0 else path_set.paths
    best: dict[str, float | None] = {
        "los": None,
        "specular_reflection": None,
        "diffuse_scatter": None,
        "transmitted": None,
    }
    for p in pool:
        cur = best.get(p.path_type)
        if cur is None or p.path_loss_db < cur:
            best[p.path_type] = round(p.path_loss_db, 2)
    return {
        "strongest_los_loss_db": best["los"],
        "strongest_reflection_loss_db": best["specular_reflection"],
        "strongest_diffuse_loss_db": best["diffuse_scatter"],
        "strongest_transmitted_loss_db": best["transmitted"],
    }


def build_simulation_frame_message(
    *,
    frame_id: int,
    sim_time_s: float,
    seed: int,
    scene_id: str,
    motion_frame: MotionFrame,
    path_set: PathSet,
    dsp_frame: DSPFrame,
    config_hash: str,
    seed_map: Mapping[str, int],
    mode: str,
    control_state: Mapping[str, Any],
    max_rf_paths: int = 72,
) -> dict[str, Any]:
    """Assemble one synchronized ``INTERFACES.md`` §8 WebSocket ``simulation_frame`` JSON message."""
    gt_dict = motion_frame.to_ground_truth_dict()
    gt_dict["action_label"] = str(gt_dict["action_label"])
    analytics_dict = dsp_frame.to_websocket_analytics_dict()
    analytics_dict["aperture_limitations"] = dsp_frame.diagnostics.get("aperture_limitations", {})
    analytics_dict["detection_branch"] = dsp_frame.diagnostics.get("detection_branch", "clutter_suppressed")

    return {
        "schema_version": "1.0",
        "type": "simulation_frame",
        "frame_id": int(frame_id),
        "sim_time_s": round(float(sim_time_s), 6),
        "seed": int(seed),
        "scene_id": scene_id,
        "ground_truth": gt_dict,
        "rf": {
            "paths": select_display_rf_paths(path_set, max_paths=max_rf_paths),
            "ray_summary": {
                **dict(path_set.diagnostics),
                "link_summary": _summarize_link_budget(path_set),
            },
        },
        "radar": dsp_frame.to_websocket_radar_dict(),
        "analytics": analytics_dict,
        "control_state": dict(control_state),
        "provenance": {
            "mode": mode,
            "fidelity_label": "preview" if mode == "preview" else "simulated",
            "config_hash": config_hash,
            "seed_map": dict(seed_map),
        },
    }
