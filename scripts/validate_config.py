#!/usr/bin/env python3
"""Validate all project configs and independently verify derived radar metrics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from radarsim.config import (  # noqa: E402
    ConfigValidationError,
    compare_derived_metrics,
    load_project_config,
)

CONFIG_PATH = ROOT / "radar_config.yaml"


def require(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing required configuration field: {'.'.join(keys)}")
        value = value[key]
    return value


def derive(config: dict[str, Any]) -> dict[str, float]:
    """Independently derive radar metrics from a raw radar_config mapping.

    Does not read ``expected_derived`` as an input.
    """
    rf = require(config, "rf")
    wf = require(config, "waveform")
    c = float(require(rf, "speed_of_light_m_per_s"))
    fc = float(require(rf, "carrier_frequency_hz"))
    slope = float(require(wf, "slope_hz_per_s"))
    fs = float(require(wf, "adc_sample_rate_hz"))
    n_adc = int(require(wf, "adc_samples_per_chirp"))
    max_if = float(require(rf, "receiver", "max_if_frequency_hz"))
    chirp_period = float(require(wf, "chirp_period_s"))
    n_tx = int(require(rf, "tx_count"))
    tdm_mode = bool(rf.get("tdm_mode", True))
    n_doppler = int(require(wf, "chirps_per_tx_per_frame"))

    if c <= 0 or fc <= 0 or slope <= 0 or fs <= 0 or n_adc <= 0:
        raise ValueError("Physical constants, slope, sample rate and ADC samples must be positive")
    if max_if <= 0 or max_if >= fs / 2:
        raise ValueError("max_if_frequency_hz must be positive and below the real-ADC Nyquist frequency")
    if chirp_period <= 0 or n_tx <= 0 or n_doppler <= 0:
        raise ValueError("Chirp period, TX count and Doppler chirps must be positive")

    effective_time = n_adc / fs
    effective_bandwidth = slope * effective_time
    wavelength = c / fc
    same_tx_interval = chirp_period * (n_tx if tdm_mode else 1)
    return {
        "effective_bandwidth_hz": effective_bandwidth,
        "range_resolution_m": c / (2.0 * effective_bandwidth),
        "max_unambiguous_range_m": c * max_if / (2.0 * slope),
        "wavelength_m": wavelength,
        "same_tx_chirp_interval_s": same_tx_interval,
        "max_unambiguous_radial_velocity_mps": wavelength / (4.0 * same_tx_interval),
        "velocity_resolution_mps": wavelength / (2.0 * n_doppler * same_tx_interval),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate radar/scene/material/motion/sim configs and derived radar metrics."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Directory containing radar_config.yaml, scene.json, material_db.yaml, motion_config.yaml, and sim_profile.yaml.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        project = load_project_config(
            args.root,
            require_expected_derived=True,
            validate_expected=False,
        )
    except ConfigValidationError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    assert project.expected_derived is not None
    comparisons = compare_derived_metrics(project.derived, project.expected_derived)

    print(f"Validated project configuration (hash={project.config_hash[:12]}):")
    print(f"  Radar profile : {project.radar.profile.id} ({project.radar.profile.name})")
    print(f"  Scene         : {project.scene.scene_id} ({project.scene.room.width_x_m}m x {project.scene.room.depth_y_m}m x {project.scene.room.height_z_m}m)")
    print(f"  Materials     : {len(project.materials.materials)} entries, {len(project.materials.aliases)} aliases")
    print(f"  Motion actor  : {project.motion.actor.actor_id} (default_mode={project.motion.scenario.default_mode}, seed={project.motion.scenario.random_seed})")
    print(f"  Sim profile   : {project.sim_profile.backend.host}:{project.sim_profile.backend.port}{project.sim_profile.backend.websocket_path} ({project.sim_profile.backend.default_mode})")
    print("\nDerived radar metrics (SI units):")

    failed: list[str] = []
    for item in comparisons:
        status = "PASS" if item.passed else "FAIL"
        print(
            f"  {item.name:44s} calculated={item.calculated:.12g} "
            f"expected={item.expected:.12g} |error|={item.abs_error:.3g} "
            f"tol={item.tolerance:.3g} {status}"
        )
        if not item.passed:
            failed.append(item.name)

    if failed:
        print("\nFAILED: " + ", ".join(failed), file=sys.stderr)
        return 1

    print("\nAll 5 configuration files are valid and all expected-derived values match tolerances.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
