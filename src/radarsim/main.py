"""Unified CLI entrypoint for interactive dashboard serving and headless batch/export runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from radarsim.motion import FALL_STYLES
from radarsim.pipeline import SimulationPipeline, export_run_npz
from radarsim.viz.protocol import VALID_MODES
from radarsim.viz.server import create_app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m radarsim.main",
        description="Indoor FMCW/TDM-MIMO Radar Room Simulator (backend server & batch exporter)",
    )
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Workspace root directory")
    parser.add_argument("--config", type=Path, default=None, help="Path to radar_config.yaml")
    parser.add_argument("--scene", type=Path, default=None, help="Path to scene.json")
    parser.add_argument("--material-db", type=Path, default=None, help="Path to material_db.yaml")
    parser.add_argument("--motion-config", type=Path, default=None, help="Path to motion_config.yaml")
    parser.add_argument("--sim-profile", type=Path, default=None, help="Path to sim_profile.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Master random seed")
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default=None,
        help="Scenario mode (e.g. normal_daily_routine, fall_demo, random_mix)",
    )
    parser.add_argument(
        "--fall-style",
        type=str,
        default=None,
        choices=FALL_STYLES,
        help="Explicit fall style override when generating a fall scenario",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="preview",
        choices=sorted(VALID_MODES),
        help="Execution fidelity mode: preview, full_sim, or analysis_export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/exports"),
        help="Output folder for exported NPZ/metadata archives",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default=None,
        help="Optional filename prefix for exported artifacts",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed frames for bounded headless runs",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Optional starting frame index within the sampled scenario",
    )
    parser.add_argument(
        "--adc-window-frames",
        type=int,
        default=16,
        help="Number of contiguous complex ADC frames retained inside the NPZ archive",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run headlessly, export NPZ/metadata, and exit without starting the HTTP server",
    )
    parser.add_argument("--host", type=str, default=None, help="Backend bind host (server mode)")
    parser.add_argument("--port", type=int, default=None, help="Backend bind port (server mode)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    pipeline = SimulationPipeline(
        root_dir=args.root_dir,
        radar_path=args.config,
        scene_path=args.scene,
        material_db_path=args.material_db,
        motion_path=args.motion_config,
        sim_profile_path=args.sim_profile,
    )

    if args.mode == "analysis_export" or args.batch:
        result = pipeline.run_scenario(
            seed=args.seed,
            scenario_mode=args.scenario_mode,
            fall_style=args.fall_style,
            mode=args.mode,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            adc_window_frames=args.adc_window_frames,
        )
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = pipeline.root_dir / out_dir
        npz_path, meta_path = export_run_npz(result, out_dir, prefix=args.export_prefix)
        payload = {
            **result.summary_dict(),
            "npz_path": str(npz_path),
            "metadata_path": str(meta_path),
        }
        print(json.dumps(payload, indent=2))
        return 0

    import uvicorn

    session = pipeline.create_session(
        mode=args.mode,
        seed=args.seed,
        scenario_mode=args.scenario_mode,
        fall_style=args.fall_style,
    )
    app = create_app(root_dir=pipeline.root_dir, session=session)
    host = args.host or pipeline.project.sim_profile.backend.host
    port = args.port or pipeline.project.sim_profile.backend.port
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
