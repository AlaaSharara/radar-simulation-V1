"""FastAPI HTTP and WebSocket server implementing ``INTERFACES.md`` §8–10."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from radarsim.viz.protocol import ControlMessageError
from radarsim.viz.session import SimulationSession


def create_app(
    root_dir: str | Path = ".",
    *,
    radar_path: str | Path | None = None,
    scene_path: str | Path | None = None,
    mode: str = "preview",
    seed: int | None = None,
    scenario_mode: str | None = None,
    session: SimulationSession | None = None,
) -> FastAPI:
    """Create the FastAPI application exposing ``/api/*``, ``/assets/avatars/*``, and ``/ws/sim``."""
    sim_session = (
        session
        if session is not None
        else SimulationSession(
            root_dir=root_dir,
            radar_path=radar_path,
            scene_path=scene_path,
            mode=mode,
            seed=seed,
            scenario_mode=scenario_mode,
        )
    )
    avatars_dir = sim_session.root_dir / "assets" / "avatars"

    app = FastAPI(
        title="Radar Room Simulation Backend",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.session = sim_session

    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        return sim_session.health_payload()

    @app.get("/api/config-summary")
    async def api_config_summary() -> dict[str, Any]:
        return sim_session.config_summary_payload()

    @app.get("/api/scenario-modes")
    async def api_scenario_modes() -> dict[str, Any]:
        return sim_session.scenario_modes_payload()

    @app.get("/assets/avatars/{filename}")
    async def api_avatar_asset(filename: str) -> FileResponse:
        # Restrict strictly to single-segment filenames inside assets/avatars/
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name:
            raise HTTPException(status_code=400, detail="Invalid asset filename")
        target = (avatars_dir / safe_name).resolve()
        if not target.exists() and safe_name == "adult_male_default.glb" and target.parent == avatars_dir.resolve():
            from radarsim.motion.avatar import build_procedural_human_glb

            build_procedural_human_glb(target, avatars_dir / "adult_male_default_rig.json")
        if not target.exists() or target.parent != avatars_dir.resolve():
            raise HTTPException(status_code=404, detail=f"Avatar asset not found: {safe_name}")
        media_type = (
            "model/gltf-binary"
            if target.suffix.lower() == ".glb"
            else "application/json"
            if target.suffix.lower() == ".json"
            else "application/octet-stream"
        )
        return FileResponse(target, media_type=media_type)

    ws_path = sim_session.project.sim_profile.backend.websocket_path

    @app.websocket(ws_path)
    async def websocket_sim(websocket: WebSocket) -> None:
        await websocket.accept()
        stop_event = asyncio.Event()
        send_lock = asyncio.Lock()

        async def _safe_send(payload: dict[str, Any]) -> bool:
            try:
                async with send_lock:
                    await websocket.send_json(payload)
                return True
            except Exception:
                stop_event.set()
                return False

        # Send initial synchronized frame immediately on connection
        initial_frame = await asyncio.to_thread(sim_session.compute_frame, advance=False)
        if not await _safe_send(initial_frame):
            return

        async def _receiver() -> None:
            try:
                while not stop_event.is_set():
                    raw = await websocket.receive_json()
                    try:
                        ack = await asyncio.to_thread(sim_session.apply_control, raw)
                        await _safe_send(ack)
                        # On step / seed / regenerate / impairment / mode change, push an immediate frame update
                        action = ack.get("action")
                        if action in (
                            "step",
                            "set_seed",
                            "regenerate_scenario",
                            "set_impairment",
                            "set_mode",
                        ):
                            frame = await asyncio.to_thread(
                                sim_session.compute_frame,
                                advance=(action == "step"),
                            )
                            await _safe_send(frame)
                    except ControlMessageError as exc:
                        await _safe_send({"type": "control_error", "error": str(exc)})
            except WebSocketDisconnect:
                stop_event.set()
            except Exception:
                stop_event.set()

        async def _producer() -> None:
            loop = asyncio.get_running_loop()
            while not stop_event.is_set():
                if not sim_session.playing:
                    await asyncio.sleep(0.05)
                    continue
                t0 = loop.time()
                frame = await asyncio.to_thread(sim_session.compute_frame, advance=True)
                ok = await _safe_send(frame)
                if not ok:
                    break
                # speed<1 slows the stream; speed>1 keeps the stream rate and skips motion frames instead
                target_fps = max(1.0, sim_session.frame_rate_hz * min(1.0, sim_session.speed))
                target_dt = 1.0 / target_fps
                elapsed = loop.time() - t0
                sleep_s = max(0.001, target_dt - elapsed)
                await asyncio.sleep(sleep_s)

        recv_task = asyncio.create_task(_receiver())
        prod_task = asyncio.create_task(_producer())
        try:
            await asyncio.wait({recv_task, prod_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_event.set()
            recv_task.cancel()
            prod_task.cancel()

    return app
