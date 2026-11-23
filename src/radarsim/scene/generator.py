"""Seeded random room-layout generator with strict geometric rejection (never silent clipping)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

from radarsim.config import (
    InitialActorConfig,
    MaterialDatabase,
    OpeningConfig,
    RadarMountConfig,
    RoomConfig,
    SceneConfig,
    SceneObjectConfig,
)
from radarsim.rng import make_generator
from radarsim.scene.model import Scene, build_scene
from radarsim.scene.validation import SceneValidationReport, validate_scene_geometry


@dataclass(frozen=True)
class LayoutGenerationBounds:
    """Configurable room-dimension and clearance bounds for seeded layout generation.

    Supports rooms up to 8.0 m x 6.0 m per ``PLAN.md`` without code changes.
    """

    width_x_range_m: tuple[float, float] = (3.8, 8.0)
    depth_y_range_m: tuple[float, float] = (4.4, 6.0)
    height_z_range_m: tuple[float, float] = (2.5, 3.2)
    min_object_clearance_m: float = 0.08
    wall_margin_m: float = 0.05
    doorway_clearance_depth_m: float = 1.0
    doorway_side_margin_m: float = 0.10
    min_walkway_width_m: float = 0.55
    actor_entry_clearance_m: float = 0.25
    allowed_yaws_rad: tuple[float, ...] = (
        0.0,
        0.5 * math.pi,
        math.pi,
        1.5 * math.pi,
    )
    continuous_yaw: bool = False


@dataclass(frozen=True)
class GeneratedLayoutResult:
    """Outcome of one seeded layout-generation run."""

    seed: int
    accepted: bool
    scene: Scene | None
    validation: SceneValidationReport
    attempts: int
    rejection_reasons: tuple[str, ...]


def _sample_candidate_scene_config(
    base_config: SceneConfig,
    *,
    rng,
    seed: int,
    attempt: int,
    bounds: LayoutGenerationBounds,
) -> SceneConfig:
    width_x = float(rng.uniform(*bounds.width_x_range_m))
    depth_y = float(rng.uniform(*bounds.depth_y_range_m))
    height_z = float(rng.uniform(*bounds.height_z_range_m))

    new_room = replace(
        base_config.room,
        width_x_m=round(width_x, 4),
        depth_y_m=round(depth_y, 4),
        height_z_m=round(height_z, 4),
    )

    # Reposition openings cleanly onto the sampled room walls while preserving dimensions/layers
    new_openings: list[OpeningConfig] = []
    door_center_x: float | None = None
    for op in base_config.openings:
        if op.wall in ("y_min", "y_max"):
            wall_len = new_room.width_x_m
            y_coord = 0.0 if op.wall == "y_min" else new_room.depth_y_m
            margin = max(0.25, 0.5 * op.width_m + 0.15)
            lo = min(margin, 0.5 * wall_len)
            hi = max(lo, wall_len - margin)
            cx = float(rng.uniform(lo, hi))
            new_center = (round(cx, 4), y_coord, op.center_m[2])
            if op.kind == "door" and op.wall == "y_min" and door_center_x is None:
                door_center_x = new_center[0]
        else:
            wall_len = new_room.depth_y_m
            x_coord = 0.0 if op.wall == "x_min" else new_room.width_x_m
            margin = max(0.25, 0.5 * op.width_m + 0.15)
            lo = min(margin, 0.5 * wall_len)
            hi = max(lo, wall_len - margin)
            cy = float(rng.uniform(lo, hi))
            new_center = (x_coord, round(cy, 4), op.center_m[2])
        new_openings.append(replace(op, center_m=new_center))

    # Position radar centered horizontally on its mount wall, respecting sampled room height
    rad_z = min(base_config.radar.position_m[2], new_room.height_z_m - 0.25)
    rad_x = round(0.5 * new_room.width_x_m, 4)
    rad_y = base_config.radar.position_m[1]
    if base_config.radar.wall == "y_max":
        rad_y = round(new_room.depth_y_m - 0.08, 4)
    new_radar: RadarMountConfig = replace(
        base_config.radar,
        position_m=(rad_x, rad_y, round(rad_z, 4)),
    )

    # Align initial actor with entrance door if present
    new_actor: InitialActorConfig | None = base_config.initial_actor
    if new_actor is not None and door_center_x is not None:
        new_actor = replace(
            new_actor,
            start_position_m=(door_center_x, new_actor.start_position_m[1], 0.0),
            entry_target_m=(door_center_x, min(new_actor.entry_target_m[1], new_room.depth_y_m - 0.5), 0.0),
        )

    # Sample furniture placements inside the room span without silent clipping
    new_objects: list[SceneObjectConfig] = []
    for obj in base_config.objects:
        if bounds.continuous_yaw:
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        else:
            yaw_idx = int(rng.integers(0, len(bounds.allowed_yaws_rad)))
            yaw = float(bounds.allowed_yaws_rad[yaw_idx])

        dx, dy, _ = obj.dimensions_m
        # Compute exact rotated footprint extents to sample the box centre cleanly;
        # if the room is too small, place at room centre and let validate_scene_geometry
        # explicitly reject OBJECT_OUT_OF_BOUNDS (never silently clip dimensions).
        eff_hx = 0.5 * (abs(dx * math.cos(yaw)) + abs(dy * math.sin(yaw)))
        eff_hy = 0.5 * (abs(dx * math.sin(yaw)) + abs(dy * math.cos(yaw)))
        cx_lo = bounds.wall_margin_m + eff_hx
        cx_hi = new_room.width_x_m - bounds.wall_margin_m - eff_hx
        cy_lo = bounds.wall_margin_m + eff_hy
        cy_hi = new_room.depth_y_m - bounds.wall_margin_m - eff_hy

        cx = float(rng.uniform(cx_lo, cx_hi)) if cx_lo <= cx_hi else 0.5 * new_room.width_x_m
        cy = float(rng.uniform(cy_lo, cy_hi)) if cy_lo <= cy_hi else 0.5 * new_room.depth_y_m
        px = cx - 0.5 * dx
        py = cy - 0.5 * dy
        new_objects.append(
            replace(
                obj,
                position_m=(round(px, 4), round(py, 4), obj.position_m[2]),
                yaw_rad=yaw,
            )
        )

    return replace(
        base_config,
        scene_id=f"{base_config.scene_id}_seed_{seed}_try_{attempt}",
        room=new_room,
        openings=tuple(new_openings),
        objects=tuple(new_objects),
        radar=new_radar,
        initial_actor=new_actor,
    )


def generate_random_layout(
    base_scene_config: SceneConfig,
    material_db: MaterialDatabase,
    seed: int,
    *,
    bounds: LayoutGenerationBounds | None = None,
    max_attempts: int = 1,
    stream_name: str = "layout",
) -> GeneratedLayoutResult:
    """Generate a seeded random room layout, strictly rejecting invalid geometries.

    If ``max_attempts == 1``, evaluates a single deterministic candidate and
    either accepts it or rejects it with explicit geometric reasons.
    If ``max_attempts > 1``, evaluates up to ``max_attempts`` deterministic
    candidates, returning the first strictly valid layout or rejecting with
    the explicit reasons from the last rejected attempt.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    cfg_bounds = LayoutGenerationBounds() if bounds is None else bounds
    rng = make_generator(seed, stream_name)

    last_report: SceneValidationReport | None = None
    for attempt in range(1, max_attempts + 1):
        candidate_cfg = _sample_candidate_scene_config(
            base_scene_config,
            rng=rng,
            seed=seed,
            attempt=attempt,
            bounds=cfg_bounds,
        )
        candidate_scene = build_scene(candidate_cfg, material_db)
        report = validate_scene_geometry(
            candidate_scene,
            min_object_clearance_m=cfg_bounds.min_object_clearance_m,
            wall_margin_m=cfg_bounds.wall_margin_m,
            doorway_clearance_depth_m=cfg_bounds.doorway_clearance_depth_m,
            doorway_side_margin_m=cfg_bounds.doorway_side_margin_m,
            min_walkway_width_m=cfg_bounds.min_walkway_width_m,
            actor_entry_clearance_m=cfg_bounds.actor_entry_clearance_m,
        )
        last_report = report
        if report.valid:
            return GeneratedLayoutResult(
                seed=seed,
                accepted=True,
                scene=candidate_scene,
                validation=report,
                attempts=attempt,
                rejection_reasons=(),
            )

    assert last_report is not None
    return GeneratedLayoutResult(
        seed=seed,
        accepted=False,
        scene=None,
        validation=last_report,
        attempts=max_attempts,
        rejection_reasons=last_report.reasons,
    )


def generate_layout_batch(
    base_scene_config: SceneConfig,
    material_db: MaterialDatabase,
    seeds: Iterable[int],
    *,
    bounds: LayoutGenerationBounds | None = None,
    max_attempts: int = 1,
    stream_name: str = "layout",
) -> list[GeneratedLayoutResult]:
    """Generate and validate a deterministic batch of seeded random layouts."""
    return [
        generate_random_layout(
            base_scene_config,
            material_db,
            seed,
            bounds=bounds,
            max_attempts=max_attempts,
            stream_name=stream_name,
        )
        for seed in seeds
    ]
