"""Scene geometry validation: object intersections, wall bounds, and doorway clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from radarsim.config import RoomConfig
from radarsim.scene.geometry import OrientedBox
from radarsim.scene.model import Scene, SceneOpening, _wall_basis


@dataclass(frozen=True)
class GeometryIssue:
    """Explicit reason why a scene geometry validation check failed."""

    code: str
    message: str
    entity_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "entity_ids": list(self.entity_ids),
        }


@dataclass(frozen=True)
class SceneValidationReport:
    """Structured result of geometric validation on a ``Scene``."""

    valid: bool
    issues: tuple[GeometryIssue, ...]
    checks: dict[str, bool]

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"[{issue.code}] {issue.message}" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": dict(self.checks),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def doorway_clearance_box(
    opening: SceneOpening,
    room: RoomConfig,
    *,
    depth_m: float = 1.0,
    side_margin_m: float = 0.10,
) -> OrientedBox:
    """Construct the interior doorway approach/swing clearance box that must stay unobstructed."""
    if opening.kind != "door":
        raise ValueError(f"Expected a door opening, got {opening.opening_id!r} ({opening.kind!r})")
    wall_len, _, _, _ = _wall_basis(opening.wall, room)
    s0 = max(0.0, opening.span_along_wall_m[0] - side_margin_m)
    s1 = min(wall_len, opening.span_along_wall_m[1] + side_margin_m)
    span_w = max(1e-6, s1 - s0)
    z0 = opening.sill_height_m
    z_h = opening.height_m
    eff_depth = min(
        depth_m,
        room.depth_y_m if opening.wall in ("y_min", "y_max") else room.width_x_m,
    )

    if opening.wall == "y_min":
        return OrientedBox.from_corner_and_dimensions(
            (s0, 0.0, z0), (span_w, eff_depth, z_h), yaw_rad=0.0
        )
    if opening.wall == "y_max":
        return OrientedBox.from_corner_and_dimensions(
            (s0, room.depth_y_m - eff_depth, z0), (span_w, eff_depth, z_h), yaw_rad=0.0
        )
    if opening.wall == "x_min":
        return OrientedBox.from_corner_and_dimensions(
            (0.0, s0, z0), (eff_depth, span_w, z_h), yaw_rad=0.0
        )
    if opening.wall == "x_max":
        return OrientedBox.from_corner_and_dimensions(
            (room.width_x_m - eff_depth, s0, z0), (eff_depth, span_w, z_h), yaw_rad=0.0
        )
    raise ValueError(f"Unsupported wall identifier: {opening.wall!r}")


def validate_scene_geometry(
    scene: Scene,
    *,
    min_object_clearance_m: float = 0.0,
    wall_margin_m: float = 0.0,
    doorway_clearance_depth_m: float = 1.0,
    doorway_side_margin_m: float = 0.10,
    min_walkway_width_m: float = 0.55,
    actor_entry_clearance_m: float = 0.25,
) -> SceneValidationReport:
    """Run wall-boundary, object-overlap, opening, doorway-clearance, and construction checks."""
    issues: list[GeometryIssue] = []
    room = scene.room

    # 1. Wall-boundary check for every furniture object
    bounds_ok = True
    for obj in scene.objects:
        if not obj.box.is_within_room(
            room.width_x_m,
            room.depth_y_m,
            room.height_z_m,
            wall_margin_m=wall_margin_m,
        ):
            bounds_ok = False
            bb = obj.box.aabb()
            issues.append(
                GeometryIssue(
                    code="OBJECT_OUT_OF_BOUNDS",
                    message=(
                        f"Object '{obj.object_id}' AABB {bb.min_m}..{bb.max_m} violates room bounds "
                        f"[0..{room.width_x_m}, 0..{room.depth_y_m}, 0..{room.height_z_m}] "
                        f"with wall_margin_m={wall_margin_m}"
                    ),
                    entity_ids=(obj.object_id,),
                )
            )

    # 2. Pairwise object-to-object intersection / clearance check
    overlap_ok = True
    objs = scene.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if objs[i].box.intersects_box(objs[j].box, clearance_m=min_object_clearance_m):
                overlap_ok = False
                dist = objs[i].box.distance_2d(objs[j].box)
                issues.append(
                    GeometryIssue(
                        code="OBJECT_OVERLAP",
                        message=(
                            f"Objects '{objs[i].object_id}' and '{objs[j].object_id}' "
                            f"overlap or violate clearance ({dist:.3f} m < {min_object_clearance_m:.3f} m)"
                        ),
                        entity_ids=(objs[i].object_id, objs[j].object_id),
                    )
                )

    # 3. Opening bounds and non-overlap on each wall
    openings_ok = True
    for wall_name in ("y_min", "y_max", "x_min", "x_max"):
        wall_len, _, _, _ = _wall_basis(wall_name, room)
        wall_ops = [op for op in scene.openings if op.wall == wall_name]
        for op in wall_ops:
            s0, s1 = op.span_along_wall_m
            if s0 < -1e-9 or s1 > wall_len + 1e-9:
                openings_ok = False
                issues.append(
                    GeometryIssue(
                        code="OPENING_OUT_OF_BOUNDS",
                        message=(
                            f"Opening '{op.opening_id}' span [{s0:.3f}, {s1:.3f}] exceeds "
                            f"wall '{wall_name}' length {wall_len:.3f} m"
                        ),
                        entity_ids=(op.opening_id,),
                    )
                )
            if op.sill_height_m < -1e-9 or (op.sill_height_m + op.height_m) > room.height_z_m + 1e-9:
                openings_ok = False
                issues.append(
                    GeometryIssue(
                        code="OPENING_OUT_OF_BOUNDS",
                        message=(
                            f"Opening '{op.opening_id}' vertical span "
                            f"[{op.sill_height_m:.3f}, {op.sill_height_m + op.height_m:.3f}] "
                            f"exceeds room height {room.height_z_m:.3f} m"
                        ),
                        entity_ids=(op.opening_id,),
                    )
                )
        wall_ops_sorted = sorted(wall_ops, key=lambda o: o.span_along_wall_m[0])
        for k in range(len(wall_ops_sorted) - 1):
            a, b = wall_ops_sorted[k], wall_ops_sorted[k + 1]
            if a.span_along_wall_m[1] > b.span_along_wall_m[0] + 1e-9:
                openings_ok = False
                issues.append(
                    GeometryIssue(
                        code="OPENING_OVERLAP",
                        message=(
                            f"Openings '{a.opening_id}' and '{b.opening_id}' overlap on wall '{wall_name}'"
                        ),
                        entity_ids=(a.opening_id, b.opening_id),
                    )
                )

    # 4. Doorway navigability & clearance check
    doorway_ok = True
    doors = scene.doors()
    if not doors:
        doorway_ok = False
        issues.append(
            GeometryIssue(
                code="NO_DOOR_OPENING",
                message="Scene has no door opening; room is not navigable",
            )
        )
    for door in doors:
        if door.clear_opening_width_m < min_walkway_width_m - 1e-9:
            doorway_ok = False
            issues.append(
                GeometryIssue(
                    code="DOOR_TOO_NARROW",
                    message=(
                        f"Door '{door.opening_id}' clear width ({door.clear_opening_width_m:.3f} m) "
                        f"is narrower than minimum navigable width ({min_walkway_width_m:.3f} m)"
                    ),
                    entity_ids=(door.opening_id,),
                )
            )
        clearance = doorway_clearance_box(
            door,
            room,
            depth_m=doorway_clearance_depth_m,
            side_margin_m=doorway_side_margin_m,
        )
        for obj in scene.objects:
            if obj.box.intersects_box(clearance, clearance_m=0.0):
                doorway_ok = False
                issues.append(
                    GeometryIssue(
                        code="DOORWAY_BLOCKED",
                        message=(
                            f"Object '{obj.object_id}' intrudes into clearance zone of door '{door.opening_id}'"
                        ),
                        entity_ids=(door.opening_id, obj.object_id),
                    )
                )

    if scene.initial_actor is not None:
        ex, ey, ez = scene.initial_actor.entry_target_m
        if not scene.is_point_inside_room(ex, ey, ez, margin_m=actor_entry_clearance_m):
            doorway_ok = False
            issues.append(
                GeometryIssue(
                    code="ENTRY_TARGET_OUT_OF_BOUNDS",
                    message=(
                        f"Initial actor entry_target_m {(ex, ey, ez)} does not lie inside room "
                        f"with clearance {actor_entry_clearance_m:.2f} m"
                    ),
                    entity_ids=(scene.initial_actor.actor_id,),
                )
            )
        elif not scene.is_point_free_2d(ex, ey, radius_m=actor_entry_clearance_m):
            doorway_ok = False
            issues.append(
                GeometryIssue(
                    code="ENTRY_TARGET_BLOCKED",
                    message=(
                        f"Initial actor entry_target_m {(ex, ey, ez)} is blocked by furniture "
                        f"(clearance radius {actor_entry_clearance_m:.2f} m)"
                    ),
                    entity_ids=(scene.initial_actor.actor_id,),
                )
            )

    # 5. Radar mount bounds & inward boresight check
    radar_ok = True
    rx, ry, rz = scene.radar_mount.position_m
    if not scene.is_point_inside_room(rx, ry, rz):
        radar_ok = False
        issues.append(
            GeometryIssue(
                code="RADAR_OUT_OF_BOUNDS",
                message=f"Radar '{scene.radar_mount.id}' position {(rx, ry, rz)} is outside room bounds",
                entity_ids=(scene.radar_mount.id,),
            )
        )
    if scene.radar_mount.wall is not None:
        _, _, _, n_in = _wall_basis(scene.radar_mount.wall, room)
        dot_in = float(
            np.dot(
                np.asarray(scene.radar_frame.boresight_world, dtype=np.float64),
                np.asarray(n_in, dtype=np.float64),
            )
        )
        if dot_in <= 0.1:
            radar_ok = False
            issues.append(
                GeometryIssue(
                    code="RADAR_BORESIGHT_OUTWARD",
                    message=(
                        f"Radar '{scene.radar_mount.id}' boresight does not point into room "
                        f"from wall '{scene.radar_mount.wall}' (dot={dot_in:.3f})"
                    ),
                    entity_ids=(scene.radar_mount.id,),
                )
            )

    # 6. Wall construction layer thickness consistency check
    construction_ok = True
    if room.construction and room.construction.walls:
        layer_sum = sum(layer.thickness_m for layer in room.construction.walls.layers)
        if abs(layer_sum - room.wall_thickness_m) > 1e-4:
            construction_ok = False
            issues.append(
                GeometryIssue(
                    code="WALL_THICKNESS_MISMATCH",
                    message=(
                        f"Sum of wall construction layer thicknesses ({layer_sum:.4f} m) does not match "
                        f"room.wall_thickness_m ({room.wall_thickness_m:.4f} m)"
                    ),
                    entity_ids=("room",),
                )
            )

    checks = {
        "wall_boundaries": bounds_ok,
        "object_non_overlap": overlap_ok,
        "opening_bounds": openings_ok,
        "doorway_clearance": doorway_ok,
        "radar_mount": radar_ok,
        "wall_construction_thickness": construction_ok,
    }
    return SceneValidationReport(
        valid=(len(issues) == 0),
        issues=tuple(issues),
        checks=checks,
    )
