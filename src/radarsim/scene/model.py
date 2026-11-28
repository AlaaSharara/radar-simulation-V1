"""Normalized 3D room scene model shared by motion, RF propagation, and visualization.

Single source of truth built strictly from ``scene.json`` (``SceneConfig``) and
``material_db.yaml`` (``MaterialDatabase``). Downstream modules must consume
``Scene`` instead of hardcoding room dimensions, furniture boxes, or material
properties.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from radarsim.config import (
    InitialActorConfig,
    MaterialDatabase,
    OpeningConfig,
    RadarMountConfig,
    RoomConfig,
    SceneConfig,
    SceneObjectConfig,
    load_material_db,
    load_scene_config,
)
from radarsim.scene.furniture_rf import build_detailed_furniture_surfaces
from radarsim.scene.geometry import (
    AABB,
    OrientedBox,
    RadarCoordinateFrame,
    SurfaceLayer,
    SurfacePatch,
    Vec3,
)
from radarsim.scene.materials import (
    MaterialProperties,
    complex_permittivity,
    lookup_material_properties,
)


@dataclass(frozen=True)
class SceneObject:
    """Normalized furniture object with collision OBB and planar RF/render surfaces."""

    object_id: str
    kind: str
    position_m: Vec3
    dimensions_m: Vec3
    yaw_rad: float
    material_id: str
    component_materials: dict[str, str]
    collision_shape: str
    box: OrientedBox
    surfaces: tuple[SurfacePatch, ...]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.object_id,
            "kind": self.kind,
            "position_m": list(self.position_m),
            "dimensions_m": list(self.dimensions_m),
            "yaw_rad": self.yaw_rad,
            "material_id": self.material_id,
            "component_materials": dict(self.component_materials),
            "collision_shape": self.collision_shape,
            "box": self.box.to_dict(),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SceneOpening:
    """Normalized door or window opening with aperture bounds and optional physical panel/leaf."""

    opening_id: str
    kind: str
    wall: str
    center_m: Vec3
    width_m: float
    height_m: float
    sill_height_m: float
    material_id: str
    hinge_side: str | None
    swing_direction: str | None
    open_fraction: float | None
    layers: tuple[SurfaceLayer, ...]
    span_along_wall_m: tuple[float, float]
    aperture_surface: SurfacePatch
    panel_surface: SurfacePatch | None
    hinge_position_m: Vec3 | None = None

    @property
    def clear_opening_width_m(self) -> float:
        """Effective navigable horizontal width for a door opening [m]."""
        if self.kind != "door":
            return 0.0
        if self.open_fraction is None:
            return self.width_m
        theta = self.open_fraction * (0.5 * math.pi)
        # Clear opening measured perpendicular to hinge side (1 - cos(theta) projection, or sin(theta))
        # Using horizontal projection along the wall plane for a swinging leaf:
        # at open_fraction=0 -> 0.0 m; at open_fraction=1 (90 deg) -> width_m
        return self.width_m * (1.0 - math.cos(theta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.opening_id,
            "kind": self.kind,
            "wall": self.wall,
            "center_m": list(self.center_m),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "sill_height_m": self.sill_height_m,
            "material_id": self.material_id,
            "hinge_side": self.hinge_side,
            "swing_direction": self.swing_direction,
            "open_fraction": self.open_fraction,
            "span_along_wall_m": list(self.span_along_wall_m),
            "layers": [layer.to_dict() for layer in self.layers],
            "hinge_position_m": list(self.hinge_position_m) if self.hinge_position_m else None,
            "aperture_surface": self.aperture_surface.to_dict(),
            "panel_surface": self.panel_surface.to_dict() if self.panel_surface else None,
        }


@dataclass(frozen=True)
class OccupancyGrid2D:
    """2D floorplan navigation grid derived directly from ``Scene``."""

    resolution_m: float
    clearance_m: float
    x_centers_m: np.ndarray
    y_centers_m: np.ndarray
    blocked: np.ndarray  # shape (nx, ny), True where occupied/blocked

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.blocked.shape[0]), int(self.blocked.shape[1]))


def _wall_basis(
    wall: str, room: RoomConfig
) -> tuple[float, Vec3, Vec3, Vec3]:
    """Return ``(wall_length_m, s0_origin_xy0, s_axis, inward_normal)`` for a room wall.

    The scalar wall coordinate ``s in [0, wall_length_m]`` runs along ``+X`` for
    ``y_min`` / ``y_max`` walls and along ``+Y`` for ``x_min`` / ``x_max`` walls.
    """
    if wall == "y_min":
        return (room.width_x_m, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    if wall == "y_max":
        return (room.width_x_m, (0.0, room.depth_y_m, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    if wall == "x_min":
        return (room.depth_y_m, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    if wall == "x_max":
        return (room.depth_y_m, (room.width_x_m, 0.0, 0.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    raise ValueError(f"Unsupported wall identifier: {wall!r}")


def _make_wall_quad(
    *,
    surface_id: str,
    parent_id: str,
    kind: str,
    wall: str,
    room: RoomConfig,
    s0: float,
    s1: float,
    z0: float,
    z1: float,
    material_id: str,
    layers: Sequence[SurfaceLayer],
    is_passable: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SurfacePatch:
    """Create a rectangular patch lying on a room wall with normal pointing INTO the room."""
    ds = s1 - s0
    dz = z1 - z0
    if wall == "y_min":
        # (-i) x k = +j (inward)
        origin = (s1, 0.0, z0)
        u_vec = (-ds, 0.0, 0.0)
    elif wall == "y_max":
        # (+i) x k = -j (inward)
        origin = (s0, room.depth_y_m, z0)
        u_vec = (ds, 0.0, 0.0)
    elif wall == "x_min":
        # (+j) x k = +i (inward)
        origin = (0.0, s0, z0)
        u_vec = (0.0, ds, 0.0)
    elif wall == "x_max":
        # (-j) x k = -i (inward)
        origin = (room.width_x_m, s1, z0)
        u_vec = (0.0, -ds, 0.0)
    else:
        raise ValueError(f"Unsupported wall identifier: {wall!r}")

    return SurfacePatch.from_quad(
        surface_id=surface_id,
        parent_id=parent_id,
        kind=kind,
        origin_m=origin,
        u_vec_m=u_vec,
        v_vec_m=(0.0, 0.0, dz),
        material_id=material_id,
        layers=layers,
        is_passable=is_passable,
        metadata=metadata,
    )


def _wall_layers(room: RoomConfig, material_db: MaterialDatabase) -> tuple[SurfaceLayer, ...]:
    if room.construction and room.construction.walls:
        return tuple(
            SurfaceLayer(material_id=layer.material_id, thickness_m=layer.thickness_m)
            for layer in room.construction.walls.layers
        )
    return (SurfaceLayer(material_id=room.wall_material_id, thickness_m=room.wall_thickness_m),)


def _floor_layers(room: RoomConfig, material_db: MaterialDatabase) -> tuple[SurfaceLayer, ...]:
    if room.construction and room.construction.floor:
        return tuple(
            SurfaceLayer(material_id=layer.material_id, thickness_m=layer.thickness_m)
            for layer in room.construction.floor.layers
        )
    default_t = material_db.resolve(room.floor_material_id).thickness_default_m or 0.15
    return (SurfaceLayer(material_id=room.floor_material_id, thickness_m=default_t),)


def _ceiling_layers(room: RoomConfig, material_db: MaterialDatabase) -> tuple[SurfaceLayer, ...]:
    if room.construction and room.construction.ceiling:
        return tuple(
            SurfaceLayer(material_id=layer.material_id, thickness_m=layer.thickness_m)
            for layer in room.construction.ceiling.layers
        )
    default_t = material_db.resolve(room.ceiling_material_id).thickness_default_m or 0.0127
    return (SurfaceLayer(material_id=room.ceiling_material_id, thickness_m=default_t),)


def _opening_layers(
    opening: OpeningConfig, material_db: MaterialDatabase
) -> tuple[SurfaceLayer, ...]:
    if opening.construction_layers:
        return tuple(
            SurfaceLayer(material_id=layer.material_id, thickness_m=layer.thickness_m)
            for layer in opening.construction_layers
        )
    default_t = material_db.resolve(opening.material_id).thickness_default_m or 0.02
    return (SurfaceLayer(material_id=opening.material_id, thickness_m=default_t),)


def _build_opening(
    opening: OpeningConfig,
    room: RoomConfig,
    material_db: MaterialDatabase,
) -> SceneOpening:
    _, base_origin, s_axis, n_in = _wall_basis(opening.wall, room)
    center_s = opening.center_m[0] if opening.wall in ("y_min", "y_max") else opening.center_m[1]
    s0 = center_s - 0.5 * opening.width_m
    s1 = center_s + 0.5 * opening.width_m
    z0 = opening.sill_height_m
    z1 = z0 + opening.height_m
    layers = _opening_layers(opening, material_db)

    aperture_surface = _make_wall_quad(
        surface_id=f"opening_{opening.id}_aperture",
        parent_id=opening.id,
        kind=opening.kind,
        wall=opening.wall,
        room=room,
        s0=s0,
        s1=s1,
        z0=z0,
        z1=z1,
        material_id="air" if opening.kind == "door" else opening.material_id,
        layers=(
            (SurfaceLayer(material_id="air", thickness_m=room.wall_thickness_m),)
            if opening.kind == "door"
            else layers
        ),
        is_passable=(opening.kind == "door"),
        metadata={"wall": opening.wall, "opening_kind": opening.kind},
    )

    hinge_pos: Vec3 | None = None
    panel_surface: SurfacePatch | None = None
    if opening.kind == "window":
        panel_surface = _make_wall_quad(
            surface_id=f"opening_{opening.id}_pane",
            parent_id=opening.id,
            kind="window",
            wall=opening.wall,
            room=room,
            s0=s0,
            s1=s1,
            z0=z0,
            z1=z1,
            material_id=opening.material_id,
            layers=layers,
            is_passable=False,
            metadata={"wall": opening.wall, "opening_kind": "window"},
        )
    elif opening.kind == "door":
        # Compute door leaf orientation when looking into the room from outside the wall.
        n_in_arr = np.asarray(n_in, dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t_right = np.cross(n_in_arr, world_up)  # right-hand axis looking into room along n_in
        s_axis_arr = np.asarray(s_axis, dtype=np.float64)
        origin_arr = np.asarray(base_origin, dtype=np.float64)

        p_s0 = origin_arr + s0 * s_axis_arr + np.array([0.0, 0.0, z0])
        p_s1 = origin_arr + s1 * s_axis_arr + np.array([0.0, 0.0, z0])
        # Determine which edge is the "left" edge when looking into the room
        if float(np.dot(p_s1 - p_s0, t_right)) >= 0.0:
            left_edge, right_edge = p_s0, p_s1
        else:
            left_edge, right_edge = p_s1, p_s0

        hinge_side = opening.hinge_side or "left"
        swing_dir = opening.swing_direction or "inward"
        open_frac = 0.0 if opening.open_fraction is None else opening.open_fraction
        theta = open_frac * (0.5 * math.pi)
        swing_vec = n_in_arr if swing_dir == "inward" else -n_in_arr

        if hinge_side == "left":
            hinge_arr = left_edge
            closed_dir = t_right
        else:
            hinge_arr = right_edge
            closed_dir = -t_right

        leaf_dir = math.cos(theta) * closed_dir + math.sin(theta) * swing_vec
        hinge_pos = (float(hinge_arr[0]), float(hinge_arr[1]), float(hinge_arr[2]))
        panel_surface = SurfacePatch.from_quad(
            surface_id=f"opening_{opening.id}_leaf",
            parent_id=opening.id,
            kind="door",
            origin_m=hinge_pos,
            u_vec_m=tuple(float(v) for v in (opening.width_m * leaf_dir)),
            v_vec_m=(0.0, 0.0, opening.height_m),
            material_id=opening.material_id,
            layers=layers,
            is_passable=False,
            metadata={
                "wall": opening.wall,
                "hinge_side": hinge_side,
                "swing_direction": swing_dir,
                "open_fraction": open_frac,
            },
        )

    return SceneOpening(
        opening_id=opening.id,
        kind=opening.kind,
        wall=opening.wall,
        center_m=opening.center_m,
        width_m=opening.width_m,
        height_m=opening.height_m,
        sill_height_m=opening.sill_height_m,
        material_id=opening.material_id,
        hinge_side=opening.hinge_side,
        swing_direction=opening.swing_direction,
        open_fraction=opening.open_fraction,
        layers=layers,
        span_along_wall_m=(s0, s1),
        aperture_surface=aperture_surface,
        panel_surface=panel_surface,
        hinge_position_m=hinge_pos,
    )


def _build_wall_surfaces(
    wall: str,
    room: RoomConfig,
    wall_openings: Sequence[SceneOpening],
    layers: tuple[SurfaceLayer, ...],
) -> list[SurfacePatch]:
    """Partition a room wall into non-overlapping solid panels around any openings."""
    wall_len, _, _, _ = _wall_basis(wall, room)
    wall_meta: dict[str, Any] = {"wall": wall}
    if room.construction and room.construction.walls:
        wall_meta["construction_id"] = room.construction.walls.construction_id
        if room.construction.walls.metal_studs is not None:
            wall_meta["metal_studs_enabled"] = room.construction.walls.metal_studs.enabled
            wall_meta["metal_studs_spacing_m"] = room.construction.walls.metal_studs.spacing_m

    primary_mat = layers[0].material_id
    ordered = sorted(wall_openings, key=lambda o: o.span_along_wall_m[0])
    patches: list[SurfacePatch] = []
    eps = 1e-6

    if not ordered:
        patches.append(
            _make_wall_quad(
                surface_id=f"wall_{wall}",
                parent_id="room",
                kind="wall",
                wall=wall,
                room=room,
                s0=0.0,
                s1=wall_len,
                z0=0.0,
                z1=room.height_z_m,
                material_id=primary_mat,
                layers=layers,
                metadata=wall_meta,
            )
        )
        return patches

    cursor = 0.0
    seg_idx = 0
    for op in ordered:
        s0 = max(0.0, min(wall_len, op.span_along_wall_m[0]))
        s1 = max(0.0, min(wall_len, op.span_along_wall_m[1]))
        z0 = max(0.0, min(room.height_z_m, op.sill_height_m))
        z1 = max(0.0, min(room.height_z_m, op.sill_height_m + op.height_m))

        if s0 - cursor > eps:
            patches.append(
                _make_wall_quad(
                    surface_id=f"wall_{wall}_seg_{seg_idx}",
                    parent_id="room",
                    kind="wall",
                    wall=wall,
                    room=room,
                    s0=cursor,
                    s1=s0,
                    z0=0.0,
                    z1=room.height_z_m,
                    material_id=primary_mat,
                    layers=layers,
                    metadata=wall_meta,
                )
            )
            seg_idx += 1

        # Below-opening sill segment
        if s1 - s0 > eps and z0 > eps:
            patches.append(
                _make_wall_quad(
                    surface_id=f"wall_{wall}_below_{op.opening_id}",
                    parent_id="room",
                    kind="wall",
                    wall=wall,
                    room=room,
                    s0=s0,
                    s1=s1,
                    z0=0.0,
                    z1=z0,
                    material_id=primary_mat,
                    layers=layers,
                    metadata=wall_meta,
                )
            )

        # Above-opening lintel/header segment
        if s1 - s0 > eps and room.height_z_m - z1 > eps:
            patches.append(
                _make_wall_quad(
                    surface_id=f"wall_{wall}_above_{op.opening_id}",
                    parent_id="room",
                    kind="wall",
                    wall=wall,
                    room=room,
                    s0=s0,
                    s1=s1,
                    z0=z1,
                    z1=room.height_z_m,
                    material_id=primary_mat,
                    layers=layers,
                    metadata=wall_meta,
                )
            )

        cursor = max(cursor, s1)

    if wall_len - cursor > eps:
        patches.append(
            _make_wall_quad(
                surface_id=f"wall_{wall}_seg_{seg_idx}",
                parent_id="room",
                kind="wall",
                wall=wall,
                room=room,
                s0=cursor,
                s1=wall_len,
                z0=0.0,
                z1=room.height_z_m,
                material_id=primary_mat,
                layers=layers,
                metadata=wall_meta,
            )
        )

    return patches


def _furniture_face_material(obj: SceneObjectConfig, face_name: str) -> str:
    """Resolve face material using ``component_materials`` when present."""
    cm = obj.component_materials
    if face_name == "top":
        for key in ("mattress", "bedding", "upholstery", "seat", "top"):
            if key in cm:
                return cm[key]
    else:
        if obj.kind == "sofa" and "upholstery" in cm:
            return cm["upholstery"]
        if "frame" in cm:
            return cm["frame"]
    return obj.material_id


def _build_furniture_surfaces(
    obj: SceneObjectConfig,
    box: OrientedBox,
    material_db: MaterialDatabase,
) -> tuple[SurfacePatch, ...]:
    """Build the RF panels of a furniture object.

    Known kinds (bed, sofa, table, chair, wardrobe) use the detailed hollow/open panel
    models in ``furniture_rf`` (see that module for the physical rationale); any other kind
    falls back to the legacy solid six-face box.
    """
    detailed = build_detailed_furniture_surfaces(obj.id, obj.kind, box, material_db)
    if detailed is not None:
        return detailed
    c = np.asarray(box.center_m, dtype=np.float64)
    hx, hy, hz = box.half_extents_m
    axes2d = box.axes_2d()
    u_x = np.array([axes2d[0, 0], axes2d[0, 1], 0.0], dtype=np.float64)
    u_y = np.array([axes2d[1, 0], axes2d[1, 1], 0.0], dtype=np.float64)
    u_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    face_specs = [
        ("top", c - hx * u_x - hy * u_y + hz * u_z, 2.0 * hx * u_x, 2.0 * hy * u_y, 2.0 * hz),
        ("bottom", c - hx * u_x + hy * u_y - hz * u_z, 2.0 * hx * u_x, -2.0 * hy * u_y, 2.0 * hz),
        ("front", c - hx * u_x - hy * u_y - hz * u_z, 2.0 * hx * u_x, 2.0 * hz * u_z, 2.0 * hy),
        ("back", c + hx * u_x + hy * u_y - hz * u_z, -2.0 * hx * u_x, 2.0 * hz * u_z, 2.0 * hy),
        ("right", c + hx * u_x - hy * u_y - hz * u_z, 2.0 * hy * u_y, 2.0 * hz * u_z, 2.0 * hx),
        ("left", c - hx * u_x + hy * u_y - hz * u_z, -2.0 * hy * u_y, 2.0 * hz * u_z, 2.0 * hx),
    ]

    surfaces: list[SurfacePatch] = []
    for face_name, origin, u_vec, v_vec, depth_m in face_specs:
        mat_id = _furniture_face_material(obj, face_name)
        mat_default_t = material_db.resolve(mat_id).thickness_default_m
        layer_thickness = mat_default_t if mat_default_t is not None else min(0.05, depth_m)
        surfaces.append(
            SurfacePatch.from_quad(
                surface_id=f"{obj.id}_{face_name}",
                parent_id=obj.id,
                kind="furniture",
                origin_m=origin,
                u_vec_m=u_vec,
                v_vec_m=v_vec,
                material_id=mat_id,
                layers=(SurfaceLayer(material_id=mat_id, thickness_m=layer_thickness),),
                is_passable=False,
                metadata={"object_kind": obj.kind, "face": face_name},
            )
        )
    return tuple(surfaces)


@dataclass(frozen=True)
class Scene:
    """Normalized room scene consumed identically by motion, RF, and frontend modules."""

    scene_id: str
    schema_version: str
    units: str
    coordinate_frame: str | None
    room: RoomConfig
    room_bounds: AABB
    materials: MaterialDatabase
    radar_mount: RadarMountConfig
    radar_frame: RadarCoordinateFrame
    openings: tuple[SceneOpening, ...]
    objects: tuple[SceneObject, ...]
    surfaces: tuple[SurfacePatch, ...]
    initial_actor: InitialActorConfig | None
    config: SceneConfig

    @property
    def width_x_m(self) -> float:
        return self.room.width_x_m

    @property
    def depth_y_m(self) -> float:
        return self.room.depth_y_m

    @property
    def height_z_m(self) -> float:
        return self.room.height_z_m

    @property
    def collision_boxes(self) -> tuple[tuple[str, OrientedBox], ...]:
        return tuple((obj.object_id, obj.box) for obj in self.objects)

    def get_object(self, object_id: str) -> SceneObject:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(f"Unknown scene object: {object_id!r}")

    def get_opening(self, opening_id: str) -> SceneOpening:
        for op in self.openings:
            if op.opening_id == opening_id:
                return op
        raise KeyError(f"Unknown scene opening: {opening_id!r}")

    def doors(self) -> tuple[SceneOpening, ...]:
        return tuple(op for op in self.openings if op.kind == "door")

    def windows(self) -> tuple[SceneOpening, ...]:
        return tuple(op for op in self.openings if op.kind == "window")

    def surfaces_by_kind(self, kind: str) -> tuple[SurfacePatch, ...]:
        return tuple(s for s in self.surfaces if s.kind == kind)

    def material_properties(
        self, material_id: str, frequency_hz: float | None = None
    ) -> MaterialProperties:
        return lookup_material_properties(self.materials, material_id, frequency_hz)

    def complex_permittivity(
        self, material_id: str, frequency_hz: float | None = None
    ) -> complex:
        return complex_permittivity(self.materials, material_id, frequency_hz)

    def is_point_inside_room(
        self, x: float, y: float, z: float = 0.0, *, margin_m: float = 0.0
    ) -> bool:
        return (
            margin_m <= x <= self.room.width_x_m - margin_m
            and margin_m <= y <= self.room.depth_y_m - margin_m
            and 0.0 <= z <= self.room.height_z_m
        )

    def is_point_free_2d(
        self,
        x: float,
        y: float,
        *,
        radius_m: float = 0.0,
        allow_doorway_exterior: bool = False,
    ) -> bool:
        """Return True if a circle at ``(x, y)`` of ``radius_m`` is free of walls and furniture."""
        in_room = (
            radius_m <= x <= self.room.width_x_m - radius_m
            and radius_m <= y <= self.room.depth_y_m - radius_m
        )
        if not in_room:
            if not allow_doorway_exterior:
                return False
            # Permit passage through any door opening band along its wall
            through_door = False
            for door in self.doors():
                s0, s1 = door.span_along_wall_m
                if door.wall == "y_min" and (s0 + radius_m <= x <= s1 - radius_m) and y <= radius_m:
                    through_door = True
                elif door.wall == "y_max" and (s0 + radius_m <= x <= s1 - radius_m) and y >= self.room.depth_y_m - radius_m:
                    through_door = True
                elif door.wall == "x_min" and (s0 + radius_m <= y <= s1 - radius_m) and x <= radius_m:
                    through_door = True
                elif door.wall == "x_max" and (s0 + radius_m <= y <= s1 - radius_m) and x >= self.room.width_x_m - radius_m:
                    through_door = True
            if not through_door:
                return False

        for obj in self.objects:
            if radius_m <= 0.0:
                if obj.box.contains_point_2d(x, y):
                    return False
            else:
                if obj.box.intersects_circle_2d((x, y), radius_m):
                    return False
        return True

    def build_occupancy_grid_2d(
        self,
        *,
        resolution_m: float = 0.10,
        clearance_m: float = 0.12,
    ) -> OccupancyGrid2D:
        """Build a 2D navigation occupancy grid directly from this normalized scene."""
        if resolution_m <= 0.0:
            raise ValueError(f"resolution_m must be > 0, got {resolution_m}")
        if clearance_m < 0.0:
            raise ValueError(f"clearance_m must be >= 0, got {clearance_m}")
        nx = max(1, int(math.ceil(self.room.width_x_m / resolution_m)))
        ny = max(1, int(math.ceil(self.room.depth_y_m / resolution_m)))
        xs = (np.arange(nx, dtype=np.float64) + 0.5) * (self.room.width_x_m / nx)
        ys = (np.arange(ny, dtype=np.float64) + 0.5) * (self.room.depth_y_m / ny)
        blocked = np.zeros((nx, ny), dtype=bool)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                if not self.is_point_free_2d(float(x), float(y), radius_m=clearance_m):
                    blocked[ix, iy] = True
        return OccupancyGrid2D(
            resolution_m=resolution_m,
            clearance_m=clearance_m,
            x_centers_m=xs,
            y_centers_m=ys,
            blocked=blocked,
        )

    def _surface_payload(self, surf: SurfacePatch, frequency_hz: float | None) -> dict[str, Any]:
        """Surface dict; furniture panels also carry their RF response so the 3D view can colour them.

        ``rf`` holds TE-polarised coherent reflection (dB) at 0/45/70 deg incidence, 45 deg
        transmission (dB) and the roughness factor, all from the SAME transfer-matrix code the
        RF engine uses (nothing is re-derived in the browser).
        """
        payload = surf.to_dict()
        if surf.kind != "furniture":
            return payload
        from radarsim.rf.em import evaluate_surface_em  # lazy: rf.em imports radarsim.scene

        f_hz = float(frequency_hz) if frequency_hz else float(self.materials.reference_frequency_hz)

        def _db(x: complex) -> float:
            return round(20.0 * math.log10(max(abs(x), 1e-6)), 2)

        em = {
            deg: evaluate_surface_em(surf.layers, self.materials, math.radians(deg), f_hz)
            for deg in (0.0, 45.0, 70.0)
        }
        payload["rf"] = {
            "frequency_hz": f_hz,
            "polarization": "TE",
            "reflection_db": {str(int(d)): _db(e.reflection_coherent) for d, e in em.items()},
            "reflection_smooth_db": {str(int(d)): _db(e.reflection_smooth) for d, e in em.items()},
            "transmission_db_45": _db(em[45.0].transmission),
            "roughness_factor_45": round(em[45.0].roughness_factor, 4),
            "part": surf.metadata.get("part"),
            "face": surf.metadata.get("face"),
        }
        return payload

    def to_dict(self, *, frequency_hz: float | None = None) -> dict[str, Any]:
        """Export a pure JSON-serializable normalized scene payload for RF, motion, and frontend."""
        mat_summary = {
            mat_id: self.material_properties(mat_id, frequency_hz).to_dict()
            for mat_id in sorted(self.config.referenced_material_ids())
        }
        return {
            "schema_version": self.schema_version,
            "scene_id": self.scene_id,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "room": {
                "width_x_m": self.room.width_x_m,
                "depth_y_m": self.room.depth_y_m,
                "height_z_m": self.room.height_z_m,
                "wall_thickness_m": self.room.wall_thickness_m,
                "floor_material_id": self.room.floor_material_id,
                "wall_material_id": self.room.wall_material_id,
                "ceiling_material_id": self.room.ceiling_material_id,
                "bounds": self.room_bounds.to_dict(),
            },
            "radar": {
                "id": self.radar_mount.id,
                "profile_id": self.radar_mount.profile_id,
                "mount_type": self.radar_mount.mount_type,
                "wall": self.radar_mount.wall,
                "frame": self.radar_frame.to_dict(),
            },
            "openings": [op.to_dict() for op in self.openings],
            "objects": [obj.to_dict() for obj in self.objects],
            "surfaces": [self._surface_payload(surf, frequency_hz) for surf in self.surfaces],
            "materials": mat_summary,
            "initial_actor": (
                {
                    "actor_id": self.initial_actor.actor_id,
                    "avatar_asset_id": self.initial_actor.avatar_asset_id,
                    "start_position_m": list(self.initial_actor.start_position_m),
                    "entry_target_m": list(self.initial_actor.entry_target_m),
                }
                if self.initial_actor is not None
                else None
            ),
        }

    def ascii_floorplan(self, cols: int = 44, rows: int = 24) -> str:
        """Render a top-down ASCII floorplan (+X right, +Y up from entrance wall at bottom)."""
        legend_chars: dict[str, str] = {
            "bed": "B",
            "sofa": "S",
            "table": "T",
            "chair": "C",
            "wardrobe": "W",
        }
        grid = [["." for _ in range(cols)] for _ in range(rows)]
        for r in range(rows):
            # Top row corresponds to y = depth_y_m (back wall), bottom row y = 0 (entrance wall)
            y = (rows - 1 - r + 0.5) / rows * self.room.depth_y_m
            for c in range(cols):
                x = (c + 0.5) / cols * self.room.width_x_m
                for obj in self.objects:
                    if obj.box.contains_point_2d(x, y):
                        grid[r][c] = legend_chars.get(obj.kind, "#")
                        break

        # Mark radar position
        rc = min(cols - 1, max(0, int(self.radar_mount.position_m[0] / self.room.width_x_m * cols)))
        rr = min(rows - 1, max(0, rows - 1 - int(self.radar_mount.position_m[1] / self.room.depth_y_m * rows)))
        grid[rr][rc] = "R"

        # Mark entry target if inside room
        if self.initial_actor is not None:
            ex, ey, _ = self.initial_actor.entry_target_m
            if 0.0 <= ex <= self.room.width_x_m and 0.0 <= ey <= self.room.depth_y_m:
                ec = min(cols - 1, max(0, int(ex / self.room.width_x_m * cols)))
                er = min(rows - 1, max(0, rows - 1 - int(ey / self.room.depth_y_m * rows)))
                if grid[er][ec] == ".":
                    grid[er][ec] = "E"

        # Top border (y_max) with window markers
        top = ["-"] * cols
        for op in self.openings:
            if op.wall == "y_max":
                s0, s1 = op.span_along_wall_m
                for c in range(cols):
                    x = (c + 0.5) / cols * self.room.width_x_m
                    if s0 <= x <= s1:
                        top[c] = "=" if op.kind == "window" else " "

        # Bottom border (y_min) with door markers
        bottom = ["-"] * cols
        for op in self.openings:
            if op.wall == "y_min":
                s0, s1 = op.span_along_wall_m
                for c in range(cols):
                    x = (c + 0.5) / cols * self.room.width_x_m
                    if s0 <= x <= s1:
                        bottom[c] = "D" if op.kind == "door" else "="

        lines = [f"+{''.join(top)}+  y={self.room.depth_y_m:.2f}m (back wall)"]
        for idx, row in enumerate(grid):
            label = "  +Y" if idx == 0 else ("  ^" if idx == 1 else "")
            lines.append(f"|{''.join(row)}|{label}")
        lines.append(f"+{''.join(bottom)}+  y=0.00m (entrance wall, 0 -> {self.room.width_x_m:.2f}m +X)")
        lines.append("Legend: B=bed S=sofa T=table C=chair W=wardrobe R=radar E=entry_target D=door ==window")
        return "\n".join(lines)

    def geometry_summary_markdown(self, *, frequency_hz: float = 61.0e9) -> str:
        """Produce a structured Markdown geometry & material summary for human inspection."""
        lines: list[str] = [
            f"# Scene Geometry Summary: `{self.scene_id}`",
            "",
            f"- **Dimensions (X x Y x Z):** {self.width_x_m:.2f} m × {self.depth_y_m:.2f} m × {self.height_z_m:.2f} m",
            f"- **Wall thickness:** {self.room.wall_thickness_m:.4f} m",
            f"- **Coordinate frame:** `{self.coordinate_frame}`",
            f"- **Radar mount:** `{self.radar_mount.id}` (`{self.radar_mount.profile_id}`) at `{self.radar_mount.position_m}` m, boresight `{tuple(round(v, 4) for v in self.radar_frame.boresight_world)}`, tilt down {self.radar_mount.tilt_down_deg:.1f}°",
            f"- **Total planar surfaces:** {len(self.surfaces)} (walls={len(self.surfaces_by_kind('wall'))}, floor={len(self.surfaces_by_kind('floor'))}, ceiling={len(self.surfaces_by_kind('ceiling'))}, doors={len(self.surfaces_by_kind('door'))}, windows={len(self.surfaces_by_kind('window'))}, furniture={len(self.surfaces_by_kind('furniture'))})",
            "",
            "## Top-down floorplan",
            "```text",
            self.ascii_floorplan(),
            "```",
            "",
            "## Furniture objects",
            "| ID | Kind | Footprint AABB X [m] | Footprint AABB Y [m] | Height Z [m] | Yaw [rad] | Material |",
            "|---|---|---|---|---|---|---|",
        ]
        for obj in self.objects:
            bb = obj.box.aabb()
            lines.append(
                f"| `{obj.object_id}` | {obj.kind} | [{bb.min_m[0]:.2f}, {bb.max_m[0]:.2f}] | "
                f"[{bb.min_m[1]:.2f}, {bb.max_m[1]:.2f}] | [{bb.min_m[2]:.2f}, {bb.max_m[2]:.2f}] | "
                f"{obj.yaw_rad:.3f} | `{obj.material_id}` |"
            )

        lines.extend(
            [
                "",
                "## Openings",
                "| ID | Kind | Wall | Span [m] | Height Z [m] | Open fraction | Layers |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for op in self.openings:
            layer_desc = ", ".join(f"{l.material_id} ({l.thickness_m*1000:.1f} mm)" for l in op.layers)
            open_str = f"{op.open_fraction:.2f}" if op.open_fraction is not None else "n/a"
            lines.append(
                f"| `{op.opening_id}` | {op.kind} | `{op.wall}` | "
                f"[{op.span_along_wall_m[0]:.2f}, {op.span_along_wall_m[1]:.2f}] | "
                f"[{op.sill_height_m:.2f}, {op.sill_height_m + op.height_m:.2f}] | {open_str} | {layer_desc} |"
            )

        lines.extend(
            [
                "",
                f"## Referenced materials at {frequency_hz * 1e-9:.2f} GHz (`epsilon_r = eps_real - j*eps_imag`)",
                "| Material ID | Resolved ID | `eps_real` | `eps_imag` | `sigma` [S/m] | Roughness [mm] | Extrapolated? |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for mat_id in sorted(self.config.referenced_material_ids()):
            props = self.material_properties(mat_id, frequency_hz)
            lines.append(
                f"| `{mat_id}` | `{props.resolved_id}` | {props.epsilon_real:.4g} | "
                f"{props.epsilon_imag:.4g} | {props.conductivity_s_per_m:.4g} | "
                f"{props.roughness_rms_m * 1000:.3g} | {props.frequency_extrapolated} |"
            )
        return "\n".join(lines) + "\n"


def build_scene(
    scene_config: SceneConfig,
    material_db: MaterialDatabase,
) -> Scene:
    """Build a normalized ``Scene`` from validated ``SceneConfig`` and ``MaterialDatabase``."""
    room = scene_config.room
    missing = sorted(
        mat_id
        for mat_id in scene_config.referenced_material_ids()
        if not material_db.has_material(mat_id)
    )
    if missing:
        raise ValueError(f"Scene references unknown material_id(s): {missing}")

    room_bounds = AABB(
        min_m=(0.0, 0.0, 0.0),
        max_m=(room.width_x_m, room.depth_y_m, room.height_z_m),
    )

    radar_frame = RadarCoordinateFrame.from_mount(
        position_m=scene_config.radar.position_m,
        boresight_unit_vector_world=scene_config.radar.boresight_unit_vector_world,
        yaw_rad=scene_config.radar.yaw_rad,
        tilt_down_deg=scene_config.radar.tilt_down_deg,
    )

    # Openings
    openings = tuple(
        _build_opening(op_cfg, room, material_db) for op_cfg in scene_config.openings
    )

    # Floor and ceiling
    floor_layers = _floor_layers(room, material_db)
    ceiling_layers = _ceiling_layers(room, material_db)
    wall_layers = _wall_layers(room, material_db)

    surfaces: list[SurfacePatch] = [
        SurfacePatch.from_quad(
            surface_id="floor",
            parent_id="room",
            kind="floor",
            origin_m=(0.0, 0.0, 0.0),
            u_vec_m=(room.width_x_m, 0.0, 0.0),
            v_vec_m=(0.0, room.depth_y_m, 0.0),
            material_id=floor_layers[0].material_id,
            layers=floor_layers,
            metadata={
                "construction_id": (
                    room.construction.floor.construction_id
                    if room.construction and room.construction.floor
                    else "default_floor"
                )
            },
        ),
        SurfacePatch.from_quad(
            surface_id="ceiling",
            parent_id="room",
            kind="ceiling",
            origin_m=(0.0, room.depth_y_m, room.height_z_m),
            u_vec_m=(room.width_x_m, 0.0, 0.0),
            v_vec_m=(0.0, -room.depth_y_m, 0.0),
            material_id=ceiling_layers[0].material_id,
            layers=ceiling_layers,
            metadata={
                "construction_id": (
                    room.construction.ceiling.construction_id
                    if room.construction and room.construction.ceiling
                    else "default_ceiling"
                )
            },
        ),
    ]

    # Layered walls partitioned around openings
    for wall_name in ("y_min", "y_max", "x_min", "x_max"):
        wall_ops = [op for op in openings if op.wall == wall_name]
        surfaces.extend(_build_wall_surfaces(wall_name, room, wall_ops, wall_layers))

    # Opening physical panels (window glass panes & swinging door leaves)
    for op in openings:
        if op.panel_surface is not None:
            surfaces.append(op.panel_surface)

    # Furniture objects + oriented faces
    scene_objects: list[SceneObject] = []
    for obj_cfg in scene_config.objects:
        box = OrientedBox.from_corner_and_dimensions(
            obj_cfg.position_m, obj_cfg.dimensions_m, obj_cfg.yaw_rad
        )
        obj_surfaces = _build_furniture_surfaces(obj_cfg, box, material_db)
        surfaces.extend(obj_surfaces)
        scene_objects.append(
            SceneObject(
                object_id=obj_cfg.id,
                kind=obj_cfg.kind,
                position_m=obj_cfg.position_m,
                dimensions_m=obj_cfg.dimensions_m,
                yaw_rad=obj_cfg.yaw_rad,
                material_id=obj_cfg.material_id,
                component_materials=dict(obj_cfg.component_materials),
                collision_shape=obj_cfg.collision_shape,
                box=box,
                surfaces=obj_surfaces,
                notes=obj_cfg.notes,
            )
        )

    return Scene(
        scene_id=scene_config.scene_id,
        schema_version=scene_config.schema_version,
        units=scene_config.units,
        coordinate_frame=scene_config.coordinate_frame,
        room=room,
        room_bounds=room_bounds,
        materials=material_db,
        radar_mount=scene_config.radar,
        radar_frame=radar_frame,
        openings=openings,
        objects=tuple(scene_objects),
        surfaces=tuple(surfaces),
        initial_actor=scene_config.initial_actor,
        config=scene_config,
    )


def load_scene(
    scene_path: str | Path = "scene.json",
    material_db_path: str | Path = "material_db.yaml",
) -> Scene:
    """Load and build a normalized ``Scene`` directly from ``scene.json`` and ``material_db.yaml``."""
    scene_cfg = load_scene_config(scene_path)
    mat_db = load_material_db(material_db_path)
    return build_scene(scene_cfg, mat_db)
