"""Coordinate transforms, oriented bounding boxes, and planar surface patches.

Follows the coordinate and angle conventions declared in ``INTERFACES.md``:
- World origin: front-left floor corner ``(0, 0, 0)``.
- World axes: ``+X`` right, ``+Y`` into room from entrance wall, ``+Z`` up.
- Radar local frame: ``+Y`` along boresight, ``+X`` right, ``+Z`` up.
- Azimuth: ``atan2(dx, dy)`` in radar local frame (positive toward local ``+X``).
- Elevation: ``atan2(dz, sqrt(dx^2 + dy^2))`` (positive above local horizontal).
- Radial velocity: ``dot(v_target - v_radar, unit_vector_radar_to_target)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

Vec3 = tuple[float, float, float]


def _normalize(vec: np.ndarray, *, label: str = "vector") -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12 or not math.isfinite(norm):
        raise ValueError(f"Cannot normalize zero or non-finite {label}: {vec!r}")
    return vec / norm


def boresight_from_yaw_tilt(yaw_rad: float, tilt_down_deg: float) -> Vec3:
    """Compute world boresight unit vector from yaw [rad] and downward tilt [deg].

    At ``yaw_rad = 0`` and ``tilt_down_deg = 0``, boresight is ``(0, 1, 0)``.
    Positive ``yaw_rad`` rotates boresight toward ``+X``; positive
    ``tilt_down_deg`` tilts boresight downward toward ``-Z``.
    """
    tilt_rad = math.radians(tilt_down_deg)
    cos_t = math.cos(tilt_rad)
    return (
        math.sin(yaw_rad) * cos_t,
        math.cos(yaw_rad) * cos_t,
        -math.sin(tilt_rad),
    )


def azimuth_elevation_from_local(local_vec: Sequence[float]) -> tuple[float, float, float]:
    """Convert a radar-local Cartesian vector ``(dx, dy, dz)`` to ``(range_m, az_rad, el_rad)``."""
    dx, dy, dz = float(local_vec[0]), float(local_vec[1]), float(local_vec[2])
    horiz = math.hypot(dx, dy)
    rng = math.hypot(horiz, dz)
    az = math.atan2(dx, dy)
    el = math.atan2(dz, horiz)
    return rng, az, el


def local_from_range_azimuth_elevation(
    range_m: float, azimuth_rad: float, elevation_rad: float
) -> Vec3:
    """Convert ``(range_m, az_rad, el_rad)`` back into radar-local ``(dx, dy, dz)``."""
    cos_el = math.cos(elevation_rad)
    dx = range_m * cos_el * math.sin(azimuth_rad)
    dy = range_m * cos_el * math.cos(azimuth_rad)
    dz = range_m * math.sin(elevation_rad)
    return (dx, dy, dz)


def radial_velocity_mps(
    radar_position_m: Sequence[float],
    target_position_m: Sequence[float],
    target_velocity_mps: Sequence[float],
    radar_velocity_mps: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """Compute radial velocity ``dot(v_target - v_radar, u_radar_to_target)`` [m/s].

    Positive values indicate a receding target; negative values indicate an
    approaching target.
    """
    r_vec = np.asarray(target_position_m, dtype=np.float64) - np.asarray(
        radar_position_m, dtype=np.float64
    )
    unit_los = _normalize(r_vec, label="radar_to_target")
    v_rel = np.asarray(target_velocity_mps, dtype=np.float64) - np.asarray(
        radar_velocity_mps, dtype=np.float64
    )
    return float(np.dot(v_rel, unit_los))


@dataclass(frozen=True)
class RadarCoordinateFrame:
    """Right-handed orthonormal coordinate transform between world and radar local frames."""

    position_m: Vec3
    boresight_world: Vec3
    axis_x_world: Vec3
    axis_y_world: Vec3
    axis_z_world: Vec3
    yaw_rad: float
    tilt_down_deg: float

    @classmethod
    def from_mount(
        cls,
        position_m: Sequence[float],
        boresight_unit_vector_world: Sequence[float],
        *,
        yaw_rad: float = 0.0,
        tilt_down_deg: float = 0.0,
    ) -> RadarCoordinateFrame:
        e_y = _normalize(
            np.asarray(boresight_unit_vector_world, dtype=np.float64),
            label="boresight_unit_vector_world",
        )
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(e_y, world_up))) > 0.9999:
            # Degenerate vertical boresight: derive local +X from yaw_rad
            e_x = _normalize(
                np.array([math.cos(yaw_rad), -math.sin(yaw_rad), 0.0], dtype=np.float64),
                label="radar_axis_x",
            )
        else:
            # Looking along e_y (+Y), right-hand +X is cross(e_y, world_up)
            e_x = _normalize(np.cross(e_y, world_up), label="radar_axis_x")
        e_z = _normalize(np.cross(e_x, e_y), label="radar_axis_z")

        pos = (float(position_m[0]), float(position_m[1]), float(position_m[2]))
        return cls(
            position_m=pos,
            boresight_world=(float(e_y[0]), float(e_y[1]), float(e_y[2])),
            axis_x_world=(float(e_x[0]), float(e_x[1]), float(e_x[2])),
            axis_y_world=(float(e_y[0]), float(e_y[1]), float(e_y[2])),
            axis_z_world=(float(e_z[0]), float(e_z[1]), float(e_z[2])),
            yaw_rad=float(yaw_rad),
            tilt_down_deg=float(tilt_down_deg),
        )

    @property
    def rotation_local_to_world(self) -> np.ndarray:
        """3x3 orthonormal matrix whose columns are ``[axis_x, axis_y, axis_z]`` in world coords."""
        return np.column_stack(
            (
                np.asarray(self.axis_x_world, dtype=np.float64),
                np.asarray(self.axis_y_world, dtype=np.float64),
                np.asarray(self.axis_z_world, dtype=np.float64),
            )
        )

    @property
    def rotation_world_to_local(self) -> np.ndarray:
        return self.rotation_local_to_world.T

    def world_to_local_vector(self, vec_world: Sequence[float]) -> Vec3:
        local = self.rotation_world_to_local @ np.asarray(vec_world, dtype=np.float64)
        return (float(local[0]), float(local[1]), float(local[2]))

    def local_to_world_vector(self, vec_local: Sequence[float]) -> Vec3:
        world = self.rotation_local_to_world @ np.asarray(vec_local, dtype=np.float64)
        return (float(world[0]), float(world[1]), float(world[2]))

    def world_to_local_point(self, point_world: Sequence[float]) -> Vec3:
        delta = np.asarray(point_world, dtype=np.float64) - np.asarray(
            self.position_m, dtype=np.float64
        )
        local = self.rotation_world_to_local @ delta
        return (float(local[0]), float(local[1]), float(local[2]))

    def local_to_world_point(self, point_local: Sequence[float]) -> Vec3:
        world = np.asarray(self.position_m, dtype=np.float64) + (
            self.rotation_local_to_world @ np.asarray(point_local, dtype=np.float64)
        )
        return (float(world[0]), float(world[1]), float(world[2]))

    def point_to_spherical(self, point_world: Sequence[float]) -> tuple[float, float, float]:
        """Return ``(range_m, azimuth_rad, elevation_rad)`` for a world-frame point."""
        return azimuth_elevation_from_local(self.world_to_local_point(point_world))

    def spherical_to_world_point(
        self, range_m: float, azimuth_rad: float, elevation_rad: float
    ) -> Vec3:
        """Inverse of :meth:`point_to_spherical`."""
        local_pt = local_from_range_azimuth_elevation(range_m, azimuth_rad, elevation_rad)
        return self.local_to_world_point(local_pt)

    def radial_velocity(
        self,
        target_position_world: Sequence[float],
        target_velocity_world: Sequence[float],
        radar_velocity_world: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> float:
        return radial_velocity_mps(
            self.position_m,
            target_position_world,
            target_velocity_world,
            radar_velocity_world,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "boresight_world": list(self.boresight_world),
            "axis_x_world": list(self.axis_x_world),
            "axis_y_world": list(self.axis_y_world),
            "axis_z_world": list(self.axis_z_world),
            "yaw_rad": self.yaw_rad,
            "tilt_down_deg": self.tilt_down_deg,
        }


# ---------------------------------------------------------------------------
# Axis-aligned and Yaw-oriented Bounding Boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AABB:
    """Axis-aligned 3D bounding box in world coordinates [m]."""

    min_m: Vec3
    max_m: Vec3

    @property
    def size_m(self) -> Vec3:
        return (
            self.max_m[0] - self.min_m[0],
            self.max_m[1] - self.min_m[1],
            self.max_m[2] - self.min_m[2],
        )

    @property
    def center_m(self) -> Vec3:
        return (
            0.5 * (self.min_m[0] + self.max_m[0]),
            0.5 * (self.min_m[1] + self.max_m[1]),
            0.5 * (self.min_m[2] + self.max_m[2]),
        )

    def contains_point(self, point_m: Sequence[float], *, eps: float = 1e-9) -> bool:
        return (
            self.min_m[0] - eps <= point_m[0] <= self.max_m[0] + eps
            and self.min_m[1] - eps <= point_m[1] <= self.max_m[1] + eps
            and self.min_m[2] - eps <= point_m[2] <= self.max_m[2] + eps
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"min_m": list(self.min_m), "max_m": list(self.max_m)}


def _point_to_segment_dist_2d(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _polygons_intersect_sat_2d(
    corners_a: np.ndarray,
    axes_a: np.ndarray,
    corners_b: np.ndarray,
    axes_b: np.ndarray,
    *,
    eps: float = 1e-9,
) -> bool:
    """Exact 2D Separating Axis Theorem overlap check for two oriented rectangles."""
    for axis in (axes_a[0], axes_a[1], axes_b[0], axes_b[1]):
        proj_a = corners_a @ axis
        proj_b = corners_b @ axis
        if float(proj_a.max()) <= float(proj_b.min()) + eps:
            return False
        if float(proj_b.max()) <= float(proj_a.min()) + eps:
            return False
    return True


@dataclass(frozen=True)
class OrientedBox:
    """3D box oriented by yaw around the vertical Z axis through its horizontal centre."""

    center_m: Vec3
    half_extents_m: Vec3
    yaw_rad: float

    @classmethod
    def from_corner_and_dimensions(
        cls,
        position_m: Sequence[float],
        dimensions_m: Sequence[float],
        yaw_rad: float = 0.0,
    ) -> OrientedBox:
        """Build from ``INTERFACES.md`` convention: lower south-west corner + ``[dx, dy, dz]``."""
        dx, dy, dz = float(dimensions_m[0]), float(dimensions_m[1]), float(dimensions_m[2])
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ValueError(f"Box dimensions must be positive, got {dimensions_m!r}")
        cx = float(position_m[0]) + 0.5 * dx
        cy = float(position_m[1]) + 0.5 * dy
        cz = float(position_m[2]) + 0.5 * dz
        return cls(
            center_m=(cx, cy, cz),
            half_extents_m=(0.5 * dx, 0.5 * dy, 0.5 * dz),
            yaw_rad=float(yaw_rad),
        )

    @property
    def dimensions_m(self) -> Vec3:
        return (
            2.0 * self.half_extents_m[0],
            2.0 * self.half_extents_m[1],
            2.0 * self.half_extents_m[2],
        )

    @property
    def z_min_m(self) -> float:
        return self.center_m[2] - self.half_extents_m[2]

    @property
    def z_max_m(self) -> float:
        return self.center_m[2] + self.half_extents_m[2]

    def axes_2d(self) -> np.ndarray:
        """Return 2x2 matrix whose rows are local ``u_x`` and ``u_y`` unit vectors in XY."""
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return np.array([[c, s], [-s, c]], dtype=np.float64)

    def corners_2d(self) -> np.ndarray:
        """Return 4x2 array of XY footprint corners in counter-clockwise order."""
        axes = self.axes_2d()
        hx, hy = self.half_extents_m[0], self.half_extents_m[1]
        center = np.array([self.center_m[0], self.center_m[1]], dtype=np.float64)
        local = np.array(
            [
                [-hx, -hy],
                [hx, -hy],
                [hx, hy],
                [-hx, hy],
            ],
            dtype=np.float64,
        )
        return center + local @ axes

    def corners_3d(self) -> np.ndarray:
        """Return 8x3 array of 3D corners (4 bottom CCW, then 4 top CCW)."""
        xy = self.corners_2d()
        z0 = np.full((4, 1), self.z_min_m, dtype=np.float64)
        z1 = np.full((4, 1), self.z_max_m, dtype=np.float64)
        return np.vstack((np.hstack((xy, z0)), np.hstack((xy, z1))))

    def aabb(self) -> AABB:
        corners = self.corners_3d()
        mins = corners.min(axis=0)
        maxs = corners.max(axis=0)
        return AABB(
            min_m=(float(mins[0]), float(mins[1]), float(mins[2])),
            max_m=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        )

    def world_to_local_2d(self, x: float, y: float) -> tuple[float, float]:
        dx = x - self.center_m[0]
        dy = y - self.center_m[1]
        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)
        return (dx * c + dy * s, -dx * s + dy * c)

    def contains_point_2d(self, x: float, y: float, *, margin_m: float = 0.0) -> bool:
        lx, ly = self.world_to_local_2d(x, y)
        return (
            abs(lx) <= self.half_extents_m[0] + margin_m
            and abs(ly) <= self.half_extents_m[1] + margin_m
        )

    def contains_point_3d(self, point_m: Sequence[float], *, margin_m: float = 0.0) -> bool:
        if not (
            self.z_min_m - margin_m <= float(point_m[2]) <= self.z_max_m + margin_m
        ):
            return False
        return self.contains_point_2d(float(point_m[0]), float(point_m[1]), margin_m=margin_m)

    def distance_to_point_2d(self, x: float, y: float) -> float:
        """Euclidean distance in XY from ``(x, y)`` to this box (0.0 if inside)."""
        lx, ly = self.world_to_local_2d(x, y)
        dx = max(0.0, abs(lx) - self.half_extents_m[0])
        dy = max(0.0, abs(ly) - self.half_extents_m[1])
        return math.hypot(dx, dy)

    def intersects_circle_2d(
        self, center_xy: Sequence[float], radius_m: float, *, eps: float = 1e-9
    ) -> bool:
        return self.distance_to_point_2d(float(center_xy[0]), float(center_xy[1])) < (
            radius_m - eps
        )

    def distance_2d(self, other: OrientedBox) -> float:
        """Exact Euclidean distance in XY between two oriented rectangles (0.0 if overlapping)."""
        ca = self.corners_2d()
        cb = other.corners_2d()
        if _polygons_intersect_sat_2d(ca, self.axes_2d(), cb, other.axes_2d(), eps=0.0):
            return 0.0
        min_dist = math.inf
        for pt in ca:
            for j in range(4):
                d = _point_to_segment_dist_2d(pt, cb[j], cb[(j + 1) % 4])
                if d < min_dist:
                    min_dist = d
        for pt in cb:
            for j in range(4):
                d = _point_to_segment_dist_2d(pt, ca[j], ca[(j + 1) % 4])
                if d < min_dist:
                    min_dist = d
        return float(min_dist)

    def intersects_box(
        self, other: OrientedBox, *, clearance_m: float = 0.0, eps: float = 1e-9
    ) -> bool:
        """Check whether two 3D yaw-oriented boxes intersect (or violate ``clearance_m``)."""
        if self.z_max_m <= other.z_min_m + eps or other.z_max_m <= self.z_min_m + eps:
            return False
        if clearance_m <= 0.0:
            return _polygons_intersect_sat_2d(
                self.corners_2d(),
                self.axes_2d(),
                other.corners_2d(),
                other.axes_2d(),
                eps=eps,
            )
        return self.distance_2d(other) < (clearance_m - eps)

    def is_within_room(
        self,
        width_x_m: float,
        depth_y_m: float,
        height_z_m: float,
        *,
        wall_margin_m: float = 0.0,
        eps: float = 1e-9,
    ) -> bool:
        corners = self.corners_3d()
        x_min, y_min, z_min = corners.min(axis=0)
        x_max, y_max, z_max = corners.max(axis=0)
        return bool(
            x_min >= wall_margin_m - eps
            and x_max <= width_x_m - wall_margin_m + eps
            and y_min >= wall_margin_m - eps
            and y_max <= depth_y_m - wall_margin_m + eps
            and z_min >= -eps
            and z_max <= height_z_m + eps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_m": list(self.center_m),
            "half_extents_m": list(self.half_extents_m),
            "dimensions_m": list(self.dimensions_m),
            "yaw_rad": self.yaw_rad,
            "corners_2d_m": [[float(x), float(y)] for x, y in self.corners_2d()],
            "aabb": self.aabb().to_dict(),
        }


# ---------------------------------------------------------------------------
# Planar layered surfaces (shared by RF geometric optics and renderer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceLayer:
    """One homogeneous layer inside a wall, floor, ceiling, door, or window stack."""

    material_id: str
    thickness_m: float

    def to_dict(self) -> dict[str, Any]:
        return {"material_id": self.material_id, "thickness_m": self.thickness_m}


@dataclass(frozen=True)
class SurfacePatch:
    """Planar rectangular surface patch with orientation, area, and material stack."""

    surface_id: str
    parent_id: str
    kind: str
    vertices_m: tuple[Vec3, Vec3, Vec3, Vec3]
    center_m: Vec3
    normal_m: Vec3
    u_axis_m: Vec3
    v_axis_m: Vec3
    width_m: float
    height_m: float
    area_m2: float
    material_id: str
    layers: tuple[SurfaceLayer, ...]
    total_thickness_m: float
    is_passable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_quad(
        cls,
        *,
        surface_id: str,
        parent_id: str,
        kind: str,
        origin_m: Sequence[float],
        u_vec_m: Sequence[float],
        v_vec_m: Sequence[float],
        material_id: str,
        layers: Sequence[SurfaceLayer],
        is_passable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> SurfacePatch:
        """Build a rectangular patch from bottom-left ``origin_m`` and edge vectors ``u_vec_m``, ``v_vec_m``.

        Vertices are ordered counter-clockwise when viewed from the ``+normal``
        side (``normal = normalize(u_vec x v_vec)``).
        """
        p0 = np.asarray(origin_m, dtype=np.float64)
        u_vec = np.asarray(u_vec_m, dtype=np.float64)
        v_vec = np.asarray(v_vec_m, dtype=np.float64)
        width = float(np.linalg.norm(u_vec))
        height = float(np.linalg.norm(v_vec))
        if width <= 1e-12 or height <= 1e-12:
            raise ValueError(f"Degenerate SurfacePatch {surface_id!r}: width={width}, height={height}")
        u_axis = u_vec / width
        v_axis = v_vec / height
        normal = _normalize(np.cross(u_axis, v_axis), label=f"normal({surface_id})")

        p1 = p0 + u_vec
        p2 = p0 + u_vec + v_vec
        p3 = p0 + v_vec
        center = p0 + 0.5 * u_vec + 0.5 * v_vec
        layer_tuple = tuple(layers)
        total_thickness = sum(layer.thickness_m for layer in layer_tuple)

        def _t3(arr: np.ndarray) -> Vec3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))

        return cls(
            surface_id=surface_id,
            parent_id=parent_id,
            kind=kind,
            vertices_m=(_t3(p0), _t3(p1), _t3(p2), _t3(p3)),
            center_m=_t3(center),
            normal_m=_t3(normal),
            u_axis_m=_t3(u_axis),
            v_axis_m=_t3(v_axis),
            width_m=width,
            height_m=height,
            area_m2=width * height,
            material_id=material_id,
            layers=layer_tuple,
            total_thickness_m=total_thickness,
            is_passable=is_passable,
            metadata=dict(metadata) if metadata else {},
        )

    def aabb(self) -> AABB:
        pts = np.asarray(self.vertices_m, dtype=np.float64)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return AABB(
            min_m=(float(mins[0]), float(mins[1]), float(mins[2])),
            max_m=(float(maxs[0]), float(maxs[1]), float(maxs[2])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "vertices_m": [list(v) for v in self.vertices_m],
            "center_m": list(self.center_m),
            "normal_m": list(self.normal_m),
            "width_m": self.width_m,
            "height_m": self.height_m,
            "area_m2": self.area_m2,
            "material_id": self.material_id,
            "layers": [layer.to_dict() for layer in self.layers],
            "total_thickness_m": self.total_thickness_m,
            "is_passable": self.is_passable,
            "metadata": dict(self.metadata),
        }
