"""Exact geometric ray-surface intersection, capsule occlusion, and image-method reflections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Collection, Sequence

import numpy as np

from radarsim.motion.types import BodyCapsule
from radarsim.scene import SurfacePatch

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceHit:
    surface: SurfacePatch
    distance_m: float
    point_m: Vec3
    incidence_angle_rad: float


@dataclass(frozen=True)
class FirstOrderReflection:
    surface: SurfacePatch
    bounce_point_m: Vec3
    incidence_angle_rad: float
    leg1_length_m: float
    leg2_length_m: float


@dataclass(frozen=True)
class SecondOrderReflection:
    surface1: SurfacePatch
    surface2: SurfacePatch
    bounce1_m: Vec3
    bounce2_m: Vec3
    incidence1_rad: float
    incidence2_rad: float
    leg1_length_m: float
    leg2_length_m: float
    leg3_length_m: float


def ray_intersect_surface(
    origin: np.ndarray,
    direction_unit: np.ndarray,
    surface: SurfacePatch,
    *,
    t_min: float = 1e-5,
    t_max: float = math.inf,
    edge_eps: float = 1e-6,
) -> SurfaceHit | None:
    """Intersect ray ``origin + t * direction_unit`` with planar rectangular ``SurfacePatch``."""
    normal = np.asarray(surface.normal_m, dtype=np.float64)
    denom = float(np.dot(direction_unit, normal))
    if abs(denom) < 1e-10:
        return None
    p0 = np.asarray(surface.vertices_m[0], dtype=np.float64)
    t = float(np.dot(p0 - origin, normal)) / denom
    if t <= t_min or t >= t_max:
        return None
    hit = origin + t * direction_unit
    rel = hit - p0
    u_ax = np.asarray(surface.u_axis_m, dtype=np.float64)
    v_ax = np.asarray(surface.v_axis_m, dtype=np.float64)
    u = float(np.dot(rel, u_ax))
    v = float(np.dot(rel, v_ax))
    if -edge_eps <= u <= surface.width_m + edge_eps and -edge_eps <= v <= surface.height_m + edge_eps:
        cos_theta = min(1.0, max(0.0, abs(denom)))
        return SurfaceHit(
            surface=surface,
            distance_m=t,
            point_m=(float(hit[0]), float(hit[1]), float(hit[2])),
            incidence_angle_rad=math.acos(cos_theta),
        )
    return None


def _segment_segment_distance(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> float:
    """Exact minimum 3D Euclidean distance between segments ``[p1, q1]`` and ``[p2, q2]``."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2))
    if a <= eps:
        s = 0.0
        t = max(0.0, min(1.0, f / e))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = max(0.0, min(1.0, -c / a))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom > eps:
                s = max(0.0, min(1.0, (b * f - c * e) / denom))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -c / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - c) / a))

    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return float(np.linalg.norm(c1 - c2))


class SurfaceRayAccelerator:
    """Pre-packed NumPy arrays for vectorized segment intersection across room surfaces."""

    def __init__(self, surfaces: Sequence[SurfacePatch]) -> None:
        solid = tuple(s for s in surfaces if not s.is_passable)
        self.surfaces: tuple[SurfacePatch, ...] = solid
        self.count = len(solid)
        self.id_to_index: dict[str, int] = {s.surface_id: i for i, s in enumerate(solid)}

        if self.count == 0:
            self.p0 = np.zeros((0, 3), dtype=np.float64)
            self.normals = np.zeros((0, 3), dtype=np.float64)
            self.u_axes = np.zeros((0, 3), dtype=np.float64)
            self.v_axes = np.zeros((0, 3), dtype=np.float64)
            self.widths = np.zeros(0, dtype=np.float64)
            self.heights = np.zeros(0, dtype=np.float64)
            self.d_planes = np.zeros(0, dtype=np.float64)
        else:
            self.p0 = np.asarray([s.vertices_m[0] for s in solid], dtype=np.float64)
            self.normals = np.asarray([s.normal_m for s in solid], dtype=np.float64)
            self.u_axes = np.asarray([s.u_axis_m for s in solid], dtype=np.float64)
            self.v_axes = np.asarray([s.v_axis_m for s in solid], dtype=np.float64)
            self.widths = np.asarray([s.width_m for s in solid], dtype=np.float64)
            self.heights = np.asarray([s.height_m for s in solid], dtype=np.float64)
            self.d_planes = np.einsum("ij,ij->i", self.p0, self.normals)

    def _candidate_hits(
        self,
        p_start: Sequence[float],
        p_end: Sequence[float],
        *,
        exclude_surface_ids: Collection[str] = (),
        endpoint_eps: float = 1e-4,
        edge_eps: float = 1e-6,
        early_exit: bool = False,
    ) -> list[SurfaceHit]:
        if self.count == 0:
            return []
        a = np.asarray(p_start, dtype=np.float64)
        b = np.asarray(p_end, dtype=np.float64)
        delta = b - a
        dist = math.sqrt(float(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]))
        if dist <= 2.0 * endpoint_eps:
            return []
        d_unit = delta / dist

        denom = self.normals @ d_unit
        n_dot_a = self.normals @ a
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.d_planes - n_dot_a) / denom

        valid = (
            (np.abs(denom) >= 1e-10)
            & (t > endpoint_eps)
            & (t < (dist - endpoint_eps))
        )
        if exclude_surface_ids:
            for sid in exclude_surface_ids:
                idx = self.id_to_index.get(sid)
                if idx is not None:
                    valid[idx] = False

        cand_indices = np.flatnonzero(valid)
        if cand_indices.size == 0:
            return []

        hits: list[SurfaceHit] = []
        a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
        d0, d1, d2 = float(d_unit[0]), float(d_unit[1]), float(d_unit[2])

        for idx in cand_indices:
            ti = float(t[idx])
            hx = a0 + ti * d0
            hy = a1 + ti * d1
            hz = a2 + ti * d2
            p0 = self.p0[idx]
            rx = hx - float(p0[0])
            ry = hy - float(p0[1])
            rz = hz - float(p0[2])
            ux = self.u_axes[idx]
            vx = self.v_axes[idx]
            u = rx * float(ux[0]) + ry * float(ux[1]) + rz * float(ux[2])
            if u < -edge_eps or u > float(self.widths[idx]) + edge_eps:
                continue
            v = rx * float(vx[0]) + ry * float(vx[1]) + rz * float(vx[2])
            if v < -edge_eps or v > float(self.heights[idx]) + edge_eps:
                continue
            cos_theta = min(1.0, max(0.0, abs(float(denom[idx]))))
            hits.append(
                SurfaceHit(
                    surface=self.surfaces[idx],
                    distance_m=ti,
                    point_m=(hx, hy, hz),
                    incidence_angle_rad=math.acos(cos_theta),
                )
            )
            if early_exit:
                return hits
        hits.sort(key=lambda h: h.distance_m)
        return hits

    def find_segment_hits(
        self,
        p_start: Sequence[float],
        p_end: Sequence[float],
        *,
        exclude_surface_ids: Collection[str] = (),
        endpoint_eps: float = 1e-4,
    ) -> list[SurfaceHit]:
        return self._candidate_hits(
            p_start,
            p_end,
            exclude_surface_ids=exclude_surface_ids,
            endpoint_eps=endpoint_eps,
            early_exit=False,
        )

    def is_segment_blocked(
        self,
        p_start: Sequence[float],
        p_end: Sequence[float],
        *,
        exclude_surface_ids: Collection[str] = (),
        endpoint_eps: float = 1e-4,
    ) -> bool:
        return bool(
            self._candidate_hits(
                p_start,
                p_end,
                exclude_surface_ids=exclude_surface_ids,
                endpoint_eps=endpoint_eps,
                early_exit=True,
            )
        )


def find_segment_surface_hits(
    p_start: Sequence[float],
    p_end: Sequence[float],
    surfaces: Sequence[SurfacePatch],
    *,
    exclude_surface_ids: Collection[str] = (),
    endpoint_eps: float = 1e-4,
) -> list[SurfaceHit]:
    """Return all non-passable scene surfaces intersected strictly between ``p_start`` and ``p_end``."""
    a = np.asarray(p_start, dtype=np.float64)
    b = np.asarray(p_end, dtype=np.float64)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if dist <= 2.0 * endpoint_eps:
        return []
    d_unit = delta / dist
    excluded = set(exclude_surface_ids)
    hits: list[SurfaceHit] = []
    for surf in surfaces:
        if surf.is_passable or surf.surface_id in excluded:
            continue
        hit = ray_intersect_surface(
            a,
            d_unit,
            surf,
            t_min=endpoint_eps,
            t_max=dist - endpoint_eps,
        )
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda h: h.distance_m)
    return hits


def is_segment_occluded_by_capsules(
    p_start: Sequence[float],
    p_end: Sequence[float],
    capsules: Sequence[BodyCapsule],
    *,
    exclude_segment_ids: Collection[str] = (),
    endpoint_trim_m: float = 0.02,
) -> bool:
    """Return True if any non-excluded human ``BodyCapsule`` blocks segment ``[p_start, p_end]``."""
    if not capsules:
        return False
    a = np.asarray(p_start, dtype=np.float64)
    b = np.asarray(p_end, dtype=np.float64)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if dist <= 2.0 * endpoint_trim_m:
        return False
    d_unit = delta / dist
    a_trim = a + endpoint_trim_m * d_unit
    b_trim = b - endpoint_trim_m * d_unit
    excluded = set(exclude_segment_ids)

    for cap in capsules:
        if cap.segment_id in excluded:
            continue
        ca = np.asarray(cap.a_m, dtype=np.float64)
        cb = np.asarray(cap.b_m, dtype=np.float64)
        if _segment_segment_distance(a_trim, b_trim, ca, cb) < cap.radius_m:
            return True
    return False


def is_segment_occluded(
    p_start: Sequence[float],
    p_end: Sequence[float],
    surfaces: Sequence[SurfacePatch],
    capsules: Sequence[BodyCapsule] = (),
    *,
    exclude_surface_ids: Collection[str] = (),
    exclude_segment_ids: Collection[str] = (),
    surface_accel: SurfaceRayAccelerator | None = None,
) -> bool:
    """Return True if any scene surface or human capsule blocks the line segment."""
    if surface_accel is not None:
        if surface_accel.is_segment_blocked(
            p_start, p_end, exclude_surface_ids=exclude_surface_ids
        ):
            return True
    elif find_segment_surface_hits(
        p_start, p_end, surfaces, exclude_surface_ids=exclude_surface_ids
    ):
        return True
    if capsules and is_segment_occluded_by_capsules(
        p_start, p_end, capsules, exclude_segment_ids=exclude_segment_ids
    ):
        return True
    return False


def mirror_point_across_plane(
    point: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    """Compute exact mirror image of ``point`` across plane ``(plane_origin, plane_normal)``."""
    signed_dist = float(np.dot(point - plane_origin, plane_normal))
    return point - 2.0 * signed_dist * plane_normal


def find_first_order_reflection(
    tx_pos: Sequence[float],
    rx_or_target_pos: Sequence[float],
    surface: SurfacePatch,
    all_surfaces: Sequence[SurfacePatch],
    capsules: Sequence[BodyCapsule] = (),
    *,
    exclude_segment_ids: Collection[str] = (),
    require_front_side: bool = True,
    surface_accel: SurfaceRayAccelerator | None = None,
) -> FirstOrderReflection | None:
    """Compute exact first-order image-method specular reflection off ``surface``.

    Verifies that both endpoints lie on the same side of ``surface``, the bounce
    point falls inside the finite rectangular patch, and both propagation legs
    are geometrically unoccluded.
    """
    if surface.is_passable:
        return None
    tx = np.asarray(tx_pos, dtype=np.float64)
    rx = np.asarray(rx_or_target_pos, dtype=np.float64)
    p0 = np.asarray(surface.vertices_m[0], dtype=np.float64)
    n = np.asarray(surface.normal_m, dtype=np.float64)

    d_tx = float(np.dot(tx - p0, n))
    d_rx = float(np.dot(rx - p0, n))
    if require_front_side:
        if d_tx <= 1e-4 or d_rx <= 1e-4:
            return None
    else:
        if d_tx * d_rx <= 1e-8:
            return None

    tx_img = mirror_point_across_plane(tx, p0, n)
    vec = rx - tx_img
    total_len = float(np.linalg.norm(vec))
    if total_len <= 1e-6:
        return None
    d_unit = vec / total_len

    hit = ray_intersect_surface(tx_img, d_unit, surface, t_min=1e-5, t_max=total_len - 1e-5)
    if hit is None:
        return None

    q = np.asarray(hit.point_m, dtype=np.float64)
    leg1 = float(np.linalg.norm(q - tx))
    leg2 = float(np.linalg.norm(rx - q))
    if leg1 <= 1e-4 or leg2 <= 1e-4:
        return None

    # Occlusion check on both legs
    if is_segment_occluded(
        tx,
        q,
        all_surfaces,
        capsules,
        exclude_surface_ids=(surface.surface_id,),
        exclude_segment_ids=exclude_segment_ids,
        surface_accel=surface_accel,
    ):
        return None
    if is_segment_occluded(
        q,
        rx,
        all_surfaces,
        capsules,
        exclude_surface_ids=(surface.surface_id,),
        exclude_segment_ids=exclude_segment_ids,
        surface_accel=surface_accel,
    ):
        return None

    return FirstOrderReflection(
        surface=surface,
        bounce_point_m=hit.point_m,
        incidence_angle_rad=hit.incidence_angle_rad,
        leg1_length_m=leg1,
        leg2_length_m=leg2,
    )


def find_second_order_reflection(
    tx_pos: Sequence[float],
    rx_pos: Sequence[float],
    surface1: SurfacePatch,
    surface2: SurfacePatch,
    all_surfaces: Sequence[SurfacePatch],
    capsules: Sequence[BodyCapsule] = (),
    *,
    surface_accel: SurfaceRayAccelerator | None = None,
) -> SecondOrderReflection | None:
    """Compute exact second-order image-method specular reflection ``TX -> S1 -> S2 -> RX``."""
    if surface1.surface_id == surface2.surface_id or surface1.is_passable or surface2.is_passable:
        return None

    tx = np.asarray(tx_pos, dtype=np.float64)
    rx = np.asarray(rx_pos, dtype=np.float64)
    p1 = np.asarray(surface1.vertices_m[0], dtype=np.float64)
    n1 = np.asarray(surface1.normal_m, dtype=np.float64)
    p2 = np.asarray(surface2.vertices_m[0], dtype=np.float64)
    n2 = np.asarray(surface2.normal_m, dtype=np.float64)

    if float(np.dot(tx - p1, n1)) <= 1e-4 or float(np.dot(rx - p2, n2)) <= 1e-4:
        return None

    tx_img1 = mirror_point_across_plane(tx, p1, n1)
    tx_img12 = mirror_point_across_plane(tx_img1, p2, n2)

    vec2 = rx - tx_img12
    len2_tot = float(np.linalg.norm(vec2))
    if len2_tot <= 1e-6:
        return None
    hit2 = ray_intersect_surface(tx_img12, vec2 / len2_tot, surface2, t_min=1e-5, t_max=len2_tot - 1e-5)
    if hit2 is None:
        return None
    q2 = np.asarray(hit2.point_m, dtype=np.float64)

    vec1 = q2 - tx_img1
    len1_tot = float(np.linalg.norm(vec1))
    if len1_tot <= 1e-6:
        return None
    hit1 = ray_intersect_surface(tx_img1, vec1 / len1_tot, surface1, t_min=1e-5, t_max=len1_tot - 1e-5)
    if hit1 is None:
        return None
    q1 = np.asarray(hit1.point_m, dtype=np.float64)

    # Confirm q1 is on front side of S2 and q2 is on front side of S1
    if float(np.dot(q2 - p1, n1)) <= 1e-4 or float(np.dot(q1 - p2, n2)) <= 1e-4:
        return None

    leg1 = float(np.linalg.norm(q1 - tx))
    leg2 = float(np.linalg.norm(q2 - q1))
    leg3 = float(np.linalg.norm(rx - q2))
    if leg1 <= 1e-4 or leg2 <= 1e-4 or leg3 <= 1e-4:
        return None

    if is_segment_occluded(
        tx, q1, all_surfaces, capsules, exclude_surface_ids=(surface1.surface_id,), surface_accel=surface_accel
    ):
        return None
    if is_segment_occluded(
        q1, q2, all_surfaces, capsules, exclude_surface_ids=(surface1.surface_id, surface2.surface_id), surface_accel=surface_accel
    ):
        return None
    if is_segment_occluded(
        q2, rx, all_surfaces, capsules, exclude_surface_ids=(surface2.surface_id,), surface_accel=surface_accel
    ):
        return None

    return SecondOrderReflection(
        surface1=surface1,
        surface2=surface2,
        bounce1_m=hit1.point_m,
        bounce2_m=hit2.point_m,
        incidence1_rad=hit1.incidence_angle_rad,
        incidence2_rad=hit2.incidence_angle_rad,
        leg1_length_m=leg1,
        leg2_length_m=leg2,
        leg3_length_m=leg3,
    )
