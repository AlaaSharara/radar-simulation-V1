"""Geometric-optics + stochastic diffuse RF propagation engine producing ``PathSet``.

Explicit physics fidelity boundary (per ``PLAN.md`` §6):
This engine implements ray-based geometric optics (direct LOS + occlusion,
1st/2nd-order image-method specular reflections with complex Fresnel/TMM
coefficients, finite-thickness/multilayer transmission, and seeded stochastic
diffuse scattering). It is **not** a full-wave electromagnetic solver.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from radarsim.config import RadarProfile, compute_derived_metrics
from radarsim.motion.types import BodyCapsule, MotionFrame
from radarsim.rf.antenna import antenna_gain_linear, tx_rx_world_positions
from radarsim.rf.em import SurfaceEMResponse, evaluate_surface_em
from radarsim.rf.raytrace import (
    SurfaceHit,
    SurfaceRayAccelerator,
    find_first_order_reflection,
    find_second_order_reflection,
    find_segment_surface_hits,
    is_segment_occluded,
    is_segment_occluded_by_capsules,
)
from radarsim.rf.types import PathSet, PointTarget, RFPath, RFPropagationConfig, Vec3
from radarsim.rng import make_generator
from radarsim.scene import Scene, SurfacePatch, lookup_material_properties


@dataclass(frozen=True)
class _Scatterer:
    scatterer_id: str
    position_m: Vec3
    velocity_mps: Vec3
    rcs_m2_base: float
    axis_unit: Vec3 | None
    exclude_capsule_id: str | None
    is_stochastic: bool
    provenance: dict[str, Any]


def _wrap_phase(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def _bistatic_radial_velocity(
    tx_pos: Sequence[float],
    rx_pos: Sequence[float],
    target_pos: Sequence[float],
    target_vel: Sequence[float],
) -> float:
    """Effective radial velocity ``0.5 * d(R_tx + R_rx)/dt`` [m/s]; positive = receding."""
    vx, vy, vz = float(target_vel[0]), float(target_vel[1]), float(target_vel[2])
    if vx == 0.0 and vy == 0.0 and vz == 0.0:
        return 0.0
    tx_dx = target_pos[0] - tx_pos[0]
    tx_dy = target_pos[1] - tx_pos[1]
    tx_dz = target_pos[2] - tx_pos[2]
    r_tx = math.sqrt(tx_dx * tx_dx + tx_dy * tx_dy + tx_dz * tx_dz)

    rx_dx = target_pos[0] - rx_pos[0]
    rx_dy = target_pos[1] - rx_pos[1]
    rx_dz = target_pos[2] - rx_pos[2]
    r_rx = math.sqrt(rx_dx * rx_dx + rx_dy * rx_dy + rx_dz * rx_dz)

    v_tx = (vx * tx_dx + vy * tx_dy + vz * tx_dz) / max(r_tx, 1e-12)
    v_rx = (vx * rx_dx + vy * rx_dy + vz * rx_dz) / max(r_rx, 1e-12)
    return 0.5 * (v_tx + v_rx)


def _aspect_rcs_m2(
    scatterer: _Scatterer,
    tx_pos: Sequence[float],
    radar_profile: RadarProfile,
) -> float:
    """Compute aspect-dependent RCS [m^2] clamped to configured human prior bounds."""
    rcs_m2 = scatterer.rcs_m2_base
    if scatterer.axis_unit is None or not radar_profile.human_scattering.aspect_dependence_enabled:
        return rcs_m2

    dx = scatterer.position_m[0] - tx_pos[0]
    dy = scatterer.position_m[1] - tx_pos[1]
    dz = scatterer.position_m[2] - tx_pos[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    if d < 1e-9:
        return rcs_m2
    cos_psi = abs(
        (dx * scatterer.axis_unit[0] + dy * scatterer.axis_unit[1] + dz * scatterer.axis_unit[2])
        / d
    )
    sin2_psi = max(0.0, 1.0 - cos_psi * cos_psi)
    # Broadside cylinder scattering strongest (1.0), end-on attenuated to 0.15 (-8.2 dB)
    aspect_factor = 0.15 + 0.85 * sin2_psi
    return rcs_m2 * aspect_factor


def _human_scatterers_from_frame(
    motion_frame: MotionFrame,
    radar_profile: RadarProfile,
    global_seed: int,
    *,
    preview_mode: bool = False,
) -> list[_Scatterer]:
    """Generate deterministic + seeded stochastic micro-scatterers from ``BodyCapsule`` proxies."""
    hs = radar_profile.human_scattering
    n_det = 1 if preview_mode else hs.deterministic_micro_scatterers_per_body_segment
    n_rand = 1 if preview_mode else hs.random_scatterers_per_body_segment
    n_total = max(1, n_det + n_rand)

    rcs_min_m2 = 10.0 ** (hs.rcs_prior_dbsm.minimum / 10.0)
    rcs_max_m2 = 10.0 ** (hs.rcs_prior_dbsm.maximum / 10.0)

    scatterers: list[_Scatterer] = []
    for cap in motion_frame.body_capsules:
        ax = cap.b_m[0] - cap.a_m[0]
        ay = cap.b_m[1] - cap.a_m[1]
        az = cap.b_m[2] - cap.a_m[2]
        seg_len = math.sqrt(ax * ax + ay * ay + az * az)
        if seg_len > 1e-9:
            axis_unit: Vec3 | None = (ax / seg_len, ay / seg_len, az / seg_len)
        else:
            axis_unit = None

        # Segment-specific deterministic stream keeps scatterers attached coherently across frames
        rng_seg = make_generator(global_seed, "scatterers", subkey=cap.segment_id)
        # Seeded RCS prior variation (+- 1.5 dB uncertainty draw) clamped to configured [min, max]
        rcs_jitter_db = float(rng_seg.normal(0.0, 1.5))
        seg_rcs_db = min(
            hs.rcs_prior_dbsm.maximum,
            max(hs.rcs_prior_dbsm.minimum, cap.rcs_prior_dbsm + rcs_jitter_db),
        )
        seg_rcs_m2 = (10.0 ** (seg_rcs_db / 10.0)) * hs.clothing_scattering_scale
        seg_rcs_m2 = min(rcs_max_m2, max(rcs_min_m2, seg_rcs_m2))
        per_pt_rcs = seg_rcs_m2 / n_total

        # Build orthonormal radial basis around capsule axis for surface offsets
        if axis_unit is not None:
            u_ax = np.asarray(axis_unit, dtype=np.float64)
            ref = np.array([0.0, 0.0, 1.0]) if abs(u_ax[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(u_ax, ref)
            e1 /= max(1e-12, float(np.linalg.norm(e1)))
            e2 = np.cross(u_ax, e1)
        else:
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])

        # Deterministic micro-scatterers evenly spaced along capsule centerline
        for k in range(n_det):
            t = (k + 1) / (n_det + 1)
            pos = (
                (1.0 - t) * cap.a_m[0] + t * cap.b_m[0],
                (1.0 - t) * cap.a_m[1] + t * cap.b_m[1],
                (1.0 - t) * cap.a_m[2] + t * cap.b_m[2],
            )
            scatterers.append(
                _Scatterer(
                    scatterer_id=f"{motion_frame.actor_id}:{cap.segment_id}:det_{k}",
                    position_m=pos,
                    velocity_mps=cap.velocity_mps,
                    rcs_m2_base=per_pt_rcs,
                    axis_unit=axis_unit,
                    exclude_capsule_id=cap.segment_id,
                    is_stochastic=False,
                    provenance={
                        "rcs_status": hs.rcs_status or cap.rcs_provenance,
                        "segment_id": cap.segment_id,
                        "scatterer_kind": "deterministic_body_micro_scatterer",
                    },
                )
            )

        # Seeded stochastic surface micro-scatterers around capsule circumference
        for k in range(n_rand):
            t = float(rng_seg.uniform(0.15, 0.85))
            phi = float(rng_seg.uniform(0.0, 2.0 * math.pi))
            rad = cap.radius_m * float(rng_seg.uniform(0.5, 0.95))
            offset = rad * (math.cos(phi) * e1 + math.sin(phi) * e2)
            pos = (
                (1.0 - t) * cap.a_m[0] + t * cap.b_m[0] + float(offset[0]),
                (1.0 - t) * cap.a_m[1] + t * cap.b_m[1] + float(offset[1]),
                (1.0 - t) * cap.a_m[2] + t * cap.b_m[2] + float(offset[2]),
            )
            w_scale = float(rng_seg.uniform(0.6, 1.4))
            scatterers.append(
                _Scatterer(
                    scatterer_id=f"{motion_frame.actor_id}:{cap.segment_id}:rnd_{k}",
                    position_m=pos,
                    velocity_mps=cap.velocity_mps,
                    rcs_m2_base=per_pt_rcs * w_scale,
                    axis_unit=axis_unit,
                    exclude_capsule_id=cap.segment_id,
                    is_stochastic=True,
                    provenance={
                        "rcs_status": hs.rcs_status or cap.rcs_provenance,
                        "segment_id": cap.segment_id,
                        "scatterer_kind": "seeded_stochastic_body_scatterer",
                    },
                )
            )

    return scatterers


class RFPropagationEngine:
    """Room-scale geometric-optics and stochastic-diffuse RF propagation engine."""

    def __init__(
        self,
        scene: Scene,
        radar_profile: RadarProfile,
        *,
        config: RFPropagationConfig | None = None,
        global_seed: int = 20261123,
    ) -> None:
        self.scene = scene
        self.radar_profile = radar_profile
        self.config = RFPropagationConfig() if config is None else config
        self.global_seed = int(global_seed)

        self.derived = compute_derived_metrics(radar_profile)
        self.c_mps = radar_profile.rf.speed_of_light_m_per_s
        self.fc_hz = radar_profile.rf.carrier_frequency_hz
        self.wavelength_m = self.derived.wavelength_m
        self.tx_world, self.rx_world = tx_rx_world_positions(radar_profile, scene.radar_frame)
        self._surface_accel = SurfaceRayAccelerator(self.scene.surfaces)

        # Pre-generate deterministic rough-surface diffuse scatter points per scene surface
        self._diffuse_surface_points = self._precompute_diffuse_surface_points()

    def _precompute_diffuse_surface_points(self) -> list[tuple[SurfacePatch, Vec3, float]]:
        """Sample deterministic seeded diffuse-scattering patches on rough scene surfaces."""
        if not self.config.enable_diffuse_scattering or self.config.diffuse_points_per_surface <= 0:
            return []

        rng = make_generator(self.global_seed, "rf_diffuse")
        pts_per_surf = 1 if self.config.preview_mode else self.config.diffuse_points_per_surface
        # Prioritize room walls, floor, ceiling, and largest furniture faces with roughness > 0
        candidates = [
            s
            for s in self.scene.surfaces
            if (not s.is_passable)
            and lookup_material_properties(self.scene.materials, s.material_id, self.fc_hz).roughness_rms_m > 0.0
        ]
        candidates.sort(key=lambda s: s.area_m2, reverse=True)
        # Room shell and furniture have SEPARATE budgets.  Furniture panels are much smaller than
        # walls/floor, so a single "largest first" list never selected any furniture and the
        # diffuse (rough-surface) return of fabric/wood furniture was silently missing.
        shell = [s for s in candidates if s.kind != "furniture"]
        furniture = [s for s in candidates if s.kind == "furniture"]
        selected = shell[: self.config.max_diffuse_surfaces] + furniture[
            : max(0, self.config.max_diffuse_furniture_surfaces)
        ]

        points: list[tuple[SurfacePatch, Vec3, float]] = []
        for surf in selected:
            p0 = np.asarray(surf.vertices_m[0], dtype=np.float64)
            u_ax = np.asarray(surf.u_axis_m, dtype=np.float64)
            v_ax = np.asarray(surf.v_axis_m, dtype=np.float64)
            n = np.asarray(surf.normal_m, dtype=np.float64)
            patch_area = surf.area_m2 / pts_per_surf
            for _ in range(pts_per_surf):
                u = float(rng.uniform(0.15, 0.85)) * surf.width_m
                v = float(rng.uniform(0.15, 0.85)) * surf.height_m
                # Slight 1 mm offset along normal so the surface does not self-occlude
                pt = p0 + u * u_ax + v * v_ax + 1e-3 * n
                points.append((surf, (float(pt[0]), float(pt[1]), float(pt[2])), patch_area))
        return points

    def _evaluate_em(self, surface: SurfacePatch, theta_i_rad: float) -> SurfaceEMResponse:
        return evaluate_surface_em(
            surface.layers,
            self.scene.materials,
            theta_i_rad,
            self.fc_hz,
            polarization=self.config.polarization,
            speed_of_light_m_per_s=self.c_mps,
        )

    def _make_target_path(
        self,
        *,
        path_id: str,
        tx_id: int,
        rx_id: int,
        target_id: str,
        surface_id: str | None,
        path_type: str,
        vertices_m: tuple[Vec3, ...],
        tx_leg_m: float,
        rx_leg_m: float,
        total_len_m: float,
        radial_vel_mps: float,
        rcs_m2: float,
        extra_coeff: complex,
        first_tx_point: Vec3,
        last_rx_point: Vec3,
        model_flags: dict[str, Any],
        provenance: dict[str, Any],
    ) -> RFPath | None:
        g_tx = antenna_gain_linear(self.radar_profile, self.scene.radar_frame, first_tx_point)
        g_rx = antenna_gain_linear(self.radar_profile, self.scene.radar_frame, last_rx_point)

        # Bistatic radar equation transfer amplitude: sqrt(P_r / P_t)
        # |h|^2 = G_tx * G_rx * lambda^2 * sigma / ((4*pi)^3 * R_tx^2 * R_rx^2) * |extra_coeff|^2
        geom_denom = (4.0 * math.pi) ** 1.5 * max(tx_leg_m, 1e-3) * max(rx_leg_m, 1e-3)
        base_amp = math.sqrt(g_tx * g_rx * max(rcs_m2, 1e-18)) * self.wavelength_m / geom_denom
        amp_linear = base_amp * abs(extra_coeff)
        if amp_linear < self.config.min_path_amplitude_linear:
            return None

        delay_s = total_len_m / self.c_mps
        doppler_hz = -2.0 * radial_vel_mps / self.wavelength_m
        prop_phase = _wrap_phase(-2.0 * math.pi * self.fc_hz * delay_s)
        total_phase = _wrap_phase(prop_phase + cmath.phase(extra_coeff))
        path_loss_db = -20.0 * math.log10(max(amp_linear, 1e-30))

        return RFPath(
            path_id=path_id,
            tx_id=tx_id,
            rx_id=rx_id,
            target_id=target_id,
            surface_id=surface_id,
            path_type=path_type,
            vertices_m=vertices_m,
            tx_leg_length_m=tx_leg_m,
            rx_leg_length_m=rx_leg_m,
            total_path_length_m=total_len_m,
            delay_s=delay_s,
            radial_velocity_mps=radial_vel_mps,
            doppler_hz=doppler_hz,
            phase_rad=prop_phase,
            path_loss_db=path_loss_db,
            reflection_coeff_real=float(extra_coeff.real),
            reflection_coeff_imag=float(extra_coeff.imag),
            rcs_m2=rcs_m2,
            amplitude_linear=amp_linear,
            amplitude_phase_rad=total_phase,
            model_flags=model_flags,
            provenance=provenance,
        )

    def _make_specular_surface_path(
        self,
        *,
        path_id: str,
        tx_id: int,
        rx_id: int,
        surface_id: str,
        vertices_m: tuple[Vec3, ...],
        tx_leg_m: float,
        rx_leg_m: float,
        total_len_m: float,
        coeff: complex,
        first_tx_point: Vec3,
        last_rx_point: Vec3,
        model_flags: dict[str, Any],
        provenance: dict[str, Any],
    ) -> RFPath | None:
        g_tx = antenna_gain_linear(self.radar_profile, self.scene.radar_frame, first_tx_point)
        g_rx = antenna_gain_linear(self.radar_profile, self.scene.radar_frame, last_rx_point)

        # Friis image-theory transfer amplitude: sqrt(P_r / P_t) = sqrt(G_tx*G_rx) * lambda / (4*pi*L) * |R|
        amp_linear = (
            math.sqrt(g_tx * g_rx)
            * self.wavelength_m
            / (4.0 * math.pi * max(total_len_m, 1e-3))
            * abs(coeff)
        )
        if amp_linear < self.config.min_path_amplitude_linear:
            return None

        delay_s = total_len_m / self.c_mps
        prop_phase = _wrap_phase(-2.0 * math.pi * self.fc_hz * delay_s)
        total_phase = _wrap_phase(prop_phase + cmath.phase(coeff))
        path_loss_db = -20.0 * math.log10(max(amp_linear, 1e-30))

        return RFPath(
            path_id=path_id,
            tx_id=tx_id,
            rx_id=rx_id,
            target_id=None,
            surface_id=surface_id,
            path_type="specular_reflection",
            vertices_m=vertices_m,
            tx_leg_length_m=tx_leg_m,
            rx_leg_length_m=rx_leg_m,
            total_path_length_m=total_len_m,
            delay_s=delay_s,
            radial_velocity_mps=0.0,
            doppler_hz=0.0,
            phase_rad=prop_phase,
            path_loss_db=path_loss_db,
            reflection_coeff_real=float(coeff.real),
            reflection_coeff_imag=float(coeff.imag),
            rcs_m2=None,
            amplitude_linear=amp_linear,
            amplitude_phase_rad=total_phase,
            model_flags=model_flags,
            provenance=provenance,
        )

    def propagate(
        self,
        frame_id: int,
        sim_time_s: float,
        *,
        motion_frame: MotionFrame | None = None,
        point_targets: Sequence[PointTarget] = (),
        tx_ids: Sequence[int] | None = None,
        rx_ids: Sequence[int] | None = None,
        include_scene_surfaces: bool = True,
    ) -> PathSet:
        """Compute all active RF propagation paths for one frame across configured TX/RX pairs."""
        active_tx = tuple(range(self.radar_profile.rf.tx_count)) if tx_ids is None else tuple(tx_ids)
        active_rx = tuple(range(self.radar_profile.rf.rx_count)) if rx_ids is None else tuple(rx_ids)

        surfaces = self.scene.surfaces if include_scene_surfaces else ()
        surface_accel = self._surface_accel if include_scene_surfaces else None
        capsules: Sequence[BodyCapsule] = (
            motion_frame.body_capsules
            if (motion_frame is not None and self.config.enable_human_body_occlusion)
            else ()
        )

        # Assemble scatterers: explicit analytical point targets + human body micro-scatterers
        scatterers: list[_Scatterer] = []
        for pt in point_targets:
            scatterers.append(
                _Scatterer(
                    scatterer_id=pt.target_id,
                    position_m=pt.position_m,
                    velocity_mps=pt.velocity_mps,
                    rcs_m2_base=pt.rcs_m2,
                    axis_unit=None,
                    exclude_capsule_id=None,
                    is_stochastic=False,
                    provenance={"source": pt.provenance},
                )
            )
        if motion_frame is not None:
            scatterers.extend(
                _human_scatterers_from_frame(
                    motion_frame,
                    self.radar_profile,
                    self.global_seed,
                    preview_mode=self.config.preview_mode,
                )
            )

        paths: list[RFPath] = []
        occluded_los_count = 0
        second_order_count = 0
        path_seq = 0

        # ------------------------------------------------------------------
        # 1. Direct LOS and Transmitted paths to scatterers / point targets
        # ------------------------------------------------------------------
        if self.config.enable_los or self.config.enable_transmission:
            for sc in scatterers:
                excl_caps = (sc.exclude_capsule_id,) if sc.exclude_capsule_id else ()
                rx_cache: dict[int, tuple[list[SurfaceHit], bool, float]] = {}
                for rx_id in active_rx:
                    rx_p = self.rx_world[rx_id]
                    r_hits = (
                        surface_accel.find_segment_hits(sc.position_m, rx_p)
                        if surface_accel is not None
                        else find_segment_surface_hits(sc.position_m, rx_p, surfaces)
                    )
                    r_blk = is_segment_occluded_by_capsules(
                        sc.position_m, rx_p, capsules, exclude_segment_ids=excl_caps
                    )
                    rx_cache[rx_id] = (r_hits, r_blk, _dist(sc.position_m, rx_p))

                for tx_id in active_tx:
                    tx_p = self.tx_world[tx_id]
                    tx_hits = (
                        surface_accel.find_segment_hits(tx_p, sc.position_m)
                        if surface_accel is not None
                        else find_segment_surface_hits(tx_p, sc.position_m, surfaces)
                    )
                    tx_body_blocked = is_segment_occluded_by_capsules(
                        tx_p, sc.position_m, capsules, exclude_segment_ids=excl_caps
                    )
                    tx_leg = _dist(tx_p, sc.position_m)
                    rcs_m2 = _aspect_rcs_m2(sc, tx_p, self.radar_profile)

                    for rx_id in active_rx:
                        rx_p = self.rx_world[rx_id]
                        rx_hits, rx_body_blocked, rx_leg = rx_cache[rx_id]
                        total_len = tx_leg + rx_leg
                        v_r = _bistatic_radial_velocity(tx_p, rx_p, sc.position_m, sc.velocity_mps)

                        if tx_body_blocked or rx_body_blocked:
                            occluded_los_count += 1
                            continue

                        if not tx_hits and not rx_hits:
                            if self.config.enable_los:
                                path_seq += 1
                                p = self._make_target_path(
                                    path_id=f"p{frame_id}_{path_seq:05d}",
                                    tx_id=tx_id,
                                    rx_id=rx_id,
                                    target_id=sc.scatterer_id,
                                    surface_id=None,
                                    path_type="los",
                                    vertices_m=(tx_p, sc.position_m, rx_p),
                                    tx_leg_m=tx_leg,
                                    rx_leg_m=rx_leg,
                                    total_len_m=total_len,
                                    radial_vel_mps=v_r,
                                    rcs_m2=rcs_m2,
                                    extra_coeff=1.0 + 0.0j,
                                    first_tx_point=sc.position_m,
                                    last_rx_point=sc.position_m,
                                    model_flags={
                                        "propagation_model": "geometric_optics_direct_los",
                                        "polarization": self.config.polarization,
                                        "is_stochastic_scatterer": sc.is_stochastic,
                                        "full_wave_em": False,
                                    },
                                    provenance=sc.provenance,
                                )
                                if p is not None:
                                    paths.append(p)
                        else:
                            occluded_los_count += 1
                            # Evaluate finite-thickness / multilayer slab transmission if configured
                            if (
                                self.config.enable_transmission
                                and len(tx_hits) <= self.config.max_transmission_slabs
                                and len(rx_hits) <= self.config.max_transmission_slabs
                            ):
                                t_total = 1.0 + 0.0j
                                crossed_ids: list[str] = []
                                extrapolated = False
                                for hit in (*tx_hits, *rx_hits):
                                    em = self._evaluate_em(hit.surface, hit.incidence_angle_rad)
                                    t_total *= em.transmission
                                    crossed_ids.append(hit.surface.surface_id)
                                    if em.frequency_extrapolated:
                                        extrapolated = True
                                if abs(t_total) > 1e-6:
                                    path_seq += 1
                                    p = self._make_target_path(
                                        path_id=f"p{frame_id}_{path_seq:05d}",
                                        tx_id=tx_id,
                                        rx_id=rx_id,
                                        target_id=sc.scatterer_id,
                                        surface_id=",".join(crossed_ids),
                                        path_type="transmitted",
                                        vertices_m=(tx_p, sc.position_m, rx_p),
                                        tx_leg_m=tx_leg,
                                        rx_leg_m=rx_leg,
                                        total_len_m=total_len,
                                        radial_vel_mps=v_r,
                                        rcs_m2=rcs_m2,
                                        extra_coeff=t_total,
                                        first_tx_point=sc.position_m,
                                        last_rx_point=sc.position_m,
                                        model_flags={
                                            "propagation_model": "geometric_optics_transfer_matrix_transmission",
                                            "polarization": self.config.polarization,
                                            "crossed_slabs": len(crossed_ids),
                                            "frequency_extrapolated": extrapolated,
                                            "full_wave_em": False,
                                        },
                                        provenance={
                                            **sc.provenance,
                                            "crossed_surfaces": crossed_ids,
                                        },
                                    )
                                    if p is not None:
                                        paths.append(p)

        # ------------------------------------------------------------------
        # 2. First-order specular reflections (static scene surfaces)
        # ------------------------------------------------------------------
        if (
            self.config.enable_first_order_reflections
            and self.config.enable_static_reflections
            and surfaces
        ):
            for surf in surfaces:
                if surf.is_passable:
                    continue
                for tx_id in active_tx:
                    tx_p = self.tx_world[tx_id]
                    for rx_id in active_rx:
                        rx_p = self.rx_world[rx_id]
                        refl = find_first_order_reflection(
                            tx_p,
                            rx_p,
                            surf,
                            surfaces,
                            capsules,
                            surface_accel=surface_accel,
                        )
                        if refl is None:
                            continue
                        em = self._evaluate_em(surf, refl.incidence_angle_rad)
                        if abs(em.reflection_coherent) <= 1e-9:
                            continue
                        path_seq += 1
                        p = self._make_specular_surface_path(
                            path_id=f"p{frame_id}_{path_seq:05d}",
                            tx_id=tx_id,
                            rx_id=rx_id,
                            surface_id=surf.surface_id,
                            vertices_m=(tx_p, refl.bounce_point_m, rx_p),
                            tx_leg_m=refl.leg1_length_m,
                            rx_leg_m=refl.leg2_length_m,
                            total_len_m=refl.leg1_length_m + refl.leg2_length_m,
                            coeff=em.reflection_coherent,
                            first_tx_point=refl.bounce_point_m,
                            last_rx_point=refl.bounce_point_m,
                            model_flags={
                                "propagation_model": "geometric_optics_first_order_image_reflection",
                                "em_method": em.method,
                                "polarization": self.config.polarization,
                                "incidence_angle_rad": refl.incidence_angle_rad,
                                "roughness_factor": em.roughness_factor,
                                "frequency_extrapolated": em.frequency_extrapolated,
                                "full_wave_em": False,
                            },
                            provenance={"layers": list(em.provenance_notes)},
                        )
                        if p is not None:
                            paths.append(p)

        # ------------------------------------------------------------------
        # 3. Target-involved first-order wall/floor multipath
        #    Both reciprocal first-order legs:
        #      (a) TX -> Surface -> target -> RX
        #      (b) TX -> target -> Surface -> RX
        # ------------------------------------------------------------------
        if self.config.enable_first_order_reflections and self.config.enable_target_multipath and surfaces and scatterers:
            # Restrict to primary room boundaries (walls, floor, ceiling) for clean bounded multipath
            major_surfaces = [s for s in surfaces if s.kind in ("wall", "floor", "ceiling")]
            # Use representative deterministic scatterers in preview mode or torso/point targets
            mp_scatterers = [
                sc for sc in scatterers if not sc.is_stochastic
            ][: (1 if self.config.preview_mode else 12)]

            for sc in mp_scatterers:
                excl_caps = (sc.exclude_capsule_id,) if sc.exclude_capsule_id else ()
                rx_occ = {
                    rx_id: is_segment_occluded(
                        sc.position_m,
                        self.rx_world[rx_id],
                        surfaces,
                        capsules,
                        exclude_segment_ids=excl_caps,
                        surface_accel=surface_accel,
                    )
                    for rx_id in active_rx
                }
                tx_occ = {
                    tx_id: is_segment_occluded(
                        self.tx_world[tx_id],
                        sc.position_m,
                        surfaces,
                        capsules,
                        exclude_segment_ids=excl_caps,
                        surface_accel=surface_accel,
                    )
                    for tx_id in active_tx
                }
                for surf in major_surfaces:
                    # (a) Outgoing surface bounce: TX -> S -> target -> RX
                    for tx_id in active_tx:
                        tx_p = self.tx_world[tx_id]
                        refl = find_first_order_reflection(
                            tx_p,
                            sc.position_m,
                            surf,
                            surfaces,
                            capsules,
                            exclude_segment_ids=excl_caps,
                            surface_accel=surface_accel,
                        )
                        if refl is None:
                            continue
                        em = self._evaluate_em(surf, refl.incidence_angle_rad)
                        if abs(em.reflection_coherent) <= 1e-6:
                            continue
                        tx_leg_tot = refl.leg1_length_m + refl.leg2_length_m
                        rcs_m2 = _aspect_rcs_m2(sc, refl.bounce_point_m, self.radar_profile)
                        for rx_id in active_rx:
                            if rx_occ[rx_id]:
                                continue
                            rx_p = self.rx_world[rx_id]
                            rx_leg = _dist(sc.position_m, rx_p)
                            v_r = _bistatic_radial_velocity(
                                refl.bounce_point_m, rx_p, sc.position_m, sc.velocity_mps
                            )
                            path_seq += 1
                            p = self._make_target_path(
                                path_id=f"p{frame_id}_{path_seq:05d}",
                                tx_id=tx_id,
                                rx_id=rx_id,
                                target_id=sc.scatterer_id,
                                surface_id=surf.surface_id,
                                path_type="specular_reflection",
                                vertices_m=(tx_p, refl.bounce_point_m, sc.position_m, rx_p),
                                tx_leg_m=tx_leg_tot,
                                rx_leg_m=rx_leg,
                                total_len_m=tx_leg_tot + rx_leg,
                                radial_vel_mps=v_r,
                                rcs_m2=rcs_m2,
                                extra_coeff=em.reflection_coherent,
                                first_tx_point=refl.bounce_point_m,
                                last_rx_point=sc.position_m,
                                model_flags={
                                    "propagation_model": "geometric_optics_target_first_order_multipath",
                                    "multipath_leg": "tx_surface_target_rx",
                                    "em_method": em.method,
                                    "polarization": self.config.polarization,
                                    "incidence_angle_rad": refl.incidence_angle_rad,
                                    "full_wave_em": False,
                                },
                                provenance={**sc.provenance, "surface_layers": list(em.provenance_notes)},
                            )
                            if p is not None:
                                paths.append(p)

                    # (b) Return surface bounce: TX -> target -> S -> RX
                    for rx_id in active_rx:
                        rx_p = self.rx_world[rx_id]
                        refl_rx = find_first_order_reflection(
                            sc.position_m,
                            rx_p,
                            surf,
                            surfaces,
                            capsules,
                            exclude_segment_ids=excl_caps,
                            surface_accel=surface_accel,
                        )
                        if refl_rx is None:
                            continue
                        em = self._evaluate_em(surf, refl_rx.incidence_angle_rad)
                        if abs(em.reflection_coherent) <= 1e-6:
                            continue
                        rx_leg_tot = refl_rx.leg1_length_m + refl_rx.leg2_length_m
                        for tx_id in active_tx:
                            if tx_occ[tx_id]:
                                continue
                            tx_p = self.tx_world[tx_id]
                            tx_leg = _dist(tx_p, sc.position_m)
                            rcs_m2 = _aspect_rcs_m2(sc, tx_p, self.radar_profile)
                            v_r = _bistatic_radial_velocity(
                                tx_p, refl_rx.bounce_point_m, sc.position_m, sc.velocity_mps
                            )
                            path_seq += 1
                            p = self._make_target_path(
                                path_id=f"p{frame_id}_{path_seq:05d}",
                                tx_id=tx_id,
                                rx_id=rx_id,
                                target_id=sc.scatterer_id,
                                surface_id=surf.surface_id,
                                path_type="specular_reflection",
                                vertices_m=(tx_p, sc.position_m, refl_rx.bounce_point_m, rx_p),
                                tx_leg_m=tx_leg,
                                rx_leg_m=rx_leg_tot,
                                total_len_m=tx_leg + rx_leg_tot,
                                radial_vel_mps=v_r,
                                rcs_m2=rcs_m2,
                                extra_coeff=em.reflection_coherent,
                                first_tx_point=sc.position_m,
                                last_rx_point=refl_rx.bounce_point_m,
                                model_flags={
                                    "propagation_model": "geometric_optics_target_first_order_multipath",
                                    "multipath_leg": "tx_target_surface_rx",
                                    "em_method": em.method,
                                    "polarization": self.config.polarization,
                                    "incidence_angle_rad": refl_rx.incidence_angle_rad,
                                    "full_wave_em": False,
                                },
                                provenance={**sc.provenance, "surface_layers": list(em.provenance_notes)},
                            )
                            if p is not None:
                                paths.append(p)

        # ------------------------------------------------------------------
        # 4. Optional bounded second-order specular reflections (TX -> S1 -> S2 -> RX)
        # ------------------------------------------------------------------
        if (
            self.config.enable_second_order_reflections
            and self.config.max_second_order_paths > 0
            and surfaces
        ):
            major = [s for s in surfaces if s.kind in ("wall", "floor", "ceiling")]
            for s1 in major:
                if second_order_count >= self.config.max_second_order_paths:
                    break
                for s2 in major:
                    if second_order_count >= self.config.max_second_order_paths:
                        break
                    if s1.surface_id == s2.surface_id:
                        continue
                    for tx_id in active_tx:
                        if second_order_count >= self.config.max_second_order_paths:
                            break
                        tx_p = self.tx_world[tx_id]
                        for rx_id in active_rx:
                            if second_order_count >= self.config.max_second_order_paths:
                                break
                            rx_p = self.rx_world[rx_id]
                            r2 = find_second_order_reflection(
                                tx_p,
                                rx_p,
                                s1,
                                s2,
                                surfaces,
                                capsules,
                                surface_accel=surface_accel,
                            )
                            if r2 is None:
                                continue
                            em1 = self._evaluate_em(s1, r2.incidence1_rad)
                            em2 = self._evaluate_em(s2, r2.incidence2_rad)
                            coeff = em1.reflection_coherent * em2.reflection_coherent
                            if abs(coeff) <= 1e-9:
                                continue
                            path_seq += 1
                            tot = r2.leg1_length_m + r2.leg2_length_m + r2.leg3_length_m
                            p = self._make_specular_surface_path(
                                path_id=f"p{frame_id}_{path_seq:05d}",
                                tx_id=tx_id,
                                rx_id=rx_id,
                                surface_id=f"{s1.surface_id}>{s2.surface_id}",
                                vertices_m=(tx_p, r2.bounce1_m, r2.bounce2_m, rx_p),
                                tx_leg_m=r2.leg1_length_m,
                                rx_leg_m=r2.leg3_length_m,
                                total_len_m=tot,
                                coeff=coeff,
                                first_tx_point=r2.bounce1_m,
                                last_rx_point=r2.bounce2_m,
                                model_flags={
                                    "propagation_model": "geometric_optics_second_order_image_reflection",
                                    "reflection_order": 2,
                                    "polarization": self.config.polarization,
                                    "full_wave_em": False,
                                },
                                provenance={
                                    "surface1": s1.surface_id,
                                    "surface2": s2.surface_id,
                                },
                            )
                            if p is not None:
                                paths.append(p)
                                second_order_count += 1

        # ------------------------------------------------------------------
        # 5. Seeded stochastic diffuse scattering on rough scene surfaces
        # ------------------------------------------------------------------
        if self.config.enable_diffuse_scattering and surfaces and self._diffuse_surface_points:
            for surf, pt_m, patch_area_m2 in self._diffuse_surface_points:
                n = np.asarray(surf.normal_m, dtype=np.float64)
                pt_arr = np.asarray(pt_m, dtype=np.float64)
                for tx_id in active_tx:
                    tx_p = self.tx_world[tx_id]
                    to_tx = np.asarray(tx_p, dtype=np.float64) - pt_arr
                    r_tx = float(np.linalg.norm(to_tx))
                    cos_i = float(np.dot(to_tx / max(r_tx, 1e-9), n))
                    if cos_i <= 1e-3:
                        continue
                    if is_segment_occluded(
                        tx_p,
                        pt_m,
                        surfaces,
                        capsules,
                        exclude_surface_ids=(surf.surface_id,),
                        surface_accel=surface_accel,
                    ):
                        continue
                    theta_i = math.acos(min(1.0, cos_i))
                    em = self._evaluate_em(surf, theta_i)
                    if em.diffuse_energy_fraction <= 1e-8:
                        continue
                    # Effective Lambertian diffuse patch RCS (passive bounded by |R_smooth|^2 * diffuse_frac * area)
                    diff_coeff = math.sqrt(em.diffuse_energy_fraction) * em.reflection_smooth
                    for rx_id in active_rx:
                        rx_p = self.rx_world[rx_id]
                        to_rx = np.asarray(rx_p, dtype=np.float64) - pt_arr
                        r_rx = float(np.linalg.norm(to_rx))
                        cos_s = float(np.dot(to_rx / max(r_rx, 1e-9), n))
                        if cos_s <= 1e-3:
                            continue
                        if is_segment_occluded(
                            pt_m,
                            rx_p,
                            surfaces,
                            capsules,
                            exclude_surface_ids=(surf.surface_id,),
                            surface_accel=surface_accel,
                        ):
                            continue
                        # Lambertian effective bistatic RCS [m^2]
                        eff_rcs_m2 = (patch_area_m2 / math.pi) * cos_i * cos_s
                        path_seq += 1
                        p = self._make_target_path(
                            path_id=f"p{frame_id}_{path_seq:05d}",
                            tx_id=tx_id,
                            rx_id=rx_id,
                            target_id=None,
                            surface_id=surf.surface_id,
                            path_type="diffuse_scatter",
                            vertices_m=(tx_p, pt_m, rx_p),
                            tx_leg_m=r_tx,
                            rx_leg_m=r_rx,
                            total_len_m=r_tx + r_rx,
                            radial_vel_mps=0.0,
                            rcs_m2=eff_rcs_m2,
                            extra_coeff=diff_coeff,
                            first_tx_point=pt_m,
                            last_rx_point=pt_m,
                            model_flags={
                                "propagation_model": "geometric_optics_stochastic_diffuse_not_full_wave_em",
                                "polarization": self.config.polarization,
                                "diffuse_energy_fraction": em.diffuse_energy_fraction,
                                "full_wave_em": False,
                            },
                            provenance={
                                "surface_id": surf.surface_id,
                                "material_id": surf.material_id,
                                "roughness_status": "assumption_stochastic_roughness_approximation",
                            },
                        )
                        if p is not None:
                            paths.append(p)

        counts_by_type = {
            "los": sum(1 for p in paths if p.path_type == "los"),
            "specular_reflection": sum(1 for p in paths if p.path_type == "specular_reflection"),
            "diffuse_scatter": sum(1 for p in paths if p.path_type == "diffuse_scatter"),
            "transmitted": sum(1 for p in paths if p.path_type == "transmitted"),
        }

        diagnostics: dict[str, Any] = {
            "model_disclaimer": "geometric_optics_and_stochastic_diffuse_not_full_wave_em",
            "polarization": self.config.polarization,
            "carrier_frequency_hz": self.fc_hz,
            "wavelength_m": self.wavelength_m,
            "seed": self.global_seed,
            "preview_mode": self.config.preview_mode,
            "total_paths": len(paths),
            "counts_by_type": counts_by_type,
            "occluded_los_count": occluded_los_count,
            "second_order_count": second_order_count,
            "conventions": {
                "delay_s": "total_path_length_m / speed_of_light_m_per_s",
                "radial_velocity_mps": "0.5 * d(tx_leg + rx_leg)/dt (positive receding)",
                "doppler_hz": "-2 * radial_velocity_mps / wavelength_m (RF carrier shift)",
                "amplitude_linear": "sqrt(P_r / P_t) channel transfer amplitude",
                "permittivity": "epsilon_real - j * epsilon_imag",
            },
        }
        return PathSet(
            frame_id=frame_id,
            sim_time_s=sim_time_s,
            paths=paths,
            diagnostics=diagnostics,
        )
