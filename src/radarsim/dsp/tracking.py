"""Radar-only detection clustering, multi-frame Kalman tracking, and sparse occupancy voxels.

Operates strictly on radar-derived ``RadarDetection`` inputs; never accesses
ground-truth actor poses or room geometry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from radarsim.dsp.types import RadarDetection, RadarOccupancyVoxels, RadarTrack


@dataclass(frozen=True)
class DetectionCluster:
    cluster_id: int
    centroid_m: tuple[float, float, float]
    mean_radial_velocity_mps: float
    max_abs_radial_velocity_mps: float
    mean_snr_db: float
    z_min_m: float
    z_max_m: float
    detection_indices: tuple[int, ...]


def cluster_detections(
    detections: Sequence[RadarDetection],
    *,
    eps_spatial_m: float = 0.70,
    eps_velocity_mps: float = 1.50,
    min_samples: int = 1,
) -> tuple[list[DetectionCluster], list[int | None]]:
    """Deterministic DBSCAN-like clustering of radar detections in 3D + radial velocity."""
    n = len(detections)
    if n == 0:
        return [], []

    labels: list[int | None] = [None] * n
    visited = [False] * n
    clusters: list[DetectionCluster] = []
    next_cid = 0

    def _neighbors(idx: int) -> list[int]:
        di = detections[idx]
        out: list[int] = []
        for j in range(n):
            dj = detections[j]
            dist = math.sqrt(
                (di.x_m - dj.x_m) ** 2 + (di.y_m - dj.y_m) ** 2 + (di.z_m - dj.z_m) ** 2
            )
            if dist <= eps_spatial_m and abs(di.radial_velocity_mps - dj.radial_velocity_mps) <= eps_velocity_mps:
                out.append(j)
        return out

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = _neighbors(i)
        if len(nbrs) < min_samples:
            continue
        cid = next_cid
        next_cid += 1
        members: list[int] = []
        queue = list(nbrs)
        in_queue = set(queue)
        while queue:
            cur = queue.pop(0)
            if not visited[cur]:
                visited[cur] = True
                cur_nbrs = _neighbors(cur)
                if len(cur_nbrs) >= min_samples:
                    for nb in cur_nbrs:
                        if nb not in in_queue:
                            in_queue.add(nb)
                            queue.append(nb)
            if labels[cur] is None:
                labels[cur] = cid
                members.append(cur)

        # SNR-weighted centroid
        weights = [max(0.1, 10.0 ** (detections[m].snr_db / 20.0)) for m in members]
        w_sum = sum(weights)
        cx = sum(w * detections[m].x_m for w, m in zip(weights, members)) / w_sum
        cy = sum(w * detections[m].y_m for w, m in zip(weights, members)) / w_sum
        cz = sum(w * detections[m].z_m for w, m in zip(weights, members)) / w_sum
        mean_vr = sum(w * detections[m].radial_velocity_mps for w, m in zip(weights, members)) / w_sum
        max_abs_vr = max(abs(detections[m].radial_velocity_mps) for m in members)
        mean_snr = sum(detections[m].snr_db for m in members) / len(members)
        zs = [detections[m].z_m for m in members]

        clusters.append(
            DetectionCluster(
                cluster_id=cid,
                centroid_m=(cx, cy, cz),
                mean_radial_velocity_mps=mean_vr,
                max_abs_radial_velocity_mps=max_abs_vr,
                mean_snr_db=mean_snr,
                z_min_m=min(zs),
                z_max_m=max(zs),
                detection_indices=tuple(members),
            )
        )

    return clusters, labels


@dataclass
class _TrackHistorySample:
    sim_time_s: float
    z_m: float
    vz_mps: float
    speed_3d_mps: float
    radial_vel_peak_mps: float
    snr_db: float
    associated_count: int


@dataclass
class _InternalTrack:
    track_id: int
    state_vec: np.ndarray  # [x, y, z, vx, vy, vz]
    cov: np.ndarray        # 6x6 covariance
    age_frames: int = 1
    hits: int = 1
    missed_frames: int = 0
    last_time_s: float = 0.0
    last_snr_db: float = 0.0
    last_det_count: int = 0
    history: deque[_TrackHistorySample] = field(default_factory=lambda: deque(maxlen=160))

    @property
    def status(self) -> str:
        if self.missed_frames > 0 and self.hits >= 2:
            return "coasting"
        return "confirmed" if self.hits >= 2 else "tentative"

    @property
    def confidence(self) -> float:
        hit_score = min(1.0, self.hits / 5.0)
        miss_penalty = max(0.0, 1.0 - 0.12 * self.missed_frames)
        snr_score = min(1.0, max(0.2, (self.last_snr_db - 8.0) / 25.0))
        return round(float(hit_score * miss_penalty * snr_score), 4)

    def to_public(self) -> RadarTrack:
        p = self.state_vec
        diag = tuple(float(v) for v in np.diag(self.cov))
        return RadarTrack(
            track_id=self.track_id,
            position_m=(float(p[0]), float(p[1]), float(p[2])),
            velocity_mps=(float(p[3]), float(p[4]), float(p[5])),
            age_frames=self.age_frames,
            hits=self.hits,
            missed_frames=self.missed_frames,
            confidence=self.confidence,
            state=self.status,
            covariance_diag=(diag[0], diag[1], diag[2], diag[3], diag[4], diag[5]),
            num_associated_detections=self.last_det_count,
            estimated_height_m=max(0.0, float(p[2])),
            mean_snr_db=float(self.last_snr_db),
        )


class RadarTracker:
    """Multi-frame constant-velocity 3D Kalman tracker operating solely on radar clusters."""

    def __init__(
        self,
        *,
        gate_distance_m: float = 1.25,
        max_missed_frames: int = 15,
        process_accel_std_mps2: float = 1.6,
        measurement_std_xy_m: float = 0.28,
        measurement_std_z_m: float = 0.30,
    ) -> None:
        self.gate_distance_m = float(gate_distance_m)
        self.max_missed_frames = int(max_missed_frames)
        self.process_accel_std = float(process_accel_std_mps2)
        self.r_mat = np.diag(
            [measurement_std_xy_m**2, measurement_std_xy_m**2, measurement_std_z_m**2]
        )
        self.h_mat = np.zeros((3, 6), dtype=np.float64)
        self.h_mat[0, 0] = 1.0
        self.h_mat[1, 1] = 1.0
        self.h_mat[2, 2] = 1.0
        self._tracks: list[_InternalTrack] = []
        self._next_track_id: int = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def internal_tracks(self) -> list[_InternalTrack]:
        return list(self._tracks)

    def update(
        self,
        sim_time_s: float,
        clusters: Sequence[DetectionCluster],
    ) -> tuple[list[RadarTrack], dict[int, int]]:
        """Predict tracks, associate radar clusters, update Kalman states, and return ``(tracks, cluster_to_track)``."""
        # 1. Predict existing tracks forward to sim_time_s
        for trk in self._tracks:
            dt = max(1e-3, min(1.0, sim_time_s - trk.last_time_s))
            f_mat = np.eye(6, dtype=np.float64)
            f_mat[0, 3] = dt
            f_mat[1, 4] = dt
            f_mat[2, 5] = dt
            q_scale = self.process_accel_std ** 2
            q_mat = np.zeros((6, 6), dtype=np.float64)
            for axis in range(3):
                q_mat[axis, axis] = 0.25 * (dt**4) * q_scale
                q_mat[axis, axis + 3] = 0.5 * (dt**3) * q_scale
                q_mat[axis + 3, axis] = 0.5 * (dt**3) * q_scale
                q_mat[axis + 3, axis + 3] = (dt**2) * q_scale

            trk.state_vec = f_mat @ trk.state_vec
            # Damp coasted velocity slightly so missed tracks do not drift out of bounds
            trk.state_vec[3:6] *= 0.92
            trk.cov = f_mat @ trk.cov @ f_mat.T + q_mat
            trk.age_frames += 1
            trk.last_time_s = sim_time_s

        # 2. Greedy nearest-neighbor gating between tracks and clusters
        cluster_to_track: dict[int, int] = {}
        matched_tracks: set[int] = set()
        matched_clusters: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for t_idx, trk in enumerate(self._tracks):
            tx, ty, tz = float(trk.state_vec[0]), float(trk.state_vec[1]), float(trk.state_vec[2])
            for c_idx, cl in enumerate(clusters):
                cx, cy, cz = cl.centroid_m
                dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2 + 0.5 * (tz - cz) ** 2)
                if dist <= self.gate_distance_m:
                    pairs.append((dist, t_idx, c_idx))
        pairs.sort(key=lambda item: item[0])

        for _, t_idx, c_idx in pairs:
            if t_idx in matched_tracks or c_idx in matched_clusters:
                continue
            matched_tracks.add(t_idx)
            matched_clusters.add(c_idx)
            trk = self._tracks[t_idx]
            cl = clusters[c_idx]
            z_meas = np.asarray(cl.centroid_m, dtype=np.float64)

            # Kalman update
            y_innov = z_meas - (self.h_mat @ trk.state_vec)
            s_mat = self.h_mat @ trk.cov @ self.h_mat.T + self.r_mat
            k_gain = trk.cov @ self.h_mat.T @ np.linalg.inv(s_mat)
            trk.state_vec = trk.state_vec + k_gain @ y_innov
            trk.cov = (np.eye(6) - k_gain @ self.h_mat) @ trk.cov

            trk.hits += 1
            trk.missed_frames = 0
            trk.last_snr_db = cl.mean_snr_db
            trk.last_det_count = len(cl.detection_indices)
            spd = float(np.linalg.norm(trk.state_vec[3:6]))
            trk.history.append(
                _TrackHistorySample(
                    sim_time_s=sim_time_s,
                    z_m=float(trk.state_vec[2]),
                    vz_mps=float(trk.state_vec[5]),
                    speed_3d_mps=spd,
                    radial_vel_peak_mps=cl.max_abs_radial_velocity_mps,
                    snr_db=cl.mean_snr_db,
                    associated_count=len(cl.detection_indices),
                )
            )
            cluster_to_track[cl.cluster_id] = trk.track_id

        # 3. Update unmatched tracks (coast or prune)
        survivors: list[_InternalTrack] = []
        for t_idx, trk in enumerate(self._tracks):
            if t_idx not in matched_tracks:
                trk.missed_frames += 1
                trk.last_det_count = 0
                spd = float(np.linalg.norm(trk.state_vec[3:6]))
                trk.history.append(
                    _TrackHistorySample(
                        sim_time_s=sim_time_s,
                        z_m=float(trk.state_vec[2]),
                        vz_mps=float(trk.state_vec[5]),
                        speed_3d_mps=spd,
                        radial_vel_peak_mps=0.0,
                        snr_db=trk.last_snr_db,
                        associated_count=0,
                    )
                )
            if trk.missed_frames <= self.max_missed_frames:
                survivors.append(trk)
        self._tracks = survivors

        # 4. Initialize new tracks for unmatched clusters
        for c_idx, cl in enumerate(clusters):
            if c_idx in matched_clusters:
                continue
            tid = self._next_track_id
            self._next_track_id += 1
            cx, cy, cz = cl.centroid_m
            state = np.array([cx, cy, cz, 0.0, 0.0, 0.0], dtype=np.float64)
            cov = np.diag([0.35, 0.35, 0.35, 1.5, 1.5, 1.5])
            new_trk = _InternalTrack(
                track_id=tid,
                state_vec=state,
                cov=cov,
                age_frames=1,
                hits=1,
                missed_frames=0,
                last_time_s=sim_time_s,
                last_snr_db=cl.mean_snr_db,
                last_det_count=len(cl.detection_indices),
            )
            new_trk.history.append(
                _TrackHistorySample(
                    sim_time_s=sim_time_s,
                    z_m=cz,
                    vz_mps=0.0,
                    speed_3d_mps=0.0,
                    radial_vel_peak_mps=cl.max_abs_radial_velocity_mps,
                    snr_db=cl.mean_snr_db,
                    associated_count=len(cl.detection_indices),
                )
            )
            self._tracks.append(new_trk)
            cluster_to_track[cl.cluster_id] = tid

        public_tracks = [t.to_public() for t in self._tracks]
        public_tracks.sort(key=lambda t: (t.state != "confirmed", -t.confidence, t.track_id))
        return public_tracks, cluster_to_track


def build_radar_occupancy_voxels(
    detections: Sequence[RadarDetection],
    *,
    voxel_size_m: float = 0.25,
) -> RadarOccupancyVoxels:
    """Build sparse radar-derived 3D occupancy voxels strictly from ``detections``.

    Never interpolates or fills unobserved body regions from ground truth.
    """
    if voxel_size_m <= 0.0:
        raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}")
    buckets: dict[tuple[int, int, int], list[RadarDetection]] = {}
    for det in detections:
        ix = int(math.floor(det.x_m / voxel_size_m))
        iy = int(math.floor(det.y_m / voxel_size_m))
        iz = int(math.floor(det.z_m / voxel_size_m))
        buckets.setdefault((ix, iy, iz), []).append(det)

    voxels: list[dict[str, Any]] = []
    for (ix, iy, iz), items in sorted(buckets.items()):
        max_conf = max(d.confidence for d in items)
        prob = min(0.99, 1.0 - math.prod(1.0 - min(0.95, max(0.05, d.confidence)) for d in items))
        voxels.append(
            {
                "ijk": [ix, iy, iz],
                "center_m": [
                    round((ix + 0.5) * voxel_size_m, 4),
                    round((iy + 0.5) * voxel_size_m, 4),
                    round((iz + 0.5) * voxel_size_m, 4),
                ],
                "probability": round(prob, 4),
                "confidence": round(max_conf, 4),
                "detection_count": len(items),
                "provenance": "radar_derived",
            }
        )

    return RadarOccupancyVoxels(
        voxel_size_m=voxel_size_m,
        voxels=tuple(voxels),
        provenance="radar_derived",
    )
