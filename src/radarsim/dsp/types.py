"""DSP output data contracts following ``INTERFACES.md`` §7.

Ground-truth positions are strictly excluded from ``RadarDetection``,
``RadarTrack``, and ``RadarOccupancyVoxels`` — all fields are derived solely
from the radar ADC cube.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RadarDetection:
    """Single radar-derived 3D point detection (no ground-truth leakage)."""

    x_m: float
    y_m: float
    z_m: float
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    radial_velocity_mps: float
    snr_db: float
    confidence: float
    track_id: int | None = None
    cluster_id: int | None = None
    range_bin: int = 0
    doppler_bin: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "range_m": self.range_m,
            "azimuth_rad": self.azimuth_rad,
            "elevation_rad": self.elevation_rad,
            "radial_velocity_mps": self.radial_velocity_mps,
            "snr_db": self.snr_db,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "cluster_id": self.cluster_id,
            "range_bin": self.range_bin,
            "doppler_bin": self.doppler_bin,
        }


@dataclass(frozen=True)
class RadarTrack:
    """Radar-derived multi-frame target track."""

    track_id: int
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float]
    age_frames: int
    hits: int
    missed_frames: int
    confidence: float
    state: str  # "tentative", "confirmed", "coasting"
    covariance_diag: tuple[float, float, float, float, float, float]
    num_associated_detections: int
    estimated_height_m: float
    mean_snr_db: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "position_m": list(self.position_m),
            "velocity_mps": list(self.velocity_mps),
            "age_frames": self.age_frames,
            "hits": self.hits,
            "missed_frames": self.missed_frames,
            "confidence": self.confidence,
            "state": self.state,
            "covariance_diag": list(self.covariance_diag),
            "num_associated_detections": self.num_associated_detections,
            "estimated_height_m": self.estimated_height_m,
            "mean_snr_db": self.mean_snr_db,
        }


@dataclass(frozen=True)
class RangeDopplerMap:
    """Range-Doppler log-magnitude power matrix [dB] and physical axes."""

    range_m: np.ndarray        # shape (N_r,)
    velocity_mps: np.ndarray   # shape (N_d,)
    values_db: np.ndarray      # shape (N_d, N_r)
    branch: str = "clutter_suppressed"

    def to_dict(self, *, downsample_r: int = 1, downsample_d: int = 1) -> dict[str, Any]:
        r_ax = self.range_m[::downsample_r]
        v_ax = self.velocity_mps[::downsample_d]
        mat = self.values_db[::downsample_d, ::downsample_r]
        return {
            "branch": self.branch,
            "range_m": [round(float(v), 4) for v in r_ax],
            "velocity_mps": [round(float(v), 4) for v in v_ax],
            "values_db": [[round(float(x), 2) for x in row] for row in mat],
        }


@dataclass(frozen=True)
class RangeAngleMap:
    """Range-angle / azimuth-elevation log-magnitude map [dB]."""

    azimuth_rad: np.ndarray    # shape (N_az,)
    elevation_rad: np.ndarray  # shape (N_el,)
    values_db: np.ndarray      # shape (N_el, N_az) azimuth-elevation map at peak range or integrated
    range_m: np.ndarray | None = None
    range_azimuth_db: np.ndarray | None = None  # shape (N_r, N_az)

    def to_dict(
        self, *, downsample_az: int = 2, downsample_el: int = 2, downsample_r: int = 2
    ) -> dict[str, Any]:
        az = self.azimuth_rad[::downsample_az]
        el = self.elevation_rad[::downsample_el]
        mat = self.values_db[::downsample_el, ::downsample_az]
        out: dict[str, Any] = {
            "azimuth_rad": [round(float(v), 4) for v in az],
            "elevation_rad": [round(float(v), 4) for v in el],
            "values_db": [[round(float(x), 2) for x in row] for row in mat],
        }
        if self.range_m is not None and self.range_azimuth_db is not None:
            r_ax = self.range_m[::downsample_r]
            ra_mat = self.range_azimuth_db[::downsample_r, ::downsample_az]
            out["range_m"] = [round(float(v), 4) for v in r_ax]
            out["range_azimuth_db"] = [[round(float(x), 2) for x in row] for row in ra_mat]
        return out


@dataclass(frozen=True)
class RadarOccupancyVoxels:
    """Sparse radar-only occupancy voxels built strictly from radar detections."""

    voxel_size_m: float
    voxels: tuple[dict[str, Any], ...]
    provenance: str = "radar_derived"
    note: str = (
        "Sparse radar-detection occupancy estimate only; missing or occluded body "
        "regions are not filled from ground truth."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel_size_m": self.voxel_size_m,
            "voxels": list(self.voxels),
            "provenance": self.provenance,
            "note": self.note,
        }


@dataclass(frozen=True)
class FallAlert:
    """Transparent rule-based fall baseline decision."""

    active: bool
    state: str  # "normal", "monitoring_drop", "fall_alert"
    confidence: float
    event_time_s: float | None
    event_label: str
    evidence: dict[str, Any]
    rule_version: str = "baseline_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "state": self.state,
            "confidence": self.confidence,
            "event_time_s": self.event_time_s,
            "event_label": self.event_label,
            "evidence": dict(self.evidence),
            "rule_version": self.rule_version,
        }


@dataclass(frozen=True)
class DSPFrame:
    """Complete radar-only DSP output for one frame (``INTERFACES.md`` §7)."""

    frame_id: int
    sim_time_s: float
    config_hash: str
    processing_mode: str
    detections: list[RadarDetection]
    tracks: list[RadarTrack]
    range_doppler: RangeDopplerMap
    range_angle: RangeAngleMap
    occupancy: RadarOccupancyVoxels
    features: dict[str, Any]
    fall_alert: FallAlert
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_websocket_radar_dict(self) -> dict[str, Any]:
        """Payload for the ``radar`` section of ``INTERFACES.md`` §8."""
        det_dicts = [d.to_dict() for d in self.detections]
        return {
            "detections": det_dicts,
            "tracks": [t.to_dict() for t in self.tracks],
            "point_cloud": det_dicts,
            "occupancy": self.occupancy.to_dict(),
            "provenance": "radar_derived",
        }

    def to_websocket_analytics_dict(self) -> dict[str, Any]:
        """Payload for the ``analytics`` section of ``INTERFACES.md`` §8."""
        return {
            "fall_alert": self.fall_alert.to_dict(),
            "features": dict(self.features),
            "range_doppler": self.range_doppler.to_dict(downsample_r=2, downsample_d=2),
            "range_angle": self.range_angle.to_dict(downsample_az=2, downsample_el=2),
        }
