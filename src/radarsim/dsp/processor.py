"""End-to-end radar-only DSP processor converting ``ADCCube`` into ``DSPFrame``.

Strict ground-truth separation: ``DSPProcessor`` accepts ONLY ``ADCCube``,
``RadarProfile``, and ``RadarCoordinateFrame`` (sensor mount pose). It never
reads ground-truth human poses, joints, or scene furniture coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from radarsim.config import RadarProfile, compute_derived_metrics
from radarsim.dsp.angle import VirtualArrayAngleEstimator
from radarsim.dsp.cfar import OSCFARConfig, run_os_cfar_2d
from radarsim.dsp.fall import FallRuleConfig, RuleBasedFallDetector, StationaryPhaseBranch
from radarsim.dsp.spectral import compute_range_doppler_cubes
from radarsim.dsp.tracking import (
    RadarTracker,
    build_radar_occupancy_voxels,
    cluster_detections,
)
from radarsim.dsp.types import (
    DSPFrame,
    RadarDetection,
    RangeAngleMap,
    RangeDopplerMap,
)
from radarsim.radar.types import ADCCube
from radarsim.scene import RadarCoordinateFrame


@dataclass(frozen=True)
class DSPConfig:
    """Configuration for the radar-only DSP chain."""

    clutter_suppression_enabled: bool = True
    apply_tdm_compensation: bool = True
    exclude_zero_doppler_when_clutter_suppressed: bool = True
    cfar: OSCFARConfig = OSCFARConfig()
    cluster_eps_spatial_m: float = 0.70
    cluster_eps_velocity_mps: float = 1.50
    occupancy_voxel_size_m: float = 0.25
    fall_rule: FallRuleConfig = FallRuleConfig()
    enable_breathing_phase_branch: bool = True


class DSPProcessor:
    """Stateful radar-only DSP pipeline: Range/Doppler -> OS-CFAR -> Angle -> Cluster -> Track -> Fall."""

    def __init__(
        self,
        radar_profile: RadarProfile,
        radar_frame: RadarCoordinateFrame,
        *,
        config: DSPConfig | None = None,
    ) -> None:
        self.radar_profile = radar_profile
        self.radar_frame = radar_frame
        self.config = DSPConfig() if config is None else config
        self.derived = compute_derived_metrics(radar_profile)
        self.angle_estimator = VirtualArrayAngleEstimator(radar_profile)
        self.tracker = RadarTracker()
        self.fall_detector = RuleBasedFallDetector(self.config.fall_rule)
        self.breathing_branch = StationaryPhaseBranch()

    def reset(self) -> None:
        """Reset multi-frame tracker, fall rule state, and stationary phase history."""
        self.tracker.reset()
        self.fall_detector.reset()
        self.breathing_branch.reset()

    def process(self, adc_cube: ADCCube) -> DSPFrame:
        """Process one ``ADCCube`` into a radar-only ``DSPFrame``."""
        cfg = self.config
        spectral = compute_range_doppler_cubes(
            adc_cube,
            self.radar_profile,
            apply_tdm_compensation=cfg.apply_tdm_compensation,
        )

        # Choose detection branch (moving clutter-suppressed vs untouched stationary)
        if cfg.clutter_suppression_enabled:
            rd_cube = spectral.rd_cube_moving
            rd_power = spectral.rd_power_moving
            branch_name = "clutter_suppressed"
            exclude_dc = cfg.exclude_zero_doppler_when_clutter_suppressed
        else:
            rd_cube = spectral.rd_cube_stationary
            rd_power = spectral.rd_power_stationary
            branch_name = "untouched_stationary"
            exclude_dc = False

        # 1. Configurable 2D OS-CFAR detection
        peaks = run_os_cfar_2d(rd_power, cfg.cfar, exclude_zero_doppler=exclude_dc)

        # 2. 2D virtual-array angle estimation & radar-only 3D world projection
        r_ax = spectral.range_axis_m
        v_ax = spectral.velocity_axis_mps
        r_step = float(r_ax[1] - r_ax[0]) if r_ax.size > 1 else 0.0
        v_step = float(v_ax[1] - v_ax[0]) if v_ax.size > 1 else 0.0

        raw_detections: list[RadarDetection] = []
        for pk in peaks:
            snap = rd_cube[pk.doppler_bin, :, :, pk.range_bin]  # shape (tx, rx)
            az_rad, el_rad, peak_ratio = self.angle_estimator.estimate_angle(snap)

            # Sub-bin interpolated range and radial velocity
            r_est = max(0.0, float(r_ax[0]) + pk.range_bin_refined * r_step)
            v_est = float(v_ax[0]) + pk.doppler_bin_refined * v_step

            # Transform (range, az, el) -> world (x, y, z) using sensor mount frame ONLY
            wx, wy, wz = self.radar_frame.spherical_to_world_point(r_est, az_rad, el_rad)

            # Confidence from SNR and angular beamformer peak sharpness
            snr_conf = min(1.0, max(0.05, (pk.snr_db - cfg.cfar.threshold_scale_db + 4.0) / 24.0))
            ang_conf = min(1.0, max(0.2, peak_ratio / 6.0))
            conf = round(float(snr_conf * (0.5 + 0.5 * ang_conf)), 4)

            raw_detections.append(
                RadarDetection(
                    x_m=float(wx),
                    y_m=float(wy),
                    z_m=float(wz),
                    range_m=float(r_est),
                    azimuth_rad=float(az_rad),
                    elevation_rad=float(el_rad),
                    radial_velocity_mps=float(v_est),
                    snr_db=round(float(pk.snr_db), 3),
                    confidence=conf,
                    range_bin=pk.range_bin,
                    doppler_bin=pk.doppler_bin,
                )
            )

        # 3. Clustering & multi-frame Kalman tracking
        clusters, cluster_labels = cluster_detections(
            raw_detections,
            eps_spatial_m=cfg.cluster_eps_spatial_m,
            eps_velocity_mps=cfg.cluster_eps_velocity_mps,
        )
        tracks, cluster_to_track = self.tracker.update(adc_cube.sim_time_s, clusters)

        detections: list[RadarDetection] = []
        for det, cid in zip(raw_detections, cluster_labels):
            tid = cluster_to_track.get(cid) if cid is not None else None
            detections.append(replace(det, cluster_id=cid, track_id=tid))

        # 4. Sparse radar-only occupancy voxels (strictly from detections)
        occupancy = build_radar_occupancy_voxels(
            detections, voxel_size_m=cfg.occupancy_voxel_size_m
        )

        # 5. Transparent rule-based fall baseline + optional stationary breathing branch
        fall_alert, features = self.fall_detector.evaluate(
            adc_cube.sim_time_s, self.tracker.internal_tracks()
        )
        if cfg.enable_breathing_phase_branch:
            features["breathing_phase_branch"] = self.breathing_branch.update(
                adc_cube.sim_time_s,
                spectral.range_cube_stationary,
                r_ax,
                self.derived.wavelength_m,
                min_range_bin=max(4, cfg.cfar.min_range_bin),
            )

        # 6. Range-Doppler and Azimuth-Elevation diagnostic maps [dB]
        rd_db = 10.0 * np.log10(np.maximum(rd_power, 1e-18))
        rd_map = RangeDopplerMap(
            range_m=r_ax,
            velocity_mps=v_ax,
            values_db=rd_db,
            branch=branch_name,
        )

        # Build azimuth-elevation map at strongest detected peak (or strongest valid RD cell)
        if peaks:
            best_d, best_r = peaks[0].doppler_bin, peaks[0].range_bin
        else:
            valid_slice = rd_power[:, cfg.cfar.min_range_bin :]
            if valid_slice.size > 0:
                flat = int(np.argmax(valid_slice))
                best_d, best_r_rel = divmod(flat, valid_slice.shape[1])
                best_r = best_r_rel + cfg.cfar.min_range_bin
            else:
                best_d, best_r = 0, 0
        az_el_power = self.angle_estimator.spectrum_2d(rd_cube[best_d, :, :, best_r])
        az_el_db = 10.0 * np.log10(np.maximum(az_el_power, 1e-18))
        ra_az_power = self.angle_estimator.compute_range_azimuth_power(
            rd_cube, exclude_zero_doppler=exclude_dc
        )
        ra_az_db = 10.0 * np.log10(np.maximum(ra_az_power, 1e-18))
        ra_map = RangeAngleMap(
            azimuth_rad=self.angle_estimator.az_grid_rad,
            elevation_rad=self.angle_estimator.el_grid_rad,
            values_db=az_el_db,
            range_m=r_ax,
            range_azimuth_db=ra_az_db,
        )

        diagnostics: dict[str, Any] = {
            "detection_branch": branch_name,
            "stationary_branch_preserved": True,
            "tdm_compensation_applied": cfg.apply_tdm_compensation,
            "same_tx_interval_s": spectral.same_tx_interval_s,
            "tx_time_offsets_s": [float(v) for v in spectral.tx_time_offsets_s],
            "fft_normalization": {
                "convention": "amplitude_preserving_window_sum_normalization",
                "range_fft_size": spectral.range_fft_size,
                "doppler_fft_size": spectral.doppler_fft_size,
                "range_window": spectral.range_window_stats.to_dict(),
                "doppler_window": spectral.doppler_window_stats.to_dict(),
                "zero_padding_note": (
                    "Zero-padding interpolates range/Doppler/angle bins only; it does NOT "
                    "improve physical range, Doppler, or Rayleigh angular resolution."
                ),
            },
            "cfar_config": cfg.cfar.to_dict(),
            "aperture_limitations": self.angle_estimator.limitations.to_dict(),
            "detection_count": len(detections),
            "cluster_count": len(clusters),
            "track_count": len(tracks),
            "voxel_count": len(occupancy.voxels),
        }

        return DSPFrame(
            frame_id=adc_cube.frame_id,
            sim_time_s=adc_cube.sim_time_s,
            config_hash=adc_cube.config_hash,
            processing_mode=adc_cube.processing_mode,
            detections=detections,
            tracks=tracks,
            range_doppler=rd_map,
            range_angle=ra_map,
            occupancy=occupancy,
            features=features,
            fall_alert=fall_alert,
            diagnostics=diagnostics,
        )
