"""Separate evaluation metrics comparing radar-derived ``DSPFrame`` outputs against ground truth.

Strictly separated from ``radarsim.dsp`` so ground truth is never consumed by
the radar processing chain. All results are explicitly labeled as simulated
synthetic evaluations, never physical-hardware accuracy claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from radarsim.dsp.types import DSPFrame
from radarsim.motion.types import MotionFrame


@dataclass(frozen=True)
class TrackerEvalMetrics:
    """Synthetic tracker evaluation summary for one simulated sequence."""

    num_frames: int
    frames_with_detections: int
    frames_with_tracks: int
    detection_rate: float
    track_coverage_rate: float
    mean_detections_per_frame: float
    xy_rmse_m: float | None
    xyz_rmse_m: float | None
    track_id_switches: int
    unique_track_ids: tuple[int, ...]
    ground_truth_leakage_detected: bool
    evaluation_type: str = "simulated_synthetic_benchmark_not_real_hardware"

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_frames": self.num_frames,
            "frames_with_detections": self.frames_with_detections,
            "frames_with_tracks": self.frames_with_tracks,
            "detection_rate": round(self.detection_rate, 4),
            "track_coverage_rate": round(self.track_coverage_rate, 4),
            "mean_detections_per_frame": round(self.mean_detections_per_frame, 3),
            "xy_rmse_m": round(self.xy_rmse_m, 4) if self.xy_rmse_m is not None else None,
            "xyz_rmse_m": round(self.xyz_rmse_m, 4) if self.xyz_rmse_m is not None else None,
            "track_id_switches": self.track_id_switches,
            "unique_track_ids": list(self.unique_track_ids),
            "ground_truth_leakage_detected": self.ground_truth_leakage_detected,
            "evaluation_type": self.evaluation_type,
        }


@dataclass(frozen=True)
class FallBaselineEvalMetrics:
    """Synthetic fall-baseline evaluation summary for one simulated sequence."""

    ground_truth_has_fall: bool
    ground_truth_fall_start_s: float | None
    ground_truth_impact_s: float | None
    alert_triggered: bool
    first_alert_time_s: float | None
    alert_latency_from_impact_s: float | None
    outcome: str  # "TP", "TN", "FP", "FN"
    hard_negative_frames: int
    hard_negative_false_alert_frames: int
    evaluation_type: str = "simulated_synthetic_benchmark_not_real_hardware"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ground_truth_has_fall": self.ground_truth_has_fall,
            "ground_truth_fall_start_s": self.ground_truth_fall_start_s,
            "ground_truth_impact_s": self.ground_truth_impact_s,
            "alert_triggered": self.alert_triggered,
            "first_alert_time_s": self.first_alert_time_s,
            "alert_latency_from_impact_s": (
                round(self.alert_latency_from_impact_s, 4)
                if self.alert_latency_from_impact_s is not None
                else None
            ),
            "outcome": self.outcome,
            "hard_negative_frames": self.hard_negative_frames,
            "hard_negative_false_alert_frames": self.hard_negative_false_alert_frames,
            "evaluation_type": self.evaluation_type,
        }


def check_no_ground_truth_leakage(
    dsp_frames: Sequence[DSPFrame],
    motion_frames: Sequence[MotionFrame],
    *,
    exact_match_eps_m: float = 1e-5,
) -> bool:
    """Return True if NO radar detection suspiciously copies exact ground-truth joint XYZ coordinates."""
    exact_hits = 0
    total_dets = 0
    for df, mf in zip(dsp_frames, motion_frames):
        gt_pts = list(mf.joints_m.values()) + [mf.root_position_m]
        for det in df.detections:
            total_dets += 1
            for gx, gy, gz in gt_pts:
                if (
                    abs(det.x_m - gx) < exact_match_eps_m
                    and abs(det.y_m - gy) < exact_match_eps_m
                    and abs(det.z_m - gz) < exact_match_eps_m
                ):
                    exact_hits += 1
                    break
    if total_dets == 0:
        return True
    return exact_hits == 0


def evaluate_tracker(
    dsp_frames: Sequence[DSPFrame],
    motion_frames: Sequence[MotionFrame],
) -> TrackerEvalMetrics:
    """Compute radar-tracker RMSE, continuity, and ground-truth non-leakage check."""
    if len(dsp_frames) != len(motion_frames):
        raise ValueError(
            f"Frame count mismatch: {len(dsp_frames)} DSPFrames vs {len(motion_frames)} MotionFrames"
        )
    n = len(dsp_frames)
    if n == 0:
        return TrackerEvalMetrics(
            num_frames=0,
            frames_with_detections=0,
            frames_with_tracks=0,
            detection_rate=0.0,
            track_coverage_rate=0.0,
            mean_detections_per_frame=0.0,
            xy_rmse_m=None,
            xyz_rmse_m=None,
            track_id_switches=0,
            unique_track_ids=(),
            ground_truth_leakage_detected=False,
        )

    det_frames = 0
    trk_frames = 0
    total_dets = 0
    xy_sq_errs: list[float] = []
    xyz_sq_errs: list[float] = []
    seen_ids: set[int] = set()
    prev_primary_id: int | None = None
    id_switches = 0

    for df, mf in zip(dsp_frames, motion_frames):
        total_dets += len(df.detections)
        if df.detections:
            det_frames += 1
        if df.tracks:
            trk_frames += 1
            primary = df.tracks[0]
            seen_ids.add(primary.track_id)
            if prev_primary_id is not None and primary.track_id != prev_primary_id:
                id_switches += 1
            prev_primary_id = primary.track_id

            # Compare primary radar track against ground-truth torso/pelvis center
            gt_p = mf.joints_m["spine"]
            dx = primary.position_m[0] - gt_p[0]
            dy = primary.position_m[1] - gt_p[1]
            dz = primary.position_m[2] - gt_p[2]
            xy_sq_errs.append(dx * dx + dy * dy)
            xyz_sq_errs.append(dx * dx + dy * dy + dz * dz)

    no_leak = check_no_ground_truth_leakage(dsp_frames, motion_frames)
    return TrackerEvalMetrics(
        num_frames=n,
        frames_with_detections=det_frames,
        frames_with_tracks=trk_frames,
        detection_rate=det_frames / n,
        track_coverage_rate=trk_frames / n,
        mean_detections_per_frame=total_dets / n,
        xy_rmse_m=math.sqrt(float(np.mean(xy_sq_errs))) if xy_sq_errs else None,
        xyz_rmse_m=math.sqrt(float(np.mean(xyz_sq_errs))) if xyz_sq_errs else None,
        track_id_switches=id_switches,
        unique_track_ids=tuple(sorted(seen_ids)),
        ground_truth_leakage_detected=(not no_leak),
    )


def evaluate_fall_baseline(
    dsp_frames: Sequence[DSPFrame],
    motion_frames: Sequence[MotionFrame],
) -> FallBaselineEvalMetrics:
    """Evaluate baseline fall alert outputs against ground-truth fall events and hard negatives."""
    if len(dsp_frames) != len(motion_frames):
        raise ValueError(
            f"Frame count mismatch: {len(dsp_frames)} DSPFrames vs {len(motion_frames)} MotionFrames"
        )

    gt_fall_frames = [mf for mf in motion_frames if mf.fall_event_id is not None]
    gt_has_fall = len(gt_fall_frames) > 0
    gt_start_s = gt_fall_frames[0].sim_time_s if gt_has_fall else None
    impact_frames = [mf for mf in gt_fall_frames if mf.action_phase == "impact_contact"]
    gt_impact_s = impact_frames[0].sim_time_s if impact_frames else gt_start_s

    alert_frames = [df for df in dsp_frames if df.fall_alert.active]
    alert_triggered = len(alert_frames) > 0
    first_alert_s = alert_frames[0].sim_time_s if alert_triggered else None

    hn_frames = 0
    hn_false_alerts = 0
    for df, mf in zip(dsp_frames, motion_frames):
        if mf.is_hard_negative:
            hn_frames += 1
            if df.fall_alert.active:
                hn_false_alerts += 1

    if gt_has_fall and alert_triggered:
        outcome = "TP"
    elif gt_has_fall and not alert_triggered:
        outcome = "FN"
    elif (not gt_has_fall) and alert_triggered:
        outcome = "FP"
    else:
        outcome = "TN"

    latency = (
        (first_alert_s - gt_impact_s)
        if (gt_impact_s is not None and first_alert_s is not None)
        else None
    )

    return FallBaselineEvalMetrics(
        ground_truth_has_fall=gt_has_fall,
        ground_truth_fall_start_s=gt_start_s,
        ground_truth_impact_s=gt_impact_s,
        alert_triggered=alert_triggered,
        first_alert_time_s=first_alert_s,
        alert_latency_from_impact_s=latency,
        outcome=outcome,
        hard_negative_frames=hn_frames,
        hard_negative_false_alert_frames=hn_false_alerts,
    )
