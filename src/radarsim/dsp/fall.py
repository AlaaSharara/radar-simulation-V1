"""Transparent rule-based fall baseline and optional untouched-branch breathing phase extractor.

Operates strictly on radar-derived track histories and untouched stationary
complex slow-time data; never consumes ground-truth labels or pose coordinates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from radarsim.dsp.tracking import _InternalTrack
from radarsim.dsp.types import FallAlert


@dataclass(frozen=True)
class FallRuleConfig:
    """Explicit, inspectable thresholds for the baseline rule-based fall detector."""

    rule_version: str = "baseline_v1"
    lookback_window_s: float = 3.5
    prior_standing_z_min_m: float = 0.72
    min_height_drop_m: float = 0.45
    rapid_drop_vz_mps: float = -0.55
    max_fall_z_m: float = 0.48
    min_stillness_s: float = 0.40
    stillness_speed_max_mps: float = 0.32
    recovery_z_reset_m: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_version": self.rule_version,
            "lookback_window_s": self.lookback_window_s,
            "prior_standing_z_min_m": self.prior_standing_z_min_m,
            "min_height_drop_m": self.min_height_drop_m,
            "rapid_drop_vz_mps": self.rapid_drop_vz_mps,
            "max_fall_z_m": self.max_fall_z_m,
            "min_stillness_s": self.min_stillness_s,
            "stillness_speed_max_mps": self.stillness_speed_max_mps,
            "recovery_z_reset_m": self.recovery_z_reset_m,
        }


class RuleBasedFallDetector:
    """Temporal height / vertical-velocity / stillness rule baseline for fall indication.

    Normal sitting, kneeling, bending, and intentional lying down on elevated
    furniture (bed/sofa) are treated as hard negatives via combined rapid-drop,
    low floor-level height, and post-drop stillness requirements.
    """

    def __init__(self, config: FallRuleConfig | None = None) -> None:
        self.config = FallRuleConfig() if config is None else config
        self._active_alert_time_s: float | None = None
        self._active_track_id: int | None = None
        self._last_drop_seen_s: float | None = None
        self._last_drop_peak_z_m: float = 0.0
        self._last_drop_min_vz_mps: float = 0.0
        self._low_still_start_s: float | None = None
        self._last_low_pos_time_s: float | None = None

    def reset(self) -> None:
        self._active_alert_time_s = None
        self._active_track_id = None
        self._last_drop_seen_s = None
        self._last_drop_peak_z_m = 0.0
        self._last_drop_min_vz_mps = 0.0
        self._low_still_start_s = None
        self._last_low_pos_time_s = None

    def evaluate(
        self,
        sim_time_s: float,
        tracks: Sequence[_InternalTrack],
    ) -> tuple[FallAlert, dict[str, Any]]:
        """Evaluate all active radar tracks and return ``(FallAlert, feature_summary)``."""
        cfg = self.config
        primary = self._select_primary_track(tracks)

        if primary is None or not primary.history:
            # Handle floor-immobile fadeout: if rapid drop followed by low pose transitioned to
            # zero-Doppler floor clutter notch, promote to alert once stillness duration elapses.
            fadeout_still_s = (
                (sim_time_s - self._low_still_start_s)
                if self._low_still_start_s is not None
                else 0.0
            )
            recent_drop_valid = (
                self._last_drop_seen_s is not None
                and (sim_time_s - self._last_drop_seen_s) <= cfg.lookback_window_s
            )
            recent_low_valid = (
                self._last_low_pos_time_s is not None
                and (sim_time_s - self._last_low_pos_time_s) <= 2.5
            )
            if (
                self._active_alert_time_s is None
                and recent_drop_valid
                and recent_low_valid
                and self._low_still_start_s is not None
                and fadeout_still_s >= cfg.min_stillness_s
            ):
                self._active_alert_time_s = self._low_still_start_s
            # Preserve active alert briefly while person remains still on floor even if coasting pruned
            if self._active_alert_time_s is not None and (sim_time_s - self._active_alert_time_s) <= 4.0:
                return (
                    FallAlert(
                        active=True,
                        state="fall_alert",
                        confidence=0.75,
                        event_time_s=self._active_alert_time_s,
                        event_label="possible_fall_baseline",
                        evidence={
                            "track_id": self._active_track_id,
                            "reason": "post_fall_stillness_track_coasted_out",
                            "thresholds": cfg.to_dict(),
                        },
                        rule_version=cfg.rule_version,
                    ),
                    {
                        "estimated_height_m": None,
                        "vertical_velocity_mps": 0.0,
                        "track_stability": 0.0,
                        "primary_track_id": None,
                    },
                )
            return (
                FallAlert(
                    active=False,
                    state="normal",
                    confidence=0.0,
                    event_time_s=None,
                    event_label="none",
                    evidence={"reason": "no_active_radar_track", "thresholds": cfg.to_dict()},
                    rule_version=cfg.rule_version,
                ),
                {
                    "estimated_height_m": None,
                    "vertical_velocity_mps": 0.0,
                    "track_stability": 0.0,
                    "primary_track_id": None,
                },
            )

        hist = list(primary.history)
        cur = hist[-1]
        window = [h for h in hist if (sim_time_s - h.sim_time_s) <= cfg.lookback_window_s]
        peak_sample = max(window, key=lambda h: h.z_m)
        peak_z = peak_sample.z_m
        min_vz = min(h.vz_mps for h in window)
        dt_drop = cur.sim_time_s - peak_sample.sim_time_s
        if dt_drop >= 0.15:
            fd_vz = (cur.z_m - peak_z) / dt_drop
            effective_min_vz = min(min_vz, fd_vz)
        else:
            effective_min_vz = min_vz
        height_drop = peak_z - cur.z_m
        track_stability = min(1.0, primary.hits / max(1.0, float(primary.age_frames)))

        # Check rapid downward drop onset from an elevated/standing height
        rapid_drop_now = (
            peak_z >= cfg.prior_standing_z_min_m
            and height_drop >= cfg.min_height_drop_m
            and effective_min_vz <= cfg.rapid_drop_vz_mps
        )
        if rapid_drop_now:
            self._last_drop_seen_s = sim_time_s
            self._last_drop_peak_z_m = max(self._last_drop_peak_z_m, peak_z)
            self._last_drop_min_vz_mps = min(self._last_drop_min_vz_mps, effective_min_vz)

        recent_drop = (
            self._last_drop_seen_s is not None
            and (sim_time_s - self._last_drop_seen_s) <= cfg.lookback_window_s
        )

        # Check if track has recovered back to standing height
        if cur.z_m >= cfg.recovery_z_reset_m:
            self._active_alert_time_s = None
            self._active_track_id = None
            self._low_still_start_s = None
            self._last_low_pos_time_s = None
            if not rapid_drop_now:
                self._last_drop_seen_s = None
                self._last_drop_peak_z_m = 0.0
                self._last_drop_min_vz_mps = 0.0

        # Track sustained low-height stillness after a rapid drop
        if recent_drop and cur.z_m <= cfg.max_fall_z_m:
            self._last_low_pos_time_s = sim_time_s

        is_low_and_still = (
            cur.z_m <= cfg.max_fall_z_m
            and cur.speed_3d_mps <= cfg.stillness_speed_max_mps
            and cur.radial_vel_peak_mps <= 0.45
        )
        if recent_drop and is_low_and_still:
            if self._low_still_start_s is None:
                self._low_still_start_s = sim_time_s
        elif not is_low_and_still and self._active_alert_time_s is None:
            self._low_still_start_s = None

        stillness_s = (
            (sim_time_s - self._low_still_start_s)
            if self._low_still_start_s is not None
            else 0.0
        )

        # Decision logic
        if self._active_alert_time_s is not None or (
            recent_drop and is_low_and_still and stillness_s >= cfg.min_stillness_s
        ):
            if self._active_alert_time_s is None:
                self._active_alert_time_s = self._low_still_start_s or sim_time_s
                self._active_track_id = primary.track_id
            drop_score = min(1.0, height_drop / max(cfg.min_height_drop_m, 1e-3))
            vz_score = min(1.0, abs(min(effective_min_vz, self._last_drop_min_vz_mps)) / max(abs(cfg.rapid_drop_vz_mps), 1e-3))
            still_score = min(1.0, stillness_s / max(cfg.min_stillness_s, 1e-3))
            conf = round(float(min(0.95, 0.35 * drop_score + 0.35 * vz_score + 0.30 * still_score)), 4)
            alert = FallAlert(
                active=True,
                state="fall_alert",
                confidence=conf,
                event_time_s=self._active_alert_time_s,
                event_label="possible_fall_baseline",
                evidence={
                    "track_id": primary.track_id,
                    "current_z_m": round(cur.z_m, 4),
                    "recent_peak_z_m": round(max(peak_z, self._last_drop_peak_z_m), 4),
                    "height_drop_m": round(height_drop, 4),
                    "min_vz_mps": round(min(effective_min_vz, self._last_drop_min_vz_mps), 4),
                    "post_drop_stillness_s": round(stillness_s, 4),
                    "thresholds": cfg.to_dict(),
                },
                rule_version=cfg.rule_version,
            )
        elif recent_drop and cur.z_m <= cfg.max_fall_z_m + 0.15:
            alert = FallAlert(
                active=False,
                state="monitoring_drop",
                confidence=round(min(0.45, 0.25 + 0.2 * (stillness_s / max(cfg.min_stillness_s, 1e-3))), 4),
                event_time_s=None,
                event_label="monitoring_rapid_descent",
                evidence={
                    "track_id": primary.track_id,
                    "current_z_m": round(cur.z_m, 4),
                    "recent_peak_z_m": round(peak_z, 4),
                    "height_drop_m": round(height_drop, 4),
                    "min_vz_mps": round(min_vz, 4),
                    "post_drop_stillness_s": round(stillness_s, 4),
                    "thresholds": cfg.to_dict(),
                },
                rule_version=cfg.rule_version,
            )
        else:
            alert = FallAlert(
                active=False,
                state="normal",
                confidence=0.0,
                event_time_s=None,
                event_label="none",
                evidence={
                    "track_id": primary.track_id,
                    "current_z_m": round(cur.z_m, 4),
                    "recent_peak_z_m": round(peak_z, 4),
                    "height_drop_m": round(height_drop, 4),
                    "min_vz_mps": round(min_vz, 4),
                    "post_drop_stillness_s": round(stillness_s, 4),
                    "thresholds": cfg.to_dict(),
                },
                rule_version=cfg.rule_version,
            )

        features = {
            "primary_track_id": primary.track_id,
            "estimated_height_m": round(cur.z_m, 4),
            "vertical_velocity_mps": round(cur.vz_mps, 4),
            "speed_3d_mps": round(cur.speed_3d_mps, 4),
            "height_drop_m": round(height_drop, 4),
            "post_drop_stillness_s": round(stillness_s, 4),
            "track_stability": round(track_stability, 4),
        }
        return alert, features

    @staticmethod
    def _select_primary_track(tracks: Sequence[_InternalTrack]) -> _InternalTrack | None:
        if not tracks:
            return None
        confirmed = [t for t in tracks if t.hits >= 2]
        pool = confirmed if confirmed else list(tracks)
        return max(pool, key=lambda t: (t.confidence, t.hits, -t.missed_frames))


class StationaryPhaseBranch:
    """Optional untouched-slow-time phase/breathing extraction branch (not a validated claim)."""

    def __init__(self, *, max_history: int = 200) -> None:
        self._times: deque[float] = deque(maxlen=max_history)
        self._phases: deque[float] = deque(maxlen=max_history)
        self._tracked_range_bin: int | None = None

    def reset(self) -> None:
        self._times.clear()
        self._phases.clear()
        self._tracked_range_bin = None

    def update(
        self,
        sim_time_s: float,
        range_cube_stationary: np.ndarray,
        range_axis_m: np.ndarray,
        wavelength_m: float,
        *,
        min_range_bin: int = 4,
    ) -> dict[str, Any]:
        """Extract slow-time complex phase from the UNTOUCHED stationary range branch."""
        # Coherent slow-time average on TX0/RX0 (preserves static target complex phase)
        profile = np.mean(range_cube_stationary[:, 0, 0, :], axis=0)
        if range_axis_m.size <= min_range_bin:
            return {
                "enabled": True,
                "status": "optional_experimental_branch_not_validated_claim",
                "tracked_range_m": None,
                "phase_rad": None,
                "displacement_mm": None,
            }
        if self._tracked_range_bin is None or self._tracked_range_bin >= profile.size:
            self._tracked_range_bin = min_range_bin + int(np.argmax(np.abs(profile[min_range_bin:])))

        z_val = complex(profile[self._tracked_range_bin])
        phase = math.atan2(z_val.imag, z_val.real)
        self._times.append(float(sim_time_s))
        self._phases.append(phase)

        unwrapped = np.unwrap(np.asarray(self._phases, dtype=np.float64))
        disp_m = (wavelength_m / (4.0 * math.pi)) * (unwrapped - unwrapped[0])
        return {
            "enabled": True,
            "status": "optional_experimental_branch_not_validated_claim",
            "source_branch": "untouched_stationary_complex_slow_time",
            "tracked_range_bin": int(self._tracked_range_bin),
            "tracked_range_m": round(float(range_axis_m[self._tracked_range_bin]), 4),
            "unwrapped_phase_rad": round(float(unwrapped[-1]), 5),
            "displacement_mm": round(float(disp_m[-1] * 1000.0), 4),
            "sample_count": len(self._phases),
        }
