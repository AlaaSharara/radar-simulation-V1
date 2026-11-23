"""Configurable 2D Ordered-Statistic CFAR (OS-CFAR) detector for Range-Doppler power maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OSCFARConfig:
    """Configuration for 2D OS-CFAR detection on Range-Doppler power maps."""

    guard_range: int = 2
    guard_doppler: int = 2
    train_range: int = 6
    train_doppler: int = 4
    rank_quantile: float = 0.75
    threshold_scale_db: float = 12.0
    min_range_bin: int = 3
    max_detections: int = 64
    local_max_suppress: bool = True
    min_power_linear: float = 1e-12

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard_range": self.guard_range,
            "guard_doppler": self.guard_doppler,
            "train_range": self.train_range,
            "train_doppler": self.train_doppler,
            "rank_quantile": self.rank_quantile,
            "threshold_scale_db": self.threshold_scale_db,
            "min_range_bin": self.min_range_bin,
            "max_detections": self.max_detections,
            "local_max_suppress": self.local_max_suppress,
        }


@dataclass(frozen=True)
class CFARPeak:
    doppler_bin: int
    range_bin: int
    doppler_bin_refined: float
    range_bin_refined: float
    power_linear: float
    noise_estimate_linear: float
    snr_db: float


def _parabolic_offset(y_prev: float, y_curr: float, y_next: float) -> float:
    """Sub-bin parabolic peak offset in ``[-0.5, +0.5]`` from 3 log-power samples."""
    denom = y_prev - 2.0 * y_curr + y_next
    if abs(denom) < 1e-12:
        return 0.0
    delta = 0.5 * (y_prev - y_next) / denom
    return max(-0.5, min(0.5, float(delta)))


def run_os_cfar_2d(
    rd_power: np.ndarray,
    config: OSCFARConfig | None = None,
    *,
    exclude_zero_doppler: bool = False,
) -> list[CFARPeak]:
    """Run 2D OS-CFAR with circular Doppler wrap and optional 3x3 non-maximum suppression."""
    cfg = OSCFARConfig() if config is None else config
    if rd_power.ndim != 2:
        raise ValueError(f"rd_power must be 2D [N_d, N_r], got shape {rd_power.shape}")

    n_d, n_r = rd_power.shape
    if n_r <= cfg.min_range_bin or n_d == 0:
        return []

    power = np.maximum(rd_power.astype(np.float64, copy=False), 0.0)
    alpha = 10.0 ** (cfg.threshold_scale_db / 10.0)
    dc_bin = n_d // 2

    # Candidate selection: 3x3 local maxima (wrapping circularly along Doppler axis)
    if cfg.local_max_suppress and n_d >= 3 and n_r >= 3:
        p_roll_up = np.roll(power, -1, axis=0)
        p_roll_dn = np.roll(power, 1, axis=0)
        is_max = (power >= p_roll_up) & (power >= p_roll_dn)
        # Compare along range neighbors
        is_max[:, 1:-1] &= (power[:, 1:-1] >= power[:, :-2]) & (power[:, 1:-1] >= power[:, 2:])
        is_max[:, 0] &= power[:, 0] >= power[:, 1]
        is_max[:, -1] &= power[:, -1] >= power[:, -2]
        # Diagonal neighbors for strict 3x3 peak
        is_max[:, 1:-1] &= (
            (power[:, 1:-1] >= p_roll_up[:, :-2])
            & (power[:, 1:-1] >= p_roll_up[:, 2:])
            & (power[:, 1:-1] >= p_roll_dn[:, :-2])
            & (power[:, 1:-1] >= p_roll_dn[:, 2:])
        )
    else:
        is_max = np.ones_like(power, dtype=bool)

    is_max[:, : cfg.min_range_bin] = False
    is_max &= power >= cfg.min_power_linear
    if exclude_zero_doppler and n_d > 1:
        is_max[dc_bin, :] = False

    cand_d, cand_r = np.nonzero(is_max)
    if cand_d.size == 0:
        return []

    gd = cfg.guard_doppler
    gr = cfg.guard_range
    td = cfg.train_doppler
    tr = cfg.train_range
    half_d = gd + td
    half_r = gr + tr

    # Prebuild relative (dd, dr) offsets of the hollow training ring
    ring_dd: list[int] = []
    ring_dr: list[int] = []
    for dd in range(-half_d, half_d + 1):
        for dr in range(-half_r, half_r + 1):
            if abs(dd) <= gd and abs(dr) <= gr:
                continue
            ring_dd.append(dd)
            ring_dr.append(dr)
    ring_dd_arr = np.asarray(ring_dd, dtype=np.int32)
    ring_dr_arr = np.asarray(ring_dr, dtype=np.int32)

    peaks: list[CFARPeak] = []
    for d_idx, r_idx in zip(cand_d.tolist(), cand_r.tolist()):
        cut_val = float(power[d_idx, r_idx])
        rr = r_idx + ring_dr_arr
        valid_r = (rr >= cfg.min_range_bin) & (rr < n_r)
        if not np.any(valid_r):
            continue
        dd = (d_idx + ring_dd_arr[valid_r]) % n_d
        rr_valid = rr[valid_r]
        train_vals = power[dd, rr_valid]
        if train_vals.size == 0:
            continue

        # Ordered-statistic rank selection (robust against multi-scatterer clutter)
        rank_idx = min(
            train_vals.size - 1,
            max(0, int(round(cfg.rank_quantile * (train_vals.size - 1)))),
        )
        z_os = float(np.partition(train_vals, rank_idx)[rank_idx])
        noise_est = max(z_os, cfg.min_power_linear)
        if cut_val > alpha * noise_est:
            snr_db = 10.0 * math.log10(cut_val / noise_est)

            # Sub-bin parabolic interpolation in log-power
            d_prev = (d_idx - 1) % n_d
            d_next = (d_idx + 1) % n_d
            d_off = _parabolic_offset(
                math.log(max(float(power[d_prev, r_idx]), 1e-30)),
                math.log(max(cut_val, 1e-30)),
                math.log(max(float(power[d_next, r_idx]), 1e-30)),
            ) if n_d >= 3 else 0.0
            if 0 < r_idx < n_r - 1:
                r_off = _parabolic_offset(
                    math.log(max(float(power[d_idx, r_idx - 1]), 1e-30)),
                    math.log(max(cut_val, 1e-30)),
                    math.log(max(float(power[d_idx, r_idx + 1]), 1e-30)),
                )
            else:
                r_off = 0.0

            peaks.append(
                CFARPeak(
                    doppler_bin=d_idx,
                    range_bin=r_idx,
                    doppler_bin_refined=float(d_idx) + d_off,
                    range_bin_refined=float(r_idx) + r_off,
                    power_linear=cut_val,
                    noise_estimate_linear=noise_est,
                    snr_db=snr_db,
                )
            )

    peaks.sort(key=lambda p: p.snr_db, reverse=True)
    return peaks[: cfg.max_detections]
