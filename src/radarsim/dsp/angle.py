"""2D virtual-array angle estimation (azimuth & elevation) with explicit aperture limitations.

Reads the virtual element positions strictly from ``RadarProfile.antenna_array``
so another vendor array substitutes cleanly. Clearly labels the dense angle grid
(configured ``angle_fft_azimuth_size x angle_fft_elevation_size``) as
interpolation rather than physical resolution improvement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from radarsim.config import RadarProfile, compute_derived_metrics


@dataclass(frozen=True)
class ApertureLimitations:
    """Explicit physical resolution boundaries of the configured virtual array."""

    layout: str
    virtual_channels: int
    azimuth_elements: int
    elevation_elements: int
    azimuth_aperture_lambda: float
    elevation_aperture_lambda: float
    azimuth_beamwidth_3db_deg: float
    elevation_beamwidth_3db_deg: float
    azimuth_interpolation_bins: int
    elevation_interpolation_bins: int
    zero_padding_note: str
    imaging_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "virtual_channels": self.virtual_channels,
            "azimuth_elements": self.azimuth_elements,
            "elevation_elements": self.elevation_elements,
            "azimuth_aperture_lambda": self.azimuth_aperture_lambda,
            "elevation_aperture_lambda": self.elevation_aperture_lambda,
            "azimuth_beamwidth_3db_deg": self.azimuth_beamwidth_3db_deg,
            "elevation_beamwidth_3db_deg": self.elevation_beamwidth_3db_deg,
            "azimuth_interpolation_bins": self.azimuth_interpolation_bins,
            "elevation_interpolation_bins": self.elevation_interpolation_bins,
            "zero_padding_note": self.zero_padding_note,
            "imaging_disclaimer": self.imaging_disclaimer,
        }


class VirtualArrayAngleEstimator:
    """2D matched-beamformer / spatial-DFT angle estimator for any configured planar virtual array."""

    def __init__(self, radar_profile: RadarProfile) -> None:
        self.radar_profile = radar_profile
        self.derived = compute_derived_metrics(radar_profile)
        arr = radar_profile.antenna_array
        wf = radar_profile.waveform

        # Virtual array coordinates in wavelengths (azimuth along u=x, elevation along v=y)
        raw_vpos = np.asarray(radar_profile.virtual_positions_lambda(), dtype=np.float64)
        if arr.coordinate_units != "wavelengths":
            raw_vpos = raw_vpos / self.derived.wavelength_m
        self.vx_lambda = raw_vpos[:, 0]  # shape (N_virt,)
        self.vz_lambda = raw_vpos[:, 1]  # shape (N_virt,)

        # Interpolated angle grids matching configured FFT sizes over +/- FoV (capped at +/- 80 deg)
        az_max = min(math.radians(80.0), 0.55 * math.radians(radar_profile.rf.antenna_pattern.azimuth_fov_3db_deg))
        el_max = min(math.radians(80.0), 0.55 * math.radians(radar_profile.rf.antenna_pattern.elevation_fov_3db_deg))
        self.az_grid_rad = np.linspace(-az_max, az_max, wf.angle_fft_azimuth_size, dtype=np.float64)
        self.el_grid_rad = np.linspace(-el_max, el_max, wf.angle_fft_elevation_size, dtype=np.float64)

        # Precompute conjugate steering matrix A_conj[n_el * n_az, n_virt]
        # Consistent with T3/T4: signal phase is -2*pi*(vx*cos(el)*sin(az) + vz*sin(el))
        # so matched correlation multiplies by exp(+j*2*pi*(vx*cos(el)*sin(az) + vz*sin(el))).
        el_mesh, az_mesh = np.meshgrid(self.el_grid_rad, self.az_grid_rad, indexing="ij")
        u_x = np.cos(el_mesh) * np.sin(az_mesh)  # (N_el, N_az)
        u_z = np.sin(el_mesh)                    # (N_el, N_az)
        phase = 2.0 * math.pi * (
            u_x.reshape(-1, 1) * self.vx_lambda[None, :]
            + u_z.reshape(-1, 1) * self.vz_lambda[None, :]
        )
        self._steering_conj = np.exp(1j * phase) / float(len(self.vx_lambda))  # (N_el*N_az, N_virt)

        # Precompute 1D zero-elevation azimuth steering matrix [N_az, N_virt] for fast Range-Azimuth B-scan
        az_phase = 2.0 * math.pi * (
            np.sin(self.az_grid_rad)[:, None] * self.vx_lambda[None, :]
        )
        self._az_steering_conj = np.exp(1j * az_phase) / float(len(self.vx_lambda))

        self.limitations = self._compute_limitations()

    def _compute_limitations(self) -> ApertureLimitations:
        wf = self.radar_profile.waveform
        ux_unique = np.unique(np.round(self.vx_lambda, 4))
        uz_unique = np.unique(np.round(self.vz_lambda, 4))
        n_az_el = int(len(ux_unique))
        n_el_el = int(len(uz_unique))
        ap_az = float(ux_unique.max() - ux_unique.min()) if n_az_el > 1 else 0.0
        ap_el = float(uz_unique.max() - uz_unique.min()) if n_el_el > 1 else 0.0

        # Effective aperture including element spacing: D_eff = N * d_lambda; 3 dB beamwidth ~ 0.886 / D_eff rad
        d_eff_az = ap_az + (ap_az / (n_az_el - 1) if n_az_el > 1 else 0.5)
        d_eff_el = ap_el + (ap_el / (n_el_el - 1) if n_el_el > 1 else 0.5)
        bw_az_deg = math.degrees(0.886 / max(d_eff_az, 0.25))
        bw_el_deg = math.degrees(0.886 / max(d_eff_el, 0.25))

        return ApertureLimitations(
            layout=self.radar_profile.antenna_array.layout,
            virtual_channels=len(self.vx_lambda),
            azimuth_elements=n_az_el,
            elevation_elements=n_el_el,
            azimuth_aperture_lambda=ap_az,
            elevation_aperture_lambda=ap_el,
            azimuth_beamwidth_3db_deg=round(bw_az_deg, 2),
            elevation_beamwidth_3db_deg=round(bw_el_deg, 2),
            azimuth_interpolation_bins=wf.angle_fft_azimuth_size,
            elevation_interpolation_bins=wf.angle_fft_elevation_size,
            zero_padding_note=(
                f"Angle grid ({wf.angle_fft_azimuth_size}x{wf.angle_fft_elevation_size}) is "
                "zero-padded / dense steering interpolation only; it does NOT improve physical "
                "Rayleigh angular resolution."
            ),
            imaging_disclaimer=(
                f"A {n_az_el}x{n_el_el} ({len(self.vx_lambda)}-virtual-channel) aperture has "
                f"coarse ~{bw_az_deg:.1f} deg azimuth and ~{bw_el_deg:.1f} deg elevation beamwidth; "
                "output is a sparse radar point cloud, not a camera-grade dense 3D human image."
            ),
        )

    def spectrum_2d(self, virtual_snapshot: np.ndarray) -> np.ndarray:
        """Compute 2D azimuth-elevation power spectrum ``[N_el, N_az]`` from a ``(tx, rx)`` or ``(N_virt,)`` vector."""
        vec = np.asarray(virtual_snapshot, dtype=np.complex128).reshape(-1)
        resp = self._steering_conj @ vec
        power = (resp.real * resp.real + resp.imag * resp.imag)
        return power.reshape(len(self.el_grid_rad), len(self.az_grid_rad))

    def compute_range_azimuth_power(
        self,
        rd_cube: np.ndarray,
        *,
        exclude_zero_doppler: bool = True,
    ) -> np.ndarray:
        """Compute Range-Azimuth power map ``[N_r, N_az]`` from each range bin's dominant moving Doppler cell."""
        n_d, n_tx, n_rx, n_r = rd_cube.shape
        n_virt = n_tx * n_rx
        flat = rd_cube.reshape(n_d, n_virt, n_r)
        if exclude_zero_doppler and n_d > 2:
            mask = np.ones(n_d, dtype=bool)
            mask[n_d // 2] = False
            flat = flat[mask]

        # Select dominant moving Doppler bin per range bin -> single (N_r, N_virt) @ (N_virt, N_az) matmul
        doppler_energy = np.sum(flat.real * flat.real + flat.imag * flat.imag, axis=1)  # (N_dm, N_r)
        best_d = np.argmax(doppler_energy, axis=0)  # (N_r,)
        peak_snaps = flat[best_d, :, np.arange(n_r)]  # (N_r, N_virt)
        resp = peak_snaps @ self._az_steering_conj.T  # (N_r, N_az)
        return resp.real * resp.real + resp.imag * resp.imag

    def estimate_angle(self, virtual_snapshot: np.ndarray) -> tuple[float, float, float]:
        """Estimate ``(azimuth_rad, elevation_rad, peak_to_mean_ratio)`` for one detection snapshot."""
        spec = self.spectrum_2d(virtual_snapshot)
        flat_idx = int(np.argmax(spec))
        el_idx, az_idx = divmod(flat_idx, spec.shape[1])

        # Parabolic sub-grid interpolation for smooth angle estimates
        az_val = float(self.az_grid_rad[az_idx])
        if 0 < az_idx < len(self.az_grid_rad) - 1:
            p0, p1, p2 = float(spec[el_idx, az_idx - 1]), float(spec[el_idx, az_idx]), float(spec[el_idx, az_idx + 1])
            denom = p0 - 2.0 * p1 + p2
            if abs(denom) > 1e-18:
                delta = max(-0.5, min(0.5, 0.5 * (p0 - p2) / denom))
                step = float(self.az_grid_rad[1] - self.az_grid_rad[0])
                az_val += delta * step

        el_val = float(self.el_grid_rad[el_idx])
        if 0 < el_idx < len(self.el_grid_rad) - 1:
            p0, p1, p2 = float(spec[el_idx - 1, az_idx]), float(spec[el_idx, az_idx]), float(spec[el_idx + 1, az_idx])
            denom = p0 - 2.0 * p1 + p2
            if abs(denom) > 1e-18:
                delta = max(-0.5, min(0.5, 0.5 * (p0 - p2) / denom))
                step = float(self.el_grid_rad[1] - self.el_grid_rad[0])
                el_val += delta * step

        peak_val = float(spec[el_idx, az_idx])
        mean_val = float(np.mean(spec)) + 1e-30
        return az_val, el_val, peak_val / mean_val
