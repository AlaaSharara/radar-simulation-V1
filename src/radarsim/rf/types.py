"""RF propagation data contracts following ``INTERFACES.md`` §5.

Never serializes Python ``complex`` values directly to JSON; stores scalar
``amplitude_linear``, ``amplitude_phase_rad``, ``reflection_coeff_real``, and
``reflection_coeff_imag`` fields alongside an internal ``complex_amplitude``
property.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass, field
import math
from typing import Any

Vec3 = tuple[float, float, float]

VALID_PATH_TYPES: frozenset[str] = frozenset(
    {"los", "specular_reflection", "diffuse_scatter", "transmitted"}
)


@dataclass(frozen=True)
class PointTarget:
    """Analytical point scatterer for direct verification and synthetic testing."""

    target_id: str
    position_m: Vec3
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    rcs_m2: float = 0.03162277660168379  # -15 dBsm default
    provenance: str = "analytical_point_target"


@dataclass(frozen=True)
class RFPropagationConfig:
    """Configurable switches and bounds for geometric-optics + stochastic diffuse RF model."""

    polarization: str = "TE"  # "TE" or "TM"
    enable_los: bool = True
    enable_first_order_reflections: bool = True
    enable_static_reflections: bool = True
    enable_target_multipath: bool = True
    enable_second_order_reflections: bool = False
    max_second_order_paths: int = 24
    enable_transmission: bool = True
    max_transmission_slabs: int = 2
    enable_diffuse_scattering: bool = True
    diffuse_points_per_surface: int = 2
    max_diffuse_surfaces: int = 12
    enable_human_body_occlusion: bool = True
    min_path_amplitude_linear: float = 1e-12
    preview_mode: bool = False

    def __post_init__(self) -> None:
        if self.polarization not in ("TE", "TM"):
            raise ValueError(f"polarization must be 'TE' or 'TM', got {self.polarization!r}")
        if self.max_second_order_paths < 0:
            raise ValueError("max_second_order_paths must be >= 0")
        if self.max_transmission_slabs < 0:
            raise ValueError("max_transmission_slabs must be >= 0")


@dataclass(frozen=True)
class RFPath:
    """Single propagation path between one TX and one RX element (``INTERFACES.md`` §5)."""

    path_id: str
    tx_id: int
    rx_id: int
    target_id: str | None
    surface_id: str | None
    path_type: str
    vertices_m: tuple[Vec3, ...]
    tx_leg_length_m: float
    rx_leg_length_m: float
    total_path_length_m: float
    delay_s: float
    radial_velocity_mps: float
    doppler_hz: float
    phase_rad: float
    path_loss_db: float
    reflection_coeff_real: float
    reflection_coeff_imag: float
    rcs_m2: float | None
    amplitude_linear: float
    amplitude_phase_rad: float
    model_flags: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path_type not in VALID_PATH_TYPES:
            raise ValueError(
                f"Invalid path_type {self.path_type!r}; expected one of {sorted(VALID_PATH_TYPES)}"
            )

    @property
    def complex_amplitude(self) -> complex:
        """Internal complex channel transfer amplitude ``amplitude_linear * exp(j * amplitude_phase_rad)``."""
        return cmath.rect(self.amplitude_linear, self.amplitude_phase_rad)

    @property
    def reflection_coeff(self) -> complex:
        return complex(self.reflection_coeff_real, self.reflection_coeff_imag)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation (no Python complex objects)."""
        return {
            "path_id": self.path_id,
            "tx_id": self.tx_id,
            "rx_id": self.rx_id,
            "target_id": self.target_id,
            "surface_id": self.surface_id,
            "path_type": self.path_type,
            "vertices_m": [list(v) for v in self.vertices_m],
            "tx_leg_length_m": self.tx_leg_length_m,
            "rx_leg_length_m": self.rx_leg_length_m,
            "total_path_length_m": self.total_path_length_m,
            "delay_s": self.delay_s,
            "radial_velocity_mps": self.radial_velocity_mps,
            "doppler_hz": self.doppler_hz,
            "phase_rad": self.phase_rad,
            "path_loss_db": self.path_loss_db,
            "reflection_coeff_real": self.reflection_coeff_real,
            "reflection_coeff_imag": self.reflection_coeff_imag,
            "rcs_m2": self.rcs_m2,
            "amplitude_linear": self.amplitude_linear,
            "amplitude_phase_rad": self.amplitude_phase_rad,
            "model_flags": dict(self.model_flags),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class PathSet:
    """All computed RF propagation paths for one simulation frame (``INTERFACES.md`` §5)."""

    frame_id: int
    sim_time_s: float
    paths: list[RFPath]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def paths_by_type(self, path_type: str) -> list[RFPath]:
        return [p for p in self.paths if p.path_type == path_type]

    def paths_for_pair(self, tx_id: int, rx_id: int) -> list[RFPath]:
        return [p for p in self.paths if p.tx_id == tx_id and p.rx_id == rx_id]

    def to_dict(self, *, max_paths: int | None = None) -> dict[str, Any]:
        selected = self.paths if max_paths is None else self.paths[:max_paths]
        return {
            "frame_id": self.frame_id,
            "sim_time_s": self.sim_time_s,
            "paths": [p.to_dict() for p in selected],
            "diagnostics": dict(self.diagnostics),
        }
